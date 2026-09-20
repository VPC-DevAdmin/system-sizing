"""Run lifecycle gaps (improvement plan D3): a stop during launch
must stop everything that was created, a hung docker run must be
cleaned up by name, a server that never answered must be terminated,
and the stale sweep must never touch a container capsim does not own.
No Docker: subprocess is faked and the calls are inspected."""

from __future__ import annotations

import os
import subprocess

import pytest

from simulator.config import EngineConfig
from simulator.engines import docker_replica as dr
from simulator.engines.base import Engine
from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine


class _FakeDocker:
    """Records every docker call; ``docker run`` returns a container id
    derived from the --name, everything else succeeds silently."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def run(self, cmd, **kw):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            name = cmd[cmd.index("--name") + 1]
            return subprocess.CompletedProcess(cmd, 0, stdout=f"cid-{name}\n",
                                               stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def of(self, verb: str) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["docker", verb]]


@pytest.fixture()
def docker(monkeypatch, tmp_path):
    fake = _FakeDocker()
    monkeypatch.setattr(dr.subprocess, "run", fake.run)
    monkeypatch.setattr(dr.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(dr.DockerReplicaEngine, "_spawn_log_streamer",
                        lambda self, cid, prefix: None)
    return fake


def _engine(n: int = 3) -> VllmCudaMultiEngine:
    return VllmCudaMultiEngine(EngineConfig(
        type="vllm_cuda_multi", model_id="org/M", port=9100,
        replica_devices=[[i] for i in range(n)], docker_volumes={},
        startup_timeout_s=5))


def test_shutdown_arriving_between_docker_run_and_append_removes_that_container(
    docker, tmp_path, monkeypatch,
):
    """asyncio.to_thread(engine.launch) cancelled: the thread keeps
    going. A container whose docker run returned after shutdown() ran
    was invisible to it and used to be orphaned."""
    eng = _engine(3)
    real_build = VllmCudaMultiEngine.build_replica_command

    def build(self, index, devices, name):
        if index == 1:
            # The caller's shutdown() lands while replica 1 is being
            # assembled -- after the flag check, before docker run.
            self.shutdown()
        return real_build(self, index, devices, name)
    monkeypatch.setattr(VllmCudaMultiEngine, "build_replica_command", build)

    with pytest.raises(RuntimeError, match="cancelled"):
        eng.launch(log_dir=tmp_path)
    runs = docker.of("run")
    assert len(runs) == 2                       # replica 2 never started
    # Replica 0 was stopped by shutdown(); replica 1, created after it,
    # was removed on the spot by the launch thread itself.
    assert ["docker", "stop", "-t", "30", "cid-vllm-r0-" + eng._run_id] \
        in docker.calls
    assert ["docker", "rm", "-f", "cid-vllm-r1-" + eng._run_id] \
        in docker.calls
    assert eng._replicas == []


def test_shutdown_before_the_next_docker_run_prevents_it(docker, tmp_path,
                                                         monkeypatch):
    eng = _engine(3)

    def streamer(self, cid, prefix):
        if cid.endswith("-r0-" + self._run_id):
            self.shutdown()                     # while r0 is unappended
        return None
    monkeypatch.setattr(dr.DockerReplicaEngine, "_spawn_log_streamer",
                        streamer)
    with pytest.raises(RuntimeError, match="cancelled"):
        eng.launch(log_dir=tmp_path)
    assert len(docker.of("run")) == 1
    assert ["docker", "rm", "-f", "cid-vllm-r0-" + eng._run_id] in docker.calls
    assert eng._replicas == []


def test_shutdown_during_the_health_wait_stops_the_wait(docker, tmp_path,
                                                        monkeypatch):
    eng = _engine(1)
    ticks = []

    def inspect_or_stop(cmd, **kw):
        if cmd[:2] == ["docker", "inspect"]:
            ticks.append(1)
            if len(ticks) == 2:
                eng.shutdown()
            return subprocess.CompletedProcess(cmd, 0, stdout="true\n",
                                               stderr="")
        return docker.run(cmd, **kw)
    monkeypatch.setattr(dr.subprocess, "run", inspect_or_stop)
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="cancelled"):
        eng.launch(log_dir=tmp_path)
    assert len(ticks) == 2
    assert eng._replicas == []


def test_a_hung_docker_run_is_removed_by_name(docker, tmp_path, monkeypatch):
    """docker run that never returns leaves a container with no id to
    remove it by. The name was chosen before the run, so it can."""
    eng = _engine(1)

    def hang(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            docker.calls.append(list(cmd))
            raise subprocess.TimeoutExpired(cmd, 120)
        return docker.run(cmd, **kw)
    monkeypatch.setattr(dr.subprocess, "run", hang)
    with pytest.raises(RuntimeError, match="did not return"):
        eng.launch(log_dir=tmp_path)
    name = "vllm-r0-" + eng._run_id
    assert ["docker", "rm", "-f", name] in docker.calls


def test_stale_sweep_is_anchored_to_capsim_prefixes(docker):
    """``name=vllm-`` is an unanchored regex and matched a user's
    my-vllm-dev; the sweep removed it."""
    filters = dr.container_name_filter()
    assert filters == [f"name=^/?{p}" for p in dr.CAPSIM_CONTAINER_PREFIXES]
    dr.remove_stale_engine_containers()
    used = [c[c.index("--filter") + 1] for c in docker.of("ps")]
    assert used == filters
    assert all(f.startswith("name=^") for f in used)
    import re
    for prefix, f in zip(dr.CAPSIM_CONTAINER_PREFIXES, used, strict=True):
        pat = re.compile(f[len("name="):])
        # Docker reports names with a leading slash; either form matches.
        assert pat.search(f"/{prefix}r0-abc") and pat.search(f"{prefix}r0-abc")
        assert not pat.search(f"/my-{prefix}dev")


def test_log_streamer_is_line_buffered(tmp_path, monkeypatch):
    """Into a file sed block-buffers, so the engine's last lines -- the
    ones a startup failure is diagnosed from -- sat in its buffer."""
    seen = {}

    class P:
        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd
    monkeypatch.setattr(dr.subprocess, "Popen", P)
    eng = _engine(1)
    eng._log_path = tmp_path / "engine.log"
    eng._spawn_log_streamer("cid", prefix="[r0] ")
    assert "sed -u " in seen["cmd"]


# ── Host-process engines (base.Engine) ────────────────────────────────


class _Sleeper(Engine):
    def _build_command(self):
        return ["sleep", "60"]

    def _build_env(self):
        return dict(os.environ)


def test_health_check_failure_terminates_the_process_group(tmp_path,
                                                           monkeypatch):
    """A server that came up but never answered held its port and its
    cores into the next launch."""
    from simulator.engines import base as base_mod

    spawned = []
    real_popen = base_mod.subprocess.Popen

    def record(*a, **kw):
        p = real_popen(*a, **kw)
        spawned.append(p)
        return p
    monkeypatch.setattr(base_mod.subprocess, "Popen", record)
    eng = _Sleeper(EngineConfig(type="vllm", model_id="org/M", port=1,
                                startup_timeout_s=1))
    monkeypatch.setattr(eng, "health_check", lambda: False)
    with pytest.raises(TimeoutError):
        eng.launch(log_dir=tmp_path)
    assert len(spawned) == 1
    assert spawned[0].poll() is not None        # terminated and reaped
    assert eng._proc is None
    assert eng._log_file is None                # closed, not leaked


def test_shutdown_survives_a_process_group_that_already_left(tmp_path,
                                                             monkeypatch):
    from simulator.engines import base as base_mod

    eng = _Sleeper(EngineConfig(type="vllm", model_id="org/M", port=1,
                                startup_timeout_s=1))
    monkeypatch.setattr(eng, "health_check", lambda: True)
    eng.launch(log_dir=tmp_path)

    def gone(pid):
        raise ProcessLookupError(pid)
    monkeypatch.setattr(base_mod.os, "getpgid", gone)
    monkeypatch.setattr(eng._proc, "wait", lambda timeout=None: 0)
    eng.shutdown()                               # no ProcessLookupError
    assert eng._proc is None
