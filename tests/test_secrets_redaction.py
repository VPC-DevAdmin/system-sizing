"""Secrets never reach a log line, an exception or a persisted failure
reason (improvement plan D1).

Every launcher hands the HF token to its container as
``-e HF_TOKEN=<value>`` and then logged the whole argv, so the token
sat in every engine log and, through the optimizer's failure_reason,
in run.json. These tests set a recognisable token and assert it is
absent from everything the launch path writes.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from simulator.config import EngineConfig
from simulator.engines.base import redact_argv

TOKEN = "hf_SECRETvalue1234567890"
REPO = Path(__file__).parent.parent


def test_redact_argv_masks_secret_values_and_keeps_names():
    cmd = ["docker", "run", "-e", f"HF_TOKEN={TOKEN}",
           "-e", "VLLM_LOGGING_LEVEL=DEBUG",
           "--env", f"OPENAI_API_KEY={TOKEN}",
           f"--env=AWS_SECRET_ACCESS_KEY={TOKEN}",
           f"-eDB_PASSWORD={TOKEN}",
           f"HUGGING_FACE_HUB_TOKEN={TOKEN}",
           "image", "--model", "org/M"]
    text = redact_argv(cmd)
    assert TOKEN not in text
    # The names stay: that a token WAS passed is diagnostic.
    for name in ("HF_TOKEN=***", "OPENAI_API_KEY=***",
                 "--env=AWS_SECRET_ACCESS_KEY=***", "-eDB_PASSWORD=***",
                 "HUGGING_FACE_HUB_TOKEN=***"):
        assert name in text
    # Non-secret env and everything else is untouched.
    assert "VLLM_LOGGING_LEVEL=DEBUG" in text
    assert text.endswith("image --model org/M")


def test_redact_argv_leaves_ordinary_arguments_alone():
    cmd = ["python", "-m", "sglang.launch_server", "--model-path", "org/M",
           "--port", "9100"]
    assert redact_argv(cmd) == " ".join(cmd)


def test_docker_replica_launch_log_carries_no_token(monkeypatch, tmp_path,
                                                   caplog):
    from simulator.engines import docker_replica as dr
    from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine

    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="cid0123456789\n",
                                           stderr="")

    monkeypatch.setattr(dr.subprocess, "run", fake_run)
    monkeypatch.setattr(dr.DockerReplicaEngine, "_spawn_log_streamer",
                        lambda self, cid, prefix: None)
    eng = VllmCudaMultiEngine(EngineConfig(
        type="vllm_cuda_multi", model_id="org/M", port=9100,
        replica_devices=[[0]], docker_volumes={}))
    eng._log_path = tmp_path / "engine.log"
    with caplog.at_level(logging.INFO, logger="simulator.engines"):
        eng._launch_replica(0, [0], "run1")
    # The token DID go to docker...
    assert any(f"HF_TOKEN={TOKEN}" in c for c in seen)
    # ...and did NOT go to the log.
    assert TOKEN not in caplog.text
    assert "HF_TOKEN=***" in caplog.text


def test_vllm_cuda_launch_log_carries_no_token(monkeypatch, tmp_path, caplog):
    from simulator.engines import vllm_cuda as vc

    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(vc.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(vc, "remove_stale_engine_containers", lambda: None)

    def fake_run(cmd, **kw):
        raise subprocess.CalledProcessError(125, cmd, stderr="daemon said no")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    eng = vc.VllmCudaEngine(EngineConfig(
        type="vllm_cuda", model_id="org/M", port=9100, docker_volumes={}))
    with caplog.at_level(logging.INFO, logger="simulator.engines"), \
            pytest.raises(RuntimeError) as e:
        eng.launch(log_dir=tmp_path)
    assert TOKEN not in caplog.text
    assert TOKEN not in str(e.value)


@pytest.fixture()
def optimizer():
    spec = importlib.util.spec_from_file_location(
        "engine_optimizer", REPO / "scripts" / "engine_optimizer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine_optimizer"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("engine_optimizer", None)


def test_optimizer_failure_reason_carries_no_token(optimizer, monkeypatch,
                                                   tmp_path):
    """The failure reason is written verbatim into run.json."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, stdout="",
                                           stderr="no such image")

    monkeypatch.setattr(optimizer.subprocess, "run", fake_run)
    cfg = optimizer.EngineConfig(
        name="c", description="d",
        replicas=[optimizer.ReplicaSpec(name="vllm-s0", port=8000,
                                        gpus="device=0")])
    with pytest.raises(RuntimeError) as e:
        optimizer.docker_launch(cfg, cfg.replicas[0])
    assert any(f"HF_TOKEN={TOKEN}" in c for c in seen)
    assert TOKEN not in str(e.value)
    assert "HF_TOKEN=***" in str(e.value)
