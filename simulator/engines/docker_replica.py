"""Shared machinery for whole-box, N-replica Docker engines.

``vllm_cuda_multi`` proved the shape: N independent GPU-pinned
containers, one per replica, sticky-routed by the pool so multi-turn
conversations keep prefix-cache locality. TensorRT-LLM needs exactly
the same lifecycle — launch N containers, wait for each to answer,
aggregate counters across them, stream every log into one file — and
differs only in the server binary it runs and the dialect its
metrics endpoint speaks.

Keeping that lifecycle in one place is not tidiness. The launch path
accumulated hard-won corrections (embedded quoting for multi-device
``--gpus``, the stale-container sweep, reading the engine's OWN
exception out of the log instead of "see the log"), and a second copy
would silently miss every one of them.

Subclasses supply:
  * ``ENGINE_NAME``      — log filename and container-name prefix
  * ``build_replica_command(index, devices, name)`` — the docker argv
  * ``_ready_url(port)`` / ``_metrics_url(port)``   — optional
  * ``parse_metrics(text)``                        — optional
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from .base import Engine

log = logging.getLogger(__name__)

# Container-name prefixes capsim owns exclusively. Anything matching
# is fair game for the pre-launch sweep.
CAPSIM_CONTAINER_PREFIXES = ("vllm-", "trtllm-")


def gpus_arg_for(device_ids: list[int]) -> str:
    """Docker ``--gpus`` value for a device list. Docker parses the
    value as CSV, so multi-device lists need EMBEDDED quotes
    (`"device=0,1"`, quote characters included) or the daemon reads
    device=0 + count=1 and refuses."""
    arg = "device=" + ",".join(str(i) for i in device_ids)
    return f'"{arg}"' if len(device_ids) > 1 else arg


def remove_stale_engine_containers() -> None:
    """rm -f any leftover capsim engine container before launching.

    A hard-killed serve leaves its engine containers RUNNING — the
    next launch then dies with "address already in use" while the
    health check happily gets 200s from the OLD engine on the same
    port, and every request 404s against the wrong model. These name
    prefixes are exclusively capsim-owned, and benchmark launches are
    mutually exclusive with the optimizer, so removal here is safe.
    """
    for prefix in CAPSIM_CONTAINER_PREFIXES:
        try:
            res = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name={prefix}"],
                capture_output=True, text=True, timeout=30,
            )
            cids = res.stdout.split()
            if cids:
                log.warning(
                    "removing %d stale %s* container(s) from a previous "
                    "run before launch", len(cids), prefix,
                )
                subprocess.run(["docker", "rm", "-f", *cids],
                               capture_output=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            log.debug("stale-container sweep for %s failed: %s", prefix, e)


def aggregate_replica_metrics(per_replica: list[dict]) -> dict[str, float]:
    """Whole-box view of N replicas' metrics: counters summed
    (including the token counters the prefill/decode rates derive
    from), gauges averaged, hit rate recomputed from the sums."""
    if not per_replica:
        return {}
    agg: dict[str, float] = {}
    for k in ("num_running", "queue_depth", "prefix_cache_hits",
              "prefix_cache_queries", "prompt_tokens_total",
              "generation_tokens_total", "preemptions_total"):
        vals = [m[k] for m in per_replica if k in m]
        if vals:
            agg[k] = sum(vals)
    # Gauges: averaged. A rate that arrives ready-made (TensorRT-LLM's
    # cacheHitRate, SGLang's cache_hit_rate) would otherwise be dropped
    # on the floor for any multi-replica engine; when the underlying
    # counters are also present the recompute below overrides it.
    for k in ("kv_cache_used_pct", "prefix_cache_hit_rate"):
        vals = [m[k] for m in per_replica if k in m]
        if vals:
            agg[k] = sum(vals) / len(vals)
    if agg.get("prefix_cache_queries"):
        agg["prefix_cache_hit_rate"] = (
            agg.get("prefix_cache_hits", 0) / agg["prefix_cache_queries"]
        )
    return agg


class DockerReplicaEngine(Engine):
    """N GPU-pinned server containers, sticky-routed by the pool."""

    # Subclasses override: names the log file and the containers.
    ENGINE_NAME = "docker"

    def __init__(self, engine_config):
        super().__init__(engine_config)
        # (index, devices, port, container_id, streamer) per replica
        self._replicas: list[tuple[int, list[int], int, str,
                                   Optional[subprocess.Popen]]] = []

    # ── Subclass interface ────────────────────────────────────────────

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        raise NotImplementedError

    def _ready_url(self, port: int) -> str:
        return f"http://{self.cfg.host}:{port}/v1/models"

    def _metrics_url(self, port: int) -> str:
        return f"http://{self.cfg.host}:{port}/metrics"

    def parse_metrics(self, text: str) -> dict[str, float]:
        return self._parse_prometheus(text)

    # ── Shape ─────────────────────────────────────────────────────────

    def _device_groups(self) -> list[list[int]]:
        groups = getattr(self.cfg, "replica_devices", None)
        if not groups or not all(isinstance(g, (list, tuple)) and g
                                 for g in groups):
            raise RuntimeError(
                f"{self.cfg.type} needs engine.replica_devices — a list "
                "of GPU-id lists, one per replica, e.g. [[0],[1],[2]]"
            )
        return [[int(d) for d in g] for g in groups]

    def _port(self, index: int) -> int:
        return self.cfg.port + index

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}/v1"

    @property
    def replica_urls(self) -> list[str]:
        n = len(self._device_groups())
        return [f"http://{self.cfg.host}:{self._port(i)}/v1"
                for i in range(n)]

    # ── Docker argv helpers (shared by every subclass) ────────────────

    def _mount_args(self) -> list[str]:
        """Volume, token and env arguments common to every engine
        image: the HF cache the weights live in, plus any hub token."""
        cfg = self.cfg
        out: list[str] = []
        mounted = set()
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            if not Path(host_path).exists():
                continue
            out += ["-v", f"{host_path}:{container_path}"]
            mounted.add(container_path)
        if "/root/.cache/huggingface" not in mounted:
            from ..models import hf_cache_dir
            cache = hf_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            out += ["-v", f"{cache}:/root/.cache/huggingface"]
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(var):
                out += ["-e", f"{var}={os.environ[var]}"]
        for k, v in (cfg.docker_extra_env or {}).items():
            out += ["-e", f"{k}={v}"]
        return out

    # ── Lifecycle ─────────────────────────────────────────────────────

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._replicas:
            raise RuntimeError("Engine already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                f"docker not found on PATH; {self.cfg.type} runs its "
                "image in Docker (needs nvidia-container-toolkit)."
            )
        groups = self._device_groups()
        flat = [d for g in groups for d in g]
        if len(set(flat)) != len(flat):
            raise RuntimeError(
                f"replica_devices assigns a GPU twice: {groups}"
            )
        remove_stale_engine_containers()

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        self._log_path = log_dir / f"engine_{self.ENGINE_NAME}_{run_id}.log"

        try:
            log.info("Starting %d %s replicas on devices %s",
                     len(groups), self.ENGINE_NAME, groups)
            for i, devices in enumerate(groups):
                self._launch_replica(i, devices, run_id)
            for i, _devices, port, cid, _ in self._replicas:
                self._wait_for_replica_ready(i, port, cid)
            log.info("All replicas ready: %s", ", ".join(self.replica_urls))
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        for i, _devices, _port, cid, streamer in self._replicas:
            log.info("Stopping replica %d (%s)", i, cid[:12])
            try:
                subprocess.run(["docker", "stop", "-t", "30", cid],
                               capture_output=True, timeout=45)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", cid],
                               capture_output=True)
            if streamer is not None:
                try:
                    streamer.terminate()
                    try:
                        streamer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        streamer.kill()
                except Exception:  # noqa: BLE001
                    pass
        self._replicas = []

    def _launch_replica(self, index: int, devices: list[int],
                        run_id: str) -> None:
        name = f"{self.ENGINE_NAME.split('_')[0]}-r{index}-{run_id}"
        cmd = self.build_replica_command(index, devices, name)
        log.info("docker run r%d: %s", index, " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            hint = ""
            if "could not select device driver" in stderr:
                hint = (" — nvidia-container-toolkit missing or the "
                        "daemon not restarted (capsim doctor checks this)")
            raise RuntimeError(
                f"docker run for replica {index} failed "
                f"(rc={e.returncode}): {stderr}{hint}"
            ) from e
        cid = result.stdout.strip()
        streamer = self._spawn_log_streamer(cid, prefix=f"[r{index}] ")
        self._replicas.append((index, devices, self._port(index), cid,
                               streamer))

    def _startup_cause(self, tail_bytes: int = 200_000) -> str:
        """The engine's OWN last error line, lifted out of the log.

        "see the log" makes every launch failure look alike and costs
        a round trip to diagnose; the exception line distinguishes an
        unsupported flag from a missing import from an OOM at a
        glance."""
        if self._log_path is None:
            return "no engine log was captured"
        try:
            with open(self._log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - tail_bytes))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return "engine log unreadable"
        # Python names its exceptions "SomeError: message". The LAST
        # one is usually the API server's generic wrapper ("Engine
        # core initialization failed. See root cause above"), which
        # says nothing — the real cause is raised earlier, in the
        # engine-core worker. Prefer the last NON-generic line.
        pat = re.compile(
            r"\b((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*)"
            r"\s*:\s+(\S.*)")
        looks_like_exc = re.compile(
            r"(Error|Exception)$|\.(errors?|exceptions?)\.", re.I)
        generic = re.compile(
            r"see root cause above|engine core initialization failed|"
            r"engine process failed to start|see stack trace",
            re.I)
        fallback = None
        for ln in reversed(lines):
            m = pat.search(ln)
            if not m or not looks_like_exc.search(m.group(1)):
                continue
            msg = f"{m.group(1)}: {m.group(2).strip()[:300]}"
            if generic.search(m.group(2)):
                fallback = fallback or msg
                continue
            return msg
        if fallback:
            return fallback
        for ln in reversed(lines):
            if ln.strip():
                return ln.strip()[:300]
        return "the engine log is empty"

    def _wait_for_replica_ready(self, index: int, port: int,
                                container_id: str) -> None:
        start = time.time()
        backoff = 1.0
        while time.time() - start < self.cfg.startup_timeout_s:
            try:
                r = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}",
                     container_id],
                    capture_output=True, text=True, timeout=5,
                )
                if r.stdout.strip() != "true":
                    raise RuntimeError(
                        f"replica {index} container exited during startup: "
                        f"{self._startup_cause()} (full log: "
                        f"{self._log_path})"
                    )
            except subprocess.TimeoutExpired:
                pass
            try:
                resp = httpx.get(self._ready_url(port), timeout=2.0)
                if resp.status_code == 200:
                    log.info("replica %d ready after %.1fs",
                             index, time.time() - start)
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.2)
        raise TimeoutError(
            f"replica {index} not healthy in {self.cfg.startup_timeout_s}s "
            f"— see {self._log_path}"
        )

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def pid(self) -> Optional[int]:
        pids = self.pids
        return pids[0] if pids else None

    @property
    def pids(self) -> list[int]:
        """Host PID of every replica container — the engine-RSS
        rollup sums the whole set, not just replica 0."""
        out: list[int] = []
        for _i, _d, _p, cid, _s in self._replicas:
            try:
                r = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Pid}}", cid],
                    capture_output=True, text=True, timeout=5,
                )
                v = r.stdout.strip()
                if v and v != "0":
                    out.append(int(v))
            except Exception:  # noqa: BLE001
                continue
        return out

    def health_check(self) -> bool:
        if not self._replicas:
            return False
        for _i, _d, port, _cid, _s in self._replicas:
            try:
                r = httpx.get(self._ready_url(port), timeout=2.0)
                if r.status_code != 200:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    def get_metrics(self) -> dict[str, float]:
        per_replica = []
        for i, _d, port, _cid, _s in self._replicas:
            try:
                r = httpx.get(self._metrics_url(port), timeout=2.0)
                if r.status_code == 200:
                    per_replica.append(self._parse_replica(i, r.text))
            except Exception:  # noqa: BLE001
                continue
        return aggregate_replica_metrics(per_replica)

    def _parse_replica(self, index: int, text: str) -> dict[str, float]:
        """Hook so stateful parsers can key accumulators per replica."""
        return self.parse_metrics(text)

    def _build_command(self) -> list[str]:
        return []          # launch() is overridden entirely

    def _build_env(self) -> dict[str, str]:
        return dict(os.environ)

    def _spawn_log_streamer(self, container_id: str, *,
                            prefix: str) -> Optional[subprocess.Popen]:
        if self._log_path is None:
            return None
        try:
            log_file = open(self._log_path, "ab")
            shell_cmd = (f"docker logs -f {container_id} 2>&1 | "
                         f"sed 's/^/{prefix}/'")
            return subprocess.Popen(shell_cmd, shell=True, stdout=log_file,
                                    stderr=subprocess.STDOUT)
        except Exception as e:  # noqa: BLE001
            log.warning("log streamer for %s failed: %s",
                        container_id[:12], e)
            return None
