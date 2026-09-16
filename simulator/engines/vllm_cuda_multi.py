"""Multi-replica CUDA vLLM engine — the whole box as N independent
replicas.

The CUDA analog of ``vllm_dual_socket``: the guided search showed
that on a PCIe box (no NVLink), N independent single- or few-GPU
replicas beat tensor-parallel spans for models that fit — dp8 of
Qwen3-30B delivered ~9.2k tok/s where TP pays an all-reduce per
layer. This engine lets the capacity benchmark MEASURE that shape
instead of extrapolating from one replica × 8.

Config: ``engine.replica_devices`` is a list of GPU-id lists, one
per replica (exactly the assignment the optimizer's placement logic
produces — e.g. dp8: ``[[0],[4],[1],[5],[2],[6],[3],[7]]``; a TP2×4
hybrid: ``[[0,1],[2,3],[4,5],[6,7]]``). Each replica serves on
``port + index`` (host networking). Everything else (image, model,
gmu, extra flags) comes from the same EngineConfig fields as
``vllm_cuda``.

Routing is the simulator's, same as dual-socket: ``replica_urls``
exposes every backend and the pool manager sticks each virtual user
to one replica (±1 balanced), so multi-turn conversations keep their
prefix-cache locality — the property the warm-KV telemetry measures.

Telemetry: ``get_metrics`` aggregates across replicas (counters
summed — including the prefill/decode token counters and
preemptions — gauges averaged); ``pids`` exposes every replica's
host PID so the engine-RSS rollup covers the whole set.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from .base import Engine

log = logging.getLogger(__name__)


def gpus_arg_for(device_ids: list[int]) -> str:
    """Docker ``--gpus`` value for a device list. Docker parses the
    value as CSV, so multi-device lists need EMBEDDED quotes
    (`"device=0,1"`, quote characters included) or the daemon reads
    device=0 + count=1 and refuses."""
    arg = "device=" + ",".join(str(i) for i in device_ids)
    return f'"{arg}"' if len(device_ids) > 1 else arg


class VllmCudaMultiEngine(Engine):
    """N GPU-pinned vLLM CUDA replicas, sticky-routed by the pool."""

    def __init__(self, engine_config):
        super().__init__(engine_config)
        # (index, devices, port, container_id, streamer) per replica
        self._replicas: list[tuple[int, list[int], int, str,
                                   Optional[subprocess.Popen]]] = []

    # ── Shape ─────────────────────────────────────────────────────────

    def _device_groups(self) -> list[list[int]]:
        groups = getattr(self.cfg, "replica_devices", None)
        if not groups or not all(isinstance(g, (list, tuple)) and g
                                 for g in groups):
            raise RuntimeError(
                "vllm_cuda_multi needs engine.replica_devices — a list "
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

    # ── Lifecycle ─────────────────────────────────────────────────────

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._replicas:
            raise RuntimeError("Engine already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker not found on PATH; vllm_cuda_multi runs the "
                "CUDA image in Docker (needs nvidia-container-toolkit)."
            )
        groups = self._device_groups()
        flat = [d for g in groups for d in g]
        if len(set(flat)) != len(flat):
            raise RuntimeError(
                f"replica_devices assigns a GPU twice: {groups}"
            )
        # Leftover replicas from a hard-killed run would hold our
        # ports and answer health checks for the WRONG model.
        from .vllm_cuda import remove_stale_engine_containers
        remove_stale_engine_containers()

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        self._log_path = log_dir / f"engine_vllm_cuda_multi_{run_id}.log"

        try:
            log.info("Starting %d CUDA replicas on devices %s",
                     len(groups), groups)
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
        name = f"vllm-r{index}-{run_id}"
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

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        cfg = self.cfg
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--gpus", gpus_arg_for(devices),
            # vLLM's CUDA workers use shared memory for tensor
            # transport; the 64MB docker default kills TP>1 init.
            "--ipc=host",
            "--network", "host",
        ]
        mounted = set()
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            if not Path(host_path).exists():
                continue
            cmd += ["-v", f"{host_path}:{container_path}"]
            mounted.add(container_path)
        if "/root/.cache/huggingface" not in mounted:
            from ..models import hf_cache_dir
            cache = hf_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            cmd += ["-v", f"{cache}:/root/.cache/huggingface"]
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(var):
                cmd += ["-e", f"{var}={os.environ[var]}"]
        for k, v in (cfg.docker_extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(cfg.gpu_image)

        model_arg = cfg.model_local_path or cfg.model_id
        inner = [
            "--model", model_arg,
            "--host", "0.0.0.0",
            "--port", str(self._port(index)),
            "--max-model-len", str(cfg.max_model_len),
            # Per-replica TP = the replica's device count; DP is the
            # replica count itself.
            "--tensor-parallel-size", str(len(devices)),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        ]
        if cfg.served_model_name:
            inner += ["--served-model-name", cfg.served_model_name]
        quantization = cfg.quantization_kind or cfg.quantization
        if quantization:
            inner += ["--quantization", quantization]
        inner += list(cfg.vllm_extra_flags or [])
        return cmd + inner

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
                        f"replica {index} container exited during startup; "
                        f"see {self._log_path}"
                    )
            except subprocess.TimeoutExpired:
                pass
            try:
                resp = httpx.get(
                    f"http://{self.cfg.host}:{port}/v1/models", timeout=2.0)
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
                r = httpx.get(
                    f"http://{self.cfg.host}:{port}/v1/models", timeout=2.0)
                if r.status_code != 200:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    def get_metrics(self) -> dict[str, float]:
        per_replica = []
        for _i, _d, port, _cid, _s in self._replicas:
            try:
                r = httpx.get(
                    f"http://{self.cfg.host}:{port}/metrics", timeout=2.0)
                if r.status_code == 200:
                    per_replica.append(self._parse_prometheus(r.text))
            except Exception:  # noqa: BLE001
                continue
        return aggregate_replica_metrics(per_replica)

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
    for k in ("kv_cache_used_pct",):
        vals = [m[k] for m in per_replica if k in m]
        if vals:
            agg[k] = sum(vals) / len(vals)
    if agg.get("prefix_cache_queries"):
        agg["prefix_cache_hit_rate"] = (
            agg.get("prefix_cache_hits", 0) / agg["prefix_cache_queries"]
        )
    return agg
