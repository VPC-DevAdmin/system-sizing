"""vLLM-CPU dual-replica + LiteLLM proxy engine for dual-socket NUMA boxes.

On dual-socket AMD EPYC, TP=2 across sockets is bottlenecked by Gloo
all-reduce traffic over the inter-socket interconnect (~1.4× scaling).
Two independent vLLM replicas, each pinned to a single NUMA node, scale
nearly linearly (~2×) because there's zero cross-socket sync — the
trade-off is that each replica has its own KV cache, so a user's
multi-turn conversation must consistently route to the same replica or
the prefix cache is missed.

A LiteLLM proxy fronts both replicas at a single OpenAI-compatible
endpoint and uses ``session_id``-based sticky routing to preserve
prefix-cache locality. The simulator's virtual-user runtime sets
``session_id = user_id`` so each user's full conversation history
stays on one replica for the lifetime of that user.

Three Docker containers come up per launch:
  * ``vllm-{name}-{uuid}`` — one per replica, NUMA-pinned via
    ``--cpuset-cpus`` + ``--cpuset-mems``.
  * ``litellm-{uuid}`` — proxy on ``litellm_port`` (default 4000).

Shutdown order: LiteLLM first, then replicas. This causes in-flight
requests to fail fast with HTTP errors rather than hang indefinitely
on backends that disappeared mid-flight.

Architecture confirmed working on R7735 (2× EPYC 9374F, 64 phys cores
across 2 NUMA nodes, ~386 GB DDR5) running Qwen3-30B-A3B-Instruct-2507
in BF16. The vllm/vllm-openai-cpu image is required — the SGLang
xeon-fixed image has a torch wheel that runs at ~3% of theoretical
compute on AMD due to missing AVX-512 BF16 dispatch.
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
import yaml

from .base import Engine

log = logging.getLogger(__name__)


class VllmDualSocketEngine(Engine):
    """Two NUMA-pinned vLLM replicas + LiteLLM session-routing proxy."""

    def __init__(self, engine_config):
        super().__init__(engine_config)
        # (ReplicaConfig, container_id, log_streamer_proc) tuples
        self._replicas: list[tuple[object, str, Optional[subprocess.Popen]]] = []
        self._litellm_container: Optional[str] = None
        self._litellm_streamer: Optional[subprocess.Popen] = None
        self._litellm_config_path: Optional[Path] = None
        self._log_dir: Optional[Path] = None

    # ── Public API ────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.litellm_port}/v1"

    @property
    def api_key(self) -> str:
        return self.cfg.litellm_master_key

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._replicas or self._litellm_container is not None:
            raise RuntimeError("Engine already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker not found on PATH; vllm_dual_socket runs require Docker."
            )
        if not self.cfg.replicas:
            raise RuntimeError(
                "engine.replicas must list at least one ReplicaConfig"
            )

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        self._log_path = self._log_dir / f"engine_vllm_dual_{run_id}.log"

        try:
            # 1. Launch all replicas (parallel docker run -d returns fast).
            log.info(
                "Starting %d vLLM replicas: %s",
                len(self.cfg.replicas),
                [r.name for r in self.cfg.replicas],
            )
            for replica in self.cfg.replicas:
                self._launch_replica(replica, run_id)

            # 2. Wait for each replica's /v1/models to return 200.
            for replica, cid, _ in self._replicas:
                self._wait_for_replica_ready(replica, cid)

            # 3. Generate LiteLLM config and start the proxy.
            self._litellm_config_path = (
                self._log_dir / f"litellm_{run_id}.yaml"
            )
            self._write_litellm_config(self._litellm_config_path)
            log.info("LiteLLM config -> %s", self._litellm_config_path)
            self._launch_litellm(run_id)

            # 4. Wait for LiteLLM /v1/models.
            self._wait_for_litellm_ready()
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        # LiteLLM first so in-flight requests fail-fast rather than hang.
        if self._litellm_container is not None:
            cid = self._litellm_container
            self._litellm_container = None
            log.info("Stopping LiteLLM container %s", cid[:12])
            self._stop_container(cid)
        if self._litellm_streamer is not None:
            self._stop_streamer(self._litellm_streamer)
            self._litellm_streamer = None

        for replica, cid, streamer in self._replicas:
            log.info("Stopping replica %s (%s)", replica.name, cid[:12])
            self._stop_container(cid)
            if streamer is not None:
                self._stop_streamer(streamer)
        self._replicas = []

    @property
    def pid(self) -> Optional[int]:
        if self._litellm_container is None:
            return None
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", self._litellm_container],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip()
            return int(v) if v and v != "0" else None
        except Exception:
            return None

    def health_check(self) -> bool:
        if self._litellm_container is None:
            return False
        try:
            url = (
                f"http://{self.cfg.host}:{self.cfg.litellm_port}/v1/models"
            )
            r = httpx.get(
                url,
                headers={"Authorization": f"Bearer {self.cfg.litellm_master_key}"},
                timeout=2.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def get_metrics(self) -> dict[str, float]:
        """Aggregate metrics from BOTH vLLM replicas.

        LiteLLM doesn't expose model-side metrics — they live on each
        replica's ``/metrics`` endpoint. We sum/average sensibly:
        running + waiting are summed across replicas; cache utilisation
        is averaged.
        """
        agg: dict[str, float] = {}
        per_replica = []
        for replica, _, _ in self._replicas:
            try:
                url = f"http://{self.cfg.host}:{replica.port}/metrics"
                r = httpx.get(url, timeout=2.0)
                if r.status_code != 200:
                    continue
                per_replica.append(self._parse_prometheus(r.text))
            except Exception:
                continue
        if not per_replica:
            return {}
        # Sum counters that should be additive
        for k in ("num_running", "queue_depth", "prefix_cache_hits",
                  "prefix_cache_queries"):
            vals = [m[k] for m in per_replica if k in m]
            if vals:
                agg[k] = sum(vals)
        # Average gauges that don't add cleanly
        for k in ("kv_cache_used_pct",):
            vals = [m[k] for m in per_replica if k in m]
            if vals:
                agg[k] = sum(vals) / len(vals)
        # Recompute hit rate from aggregated counters
        if agg.get("prefix_cache_queries"):
            agg["prefix_cache_hit_rate"] = (
                agg.get("prefix_cache_hits", 0) / agg["prefix_cache_queries"]
            )
        return agg

    # Required-by-base abstract API; not used directly because we
    # override launch() entirely.
    def _build_command(self) -> list[str]:
        return []

    def _build_env(self) -> dict[str, str]:
        return dict(os.environ)

    # ── Replica lifecycle ─────────────────────────────────────────────

    def _launch_replica(self, replica, run_id: str) -> None:
        name = f"vllm-{replica.name}-{run_id}"
        cmd = self._build_replica_command(replica, name)
        log.info("docker run %s: %s", replica.name, " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"docker run for replica {replica.name} failed "
                f"(rc={e.returncode}): {e.stderr.strip()}"
            ) from e
        cid = result.stdout.strip()
        log.info("replica %s container=%s", replica.name, cid[:12])

        # Stream the replica's stdout/stderr to the shared engine log
        # with a per-replica prefix so multi-replica logs are readable.
        streamer = self._spawn_log_streamer(cid, prefix=f"[{replica.name}] ")
        self._replicas.append((replica, cid, streamer))

    def _build_replica_command(self, replica, container_name: str) -> list[str]:
        cfg = self.cfg
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--network", "host",
            "--shm-size", "4g",
            # SYS_NICE + seccomp=unconfined are required by vLLM CPU's
            # OpenMP runtime to apply thread priorities; without them
            # OMP scheduling degrades quietly.
            "--security-opt", "seccomp=unconfined",
            "--cap-add", "SYS_NICE",
        ]
        if replica.cpuset_cpus:
            cmd += ["--cpuset-cpus", replica.cpuset_cpus]
        if replica.cpuset_mems:
            cmd += ["--cpuset-mems", replica.cpuset_mems]
        # Volume mounts (model weights, HF cache).
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            if not Path(host_path).exists():
                continue
            cmd += ["-v", f"{host_path}:{container_path}"]
        # Per-replica env (NUMA-specific OpenMP settings).
        for k, v in (replica.env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(cfg.vllm_image)

        # Inner: vLLM serve flags.
        model_arg = cfg.model_local_path or cfg.model_id
        served_name = cfg.served_model_name or cfg.model_id
        cmd += [
            "--model", model_arg,
            "--dtype", "bfloat16",
            "--max-model-len", str(cfg.max_model_len),
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", str(replica.port),
            "--served-model-name", served_name,
        ]
        if cfg.quantization_kind:
            cmd += ["--quantization", cfg.quantization_kind]
        cmd += list(cfg.vllm_extra_flags or [])
        return cmd

    def _wait_for_replica_ready(self, replica, container_id: str) -> None:
        log.info(
            "Waiting for replica %s on port %d (timeout %ds)",
            replica.name, replica.port, self.cfg.startup_timeout_s,
        )
        start = time.time()
        backoff = 1.0
        while time.time() - start < self.cfg.startup_timeout_s:
            if not self._container_running(container_id):
                raise RuntimeError(
                    f"Replica {replica.name} container exited during startup; "
                    f"see {self._log_path}"
                )
            try:
                url = f"http://{self.cfg.host}:{replica.port}/v1/models"
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    log.info("Replica %s ready after %.1fs",
                             replica.name, time.time() - start)
                    return
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.2)
        raise TimeoutError(
            f"Replica {replica.name} did not become ready within "
            f"{self.cfg.startup_timeout_s}s — see {self._log_path}"
        )

    # ── LiteLLM lifecycle ─────────────────────────────────────────────

    def _write_litellm_config(self, path: Path) -> None:
        cfg = self.cfg
        served_name = cfg.served_model_name or cfg.model_id
        model_list = []
        for replica in cfg.replicas:
            model_list.append({
                "model_name": served_name,
                "litellm_params": {
                    "model": f"openai/{served_name}",
                    "api_base": f"http://127.0.0.1:{replica.port}/v1",
                    "api_key": "dummy",
                },
            })
        document = {
            "model_list": model_list,
            "router_settings": {
                "routing_strategy": "session-state-based",
                "num_retries": 1,
                "timeout": 600,
                "enable_pre_call_checks": True,
            },
            "general_settings": {
                "master_key": cfg.litellm_master_key,
            },
        }
        path.write_text(yaml.safe_dump(document, sort_keys=False))

    def _launch_litellm(self, run_id: str) -> None:
        cfg = self.cfg
        name = f"litellm-{run_id}"
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", name,
            "--network", "host",
            "-v", f"{self._litellm_config_path}:/app/config.yaml:ro",
            cfg.litellm_image,
            "--config", "/app/config.yaml",
            "--port", str(cfg.litellm_port),
        ]
        log.info("docker run litellm: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=60,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"LiteLLM docker run failed (rc={e.returncode}): {e.stderr.strip()}"
            ) from e
        self._litellm_container = result.stdout.strip()
        log.info("litellm container=%s", self._litellm_container[:12])
        self._litellm_streamer = self._spawn_log_streamer(
            self._litellm_container, prefix="[litellm] ",
        )

    def _wait_for_litellm_ready(self) -> None:
        log.info(
            "Waiting for LiteLLM on port %d", self.cfg.litellm_port,
        )
        start = time.time()
        timeout = min(120, self.cfg.startup_timeout_s)  # LiteLLM is fast
        backoff = 1.0
        while time.time() - start < timeout:
            if not self._container_running(self._litellm_container):
                raise RuntimeError(
                    f"LiteLLM container exited during startup; see {self._log_path}"
                )
            try:
                url = f"http://{self.cfg.host}:{self.cfg.litellm_port}/v1/models"
                r = httpx.get(
                    url,
                    headers={"Authorization": f"Bearer {self.cfg.litellm_master_key}"},
                    timeout=2.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    models = [m.get("id") for m in data.get("data", [])]
                    log.info(
                        "LiteLLM ready after %.1fs; serving: %s",
                        time.time() - start, models,
                    )
                    return
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(3.0, backoff * 1.2)
        raise TimeoutError(
            f"LiteLLM did not become ready within {timeout}s"
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _spawn_log_streamer(self, container_id: str, *, prefix: str) -> Optional[subprocess.Popen]:
        """Stream ``docker logs -f`` into the shared engine log, prefixing
        each line so multi-replica output is greppable by container."""
        if self._log_path is None:
            return None
        try:
            log_file = open(self._log_path, "ab")
            # ``stdbuf -oL`` on docker logs ensures line-buffered output;
            # we then prefix via sed in a shell pipeline.
            shell_cmd = (
                f"docker logs -f {container_id} 2>&1 | "
                f"sed 's/^/{prefix}/'"
            )
            return subprocess.Popen(
                shell_cmd, shell=True, stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            log.warning("Could not start log streamer for %s: %s", container_id[:12], e)
            return None

    def _container_running(self, container_id: str | None) -> bool:
        if not container_id:
            return False
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip() == "true"
        except Exception:
            return False

    def _stop_container(self, container_id: str) -> None:
        grace = self.cfg.shutdown_grace_s
        try:
            subprocess.run(
                ["docker", "stop", "-t", str(grace), container_id],
                capture_output=True, timeout=grace + 15,
            )
        except subprocess.TimeoutExpired:
            log.warning("docker stop timed out for %s; forcing rm", container_id[:12])
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
        except Exception as e:
            log.warning("docker stop %s: %s", container_id[:12], e)

    def _stop_streamer(self, p: subprocess.Popen) -> None:
        try:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        except Exception:
            pass
