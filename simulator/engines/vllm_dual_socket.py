"""vLLM-CPU dual-replica engine for dual-socket NUMA boxes.

On dual-socket AMD EPYC, TP=2 across sockets is bottlenecked by Gloo
all-reduce traffic over the inter-socket interconnect (~1.4× scaling).
Two independent vLLM replicas, each pinned to one NUMA node, scale
nearly linearly (~2×) because there's zero cross-socket sync. Trade-
off: each replica has its own KV cache, so a user's multi-turn
conversation must consistently route to the same replica or the
prefix cache is missed on every turn.

The simulator handles routing itself: ``Engine.replica_urls`` returns
one URL per replica, the pool manager builds one ``AsyncOpenAI`` client
per replica, and each virtual user is hashed to a stable replica via
``hash(user_id) % len(replicas)``. Multi-turn conversations stay on
one backend for free, with no proxy in the request path.

The original architecture used a LiteLLM proxy at port 4000 with
``routing_strategy: "session-state-based"`` for sticky routing.
Recent LiteLLM releases removed that strategy name, and pinning a
specific tag for stability is fragile. Dropping LiteLLM removed an
external dependency and a process boundary on the request hot path.

Two Docker containers per launch — one ``vllm-{name}-{run_id}`` per
replica, NUMA-pinned via ``--cpuset-cpus`` + ``--cpuset-mems``.
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


class VllmDualSocketEngine(Engine):
    """Two NUMA-pinned vLLM replicas, hash-routed by virtual user id."""

    def __init__(self, engine_config):
        super().__init__(engine_config)
        # (ReplicaConfig, container_id, log_streamer_proc) tuples
        self._replicas: list[tuple[object, str, Optional[subprocess.Popen]]] = []
        self._log_dir: Optional[Path] = None

    # ── Public API ────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        # Health-check / single-endpoint default. The pool manager uses
        # ``replica_urls`` for actual request routing.
        if self._replicas:
            replica = self._replicas[0][0]
            return f"http://{self.cfg.host}:{replica.port}/v1"
        if self.cfg.replicas:
            return f"http://{self.cfg.host}:{self.cfg.replicas[0].port}/v1"
        raise RuntimeError("No replicas configured for vllm_dual_socket")

    @property
    def replica_urls(self) -> list[str]:
        """One URL per replica. The simulator hash-routes virtual users
        across these to preserve prefix-cache locality per user."""
        return [
            f"http://{self.cfg.host}:{r.port}/v1" for r in self.cfg.replicas
        ]

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._replicas:
            raise RuntimeError("Engine already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker not found on PATH; vllm_dual_socket requires Docker."
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
            log.info(
                "Starting %d vLLM replicas: %s",
                len(self.cfg.replicas),
                [r.name for r in self.cfg.replicas],
            )
            for replica in self.cfg.replicas:
                self._launch_replica(replica, run_id)
            for replica, cid, _ in self._replicas:
                self._wait_for_replica_ready(replica, cid)
            log.info(
                "All replicas ready. Routing across: %s",
                ", ".join(self.replica_urls),
            )
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        for replica, cid, streamer in self._replicas:
            log.info("Stopping replica %s (%s)", replica.name, cid[:12])
            self._stop_container(cid)
            if streamer is not None:
                self._stop_streamer(streamer)
        self._replicas = []

    @property
    def pid(self) -> Optional[int]:
        # No single PID — return the first replica's host PID for
        # telemetry's RSS rollup; psutil can walk children from there.
        if not self._replicas:
            return None
        try:
            cid = self._replicas[0][1]
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", cid],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip()
            return int(v) if v and v != "0" else None
        except Exception:
            return None

    def health_check(self) -> bool:
        """Healthy when all replicas serve ``/v1/models``."""
        if not self._replicas:
            return False
        for replica, _, _ in self._replicas:
            try:
                url = f"http://{self.cfg.host}:{replica.port}/v1/models"
                r = httpx.get(url, timeout=2.0)
                if r.status_code != 200:
                    return False
            except Exception:
                return False
        return True

    def get_metrics(self) -> dict[str, float]:
        """Aggregate metrics across all replicas.

        Counters (running, queue depth, prefix-cache hits/queries) are
        summed; gauges (KV usage %) are averaged. Hit rate is
        recomputed from the aggregated counters.
        """
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
        agg: dict[str, float] = {}
        for k in ("num_running", "queue_depth", "prefix_cache_hits",
                  "prefix_cache_queries"):
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

    def _build_command(self) -> list[str]:
        return []  # we override launch() entirely

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
        streamer = self._spawn_log_streamer(cid, prefix=f"[{replica.name}] ")
        self._replicas.append((replica, cid, streamer))

    def _build_replica_command(self, replica, container_name: str) -> list[str]:
        cfg = self.cfg
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--network", "host",
            "--shm-size", "4g",
            # SYS_NICE + seccomp=unconfined per vLLM CPU docs — without
            # them the OMP runtime can't apply thread priorities and
            # scheduling degrades silently.
            "--security-opt", "seccomp=unconfined",
            "--cap-add", "SYS_NICE",
        ]
        if replica.cpuset_cpus:
            cmd += ["--cpuset-cpus", replica.cpuset_cpus]
        if replica.cpuset_mems:
            cmd += ["--cpuset-mems", replica.cpuset_mems]
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            if not Path(host_path).exists():
                continue
            cmd += ["-v", f"{host_path}:{container_path}"]
        for k, v in (replica.env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(cfg.vllm_image)

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

    # ── Helpers ───────────────────────────────────────────────────────

    def _spawn_log_streamer(self, container_id: str, *, prefix: str) -> Optional[subprocess.Popen]:
        """Stream ``docker logs -f`` into the engine log file with a
        per-replica prefix so multi-replica output is greppable."""
        if self._log_path is None:
            return None
        try:
            log_file = open(self._log_path, "ab")
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
