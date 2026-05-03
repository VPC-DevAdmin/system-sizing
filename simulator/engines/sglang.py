"""SGLang engine launcher — Docker-based.

SGLang's CPU support depends on a binary ``sgl_kernel`` that the mainline
pip wheel only publishes in CUDA variants. Importing the package on a
GPU-less host fails at load-time. The working CPU path is the upstream
``sglang-cpu`` Docker image; we layer ``sentencepiece`` / ``tiktoken`` /
``protobuf`` on top so modern HF tokenizers (Qwen3, GLM, Mistral, Llama 3)
load cleanly. See Dockerfile.xeon-fixed in the repo root for the layer.

Lessons from the Granite Rapids 64C runbook, baked in:

  * SGLang's CPU scheduler asserts ``tp == len(SGLANG_CPU_OMP_THREADS_BIND.split('|'))``.
    We derive the env var from the engine's ``cpu_bind`` field via
    ``derive_sglang_thread_binding`` so the alignment is enforced at
    launch time, not 20 minutes into a model load.
  * Healthcheck on ``/health`` returns 200 BEFORE the model is loaded —
    you'd send requests into the void. We poll ``/v1/models`` instead.
  * ``--shm-size 32g`` is required for TP>1; the docker default 64 MB
    kills multi-worker init.
  * On graceful shutdown allow 30s — SGLang's worker teardown takes a
    while to release shared memory and ports.
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

from ..cpu_binding import derive_sglang_thread_binding, flatten_for_taskset
from .base import Engine

log = logging.getLogger(__name__)


class SGLangEngine(Engine):
    """Launches SGLang in a Docker container with CPU-correct flags."""

    def __init__(self, engine_config):
        super().__init__(engine_config)
        self._container_id: Optional[str] = None
        self._log_streamer: Optional[subprocess.Popen] = None

    # -- Public API ---------------------------------------------------------

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._container_id is not None:
            raise RuntimeError("SGLang container already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker not found on PATH; SGLang CPU runs require Docker. "
                "See Dockerfile.xeon-fixed in the repo root for the image."
            )

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"engine_sglang_{int(time.time())}.log"
        self._log_path = log_path

        cmd = self._build_docker_command()
        log.info("Launching sglang (docker): %s", " ".join(cmd))
        log.info("Engine logs -> %s", log_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"docker run failed (rc={e.returncode}): {e.stderr.strip()}"
            ) from e
        self._container_id = result.stdout.strip()
        log.info("Container id: %s", self._container_id)

        # Stream container logs to file in the background. The streamer
        # exits when the container does, so we rely on that for cleanup.
        self._log_streamer = subprocess.Popen(
            ["docker", "logs", "-f", self._container_id],
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
        )

        try:
            self._wait_for_health(self.cfg.startup_timeout_s)
        except Exception:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        if self._container_id is None:
            return
        cid = self._container_id
        self._container_id = None
        log.info("Stopping container %s (30s grace)", cid)
        try:
            subprocess.run(
                ["docker", "stop", "-t", "30", cid],
                capture_output=True, timeout=45,
            )
        except subprocess.TimeoutExpired:
            log.warning("docker stop timed out; forcing rm")
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        if self._log_streamer is not None:
            try:
                self._log_streamer.terminate()
                try:
                    self._log_streamer.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._log_streamer.kill()
            except Exception:
                pass
            self._log_streamer = None

    @property
    def pid(self) -> Optional[int]:
        """Host-visible PID of the container's main process, when running.

        ``docker inspect -f '{{.State.Pid}}'`` walks the namespace boundary,
        so ``psutil.Process(pid).children(recursive=True)`` from the host
        will see the worker subprocesses for accurate engine-RSS rollup.
        """
        if self._container_id is None:
            return None
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", self._container_id],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip()
            return int(v) if v and v != "0" else None
        except Exception:
            return None

    def health_check(self) -> bool:
        """``/v1/models`` only returns 200 once the model is loaded.

        ``/health`` returns 200 the moment the HTTP server is up — for
        a 30B+ model that's a 30+ second window where requests will fail.
        Don't use it.
        """
        if self._container_id is None:
            return False
        # First confirm the container hasn't already died.
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_id],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip() != "true":
                raise RuntimeError(
                    f"container {self._container_id} exited; "
                    f"see {self._log_path} for details"
                )
        except subprocess.TimeoutExpired:
            return False

        try:
            url = f"http://{self.cfg.host}:{self.cfg.port}/v1/models"
            r = httpx.get(url, timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _wait_for_health(self, timeout_s: int) -> None:
        """Override base — the base method poll-checks via ``health_check``
        which is fine, but it also pokes the subprocess.poll() of the
        ``self._proc`` we never set. Bypass that path entirely."""
        start = time.time()
        backoff = 1.0
        while time.time() - start < timeout_s:
            try:
                if self.health_check():
                    log.info("SGLang ready after %.1fs", time.time() - start)
                    return
            except RuntimeError:
                # container died — fail fast
                raise
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.2)
        raise TimeoutError(
            f"SGLang did not become healthy in {timeout_s}s — "
            f"check {self._log_path}"
        )

    # -- Command construction ----------------------------------------------

    def _build_docker_command(self) -> list[str]:
        cfg = self.cfg
        name = f"sglang-{uuid.uuid4().hex[:12]}"

        # cpuset for the container is the flat union of cpu_bind groups;
        # the per-worker thread binding goes into SGLANG_CPU_OMP_THREADS_BIND.
        cpuset = flatten_for_taskset(cfg.cpu_bind)

        # Thread binding aligned to TP. Raises with a clear error if
        # TP and cpu_bind are inconsistent.
        sglang_bind = derive_sglang_thread_binding(
            cfg.cpu_bind, cfg.tensor_parallel_size,
        )

        # ── docker args ───────────────────────────────────────────────
        cmd: list[str] = [
            "docker", "run", "-d", "--rm",
            "--name", name,
            "--network", cfg.docker_network,
            "--shm-size", cfg.docker_shm_size,
        ]
        if cpuset:
            cmd += ["--cpuset-cpus", cpuset]
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            # Skip mounts whose host path doesn't exist — common on dev
            # machines where the standard /data/ml layout isn't present.
            if not Path(host_path).exists():
                continue
            cmd += ["-v", f"{host_path}:{container_path}"]
        if sglang_bind:
            cmd += ["-e", f"SGLANG_CPU_OMP_THREADS_BIND={sglang_bind}"]
        for k, v in (cfg.docker_extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        # OMP knobs for the in-container OpenMP runtime.
        for k, v in (cfg.sglang_extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(cfg.docker_image)

        # ── inner SGLang launch command ───────────────────────────────
        # The image's venv is at /opt/.venv per upstream convention.
        model_arg = cfg.model_local_path or cfg.model_id
        served_name = cfg.served_model_name or cfg.model_id
        # Match the minimal-but-working launch shape from the runbook;
        # optional knobs are only emitted when explicitly set in config.
        inner = [
            "/opt/.venv/bin/python", "-m", "sglang.launch_server",
            "--model", model_arg,
            "--device", "cpu",
            "--host", "0.0.0.0",
            "--port", str(cfg.port),
            "--tp", str(cfg.tensor_parallel_size),
            "--served-model-name", served_name,
            "--trust-remote-code",
        ]
        if cfg.disable_overlap_schedule:
            inner.append("--disable-overlap-schedule")
        if cfg.enable_metrics:
            inner.append("--enable-metrics")
        if cfg.context_length is not None:
            inner += ["--context-length", str(cfg.context_length)]
        if cfg.max_total_tokens is not None:
            inner += ["--max-total-tokens", str(cfg.max_total_tokens)]
        if cfg.chunked_prefill_size is not None:
            inner += ["--chunked-prefill-size", str(cfg.chunked_prefill_size)]
        if cfg.mem_fraction_static is not None:
            inner += ["--mem-fraction-static", str(cfg.mem_fraction_static)]
        if cfg.attention_backend:
            inner += ["--attention-backend", cfg.attention_backend]
        quantization = cfg.quantization_kind or cfg.quantization
        if quantization:
            inner += ["--quantization", quantization]
        else:
            # Don't pass --dtype when quantization is in play; SGLang
            # wants the model in its native quantized form.
            inner += ["--dtype", "bfloat16"]
        inner += list(cfg.sglang_extra_flags or [])

        return cmd + inner

    # ``base.Engine`` delegates command construction to subclasses for the
    # subprocess path; we don't use that here, so satisfy the abstract API.
    def _build_command(self) -> list[str]:
        return self._build_docker_command()

    def _build_env(self) -> dict[str, str]:
        # Env is passed to docker -e; the host env doesn't matter for the
        # contained process. Return an empty dict so the base class doesn't
        # accidentally pollute anything.
        return dict(os.environ)

    def _health_path(self) -> str:
        return "/v1/models"

    def _metrics_path(self) -> str:
        return "/metrics"
