"""vLLM CUDA engine launcher — Docker-based (roadmap 1.1).

Runs the upstream ``vllm/vllm-openai`` CUDA image with ``--gpus``,
for the Xeon+NVIDIA host class. Container lifecycle mirrors the
SGLang launcher (detached ``docker run``, streamed logs, ``/v1/models``
health gate, PID via ``docker inspect`` for engine-RSS rollup).

Launch shape notes:
  * ``--ipc=host`` — vLLM's CUDA workers use shared memory for tensor
    transport; the docker default 64 MB /dev/shm kills TP>1 init.
  * Health gate is ``/v1/models``: the OpenAI server binds late in
    startup, but polling the model list is unambiguous across image
    versions and matches the SGLang launcher's behavior.
  * ``--gpus`` defaults to ``all``; ``gpu_device_ids`` narrows to
    specific devices (e.g. ``[0, 1]`` → ``--gpus "device=0,1"``).
  * The HF cache mount comes from ``docker_volumes`` like every other
    engine, so pre-downloaded weights are shared across engines.
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

# Stale-container sweep is shared: a leftover trtllm-* container would
# hold port 9100 and answer health checks for the WRONG engine, so it
# must be swept before a vLLM launch too.
from .docker_replica import remove_stale_engine_containers  # noqa: F401

log = logging.getLogger(__name__)


class VllmCudaEngine(Engine):
    """Launches CUDA vLLM in a Docker container with --gpus."""

    def __init__(self, engine_config):
        super().__init__(engine_config)
        self._container_id: Optional[str] = None
        self._log_streamer: Optional[subprocess.Popen] = None

    # -- Public API ---------------------------------------------------------

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._container_id is not None:
            raise RuntimeError("vLLM CUDA container already launched")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker not found on PATH; the vllm_cuda engine runs the "
                "upstream CUDA image in Docker (needs nvidia-container-toolkit)."
            )
        remove_stale_engine_containers()

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"engine_vllm_cuda_{int(time.time())}.log"
        self._log_path = log_path

        cmd = self._build_docker_command()
        log.info("Launching vllm_cuda (docker): %s", " ".join(cmd))
        log.info("Engine logs -> %s", log_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip()
            hint = ""
            if "could not select device driver" in stderr or "nvidia" in stderr.lower():
                hint = (
                    " — Docker couldn't hand the container a GPU; check "
                    "nvidia-container-toolkit is installed and the daemon "
                    "restarted (capsim doctor verifies this)."
                )
            raise RuntimeError(
                f"docker run failed (rc={e.returncode}): {stderr}{hint}"
            ) from e
        self._container_id = result.stdout.strip()
        log.info("Container id: %s", self._container_id)

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
            except Exception:  # noqa: BLE001
                pass
            self._log_streamer = None

    @property
    def pid(self) -> Optional[int]:
        """Host-visible PID of the container's main process (crosses the
        namespace boundary so engine-RSS rollup sees the workers)."""
        if self._container_id is None:
            return None
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Pid}}", self._container_id],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip()
            return int(v) if v and v != "0" else None
        except Exception:  # noqa: BLE001
            return None

    def health_check(self) -> bool:
        """``/v1/models`` returns 200 only once the model is loaded."""
        if self._container_id is None:
            return False
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
        """Container-based poll (base's version also polls the
        subprocess handle we never set)."""
        start = time.time()
        backoff = 1.0
        while time.time() - start < timeout_s:
            try:
                if self.health_check():
                    log.info("vLLM CUDA ready after %.1fs", time.time() - start)
                    return
            except RuntimeError:
                raise  # container died — fail fast
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.2)
        raise TimeoutError(
            f"vLLM CUDA did not become healthy in {timeout_s}s — "
            f"check {self._log_path}"
        )

    # -- Command construction ----------------------------------------------

    def _build_docker_command(self) -> list[str]:
        cfg = self.cfg
        name = f"vllm-cuda-{uuid.uuid4().hex[:12]}"

        if cfg.gpu_device_ids:
            gpus_arg = "device=" + ",".join(str(i) for i in cfg.gpu_device_ids)
            # Docker parses the --gpus value as CSV, so a multi-device
            # list needs EMBEDDED quotes (`"device=0,1"`, quote chars
            # included) or it reads as device=0 + count=1 and the
            # daemon refuses ("cannot set both Count and DeviceIDs").
            if len(cfg.gpu_device_ids) > 1:
                gpus_arg = f'"{gpus_arg}"'
        else:
            gpus_arg = "all"

        cmd: list[str] = [
            "docker", "run", "-d", "--rm",
            "--name", name,
            "--gpus", gpus_arg,
            "--ipc=host",
            "--network", cfg.docker_network,
        ]
        # host networking exposes the server port directly; bridged
        # setups need an explicit publish (image serves on 8000).
        if cfg.docker_network != "host":
            cmd += ["-p", f"{cfg.port}:8000"]
        mounted_targets = set()
        for host_path, container_path in (cfg.docker_volumes or {}).items():
            if not Path(host_path).exists():
                continue
            cmd += ["-v", f"{host_path}:{container_path}"]
            mounted_targets.add(container_path)
        # Weights cache: when the profile's HF-cache mount didn't apply
        # (host path absent, or a UI-chosen storage location is in
        # play), mount the RESOLVED cache so weights land / are found
        # on the disk the user actually picked.
        if "/root/.cache/huggingface" not in mounted_targets:
            from ..models import hf_cache_dir
            cache = hf_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            cmd += ["-v", f"{cache}:/root/.cache/huggingface"]
        # Pass through an HF token for gated models.
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(var):
                cmd += ["-e", f"{var}={os.environ[var]}"]
        for k, v in (cfg.docker_extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(cfg.gpu_image)

        # Inner vLLM server args (the image's entrypoint is the server).
        model_arg = cfg.model_local_path or cfg.model_id
        inner = [
            "--model", model_arg,
            "--host", "0.0.0.0",
            "--port", str(cfg.port if cfg.docker_network == "host" else 8000),
            "--max-model-len", str(cfg.max_model_len),
            "--tensor-parallel-size", str(cfg.tensor_parallel_size),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        ]
        if cfg.served_model_name:
            inner += ["--served-model-name", cfg.served_model_name]
        quantization = cfg.quantization_kind or cfg.quantization
        if quantization:
            inner += ["--quantization", quantization]
        inner += list(cfg.vllm_extra_flags or [])

        return cmd + inner

    # Satisfy the base's abstract API (unused — we own launch()).
    def _build_command(self) -> list[str]:
        return self._build_docker_command()

    def _build_env(self) -> dict[str, str]:
        return dict(os.environ)
