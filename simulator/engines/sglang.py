"""SGLang engine launcher.

CPU support in SGLang is newer than vLLM's; tune conservatively. Users may
override flags via config.
"""

from __future__ import annotations

import os
import sys

from .base import Engine


class SGLangEngine(Engine):
    def _build_command(self) -> list[str]:
        cfg = self.cfg
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", cfg.model_id,
            "--dtype", "bfloat16",
            "--tp", str(cfg.tensor_parallel_size),
            "--port", str(cfg.port),
            "--host", cfg.host,
            "--context-length", str(cfg.max_model_len),
        ]
        if cfg.quantization:
            cmd += ["--quantization", cfg.quantization]
        cmd += list(cfg.sglang_extra_flags)
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "64")
        env.update(self.cfg.sglang_extra_env)
        return env

    def _health_path(self) -> str:
        return "/health"

    def _metrics_path(self) -> str:
        # SGLang exposes Prometheus metrics at /metrics when --enable-metrics
        # is set. Health endpoint also returns scheduler state on some builds.
        return "/metrics"
