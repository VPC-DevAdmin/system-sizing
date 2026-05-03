"""vLLM engine launcher with Xeon CPU tuning recipe."""

from __future__ import annotations

import os

from .base import Engine


class VLLMEngine(Engine):
    def _build_command(self) -> list[str]:
        cfg = self.cfg
        cmd = [
            "vllm", "serve", cfg.model_id,
            "--dtype", "bfloat16",
            "--max-model-len", str(cfg.max_model_len),
            "--tensor-parallel-size", str(cfg.tensor_parallel_size),
            "--port", str(cfg.port),
            "--host", cfg.host,
        ]
        if cfg.quantization:
            cmd += ["--quantization", cfg.quantization]
        cmd += list(cfg.vllm_extra_flags)
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Default tuning for Xeon 6761P (64-core Granite Rapids)
        defaults = {
            "VLLM_CPU_KVCACHE_SPACE": str(self.cfg.kv_cache_gb),
            # ONEDNN verbose dispatch logging — read at end of run by
            # ``amx_utilization.parse_amx_utilization`` to compute the
            # AMX dispatch fraction. Cost on the engine is small (a line
            # per matmul); the log can grow multi-GB on long runs but
            # the parser uses a 256 MiB byte budget by default.
            "ONEDNN_VERBOSE": "1",
        }
        # tcmalloc preload if available
        tcmalloc = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
        if os.path.exists(tcmalloc) and "LD_PRELOAD" not in env:
            defaults["LD_PRELOAD"] = tcmalloc
        for k, v in defaults.items():
            env.setdefault(k, v)
        env.update(self.cfg.vllm_extra_env)
        # Engine-agnostic cpu_bind backfills VLLM_CPU_OMP_THREADS_BIND
        # only if the caller didn't set it explicitly — explicit env wins
        # so existing configs don't change behaviour.
        if self.cfg.cpu_bind and "VLLM_CPU_OMP_THREADS_BIND" not in env:
            env["VLLM_CPU_OMP_THREADS_BIND"] = self.cfg.cpu_bind
        return env

    def _health_path(self) -> str:
        return "/health"

    def _metrics_path(self) -> str:
        return "/metrics"
