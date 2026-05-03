"""SGLang engine launcher.

CPU support in SGLang is newer than vLLM's; tune conservatively. Users may
override flags via config. Honours the engine-agnostic ``cpu_bind`` field
by wrapping the launch with ``taskset -c <list>`` so the bound-CPU set
matches what the simulator's frequency collector aggregates over.
"""

from __future__ import annotations

import os
import shutil
import sys

from ..cpu_binding import flatten_for_taskset
from .base import Engine


class SGLangEngine(Engine):
    def _build_command(self) -> list[str]:
        cfg = self.cfg
        inner = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", cfg.model_id,
            "--dtype", "bfloat16",
            "--tp", str(cfg.tensor_parallel_size),
            "--port", str(cfg.port),
            "--host", cfg.host,
            "--context-length", str(cfg.max_model_len),
            # SGLang's get_device() probes CUDA/XPU/HPU/NPU/MUSA/MPS and
            # raises if none are present — it does NOT fall through to
            # CPU on its own. The simulator is CPU-targeted, so be
            # explicit. Override via sglang_extra_flags if you ever want
            # to point at a GPU box.
            "--device", "cpu",
        ]
        if cfg.quantization:
            inner += ["--quantization", cfg.quantization]
        # ``--enable-metrics`` is what makes /metrics return Prometheus
        # data; without it the engine metrics sampler reads an empty body
        # and engine-side telemetry is NULL.
        if "--enable-metrics" not in cfg.sglang_extra_flags:
            inner.append("--enable-metrics")
        # Many Qwen / community model repos ship custom modelling code;
        # SGLang requires opting in. Cheap to allow by default — users
        # can override via sglang_extra_flags if they need to lock it down.
        if "--trust-remote-code" not in cfg.sglang_extra_flags:
            inner.append("--trust-remote-code")
        inner += list(cfg.sglang_extra_flags)

        # SGLang on CPU has no first-class thread-binding flag, so we wrap
        # the process with ``taskset -c`` when ``cpu_bind`` is set.
        # ``numactl`` would also work and pins memory locality more
        # tightly, but taskset is universally available and the
        # frequency aggregator only cares about the CPU set.
        cpu_list = flatten_for_taskset(cfg.cpu_bind)
        if cpu_list and shutil.which("taskset"):
            return ["taskset", "-c", cpu_list, *inner]
        return inner

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "64")
        # Sensible defaults for OpenMP thread placement when a CPU bind
        # is in effect — keeps threads on adjacent cores rather than
        # scattered across the socket.
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
        env.update(self.cfg.sglang_extra_env)
        return env

    def _health_path(self) -> str:
        return "/health"

    def _metrics_path(self) -> str:
        return "/metrics"
