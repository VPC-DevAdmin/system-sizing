"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .preflight import HardwareRequirements


@dataclass
class ReplicaConfig:
    """One vLLM-CPU replica for multi-replica deployments.

    Used by ``vllm_dual_socket`` engine type — pins one vLLM container
    to a specific NUMA node via cpuset-cpus + cpuset-mems, exposes it
    on a per-replica port, and registers it as a backend with LiteLLM.
    """
    name: str = ""
    cpuset_cpus: str = ""
    cpuset_mems: str = ""
    port: int = 0
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class EngineConfig:
    type: str = "vllm"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    quantization: str | None = None
    max_model_len: int = 8192
    tensor_parallel_size: int = 4
    kv_cache_gb: int = 64
    port: int = 9100
    host: str = "127.0.0.1"
    startup_timeout_s: int = 600
    # Engine-agnostic CPU pinning. Same syntax as vLLM's
    # ``VLLM_CPU_OMP_THREADS_BIND`` (``|`` separates per-worker groups,
    # ``,`` and ``-`` work inside a group). Used for:
    #   * frequency-collector filter
    #   * vLLM: passed straight to VLLM_CPU_OMP_THREADS_BIND if unset
    #   * SGLang: flattened and used to wrap launch with ``taskset -c``
    cpu_bind: str | None = None
    vllm_extra_flags: list[str] = field(default_factory=list)
    vllm_extra_env: dict[str, str] = field(default_factory=dict)
    sglang_extra_flags: list[str] = field(default_factory=list)
    sglang_extra_env: dict[str, str] = field(default_factory=dict)

    # ── Docker-specific config (SGLang only) ──────────────────────────
    # SGLang's CPU support depends on a binary ``sgl_kernel`` that the
    # mainline pip wheel only ships in CUDA variants. On CPU the working
    # path is the upstream ``sglang-cpu`` Docker image, which we extend
    # with a small ``-fixed`` layer adding sentencepiece/tiktoken/protobuf.
    # See Dockerfile.xeon-fixed in the repo root.
    docker_image: str = "sglang-cpu:xeon-fixed"
    docker_shm_size: str = "32g"
    # Each entry maps host_path -> container_path. Models and HF cache
    # are the typical mounts.
    docker_volumes: dict[str, str] = field(default_factory=lambda: {
        "/data/ml/models": "/models",
        "/data/ml/huggingface": "/root/.cache/huggingface",
    })
    docker_network: str = "host"
    docker_extra_args: list[str] = field(default_factory=list)
    docker_extra_env: dict[str, str] = field(default_factory=dict)

    # ── SGLang launch flags (set sensible CPU defaults) ───────────────
    # Per the GNR runbook: TP=1 baseline + TP=4 are the only known-good
    # parallelism shapes on CPU. Anything else (DP/EP combos) is broken
    # upstream as of late 2025.
    served_model_name: str | None = None
    quantization_kind: str | None = None  # e.g. "fp8"; passed via --quantization
    # ``intel_amx`` on Xeon GNR/SPR/EMR; on AMD let SGLang fall back
    # (set to None to omit the flag entirely — SGLang picks torch_native).
    attention_backend: str | None = None
    # All four below default to None → flag omitted, SGLang picks its own
    # default. Set explicitly only when you need to cap KV pool / fix a
    # tuning knob. The minimal "known-working" launch from the GNR
    # runbook uses only max_total_tokens=16384 and disable_overlap_schedule.
    mem_fraction_static: float | None = None
    max_total_tokens: int | None = 16384
    chunked_prefill_size: int | None = None
    context_length: int | None = None  # if None we use max_model_len; set explicitly to skip
    disable_overlap_schedule: bool = True
    enable_metrics: bool = True       # required for /metrics telemetry path
    # Path INSIDE the container where the model lives. When set, used
    # in place of ``model_id`` for the SGLang ``--model`` flag — useful
    # for pre-downloaded weights mounted into /models.
    model_local_path: str | None = None

    # Hardware requirements that the host must satisfy. Validated by
    # ``simulator.preflight.preflight_check`` before launch — fails fast
    # so we don't waste minutes loading weights only to crash on, say,
    # SGLang's ``Fp8LinearMethod requires CPU AMX support`` deep in
    # weight processing on AMD.
    hardware_requirements: HardwareRequirements = field(
        default_factory=HardwareRequirements
    )

    # ── vllm_dual_socket: per-replica + LiteLLM-proxy fields ──────────
    # Used when ``type == "vllm_dual_socket"`` — two vLLM-CPU containers
    # pinned to different NUMA nodes, fronted by a LiteLLM proxy that
    # does session-based sticky routing. Image defaults match the
    # working AMD R7735 runbook.
    vllm_image: str = "vllm/vllm-openai-cpu:latest-x86_64"
    litellm_image: str = "ghcr.io/berriai/litellm:main-latest"
    litellm_master_key: str = "sk-local-dev-only"
    litellm_port: int = 4000
    replicas: list = field(default_factory=list)  # list[ReplicaConfig]
    shutdown_grace_s: int = 30

    @property
    def base_url(self) -> str:
        if self.type == "vllm":
            return f"http://{self.host}:{self.port}/v1"
        if self.type == "sglang":
            return f"http://{self.host}:{self.port}/v1"
        if self.type == "vllm_dual_socket":
            return f"http://{self.host}:{self.litellm_port}/v1"
        raise ValueError(f"Unknown engine type {self.type!r}")

    @property
    def api_key(self) -> str:
        """OpenAI-compatible API key. ``vllm_dual_socket`` requires the
        LiteLLM master key in the Authorization header; direct vLLM /
        SGLang accept any non-empty value."""
        if self.type == "vllm_dual_socket":
            return self.litellm_master_key
        return "EMPTY"


@dataclass
class SimulationConfig:
    initial_pool_size: int = 4
    max_pool_size: int = 1024
    target_samples_per_step: int = 500
    measurement_timeout_s: int = 300
    # ── Soft-start ramp ──
    # Spawning N users in a tight loop produces a synchronised burst
    # that takes minutes to dissolve into independent cycles (or never
    # does, on tight engines). Pace new spawns instead — one virtual
    # user per ``ramp_spawn_interval_s`` seconds — so users land in the
    # request/think cycle staggered.
    ramp_spawn_interval_s: float = 1.0

    # ── Initial phase offset ──
    # Each new virtual user sleeps for a random fraction of one
    # think-time sample before issuing its first request. This phases
    # users uniformly across their request/think cycle from spawn,
    # eliminating the "all 8 users hit /chat at the same instant"
    # problem even within a single ramp burst.
    initial_phase_offset_enabled: bool = True

    # ── Warmup floor ──
    # Don't even consider convergence until at least this many seconds
    # have passed (engine cold start, prefix cache warming, etc.).
    warmup_min_duration_s: int = 30
    # Hard ceiling — after this much time we proceed to measurement
    # whether convergence is detected or not. Better warmup-tail noise
    # than no data.
    warmup_max_duration_s: int = 300

    # ── Throughput convergence ──
    # Replaces the in-flight-CV detector. Compare completions/sec over
    # two consecutive ``convergence_window_s``-second windows; declare
    # converged when relative change drops below threshold. Throughput
    # actually converges (unlike in_flight which oscillates with the
    # closed-loop cycle period) so this is a meaningful signal.
    convergence_window_s: int = 60
    convergence_threshold: float = 0.20
    # Skip comparison until each window has at least this many
    # completions — avoids declaring "converged" on noise when the
    # engine is so slow only a handful of requests have finished.
    convergence_min_completions_per_window: int = 5
    knee_slope_threshold: float = 0.005
    stop_violation_threshold: float = 0.5
    # Above this violation rate, the adaptive stepper switches from
    # coarse-ramp doubling to bisection. Distinct from
    # stop_violation_threshold (the upper measurement bound) so that
    # marginal-zone measurements (e.g. 36% violation) bracket the knee
    # without halting further measurement. Default 0.20 matches the
    # ``capacity_status='fail'`` boundary in measurement.py.
    knee_zone_threshold: float = 0.20
    max_total_duration_minutes: int = 180
    enable_token_timestamps: bool = False
    snapshot_interval_s: int = 1
    request_timeout_s: int = 300


@dataclass
class TelemetryConfig:
    enable_pmu: bool = True
    enable_memory_bandwidth: bool = True
    enable_power: bool = True
    enable_engine_metrics: bool = True
    enable_amx_utilization: bool = True
    perf_events: list[str] = field(default_factory=lambda: [
        "cycles",
        "instructions",
        "cycle_activity.stalls_mem_any",
        "cycle_activity.stalls_l3_miss",
        "cache-misses",
        "cache-references",
        "longest_lat_cache.reference",
        "longest_lat_cache.miss",
        "offcore_requests.all_data_rd",
        "offcore_requests.demand_data_rd",
        "mem_load_l3_miss_retired.local_dram",
        "mem_load_l3_miss_retired.remote_dram",
    ])


@dataclass
class OutputConfig:
    db_directory: str = "runs"
    json_export: bool = True


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dataclass(target: Any, source: dict[str, Any]) -> None:
    for key, value in source.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    _merge_dataclass(cfg, raw)
    # ``engine.replicas`` is a list[ReplicaConfig] — yaml gives us list[dict].
    # The generic merger doesn't know to construct dataclasses from dicts
    # inside a list; do it explicitly here.
    if cfg.engine.replicas and isinstance(cfg.engine.replicas[0], dict):
        cfg.engine.replicas = [ReplicaConfig(**r) for r in cfg.engine.replicas]
    return cfg


def apply_cli_overrides(
    cfg: Config,
    *,
    engine: str | None = None,
    model: str | None = None,
) -> Config:
    if engine:
        cfg.engine.type = engine
    if model:
        cfg.engine.model_id = model
    # Allow env overrides for non-Make CLI paths
    if os.getenv("SIMULATOR_ENGINE"):
        cfg.engine.type = os.environ["SIMULATOR_ENGINE"]
    if os.getenv("SIMULATOR_MODEL"):
        cfg.engine.model_id = os.environ["SIMULATOR_MODEL"]
    return cfg
