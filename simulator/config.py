"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    vllm_extra_flags: list[str] = field(default_factory=list)
    vllm_extra_env: dict[str, str] = field(default_factory=dict)
    sglang_extra_flags: list[str] = field(default_factory=list)
    sglang_extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        if self.type == "vllm":
            return f"http://{self.host}:{self.port}/v1"
        if self.type == "sglang":
            return f"http://{self.host}:{self.port}/v1"
        raise ValueError(f"Unknown engine type {self.type!r}")


@dataclass
class SimulationConfig:
    initial_pool_size: int = 4
    max_pool_size: int = 1024
    target_samples_per_step: int = 500
    measurement_timeout_s: int = 300
    stabilization_cv_threshold: float = 0.15
    stabilization_min_duration_s: int = 60
    stabilization_max_duration_s: int = 300
    knee_slope_threshold: float = 0.005
    stop_violation_threshold: float = 0.5
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
