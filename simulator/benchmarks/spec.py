"""Public recipe contract. Unknown settings are errors, never silently ignored."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSIONS = {"aiperf": "0.13.0", "guidellm": "0.7.4"}
ADAPTER_VERSION = 4


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    harnesses: list[Literal["aiperf", "guidellm"]] = Field(default_factory=list)
    recipe: Literal["reproduce", "sweep", "slo", "sustained"] = "reproduce"
    duration_s: int = Field(default=300, ge=10, le=86400)
    warmup_s: int = Field(default=30, ge=0, le=3600)
    repetitions: int = Field(default=3, ge=1, le=20)
    concurrency: list[int] = Field(default_factory=list, max_length=24)
    rates: list[float] = Field(default_factory=list, max_length=24)
    input_tokens: int | None = Field(default=None, ge=8, le=131072)
    output_tokens: int | None = Field(default=None, ge=1, le=131072)
    seed: int = Field(default=42, ge=0, le=2147483647)
    dataset_size: int = Field(default=256, ge=16, le=4096)
    cache_policy: Literal["warm", "restart"] = "warm"
    ignore_eos: bool = True
    ttft_p95_ms: float | None = Field(default=None, gt=0)
    tpot_p95_ms: float | None = Field(default=None, gt=0)
    max_error_fraction: float = Field(default=0.01, ge=0, le=1)
    min_requests: int = Field(default=100, ge=1, le=10000000)
    max_attempts: int = Field(default=2, ge=1, le=4)
    timeout_grace_s: int = Field(default=300, ge=10, le=3600)
    processing_grace_s: int = Field(default=900, ge=30, le=7200)
    min_free_gb: float = Field(default=10, ge=0.1, le=10000)
    disk_budget_gb: float = Field(default=20, ge=0.1, le=10000)
    max_hours: float = Field(default=72, gt=0, le=720)

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.harnesses)) != len(self.harnesses):
            raise ValueError("Select each harness once")
        if any(isinstance(v, bool) or v < 1 or v > 65536 for v in self.concurrency):
            raise ValueError("Concurrency must be between 1 and 65536")
        if any(not math.isfinite(v) or v <= 0 or v > 100000 for v in self.rates):
            raise ValueError("Request rates must be positive and at most 100000")
        if len(set(self.concurrency)) != len(self.concurrency) or len(set(self.rates)) != len(
            self.rates
        ):
            raise ValueError("Load points must be unique")
        if self.recipe == "slo":
            if not self.rates or (self.ttft_p95_ms is None and self.tpot_p95_ms is None):
                raise ValueError("SLO recipe requires request rates and a latency threshold")
        elif self.rates:
            raise ValueError("Request rates are only used by the SLO recipe")
        if self.recipe == "sweep" and not self.concurrency:
            raise ValueError("Sweep requires explicit concurrency points")
        if self.recipe in ("reproduce", "sustained") and len(self.concurrency) > 1:
            raise ValueError("Reproduce and sustained recipes use one load point")
        if self.recipe == "reproduce" and (self.input_tokens or self.output_tokens):
            raise ValueError("Reproduce uses recorded token lengths; choose sweep to override them")
        return self


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def winners(results: list[dict], catalog: list[dict]) -> list[dict]:
    """One confirmed throughput winner per explicit family and parameter size."""
    by_id = {m["id"]: m for m in catalog}
    groups: dict[str, list[dict]] = {}
    for row in results:
        rate = row.get("out_tok_s")
        if (
            not row.get("confirmed")
            or row.get("error")
            or row.get("steady_state") is False
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            continue
        meta = by_id.get(row["model"], {})
        # Do not infer equivalence from vendor/series or strip model names heuristically.
        family = meta.get("family") or row["model"]
        key = f"{family}|{meta.get('params_b', row['model'])}"
        groups.setdefault(key, []).append(row)
    return [
        sorted(rows, key=lambda r: (-r["out_tok_s"], digest(r)))[0]
        for _, rows in sorted(groups.items())
    ]


def resolve_spec(spec: BenchmarkSpec, row: dict) -> dict:
    d = spec.model_dump()
    d["input_tokens"] = spec.input_tokens or row.get("input_tokens")
    d["output_tokens"] = spec.output_tokens or row.get("output_tokens")
    if not d["input_tokens"] or not d["output_tokens"]:
        raise ValueError("This legacy result lacks recorded token lengths; choose a newer result")
    if not d["concurrency"] and spec.recipe != "slo":
        c = row.get("concurrency") or row.get("in_flight")
        if c is None or c < 1:
            raise ValueError("Recorded concurrency unavailable; specify a load point")
        d["concurrency"] = [max(1, round(c))]
    return d
