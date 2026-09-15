#!/usr/bin/env python3
"""Build slim data files for the in-repo capacity site.

Reads the full simulator exports (20 MB+ each) and writes small JS data
files the static pages load via <script> (works from file:// — no fetch/CORS).

Usage:
    python3 site/build_data.py            # uses artifacts/*.json
    python3 site/build_data.py --intel path.json --amd path.json
"""
import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"

CURVE_FIELDS = [
    "pool_size", "sample_size", "violation_rate", "ttft_violation_rate",
    "tpot_violation_rate", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
    "tpot_p95_ms", "kv_cache_used_pct", "status",
    "prompt_tok_per_s", "visible_output_tok_per_s", "measurement_duration_s",
]

COHORT_FIELDS = [
    "id", "name", "description", "category", "persona_weights",
    "capacity_pool_size", "soft_capacity_pool_size", "fail_pool_size",
    "headroom_pool_size", "cliff_pool_size", "deployment_band_shape",
    "bottleneck", "bottleneck_evidence", "capacity_landing_zones",
    "capacity_throughput",
]

TURN_FIELDS = ["persona_id", "ttft_ms", "tpot_ms", "input_tokens",
               "output_tokens", "end_to_end_ms", "in_flight_at_submit",
               "sla_ttft_violation", "sla_tpot_violation"]


def rnd(v, nd=2):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd)
    return v


def slim_point(pt):
    return {k: rnd(pt.get(k)) for k in CURVE_FIELDS}


def slim_turns(pt):
    """Replay feed: turns ordered by submit time, offsets in seconds."""
    turns = pt.get("turns") or []
    if not turns:
        return []
    turns = sorted(turns, key=lambda t: t.get("submitted_at_ms") or 0)
    t0 = turns[0].get("submitted_at_ms") or 0
    out = []
    for t in turns:
        rec = {k: rnd(t.get(k), 1) for k in TURN_FIELDS}
        rec["t_s"] = rnd(((t.get("submitted_at_ms") or t0) - t0) / 1000.0, 2)
        out.append(rec)
    return out


def downsample_telemetry(pt, max_points=72):
    samples = pt.get("telemetry_samples") or []
    if not samples:
        return []
    step = max(1, len(samples) // max_points)
    picked = samples[::step][:max_points]
    t0 = picked[0].get("sampled_at_ms") or 0
    out = []
    for s in picked:
        out.append({
            "t_s": rnd(((s.get("sampled_at_ms") or t0) - t0) / 1000.0, 1),
            "freq_ghz": rnd((s.get("freq_mhz_mean") or 0) / 1000.0, 3) or None,
            "kv_pct": rnd(s.get("kv_cache_used_pct"), 2),
            "mem_gb": rnd(s.get("memory_used_gb"), 1),
        })
    return out


def replay_point(cohort):
    """Pick the curve point used for the live replay: the measured point at the
    soft-capacity knee, else the highest passing point, else the densest."""
    curve = cohort.get("curve") or []
    if not curve:
        return None
    soft = cohort.get("soft_capacity_pool_size")
    by_pool = {p["pool_size"]: p for p in curve}
    if soft in by_pool:
        return by_pool[soft]
    passing = [p for p in curve if p.get("status") == "pass"]
    if passing:
        return max(passing, key=lambda p: p["pool_size"])
    return curve[-1]


def build(src_path, label):
    data = json.loads(Path(src_path).read_text())
    meta = data.get("meta", {})
    eng = meta.get("engine_config", {}) or {}
    out = {
        "platform": label,
        "generated_at": meta.get("generated_at"),
        "engine": (meta.get("engines") or [None])[0],
        "model": (meta.get("models") or [None])[0],
        "engine_config": {
            "max_model_len": eng.get("max_model_len"),
            "kv_cache_gb": eng.get("kv_cache_gb"),
            "quantization_kind": eng.get("quantization_kind"),
            "attention_backend": eng.get("attention_backend"),
        },
        "cohorts": [],
    }
    for c in data.get("cohorts", []):
        sc = {k: c.get(k) for k in COHORT_FIELDS}
        sc["curve"] = [slim_point(p) for p in (c.get("curve") or [])]
        rp = replay_point(c)
        if rp is not None:
            sc["replay"] = {
                "pool_size": rp.get("pool_size"),
                "status": rp.get("status"),
                "duration_s": rnd(rp.get("measurement_duration_s"), 1),
                "turns": slim_turns(rp),
                "telemetry": downsample_telemetry(rp),
            }
        out["cohorts"].append(sc)
    return out


def write_js(obj, var, path):
    payload = json.dumps(obj, separators=(",", ":"))
    path.write_text(f"window.{var} = {payload};\n")
    print(f"wrote {path}  ({path.stat().st_size/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intel", default=str(REPO / "artifacts/Intel_sizing_qwen3.json"))
    ap.add_argument("--amd", default=str(REPO / "artifacts/AMD_sizing_qwen3.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_js(build(args.intel, "intel"), "SIZING_DATA", OUT / "intel-data.js")
    write_js(build(args.amd, "amd"), "SIZING_DATA", OUT / "amd-data.js")


if __name__ == "__main__":
    main()
