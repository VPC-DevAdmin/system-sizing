"""Export simplified JSON for the buyer-facing capacity-planning page.

Walks all .db files in a directory and emits one JSON document with:
  * meta — engine, model, generation timestamp
  * cohorts — array of cohort summaries, each with:
      * id, name, description
      * curve — list of {pool_size, violation_rate, ci_lower, ci_upper,
                          ttft_p95_ms, tpot_p95_ms, status}
      * capacity_pool_size — last 'pass' point before knee
      * knee_pool_size — first 'fail' point (or last marginal if no fail)
      * bottleneck — best-effort attribution string
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _read_run(path: Path) -> dict | None:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception:
        return None
    try:
        run = conn.execute(
            "SELECT * FROM cohort_run ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if run is None:
            return None
        run = dict(run)
        ms = conn.execute(
            """SELECT step_index, target_pool_size, sample_size,
                      ttft_violation_rate, tpot_violation_rate,
                      combined_violation_rate,
                      violation_rate_ci_lower, violation_rate_ci_upper,
                      ttft_p50_ms, ttft_p95_ms, tpot_p50_ms, tpot_p95_ms,
                      avg_kv_cache_pct, capacity_status, measurement_id
               FROM cohort_measurements
               WHERE cohort_run_id = ?
               ORDER BY step_index ASC""",
            (run["cohort_run_id"],),
        ).fetchall()
        run["measurements"] = [dict(m) for m in ms]
        # Aggregate telemetry per measurement (averages we care about)
        for m in run["measurements"]:
            tele = conn.execute(
                """SELECT AVG(kv_cache_used_pct) AS kv,
                          AVG(memory_bw_read_gb_s) AS bw_r,
                          AVG(memory_bw_write_gb_s) AS bw_w,
                          AVG(cpu_util_avg) AS cpu,
                          AVG(CAST(pmu_stalls_mem_any AS REAL) /
                              NULLIF(pmu_cycles, 0)) AS mem_stall_ratio
                   FROM measurement_telemetry
                   WHERE measurement_id = ?""",
                (m["measurement_id"],),
            ).fetchone()
            m["telemetry"] = dict(tele) if tele else {}
        return run
    finally:
        conn.close()


def _bottleneck(measurements: list[dict]) -> str:
    """Attribute the most likely bottleneck at the knee, best effort.

    Heuristics, in order:
      * If KV cache > 90% near knee -> "kv_cache"
      * If memory stall ratio > 0.5 -> "memory_bandwidth"
      * If TTFT-violation dominates TPOT-violation -> "prefill_throughput"
      * Else -> "decode_throughput"
    """
    knee = next((m for m in measurements if m["capacity_status"] in ("fail", "marginal")), None)
    if knee is None and measurements:
        knee = measurements[-1]
    if knee is None:
        return "unknown"
    tele = knee.get("telemetry") or {}
    kv = tele.get("kv")
    if kv is not None and kv > 90.0:
        return "kv_cache"
    mem_stall = tele.get("mem_stall_ratio")
    if mem_stall is not None and mem_stall > 0.5:
        return "memory_bandwidth"
    ttft_v = knee["ttft_violation_rate"] or 0
    tpot_v = knee["tpot_violation_rate"] or 0
    if ttft_v > tpot_v * 1.5:
        return "prefill_throughput"
    return "decode_throughput"


def _hardware_recommendation(bottleneck: str) -> str:
    return {
        "kv_cache": "Increase KV cache space (VLLM_CPU_KVCACHE_SPACE) or add RAM.",
        "memory_bandwidth": "Memory-bandwidth bound: prefer faster DDR5 / more channels, or scale out.",
        "prefill_throughput": "Prefill bound: more cores / AMX-capable SKU, or shorter prompts.",
        "decode_throughput": "Decode bound: faster cores or smaller / quantised model.",
        "unknown": "Insufficient telemetry to attribute. Re-run with PMU enabled.",
    }.get(bottleneck, "")


def _summarise_cohort(run: dict) -> dict:
    measurements = run.get("measurements", [])
    curve = []
    capacity_pool = None
    knee_pool = None
    for m in measurements:
        curve.append({
            "pool_size": m["target_pool_size"],
            "sample_size": m["sample_size"],
            "violation_rate": m["combined_violation_rate"],
            "ttft_violation_rate": m["ttft_violation_rate"],
            "tpot_violation_rate": m["tpot_violation_rate"],
            "ci_lower": m["violation_rate_ci_lower"],
            "ci_upper": m["violation_rate_ci_upper"],
            "ttft_p50_ms": m["ttft_p50_ms"],
            "ttft_p95_ms": m["ttft_p95_ms"],
            "tpot_p50_ms": m["tpot_p50_ms"],
            "tpot_p95_ms": m["tpot_p95_ms"],
            "kv_cache_pct": m["avg_kv_cache_pct"],
            "status": m["capacity_status"],
        })
        if m["capacity_status"] == "pass":
            capacity_pool = m["target_pool_size"]
        if m["capacity_status"] in ("fail", "marginal") and knee_pool is None:
            knee_pool = m["target_pool_size"]

    cohort_def = json.loads(run["cohort_definition_json"])
    bottleneck = _bottleneck(measurements)
    return {
        "id": run["cohort_id"],
        "name": cohort_def.get("name", run["cohort_id"]),
        "description": cohort_def.get("description", ""),
        "persona_weights": cohort_def.get("persona_weights", {}),
        "engine": run["engine_type"],
        "model": run["model_id"],
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
        "final_status": run["final_status"],
        "capacity_pool_size": capacity_pool,
        "knee_pool_size": knee_pool,
        "curve": curve,
        "bottleneck": bottleneck,
        "hardware_recommendation": _hardware_recommendation(bottleneck),
    }


def export_dir(input_dir: str | Path, output_path: str | Path) -> dict:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    cohorts: list[dict] = []
    engine_seen: set[str] = set()
    model_seen: set[str] = set()
    for db in sorted(input_dir.glob("*.db")):
        run = _read_run(db)
        if run is None:
            continue
        engine_seen.add(run["engine_type"])
        model_seen.add(run["model_id"])
        cohorts.append(_summarise_cohort(run))

    doc = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engines": sorted(engine_seen),
            "models": sorted(model_seen),
            "source_dir": str(input_dir),
            "cohort_count": len(cohorts),
        },
        "cohorts": cohorts,
    }
    output_path.write_text(json.dumps(doc, indent=2))
    return doc
