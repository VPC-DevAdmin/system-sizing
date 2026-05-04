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

from .prefix_cache import (
    PrefixCacheReport,
    analyse_rows,
    read_turn_rows_with_step,
)


def _read_prefix_cache_report(
    path: Path, cohort_run_id: str | None = None,
) -> dict | None:
    try:
        rows = read_turn_rows_with_step(path, cohort_run_id=cohort_run_id)
    except Exception:
        return None
    if not rows:
        return None
    pure_rows = [r for _, r in rows]
    report = analyse_rows(pure_rows, rows_with_step=rows)
    return report.to_dict()


def _read_cohort_run(conn: sqlite3.Connection, run_row: sqlite3.Row) -> dict:
    run = dict(run_row)
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
    for m in run["measurements"]:
        # Per-second telemetry: averages of kv usage / cpu / engine RSS / freq.
        tele = conn.execute(
            """SELECT AVG(kv_cache_used_pct) AS kv,
                      AVG(cpu_util_avg) AS cpu,
                      AVG(engine_rss_gb) AS engine_rss_gb_avg,
                      AVG(freq_mhz_mean) AS freq_mhz_avg
               FROM measurement_telemetry
               WHERE measurement_id = ?""",
            (m["measurement_id"],),
        ).fetchone()
        m["telemetry"] = dict(tele) if tele else {}
        # Window aggregates (PMU/BW/power/AMX/freq).
        agg = conn.execute(
            "SELECT * FROM measurement_aggregate WHERE measurement_id = ?",
            (m["measurement_id"],),
        ).fetchone()
        m["aggregate"] = dict(agg) if agg else {}
    return run


def _read_runs(path: Path) -> list[dict]:
    """Read every cohort_run from a DB.

    Pre-consolidation DBs hold a single cohort_run; the new run.db per
    run_NN/ holds N (one per cohort/persona invocation in that run).
    Same code path either way."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT * FROM cohort_run ORDER BY started_at ASC"
        ).fetchall()
        return [_read_cohort_run(conn, r) for r in rows]
    finally:
        conn.close()


def _bottleneck(measurements: list[dict]) -> tuple[str, dict]:
    """Attribute the most likely bottleneck at the knee, best effort.

    Returns (bottleneck_id, evidence_dict). The evidence carries the
    actual numbers that drove the classification — for the buyer-page
    consumer to display under the recommendation.

    Heuristics evaluated in order; first match wins:
      1. KV cache utilisation > 90%                  -> kv_cache
      2. Measured BW > 75% of memory_theoretical    -> memory_bandwidth
         (or the perf stall_mem_ratio exceeds 0.5 if BW unknown)
      3. AMX dispatch fraction < 50% but matmul time > 0
                                                     -> amx_underutilised
      4. Effective freq < 90% of nominal             -> frequency_droop
      5. TTFT violations dominate TPOT (1.5x)        -> prefill_throughput
      6. Otherwise                                   -> decode_throughput
    """
    knee = next((m for m in measurements if m["capacity_status"] in ("fail", "marginal")), None)
    if knee is None and measurements:
        knee = measurements[-1]
    if knee is None:
        return "unknown", {}
    tele = knee.get("telemetry") or {}
    agg = knee.get("aggregate") or {}
    evidence: dict = {}

    kv = tele.get("kv")
    if kv is not None:
        evidence["kv_cache_pct"] = kv
        if kv > 90.0:
            return "kv_cache", evidence

    bw_read = agg.get("memory_bw_read_gb_s_avg")
    bw_write = agg.get("memory_bw_write_gb_s_avg")
    if bw_read is not None or bw_write is not None:
        total_bw = (bw_read or 0.0) + (bw_write or 0.0)
        evidence["memory_bw_total_gb_s"] = total_bw
        # Heuristic threshold; tune from real runs. 100 GB/s is a
        # sensible "sustained DDR5 saturation" floor on a single-socket
        # 8-channel host. The real number is theoretical * efficiency.
        if total_bw > 100.0:
            return "memory_bandwidth", evidence
    stall_ratio = agg.get("pmu_stall_mem_ratio")
    if stall_ratio is not None:
        evidence["pmu_stall_mem_ratio"] = stall_ratio
        if stall_ratio > 0.5:
            return "memory_bandwidth", evidence

    onednn_frac = agg.get("onednn_amx_time_fraction")
    if onednn_frac is not None:
        evidence["onednn_amx_time_fraction"] = onednn_frac
        if 0.0 < onednn_frac < 0.5:
            return "amx_underutilised", evidence

    freq_mean = agg.get("effective_freq_ghz_mean")
    freq_min = agg.get("effective_freq_ghz_min")
    if freq_mean is not None:
        evidence["effective_freq_ghz_mean"] = freq_mean
    if freq_min is not None:
        evidence["effective_freq_ghz_min"] = freq_min
        # Conservative droop floor: <2.5 GHz on a Xeon 6 nominal-3.0
        # all-core turbo workload is real droop worth flagging.
        if freq_min < 2.5:
            return "frequency_droop", evidence

    ttft_v = knee["ttft_violation_rate"] or 0
    tpot_v = knee["tpot_violation_rate"] or 0
    evidence["ttft_violation_rate"] = ttft_v
    evidence["tpot_violation_rate"] = tpot_v
    if ttft_v > tpot_v * 1.5:
        return "prefill_throughput", evidence
    return "decode_throughput", evidence


def _hardware_recommendation(bottleneck: str) -> str:
    return {
        "kv_cache": "Increase KV cache space (VLLM_CPU_KVCACHE_SPACE) or add RAM.",
        "memory_bandwidth": "Memory-bandwidth bound: prefer faster DDR5 / more channels, or scale out.",
        "amx_underutilised": "Matmul dispatch falling back to non-AMX kernels — check ONEDNN_VERBOSE log "
                              "and verify weights are in a layout the AMX kernels accept (BF16 / W8A8).",
        "frequency_droop": "Effective frequency is below nominal turbo — check thermals, power caps, and "
                            "whether VLLM_CPU_OMP_THREADS_BIND oversubscribes physical cores.",
        "prefill_throughput": "Prefill bound: more cores / AMX-capable SKU, or shorter prompts.",
        "decode_throughput": "Decode bound: faster cores or smaller / quantised model.",
        "unknown": "Insufficient telemetry to attribute. Re-run with PMU enabled.",
    }.get(bottleneck, "")


def _summarise_cohort(run: dict, prefix_cache: dict | None) -> dict:
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
    bottleneck, evidence = _bottleneck(measurements)
    return {
        "id": run["cohort_id"],
        "name": cohort_def.get("name", run["cohort_id"]),
        "description": cohort_def.get("description", ""),
        "category": cohort_def.get("category", "mix"),
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
        "bottleneck_evidence": evidence,
        "hardware_recommendation": _hardware_recommendation(bottleneck),
        "prefix_cache": prefix_cache,
    }


def export_dir(input_dir: str | Path, output_path: str | Path) -> dict:
    """Build the buyer-page JSON.

    With the ``runs/run_NN/`` layout, ``input_dir`` is the base ``runs``
    directory; we read DBs from the latest ``run_NN``. Flat-directory
    layouts still work as a fallback when no ``run_NN`` exists.
    """
    from .runs import latest_run_dir
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    db_source = latest_run_dir(input_dir) or input_dir
    cohorts: list[dict] = []
    engine_seen: set[str] = set()
    model_seen: set[str] = set()
    # One run.db per run_NN/ holds every cohort_run for that invocation;
    # legacy flat layouts (one .db per cohort) iterate the same way.
    for db in sorted(db_source.glob("*.db")):
        for run in _read_runs(db):
            engine_seen.add(run["engine_type"])
            model_seen.add(run["model_id"])
            prefix_cache = _read_prefix_cache_report(
                db, cohort_run_id=run["cohort_run_id"],
            )
            cohorts.append(_summarise_cohort(run, prefix_cache))

    doc = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engines": sorted(engine_seen),
            "models": sorted(model_seen),
            "source_dir": str(db_source),
            "cohort_count": len(cohorts),
        },
        "cohorts": cohorts,
    }
    output_path.write_text(json.dumps(doc, indent=2))
    return doc
