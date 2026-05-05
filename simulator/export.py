"""Export the buyer-facing JSON document.

One ``buyer_page_data.json`` per call, structured for direct consumption
by [web/index.html](../web/index.html) (or any downstream parser):

    {
      "meta": {generated_at, engines, models, source_dir, cohort_count},
      "cohorts": [
        {
          id, name, description, category, persona_weights,
          engine, model, started_at, completed_at, final_status,
          capacity_pool_size,    // last 'pass' before the knee
          knee_pool_size,        // first 'fail'/'marginal' step
          bottleneck,            // attributed at the knee
          bottleneck_evidence,
          hardware_recommendation,
          prefix_cache,          // analysis from turn_events

          curve: [               // per-step rollup (one entry per ramp step)
            {
              pool_size, sample_size, violation_rate, ttft_p95_ms, ...,
              telemetry_samples: [               // per-second within-window
                {sampled_at_ms, kv_cache_used_pct, queue_depth,
                 prefix_cache_hits, cpu_util_avg, engine_rss_gb,
                 freq_mhz_mean, freq_mhz_min, ...},
                ...
              ],
              turns: [                           // per-turn events in window
                {persona_id, session_id, turn_index, submitted_at_ms,
                 ttft_ms, tpot_ms, end_to_end_ms, input_tokens,
                 history_tokens, output_tokens, in_flight_at_submit,
                 sla_ttft_violation, sla_tpot_violation, ...},
                ...
              ]
            },
            ...
          ],

          snapshots: [           // 1 Hz pool/in-flight heartbeat for
                                 // the whole cohort run
            {snapshot_at_ms, phase, pool_size, in_flight,
             requests_completed, errors,
             step_samples, step_target_samples},
            ...
          ]
        },
        ...
      ]
    }

The website iterates ``cohorts``, then ``curve`` for the rollup chart,
then drills into ``curve[i].turns`` / ``curve[i].telemetry_samples``
for per-step detail, or ``snapshots`` for whole-run timeline plots.
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


def _attach_engine_hit_rate(prefix_cache: dict | None, run: dict) -> dict | None:
    """Fold the engine-reported (token-level) prefix-cache hit rate
    into the per-session analysis dict.

    The two numbers measure different things and should be read
    together — the per-session ``overall_hit_rate`` asks "did
    multi-turn sessions get a TTFT speedup?" (a stricter buyer-
    facing question), while ``engine_hit_rate`` asks "did any KV
    block get reused?" (the engine's own counter, including
    chat-template prefixes shared across users). On AMD without AMX,
    engine reuse can be high (~50%+) yet the TTFT speedup may not
    cross the analysis threshold — surfacing both keeps the
    downstream story honest.
    """
    eng_hits = run.get("prefix_cache_engine_hits")
    eng_queries = run.get("prefix_cache_engine_queries")
    eng_rate = run.get("prefix_cache_engine_hit_rate")
    if eng_hits is None and eng_queries is None and eng_rate is None:
        return prefix_cache
    out = dict(prefix_cache or {})
    out["engine_hits"] = int(eng_hits) if eng_hits is not None else None
    out["engine_queries"] = int(eng_queries) if eng_queries is not None else None
    out["engine_hit_rate"] = (
        float(eng_rate) if eng_rate is not None else None
    )
    return out


# Columns surfaced in the per-step ``telemetry_samples`` array. Kept
# explicit so downstream consumers see a stable schema even if new
# columns are added to ``measurement_telemetry`` later.
_TELEMETRY_SAMPLE_COLUMNS = (
    "sampled_at_ms",
    "kv_cache_used_pct",
    "queue_depth",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "cpu_util_avg",
    "memory_used_gb",
    "engine_rss_gb",
    "freq_mhz_mean",
    "freq_mhz_stddev",
    "freq_mhz_min",
)

# Columns surfaced in the per-step ``turns`` array. ``token_timestamps_json``
# is intentionally excluded — it can be very large (one entry per emitted
# token) and is opt-in tier-3 capture; consumers that want it should
# query the DB directly.
_TURN_EVENT_COLUMNS = (
    "persona_id",
    "user_id",
    "session_id",
    "turn_index",
    "submitted_at_ms",
    "ttft_ms",
    "completed_at_ms",
    "input_tokens",
    "history_tokens",
    "output_tokens",
    "tpot_ms",
    "end_to_end_ms",
    "in_flight_at_submit",
    "in_flight_avg_during",
    "in_flight_peak_during",
    "sla_ttft_violation",
    "sla_tpot_violation",
)

# Columns surfaced in the per-cohort ``snapshots`` array.
_SNAPSHOT_COLUMNS = (
    "snapshot_at_ms",
    "phase",
    "pool_size",
    "in_flight",
    "requests_completed",
    "errors",
    "step_samples",
    "step_target_samples",
)


def _read_cohort_run(conn: sqlite3.Connection, run_row: sqlite3.Row) -> dict:
    run = dict(run_row)
    # Window aggregates (PMU / BW / power / AMX / freq) live as columns
    # on cohort_measurements directly — no JOIN needed.
    ms = conn.execute(
        """SELECT * FROM cohort_measurements
           WHERE cohort_run_id = ?
           ORDER BY step_index ASC""",
        (run["cohort_run_id"],),
    ).fetchall()
    run["measurements"] = [dict(m) for m in ms]
    for m in run["measurements"]:
        # Per-second telemetry rollup: averages of kv usage / cpu /
        # engine RSS / freq across the window.
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
        # Raw per-second telemetry samples (drives in-window time-series
        # charts on the website).
        tele_cols = ", ".join(_TELEMETRY_SAMPLE_COLUMNS)
        m["telemetry_samples"] = [
            {k: r[k] for k in _TELEMETRY_SAMPLE_COLUMNS}
            for r in conn.execute(
                f"SELECT {tele_cols} FROM measurement_telemetry "
                f"WHERE measurement_id = ? ORDER BY sampled_at_ms",
                (m["measurement_id"],),
            )
        ]
        # Per-turn events captured during the window.
        turn_cols = ", ".join(_TURN_EVENT_COLUMNS)
        m["turns"] = [
            {k: r[k] for k in _TURN_EVENT_COLUMNS}
            for r in conn.execute(
                f"SELECT {turn_cols} FROM turn_events "
                f"WHERE measurement_id = ? ORDER BY submitted_at_ms",
                (m["measurement_id"],),
            )
        ]
    # Whole-run heartbeat (1 Hz pool/in-flight ticks).
    snap_cols = ", ".join(_SNAPSHOT_COLUMNS)
    run["snapshots"] = [
        {k: r[k] for k in _SNAPSHOT_COLUMNS}
        for r in conn.execute(
            f"SELECT {snap_cols} FROM simulation_snapshots "
            f"WHERE cohort_run_id = ? ORDER BY snapshot_at_ms",
            (run["cohort_run_id"],),
        )
    ]
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
      4. Effective freq MEAN < 2.5 GHz across bound CPUs -> frequency_droop
         (mean, not min — see the inline comment for why min is misleading)
      5. TTFT violations dominate TPOT (1.5x)        -> prefill_throughput
      6. Otherwise                                   -> decode_throughput
    """
    return _attribute_bottleneck(
        measurements,
        status_field="capacity_status",
        ttft_rate_field="ttft_violation_rate",
        tpot_rate_field="tpot_violation_rate",
    )


def _attribute_bottleneck(
    measurements: list[dict],
    *,
    status_field: str,
    ttft_rate_field: str,
    tpot_rate_field: str,
) -> tuple[str, dict]:
    """Generic bottleneck attributor — pick the knee per the given
    status field, read evidence, classify. Called twice from the
    export path: once for the failure-bound knee (capacity_status +
    violation rates) and once for the target-bound knee (target_status
    + target_miss rates). The two attributions can differ; if they
    do, the buyer page can show "first thing to bend = X, hard
    ceiling reason = Y" — sharper diagnostic than a single label."""
    knee = next(
        (m for m in measurements if m.get(status_field) in ("fail", "marginal")),
        None,
    )
    if knee is None and measurements:
        knee = measurements[-1]
    if knee is None:
        return "unknown", {}
    tele = knee.get("telemetry") or {}
    # Aggregate columns live directly on the measurement row now.
    agg = knee
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
    # Use the MEAN frequency, not the min, to detect droop.
    #
    # The frequency collector samples every CPU in ``bound_cpus`` —
    # but a workload (single-worker TP=1 with chunked prefill, say)
    # may not keep all cores busy at every sample tick, and Linux
    # parks idle cores into deep C-states at ~0.5 GHz. So
    # ``freq_min`` of 0.5 GHz commonly reflects parked-not-throttled
    # cores, not actual frequency throttling, and a min-based
    # heuristic over-attributes ``frequency_droop`` to cohorts
    # whose real bottleneck is decode or prefill compute.
    #
    # ``freq_mean`` averaged across the bound set is the honest
    # "is the active work happening below base clock?" signal.
    # Threshold of 2.5 GHz works for both Xeon 6761P (base 2.5,
    # all-core turbo ~3.0) and EPYC 9374F (base 3.85) — a mean
    # below 2.5 GHz on either CPU is unambiguously degraded.
    if freq_mean is not None and freq_mean < 2.5:
        return "frequency_droop", evidence

    ttft_v = knee.get(ttft_rate_field) or 0
    tpot_v = knee.get(tpot_rate_field) or 0
    evidence[ttft_rate_field] = ttft_v
    evidence[tpot_rate_field] = tpot_v
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


def _capacity_and_knee(
    measurements: list[dict],
    status_field: str = "capacity_status",
) -> tuple[int | None, int | None]:
    """Pick (capacity_pool_size, knee_pool_size) from a measurement
    list, gating on ``status_field``.

    Use ``status_field="capacity_status"`` for the failure-bound
    capacity (hard SLA — the headline buyer-page number) and
    ``status_field="target_status"`` for the target-bound capacity
    (premium-quality bar — the higher-quality number).

    Adaptive bisection produces non-monotonic curves — a measurement
    can flip 'pass' → 'marginal' → 'pass' again as the stepper
    re-samples around a noisy boundary. The previous "first non-pass"
    rule reported a low knee even when the cohort genuinely passed at
    higher pools, and "last pass observed" reported a capacity
    *greater* than the knee. Sort by pool_size and use the *sustained*
    non-pass (no later pool passes) as the knee:

      * knee = smallest pool_size where status is non-pass AND no
        higher pool size has status='pass'. Tolerates noise marginals
        the bisection later disproved.
      * capacity = largest pool_size with status='pass' that is
        strictly below the knee. Always ≤ knee.

    Returns (None, None) when no measurements exist or the status
    field isn't populated (e.g. legacy DB without target_status).
    """
    if not measurements:
        return None, None
    # Drop measurements missing the requested status field — happens
    # for legacy (pre-target-status) data when querying target_status.
    typed = [m for m in measurements if m.get(status_field)]
    if not typed:
        return None, None
    sorted_m = sorted(typed, key=lambda m: m["target_pool_size"])
    knee_pool: int | None = None
    for i, m in enumerate(sorted_m):
        if m[status_field] == "pass":
            continue
        # Non-pass: only counts as the knee if no pool above also passes.
        if any(later[status_field] == "pass" for later in sorted_m[i + 1:]):
            continue
        knee_pool = m["target_pool_size"]
        break
    pass_pools = [
        m["target_pool_size"] for m in sorted_m if m[status_field] == "pass"
    ]
    if knee_pool is not None:
        pass_pools = [p for p in pass_pools if p < knee_pool]
    capacity_pool = max(pass_pools) if pass_pools else None
    return capacity_pool, knee_pool


def _summarise_cohort(run: dict, prefix_cache: dict | None) -> dict:
    measurements = run.get("measurements", [])
    curve = []
    for m in measurements:
        curve.append({
            "step_index": m["step_index"],
            "pool_size": m["target_pool_size"],
            "sample_size": m["sample_size"],
            "violation_rate": m["combined_violation_rate"],
            "ttft_violation_rate": m["ttft_violation_rate"],
            "tpot_violation_rate": m["tpot_violation_rate"],
            # Target-miss rates: same axis but against the looser
            # target threshold. Surfaced per-step so the website
            # can plot both curves together.
            "target_miss_rate": m.get("combined_target_miss_rate"),
            "ttft_target_miss_rate": m.get("ttft_target_miss_rate"),
            "tpot_target_miss_rate": m.get("tpot_target_miss_rate"),
            "ci_lower": m["violation_rate_ci_lower"],
            "ci_upper": m["violation_rate_ci_upper"],
            "ttft_p50_ms": m["ttft_p50_ms"],
            "ttft_p95_ms": m["ttft_p95_ms"],
            "tpot_p50_ms": m["tpot_p50_ms"],
            "tpot_p95_ms": m["tpot_p95_ms"],
            "kv_cache_pct": m["avg_kv_cache_pct"],
            "status": m["capacity_status"],          # failure-bound
            "target_status": m.get("target_status"), # target-bound
            "measurement_started_at": m.get("measurement_started_at"),
            "measurement_duration_s": m.get("measurement_duration_s"),
            # Per-step time-series; lists may be empty (e.g. a 'pending'
            # row for an in-progress step had no events captured).
            "telemetry_samples": m.get("telemetry_samples", []),
            "turns": m.get("turns", []),
        })
    # Two parallel capacity readings — failure-bound (hard SLA, the
    # buyer-page headline) and target-bound (premium quality bar).
    capacity_pool, knee_pool = _capacity_and_knee(measurements, "capacity_status")
    target_capacity_pool, target_knee_pool = _capacity_and_knee(
        measurements, "target_status",
    )

    cohort_def = json.loads(run["cohort_definition_json"])
    # Two parallel bottleneck attributions — what limits SLA capacity
    # vs what limits premium-quality capacity. They can differ; if
    # they do, the buyer page can show "first thing to bend = X,
    # hard ceiling reason = Y" — sharper diagnostic than one label.
    bottleneck, evidence = _bottleneck(measurements)
    target_bottleneck, target_evidence = _attribute_bottleneck(
        measurements,
        status_field="target_status",
        ttft_rate_field="ttft_target_miss_rate",
        tpot_rate_field="tpot_target_miss_rate",
    )
    return {
        "id": run["cohort_id"],
        "cohort_run_id": run["cohort_run_id"],
        "name": cohort_def.get("name", run["cohort_id"]),
        "description": cohort_def.get("description", ""),
        "category": cohort_def.get("category", "mix"),
        "persona_weights": cohort_def.get("persona_weights", {}),
        "engine": run["engine_type"],
        "model": run["model_id"],
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
        "final_status": run["final_status"],
        "capacity_pool_size": capacity_pool,           # failure-bound (SLA)
        "knee_pool_size": knee_pool,
        "target_capacity_pool_size": target_capacity_pool,  # target-bound (quality)
        "target_knee_pool_size": target_knee_pool,
        "curve": curve,
        "snapshots": run.get("snapshots", []),
        "bottleneck": bottleneck,                      # what limits SLA capacity
        "bottleneck_evidence": evidence,
        "target_bottleneck": target_bottleneck,        # what limits quality capacity
        "target_bottleneck_evidence": target_evidence,
        "hardware_recommendation": _hardware_recommendation(bottleneck),
        "prefix_cache": prefix_cache,
    }


def export_dir(
    input_dir: str | Path,
    output_path: str | Path | None = None,
) -> tuple[dict, Path]:
    """Build the buyer-page JSON. Returns ``(doc, output_path)``.

    With the ``runs/run_NN/`` layout, ``input_dir`` is the base ``runs``
    directory; we read DBs from the latest ``run_NN``. Flat-directory
    layouts still work as a fallback when no ``run_NN`` exists.

    When ``output_path`` is None, the JSON lands at
    ``<source_dir>/buyer_page_data.json`` — same directory as the
    ``run.db`` it came from, so per-run artifacts stay grouped.
    """
    from .runs import latest_run_dir
    input_dir = Path(input_dir)
    db_source = latest_run_dir(input_dir) or input_dir
    if output_path is None:
        output_path = db_source / "buyer_page_data.json"
    else:
        output_path = Path(output_path)
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
            prefix_cache = _attach_engine_hit_rate(prefix_cache, run)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2))
    return doc, output_path
