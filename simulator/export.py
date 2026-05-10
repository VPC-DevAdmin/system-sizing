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
              ],
              timeline: {                        // phase distribution
                resolution_ms: 1000,             // per second
                schema: ["t_offset_s", "prefill", "decode",
                         "think", "idle"],
                rows: [[0,0,0,0,8], [1,2,0,0,6], ...]
              }
            },
            ...
          ]
        },
        ...
      ]
    }

The website iterates ``cohorts``, then ``curve`` for the rollup chart,
then drills into ``curve[i].turns`` / ``curve[i].telemetry_samples``
for per-step detail, or ``curve[i].timeline`` for phase-distribution
plots (storm/oscillation diagnostic — prior schemas had a cohort-level
``snapshots`` field, now superseded).
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

    **Cumulative-counter requirement.** Only emit ``engine_hit_rate``
    when both ``engine_hits`` and ``engine_queries`` (cumulative
    counters since engine launch) were scraped. SGLang's
    ``sglang:cache_hit_rate`` is a moving-window value, not a
    cumulative counter — taking that as the run-wide rate produces a
    snapshot of "what was happening at the end of the cohort," not
    "how often was the cache useful overall." The May 2026 Intel run
    surfaced this as wildly bimodal hit rates (0.0 for cohorts that
    ramped to high concurrency, 0.84–0.95 for cohorts that ended at
    low concurrency). When only the rate was scraped, drop it
    rather than mislead the buyer page.
    """
    eng_hits = run.get("prefix_cache_engine_hits")
    eng_queries = run.get("prefix_cache_engine_queries")
    eng_rate = run.get("prefix_cache_engine_hit_rate")
    if eng_hits is None and eng_queries is None and eng_rate is None:
        return prefix_cache
    out = dict(prefix_cache or {})
    # Counters are the trustworthy run-wide signal. Only when both
    # are present do we surface engine_hit_rate.
    if eng_hits is not None and eng_queries is not None:
        out["engine_hits"] = int(eng_hits)
        out["engine_queries"] = int(eng_queries)
        out["engine_hit_rate"] = (
            float(eng_rate) if eng_rate is not None else (
                float(eng_hits) / float(eng_queries) if eng_queries else None
            )
        )
    else:
        # Snapshot-only (e.g. SGLang) — record the source signal so
        # downstream consumers can see why engine_hit_rate is
        # missing, but DO NOT publish a misleading rate.
        out["engine_hit_rate_unavailable_reason"] = (
            "engine reports hit_rate as a moving-window value "
            "(e.g. sglang:cache_hit_rate); cumulative hits/queries "
            "counters were not exposed, so a run-wide rate cannot "
            "be computed reliably"
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

# NOTE: prior schema surfaced a per-cohort ``snapshots`` array
# (columns from ``simulation_snapshots`` — just aggregate in_flight
# over time). Removed in favor of per-step ``curve[i].timeline`` which
# carries finer-grain phase distribution (prefill/decode/think/idle).
# The ``simulation_snapshots`` DB table is still written at run time
# because the live dashboard reads it — but it's no longer echoed
# into the export JSON.


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
        # Token-volume aggregates per measurement window. Driven by
        # the user's question "how do I derive total token usage per
        # cohort over a fixed time" — surfacing these directly in the
        # export saves the consumer an SQL roundtrip and keeps the
        # buyer page free of derived-on-the-fly arithmetic. Reasoning
        # tokens are summed separately so reasoning-model overhead is
        # visible alongside content output.
        token_agg = conn.execute(
            """SELECT
                  COALESCE(SUM(input_tokens + history_tokens), 0)  AS prompt_tok,
                  COALESCE(SUM(output_tokens), 0)                  AS content_tok,
                  COALESCE(SUM(reasoning_tokens), 0)               AS reasoning_tok
               FROM turn_events
               WHERE measurement_id = ?""",
            (m["measurement_id"],),
        ).fetchone()
        m["tokens"] = dict(token_agg) if token_agg else {
            "prompt_tok": 0, "content_tok": 0, "reasoning_tok": 0,
        }
        # Per-step phase-distribution timeline: how many users in
        # each of prefill/decode/think/idle, sampled every second.
        # Replaces the per-cohort ``snapshots`` array (which only
        # carried aggregate in_flight). Built from turn_events so the
        # raw data already in the DB; no new run-time write needed.
        from .timeline import compute_timeline, timeline_to_export_dict
        m["timeline"] = timeline_to_export_dict(
            compute_timeline(conn, m["measurement_id"], resolution_ms=1000),
            resolution_ms=1000,
        )
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
        evidence["kv_cache_used_pct"] = kv
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


def _landing_zones(
    measurements: list[dict],
    status_field: str = "capacity_status",
) -> tuple[int | None, int | None, int | None]:
    """Pick the three landing-zone boundaries from a measurement list:

    Returns (capacity, soft_capacity, fail) where:

      * **capacity**: largest pool with status='pass' below the first
        non-pass that's never later "rescued" by a higher-pool pass.
        Premium quality — no SLA violations. The pre-existing
        ``capacity_pool_size`` semantic.

      * **soft_capacity**: largest pool with status in {'pass',
        'marginal'} below the first sustained 'fail'. Tolerable
        operating point — some fraction of users see slower
        responses, but no one's actually breaking the SLA.

      * **fail**: smallest pool with status='fail' AND no higher-pool
        non-fail status (i.e. the *sustained* fail point — bisection
        marginals that recover don't count). The hard cliff.

    These three numbers tell the buyer-page story directly:
        "supports up to {capacity} users at premium quality;
         tolerable up to {soft_capacity};
         degrades past {fail}."

    Use ``status_field="capacity_status"`` for the failure-bound
    landing zones (hard SLA — TTFT/TPOT past the *failure* threshold)
    and ``status_field="target_status"`` for the target-bound
    landing zones (premium-quality SLA — past the *target* threshold).

    Adaptive bisection produces non-monotonic curves — a measurement
    can flip 'pass' → 'marginal' → 'pass' again as the stepper
    re-samples around a noisy boundary. The "sustained" criterion
    (no later pool returns to a better band) tolerates that noise.

    Returns (None, None, None) when no measurements exist or the
    status field isn't populated.
    """
    if not measurements:
        return None, None, None
    # Drop measurements missing the requested status field — happens
    # for legacy (pre-target-status) data when querying target_status.
    typed = [m for m in measurements if m.get(status_field)]
    if not typed:
        return None, None, None
    sorted_m = sorted(typed, key=lambda m: m["target_pool_size"])

    # capacity: largest pass below first sustained non-pass.
    capacity_knee: int | None = None
    for i, m in enumerate(sorted_m):
        if m[status_field] == "pass":
            continue
        # Non-pass at i: counts as the capacity boundary only if no
        # higher pool returns to pass.
        if any(later[status_field] == "pass" for later in sorted_m[i + 1:]):
            continue
        capacity_knee = m["target_pool_size"]
        break
    pass_pools = [
        m["target_pool_size"] for m in sorted_m if m[status_field] == "pass"
    ]
    if capacity_knee is not None:
        pass_pools = [p for p in pass_pools if p < capacity_knee]
    capacity = max(pass_pools) if pass_pools else None

    # fail: smallest pool with status='fail' that's sustained
    # (no higher pool returns to pass-or-marginal).
    fail_pool: int | None = None
    for i, m in enumerate(sorted_m):
        if m[status_field] != "fail":
            continue
        if any(
            later[status_field] in ("pass", "marginal")
            for later in sorted_m[i + 1:]
        ):
            continue
        fail_pool = m["target_pool_size"]
        break

    # soft_capacity: largest pass-or-marginal below fail_pool.
    tolerable_pools = [
        m["target_pool_size"] for m in sorted_m
        if m[status_field] in ("pass", "marginal")
    ]
    if fail_pool is not None:
        tolerable_pools = [p for p in tolerable_pools if p < fail_pool]
    soft_capacity = max(tolerable_pools) if tolerable_pools else None

    return capacity, soft_capacity, fail_pool


# Backwards-compat alias for callers that want the old (capacity, knee)
# pair. Kept thin — new code should use ``_landing_zones`` directly.
def _capacity_and_knee(
    measurements: list[dict],
    status_field: str = "capacity_status",
) -> tuple[int | None, int | None]:
    capacity, _soft, _fail = _landing_zones(measurements, status_field)
    # Legacy "knee" was "first sustained non-pass" — equivalent to
    # the capacity-side boundary. Compute the same way for shim.
    if not measurements:
        return None, None
    typed = [m for m in measurements if m.get(status_field)]
    if not typed:
        return None, None
    sorted_m = sorted(typed, key=lambda m: m["target_pool_size"])
    knee_pool: int | None = None
    for i, m in enumerate(sorted_m):
        if m[status_field] == "pass":
            continue
        if any(later[status_field] == "pass" for later in sorted_m[i + 1:]):
            continue
        knee_pool = m["target_pool_size"]
        break
    return capacity, knee_pool


def _summarise_cohort(
    run: dict, prefix_cache: dict | None, slim: bool = False,
) -> dict:
    """Build the per-cohort export dict.

    ``slim=True`` drops the heavy time-series fields:
      * curve[i].telemetry_samples (per-second within-window samples)
      * curve[i].turns             (per-turn events)
      * curve[i].timeline          (phase-distribution per second)

    Headline summary, per-step rollup, capacity landing zones,
    bottleneck attribution, and prefix-cache verdict all stay.
    Typical size reduction: ~99% (35 MB → ~200-500 KB on a full
    AMD/Intel sweep). Use slim for buyer-page summary distribution;
    use the full export when drilling into per-step diagnostics."""
    measurements = run.get("measurements", [])
    curve = []
    for m in measurements:
        entry = {
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
            # TTFCT — Time to First Content Token. For non-reasoning
            # models == ttft trivially; for reasoning models lags
            # ttft by the chain-of-thought duration. Diagnostic only;
            # capacity_status still gates on ttft.
            "ttfct_p50_ms": m.get("ttfct_p50_ms"),
            "ttfct_p95_ms": m.get("ttfct_p95_ms"),
            "avg_reasoning_tokens": m.get("avg_reasoning_tokens"),
            # 0–100 percent (matches the engine's prometheus
            # ``kv_cache_used_pct`` field — naming aligned so a
            # downstream consumer doesn't multiply by 100 thinking
            # this is a 0..1 fraction).
            "kv_cache_used_pct": m["avg_kv_cache_pct"],
            "status": m["capacity_status"],          # failure-bound
            "target_status": m.get("target_status"), # target-bound
            "measurement_started_at": m.get("measurement_started_at"),
            "measurement_duration_s": m.get("measurement_duration_s"),
        }
        # Token-volume + per-second rates for this step. Always in
        # the export (slim or full) — small per-step overhead, big
        # value for buyer-page throughput sizing. Reasoning tokens
        # surfaced separately so consumers can see chain-of-thought
        # overhead vs the user-facing answer rate.
        tokens = m.get("tokens") or {}
        prompt_tok = tokens.get("prompt_tok") or 0
        content_tok = tokens.get("content_tok") or 0
        reasoning_tok = tokens.get("reasoning_tok") or 0
        dur_s = m.get("measurement_duration_s") or 0
        entry["prompt_tokens"] = prompt_tok
        entry["content_tokens"] = content_tok
        entry["reasoning_tokens"] = reasoning_tok
        entry["total_visible_output_tokens"] = content_tok + reasoning_tok
        if dur_s and dur_s > 0:
            entry["prompt_tok_per_s"] = round(prompt_tok / dur_s, 2)
            entry["content_tok_per_s"] = round(content_tok / dur_s, 2)
            entry["visible_output_tok_per_s"] = round(
                (content_tok + reasoning_tok) / dur_s, 2,
            )
        else:
            entry["prompt_tok_per_s"] = None
            entry["content_tok_per_s"] = None
            entry["visible_output_tok_per_s"] = None
        if not slim:
            # Per-step time-series; lists may be empty (e.g. a 'pending'
            # row for an in-progress step had no events captured).
            entry["telemetry_samples"] = m.get("telemetry_samples", [])
            entry["turns"] = m.get("turns", [])
            # Phase-distribution timeline (prefill/decode/think/idle
            # per second). Replaces the cohort-level ``snapshots``
            # field from the prior schema — finer grain (per-step
            # rather than per-cohort) and richer info (4 phases vs.
            # 1 aggregate counter). Slim mode drops it since it's a
            # diagnostic for storm detection, not a buyer-page metric.
            entry["timeline"] = m.get("timeline", {})
        curve.append(entry)
    # Three landing-zone boundaries on each axis:
    #   capacity  — premium / no SLA violations
    #   soft_cap  — acceptable / tolerable degradation, no actual fails
    #   fail_pool — degraded / hard cliff
    # On both the failure-bound (hard SLA) and target-bound (premium
    # SLA) axes — six numbers per cohort that map directly onto the
    # buyer-page narrative "fast / acceptable / degraded".
    capacity_pool, soft_capacity_pool, fail_pool = _landing_zones(
        measurements, "capacity_status",
    )
    target_capacity_pool, target_soft_capacity_pool, target_fail_pool = (
        _landing_zones(measurements, "target_status")
    )

    # Derived deployment-shape fields. These are pure post-processing
    # of the landing-zone numbers above — let the buyer-page frontend
    # render the right warnings without parsing the underlying numbers
    # itself.
    #
    # Computed on the TARGET axis (not the failure axis), because the
    # buyer-page narrative is "premium experience → tolerable → hard
    # fail." The target axis fires earlier and gives a wider band to
    # measure shape on; the failure axis is structurally narrow for
    # cohorts where the cliff is steep (capacity == soft_capacity on
    # the failure axis hides a meaningful target-band span). For
    # example, the May 2026 Intel quick_lookup run had failure-axis
    # capacity == soft_capacity == 232 (headroom=0) while the target
    # axis showed target_capacity=64 / target_soft=192 (real headroom
    # of 128 pool slots).
    #
    #   headroom = target_soft_capacity − target_capacity
    #              "graceful degradation zone" — how many users you
    #              can add past the premium cap before quality starts
    #              dropping below the target SLA.
    #
    #   cliff    = target_fail − target_soft_capacity
    #              "warning zone width" — pool slots between first
    #              target miss (5%) and substantial target miss
    #              (>30%). Wide = slow degradation; narrow = sharp.
    #
    #   deployment_band_shape:
    #     graceful  — cliff > 16 (gentle slope; oversize forgiving)
    #     moderate  — 8 ≤ cliff ≤ 16
    #     sharp     — cliff < 8 (oversize unforgiving — small pool
    #                            increase past soft_capacity collapses)
    #     unbounded — fail not observed within probed range
    #     unmeasured — capacity / soft_capacity not located
    headroom_pool = (
        target_soft_capacity_pool - target_capacity_pool
        if (
            target_soft_capacity_pool is not None
            and target_capacity_pool is not None
        )
        else None
    )
    cliff_pool = (
        target_fail_pool - target_soft_capacity_pool
        if (
            target_fail_pool is not None
            and target_soft_capacity_pool is not None
        )
        else None
    )
    if cliff_pool is None:
        if target_fail_pool is None and target_soft_capacity_pool is not None:
            band_shape = "unbounded"
        elif target_soft_capacity_pool is None:
            band_shape = "unmeasured"
        else:
            band_shape = "unmeasured"
    elif cliff_pool > 16:
        band_shape = "graceful"
    elif cliff_pool >= 8:
        band_shape = "moderate"
    else:
        band_shape = "sharp"

    # Measurement coverage describes how confident we are in the
    # capacity numbers. Currently a coarse three-way split — the
    # two-knee algorithm (forthcoming) will populate finer values:
    #
    #   full_curve       — the algorithm reached a sustained fail
    #                      AND has at least one passing measurement.
    #   downward_search  — initial pool size already failed; the
    #                      algorithm searched downward (NOT YET
    #                      IMPLEMENTED — placeholder for the two-knee
    #                      stepper that ships next).
    #   single_point     — only one measurement exists (cohort
    #                      collapsed at initial_pool_size with no
    #                      adaptive bisection).
    #   capped           — algorithm reached max_pool_size without
    #                      finding a sustained fail.
    if len(measurements) <= 1:
        coverage = "single_point"
    elif fail_pool is not None and capacity_pool is not None:
        coverage = "full_curve"
    elif fail_pool is None:
        coverage = "capped"
    else:
        coverage = "single_point"

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
        # Failure-bound landing zones (hard SLA — past the FAILURE
        # threshold). The "fast / acceptable / degraded" story.
        "capacity_pool_size": capacity_pool,                # premium / fast
        "soft_capacity_pool_size": soft_capacity_pool,      # acceptable / tolerable
        "fail_pool_size": fail_pool,                        # degraded / cliff
        # Target-bound landing zones (premium SLA — past the TARGET
        # threshold). Same three-zone shape, tighter bar.
        "target_capacity_pool_size": target_capacity_pool,
        "target_soft_capacity_pool_size": target_soft_capacity_pool,
        "target_fail_pool_size": target_fail_pool,
        # Derived deployment-shape fields — saves the buyer page from
        # having to compute them. See the comment block above the
        # ``headroom_pool`` / ``cliff_pool`` / ``band_shape`` /
        # ``coverage`` definitions.
        "headroom_pool_size": headroom_pool,
        "cliff_pool_size": cliff_pool,
        "deployment_band_shape": band_shape,
        "measurement_coverage": coverage,
        # Self-documenting band labels so the buyer page can render
        # the three-zone narrative without hardcoding the language
        # in the frontend.
        "capacity_landing_zones": {
            "fast": (
                f"≤{capacity_pool} concurrent users — no SLA violations"
                if capacity_pool is not None
                else "no clean-pass operating point — workload is "
                     "always at least marginal"
            ),
            "acceptable": (
                f"≤{soft_capacity_pool} concurrent users — some users "
                f"see slower-than-target responses but no failures"
                if soft_capacity_pool is not None
                else "no acceptable operating point — workload fails "
                     "at the smallest probed concurrency"
            ),
            "degraded": (
                f"{fail_pool}+ concurrent users — substantial fraction "
                f"of users hit the failure threshold"
                if fail_pool is not None
                else "no sustained-fail point observed within the "
                     "tested concurrency range"
            ),
        },
        "curve": curve,
        # NOTE: the prior ``snapshots`` cohort-level heartbeat field is
        # gone — superseded by per-step ``curve[i].timeline`` which
        # carries finer-grain phase distribution (prefill/decode/
        # think/idle) instead of just aggregate in_flight. The
        # ``simulation_snapshots`` table is still populated at run
        # time (the live dashboard reads it) but it's no longer
        # echoed into the export JSON.
        "bottleneck": bottleneck,                      # what limits SLA capacity
        "bottleneck_evidence": evidence,
        "target_bottleneck": target_bottleneck,        # what limits quality capacity
        "target_bottleneck_evidence": target_evidence,
        "hardware_recommendation": _hardware_recommendation(bottleneck),
        "prefix_cache": prefix_cache,
        # Throughput at the SLA-bound capacity pool — the headline
        # buyer-page question "how many tokens/sec does this hardware
        # sustain at the recommended deployment concurrency."
        "capacity_throughput": _capacity_throughput(
            measurements, capacity_pool,
        ),
    }


def _capacity_throughput(
    measurements: list[dict], capacity_pool: int | None,
) -> dict | None:
    """Aggregate token-volume + per-second rates at the cohort's
    capacity_pool_size measurement. Returns None when capacity isn't
    located (cohort never produced a clean-pass measurement).

    The block answers "deploy at this concurrency to sustain X
    tokens/sec at SLA." Tokens are split by phase:
      * prompt_tokens — input prefill load on the engine
      * content_tokens — user-facing answer output
      * reasoning_tokens — chain-of-thought (zero for non-reasoning)
      * total_visible_output_tokens = content + reasoning
    """
    if capacity_pool is None:
        return None
    matching = [
        m for m in measurements if m.get("target_pool_size") == capacity_pool
    ]
    if not matching:
        return None
    # If multiple measurements landed on the same pool size (e.g. a
    # spot-check appended a re-measurement), prefer the one with the
    # most samples — that's the more statistically meaningful row.
    m = max(matching, key=lambda r: r.get("sample_size") or 0)
    tokens = m.get("tokens") or {}
    prompt_tok = tokens.get("prompt_tok") or 0
    content_tok = tokens.get("content_tok") or 0
    reasoning_tok = tokens.get("reasoning_tok") or 0
    dur_s = m.get("measurement_duration_s") or 0
    sample_size = m.get("sample_size") or 0

    out = {
        "pool_size": capacity_pool,
        "measurement_duration_s": dur_s,
        "sample_size": sample_size,
        "prompt_tokens": prompt_tok,
        "content_tokens": content_tok,
        "reasoning_tokens": reasoning_tok,
        "total_visible_output_tokens": content_tok + reasoning_tok,
    }
    if dur_s and dur_s > 0:
        out["prompt_tok_per_s"] = round(prompt_tok / dur_s, 2)
        out["content_tok_per_s"] = round(content_tok / dur_s, 2)
        out["visible_output_tok_per_s"] = round(
            (content_tok + reasoning_tok) / dur_s, 2,
        )
    else:
        out["prompt_tok_per_s"] = None
        out["content_tok_per_s"] = None
        out["visible_output_tok_per_s"] = None
    return out


def export_dir(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    slim: bool = False,
) -> tuple[dict, Path]:
    """Build the buyer-page JSON. Returns ``(doc, output_path)``.

    With the ``runs/run_NN/`` layout, ``input_dir`` is the base ``runs``
    directory; we read DBs from the latest ``run_NN``. Flat-directory
    layouts still work as a fallback when no ``run_NN`` exists.

    ``slim=True`` produces a summary-only document (~99% smaller —
    typical 35 MB → 200-500 KB). Drops per-step time-series fields
    (telemetry_samples, turns) and the cohort-level 1 Hz heartbeat
    (snapshots). Headline summary, per-step rollup, capacity landing
    zones, bottleneck attribution, and prefix-cache verdict all stay.
    Default output filename in slim mode is
    ``buyer_page_data_slim.json`` so the full and slim exports can
    coexist in the same run dir.

    When ``output_path`` is None, the JSON lands at
    ``<source_dir>/buyer_page_data{_slim}.json`` — same directory
    as the ``run.db`` it came from, so per-run artifacts stay grouped.
    """
    from .runs import latest_run_dir
    input_dir = Path(input_dir)
    db_source = latest_run_dir(input_dir) or input_dir
    if output_path is None:
        fname = "buyer_page_data_slim.json" if slim else "buyer_page_data.json"
        output_path = db_source / fname
    else:
        output_path = Path(output_path)
    cohorts: list[dict] = []
    engine_seen: set[str] = set()
    model_seen: set[str] = set()
    # First-seen engine config (from config_json) becomes the run-wide
    # ``engine_config`` block in meta. All cohort_runs in a single
    # sweep share one config, so the first row is canonical.
    engine_config: dict | None = None
    # One run.db per run_NN/ holds every cohort_run for that invocation;
    # legacy flat layouts (one .db per cohort) iterate the same way.
    for db in sorted(db_source.glob("*.db")):
        for run in _read_runs(db):
            engine_seen.add(run["engine_type"])
            model_seen.add(run["model_id"])
            if engine_config is None:
                engine_config = _extract_engine_config(run)
            prefix_cache = _read_prefix_cache_report(
                db, cohort_run_id=run["cohort_run_id"],
            )
            prefix_cache = _attach_engine_hit_rate(prefix_cache, run)
            cohorts.append(_summarise_cohort(run, prefix_cache, slim=slim))

    doc = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engines": sorted(engine_seen),
            "models": sorted(model_seen),
            "source_dir": str(db_source),
            "cohort_count": len(cohorts),
            "slim": slim,
            # Engine launch parameters that affect throughput / capacity
            # numbers — KV pool size, chunked prefill, attention backend,
            # quantization, parallelism. Pinning them in the export lets
            # cross-host comparisons (Intel vs AMD, etc.) confirm the
            # configs were actually equivalent before reading the
            # capacity differences as host effects.
            "engine_config": engine_config,
        },
        "cohorts": cohorts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2))
    return doc, output_path


# Whitelisted engine config fields that materially affect throughput /
# capacity. Anything outside this list is excluded from the export
# either because it's noisy (timestamps, ports), redundant (model
# name is already in meta.models), or irrelevant to a buyer-page
# comparison (docker mounts).
_ENGINE_CONFIG_FIELDS = (
    "type",
    "model_id",
    "model_local_path",
    "served_model_name",
    "quantization",
    "quantization_kind",
    "max_model_len",
    "context_length",
    "max_total_tokens",
    "tensor_parallel_size",
    "kv_cache_gb",
    "chunked_prefill_size",
    "mem_fraction_static",
    "attention_backend",
    "disable_overlap_schedule",
    "enable_metrics",
    "cpu_bind",
    "docker_image",
    "vllm_extra_flags",
    "sglang_extra_flags",
)


def _extract_engine_config(run: dict) -> dict | None:
    """Pull a curated subset of the engine config from ``config_json``.

    Returns ``None`` if config_json can't be parsed (legacy / corrupt
    rows). The returned dict only contains fields from
    ``_ENGINE_CONFIG_FIELDS`` that were actually present — no
    placeholders for missing keys, so additions to ``EngineConfig``
    over time don't leave empty slots in old exports.
    """
    raw = run.get("config_json")
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except (TypeError, ValueError):
        return None
    engine = cfg.get("engine") if isinstance(cfg, dict) else None
    if not isinstance(engine, dict):
        return None
    return {k: engine[k] for k in _ENGINE_CONFIG_FIELDS if k in engine}
