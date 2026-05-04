"""Measurement controller — three-phase orchestration: ramp → stabilize → measure."""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .virtual_user import SharedState, TurnEvent

log = logging.getLogger(__name__)


PHASE_IDLE = "idle"
PHASE_RAMPING = "ramping"
PHASE_STABILIZING = "stabilizing"
PHASE_MEASURING = "measuring"


@dataclass
class MeasurementResult:
    status: str   # 'ok' | 'unstable' | 'no_samples' | 'aborted'
    target_pool_size: int
    sample_size: int = 0
    measurement_id: Optional[int] = None
    measured_avg_pool_size: float = 0.0
    measured_avg_in_flight: float = 0.0
    measurement_duration_s: int = 0
    ttft_violation_rate: float = 0.0
    tpot_violation_rate: float = 0.0
    combined_violation_rate: float = 0.0
    violation_rate_ci_lower: float = 0.0
    violation_rate_ci_upper: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p75_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p75_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    avg_kv_cache_pct: Optional[float] = None
    estimated_prefix_hit_rate: Optional[float] = None
    capacity_status: str = "pass"
    events: list[TurnEvent] = field(default_factory=list)


class PhaseTracker:
    """Holds the current phase string for telemetry to read."""
    def __init__(self):
        self.phase = PHASE_IDLE

    def set(self, phase: str) -> None:
        self.phase = phase

    def get(self) -> str:
        return self.phase


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


async def _wait_for_throughput_convergence(
    state: SharedState,
    *,
    min_warmup_s: int,
    max_wait_s: int,
    window_s: int,
    threshold: float,
    min_completions_per_window: int,
    in_flight_history: list[int] | None = None,
) -> tuple[bool, str]:
    """Wait until completion-throughput has converged.

    Closed-loop persona workloads have intrinsically bursty in_flight
    counts (each user oscillates between request and think states), so
    ``CV(in_flight)`` never converges regardless of threshold or pool
    size. Throughput (completions/sec) is the right signal: it
    converges to the system's max sustainable rate as warmup completes.

    We track per-second snapshots of ``state.completed`` and compare
    the count of completions in the most-recent ``window_s`` seconds
    against the count in the previous ``window_s`` seconds. Converged
    when relative change drops below ``threshold``.

    Both windows must have at least ``min_completions_per_window``
    completions to be compared — protects against declaring "converged"
    on noise when the engine is so slow only a handful of requests
    have finished.

    Returns ``(converged, reason)``. ``converged`` is True if the
    threshold was crossed within ``max_wait_s``, False otherwise.
    ``reason`` is a short human-readable explanation for the run log.
    The caller proceeds to measurement either way — better warmup-tail
    noise than no data.
    """
    start = time.monotonic()
    snapshots: deque[int] = deque()
    snapshots.append(state.completed)
    last_reason = "still warming up"

    while time.monotonic() - start < max_wait_s:
        if in_flight_history is not None:
            in_flight_history.append(state.in_flight)
        await asyncio.sleep(1.0)
        elapsed = time.monotonic() - start
        snapshots.append(state.completed)
        # Keep ~2*window_s + 1 entries (one per second).
        while len(snapshots) > 2 * window_s + 1:
            snapshots.popleft()

        if elapsed < min_warmup_s:
            last_reason = f"warmup floor ({elapsed:.0f}s < {min_warmup_s}s)"
            continue
        if len(snapshots) < 2 * window_s + 1:
            last_reason = (
                f"need {2*window_s}s of history, have {len(snapshots)-1}s"
            )
            continue

        # snapshots[0] == count 2*window_s ago; snapshots[-window_s-1] == window_s ago; snapshots[-1] == now.
        now_completed = snapshots[-1]
        window_ago = snapshots[-window_s - 1]
        two_window_ago = snapshots[0]
        recent = now_completed - window_ago
        prior = window_ago - two_window_ago

        if recent < min_completions_per_window or prior < min_completions_per_window:
            last_reason = (
                f"throughput too low to compare: prior={prior}, "
                f"recent={recent}, need {min_completions_per_window}"
            )
            continue

        rel_diff = abs(recent - prior) / max(recent, prior)
        last_reason = (
            f"prior={prior}/{window_s}s ({prior/window_s:.2f} req/s), "
            f"recent={recent}/{window_s}s ({recent/window_s:.2f} req/s), "
            f"rel_diff={rel_diff:.3f}"
        )
        if rel_diff < threshold:
            return True, f"converged: {last_reason}"

    return False, f"timed out at {max_wait_s}s: {last_reason}"


async def run_measurement_step(
    pool,
    state: SharedState,
    phase_tracker: PhaseTracker,
    telemetry,
    db,
    *,
    cohort_run_id: str,
    step_index: int,
    target_pool_size: int,
    target_samples: int,
    measurement_timeout_s: int,
    warmup_min_duration_s: int,
    warmup_max_duration_s: int,
    convergence_window_s: int,
    convergence_threshold: float,
    convergence_min_completions_per_window: int,
) -> MeasurementResult:
    """Ramp to target_pool_size with soft-start, await throughput
    convergence, then measure."""
    log.info(
        "Step %d: soft-ramping pool to %d (interval %.1fs)",
        step_index, target_pool_size, getattr(pool, "_ramp_spawn_interval_s", 0.0),
    )
    phase_tracker.set(PHASE_RAMPING)
    await pool.set_target_size(target_pool_size)

    phase_tracker.set(PHASE_STABILIZING)  # phase name kept for dashboard back-compat
    log.info(
        "Step %d: awaiting throughput convergence "
        "(window=%ds, threshold=%.2f, min_warmup=%ds, max_wait=%ds)",
        step_index, convergence_window_s, convergence_threshold,
        warmup_min_duration_s, warmup_max_duration_s,
    )
    in_flight_track: list[int] = []
    converged, reason = await _wait_for_throughput_convergence(
        state,
        min_warmup_s=warmup_min_duration_s,
        max_wait_s=warmup_max_duration_s,
        window_s=convergence_window_s,
        threshold=convergence_threshold,
        min_completions_per_window=convergence_min_completions_per_window,
        in_flight_history=in_flight_track,
    )
    if converged:
        log.info("Step %d: %s", step_index, reason)
    else:
        log.warning(
            "Step %d: throughput not converged — measuring anyway (%s)",
            step_index, reason,
        )

    # Drain queue of pre-measurement events
    while not state.events.empty():
        try:
            state.events.get_nowait()
        except asyncio.QueueEmpty:
            break

    phase_tracker.set(PHASE_MEASURING)
    measurement_started_at = datetime.now(timezone.utc).isoformat()
    measurement_start_mono = time.monotonic()
    log.info("Step %d: measuring (target_samples=%d, timeout=%ds)",
             step_index, target_samples, measurement_timeout_s)

    # Insert measurement row up-front so telemetry can attach
    pre_row = {
        "cohort_run_id": cohort_run_id,
        "step_index": step_index,
        "target_pool_size": target_pool_size,
        "measured_avg_pool_size": float(target_pool_size),
        "measured_avg_in_flight": 0.0,
        "measurement_started_at": measurement_started_at,
        "measurement_duration_s": 0,
        "sample_size": 0,
        "ttft_violation_rate": 0.0,
        "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0,
        "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 0.0,
        "ttft_p50_ms": 0.0, "ttft_p75_ms": 0.0, "ttft_p95_ms": 0.0,
        "tpot_p50_ms": 0.0, "tpot_p75_ms": 0.0, "tpot_p95_ms": 0.0,
        "avg_kv_cache_pct": None,
        "estimated_prefix_hit_rate": None,
        "capacity_status": "pending",
    }
    measurement_id = db.insert_measurement(pre_row)

    telemetry.start(measurement_id)
    in_flight_during: list[int] = []
    sampler_task = asyncio.create_task(_in_flight_sampler(state, in_flight_during))

    buffer: list[TurnEvent] = []
    deadline = measurement_start_mono + measurement_timeout_s
    try:
        while len(buffer) < target_samples:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                event = await asyncio.wait_for(state.events.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            buffer.append(event)
    finally:
        sampler_task.cancel()
        try:
            await sampler_task
        except (asyncio.CancelledError, Exception):
            pass

    duration = int(time.monotonic() - measurement_start_mono)
    _, telemetry_rows, telemetry_agg = await telemetry.stop()
    db.insert_telemetry(telemetry_rows)
    if telemetry_agg:
        agg_row = {"measurement_id": measurement_id}
        # Restrict to columns that exist in measurement_aggregate.
        for k in (
            "pmu_cycles", "pmu_instructions", "pmu_ipc",
            "pmu_stalls_mem_any", "pmu_stalls_l3_miss", "pmu_stall_mem_ratio",
            "pmu_amx_ops", "amx_perf_event_name",
            "pmu_llc_reference", "pmu_llc_miss",
            "mem_local_fraction", "mem_remote_fraction",
            "memory_bw_read_gb_s_avg", "memory_bw_read_gb_s_peak",
            "memory_bw_write_gb_s_avg", "memory_bw_write_gb_s_peak",
            "bandwidth_status",
            "power_w_avg", "power_w_peak", "power_status",
            "effective_freq_ghz_mean", "effective_freq_ghz_stddev",
            "effective_freq_ghz_min",
        ):
            if telemetry_agg.get(k) is not None:
                agg_row[k] = telemetry_agg[k]
        if len(agg_row) > 1:
            db.upsert_aggregate(agg_row)
    phase_tracker.set(PHASE_IDLE)

    if not buffer:
        log.warning("Step %d: no samples captured", step_index)
        return MeasurementResult(
            status="no_samples",
            target_pool_size=target_pool_size,
            measurement_id=measurement_id,
        )

    # Aggregate
    ttft_values = [e.ttft_ms for e in buffer]
    tpot_values = [e.tpot_ms for e in buffer]
    ttft_violations = sum(1 for e in buffer if e.ttft_violation())
    tpot_violations = sum(1 for e in buffer if e.tpot_violation())
    combined_violations = sum(
        1 for e in buffer if e.ttft_violation() or e.tpot_violation()
    )
    n = len(buffer)
    ttft_rate = ttft_violations / n
    tpot_rate = tpot_violations / n
    combined_rate = combined_violations / n
    ci_lo, ci_hi = _wilson_ci(combined_violations, n)

    avg_in_flight = statistics.fmean(in_flight_during) if in_flight_during else 0.0
    avg_kv = _avg_field(telemetry_rows, "kv_cache_used_pct")
    est_prefix = _estimate_prefix_hit_rate(telemetry_rows)

    if combined_rate < 0.05:
        capacity_status = "pass"
    elif combined_rate < 0.20:
        capacity_status = "marginal"
    else:
        capacity_status = "fail"

    final_row = {
        "measured_avg_pool_size": float(target_pool_size),
        "measured_avg_in_flight": avg_in_flight,
        "measurement_duration_s": duration,
        "sample_size": n,
        "ttft_violation_rate": ttft_rate,
        "tpot_violation_rate": tpot_rate,
        "combined_violation_rate": combined_rate,
        "violation_rate_ci_lower": ci_lo,
        "violation_rate_ci_upper": ci_hi,
        "ttft_p50_ms": _percentile(ttft_values, 0.5),
        "ttft_p75_ms": _percentile(ttft_values, 0.75),
        "ttft_p95_ms": _percentile(ttft_values, 0.95),
        "tpot_p50_ms": _percentile(tpot_values, 0.5),
        "tpot_p75_ms": _percentile(tpot_values, 0.75),
        "tpot_p95_ms": _percentile(tpot_values, 0.95),
        "avg_kv_cache_pct": avg_kv,
        "estimated_prefix_hit_rate": est_prefix,
        "capacity_status": capacity_status,
    }
    cols = ",".join(f"{k}=?" for k in final_row)
    with db.cursor() as c:
        c.execute(
            f"UPDATE cohort_measurements SET {cols} WHERE measurement_id = ?",
            list(final_row.values()) + [measurement_id],
        )

    # Persist events
    event_rows = [_event_to_row(e, measurement_id) for e in buffer]
    db.insert_events(event_rows)

    log.info(
        "Step %d done: pool=%d samples=%d violation=%.1f%% (ttft=%.1f%%, tpot=%.1f%%) status=%s",
        step_index, target_pool_size, n, combined_rate * 100,
        ttft_rate * 100, tpot_rate * 100, capacity_status,
    )

    return MeasurementResult(
        status="ok",
        target_pool_size=target_pool_size,
        sample_size=n,
        measurement_id=measurement_id,
        measured_avg_pool_size=float(target_pool_size),
        measured_avg_in_flight=avg_in_flight,
        measurement_duration_s=duration,
        ttft_violation_rate=ttft_rate,
        tpot_violation_rate=tpot_rate,
        combined_violation_rate=combined_rate,
        violation_rate_ci_lower=ci_lo,
        violation_rate_ci_upper=ci_hi,
        ttft_p50_ms=final_row["ttft_p50_ms"],
        ttft_p75_ms=final_row["ttft_p75_ms"],
        ttft_p95_ms=final_row["ttft_p95_ms"],
        tpot_p50_ms=final_row["tpot_p50_ms"],
        tpot_p75_ms=final_row["tpot_p75_ms"],
        tpot_p95_ms=final_row["tpot_p95_ms"],
        avg_kv_cache_pct=avg_kv,
        estimated_prefix_hit_rate=est_prefix,
        capacity_status=capacity_status,
        events=buffer,
    )


async def _in_flight_sampler(state: SharedState, store: list[int]) -> None:
    try:
        while True:
            store.append(state.in_flight)
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass


def _event_to_row(e: TurnEvent, measurement_id: int) -> dict:
    import json as _json
    return {
        "measurement_id": measurement_id,
        "persona_id": e.persona_id,
        "user_id": e.user_id,
        "session_id": e.session_id,
        "turn_index": e.turn_index,
        "submitted_at_ms": e.submitted_at_ms,
        "ttft_ms": e.ttft_ms,
        "completed_at_ms": e.completed_at_ms,
        "input_tokens": e.input_tokens,
        "history_tokens": e.history_tokens,
        "output_tokens": e.output_tokens,
        "tpot_ms": e.tpot_ms,
        "end_to_end_ms": e.end_to_end_ms,
        "in_flight_at_submit": e.in_flight_at_submit,
        "in_flight_avg_during": e.in_flight_avg_during,
        "in_flight_peak_during": e.in_flight_peak_during,
        "sla_ttft_violation": int(e.ttft_violation()),
        "sla_tpot_violation": int(e.tpot_violation()),
        "token_timestamps_json": (
            _json.dumps(e.token_timestamps) if e.token_timestamps else None
        ),
    }


def _avg_field(rows: list[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(statistics.fmean(vals))


def _estimate_prefix_hit_rate(rows: list[dict]) -> Optional[float]:
    # Best-effort: if engine reports running counters, use deltas.
    hits = [r["prefix_cache_hits"] for r in rows if r.get("prefix_cache_hits") is not None]
    if len(hits) < 2:
        return None
    delta_hits = max(0, hits[-1] - hits[0])
    return float(delta_hits) if delta_hits >= 0 else None
