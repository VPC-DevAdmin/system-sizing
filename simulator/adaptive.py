"""Adaptive ramp-step selection.

Two phases:

1. **Coarse ramp** (no failing measurement yet): start at
   ``initial_pool_size``, then double / take small steps based on slope
   until either ``max_pool_size`` is hit or a measurement crosses
   ``stop_violation_threshold``.

2. **Bidirectional bisection** (once a fail has been measured): keep
   bisecting between the largest known pass and the smallest known
   fail until the gap drops below ``min_bisect_gap``. This nails down
   the actual knee location instead of stopping the moment we see a
   violation cliff.

The previous implementation returned None on the first failing step
without bisecting, so a 32→64 jump that crossed the cliff produced a
single fail data point with no localization. Now we'd bisect to 48,
and depending on the result keep narrowing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    pool_size: int
    violation_rate: float


def _smoothed_slope(history: list[StepResult]) -> float:
    """Slope between the last two strictly-increasing pool-size points.

    Bisection produces non-monotonic histories — a slope between a
    larger fail and a smaller bisect-pass is meaningless. Pick the
    last pair where pool size strictly increased.
    """
    if len(history) < 2:
        return 0.0
    for i in range(len(history) - 1, 0, -1):
        a, b = history[i - 1], history[i]
        if b.pool_size > a.pool_size:
            delta_pool = b.pool_size - a.pool_size
            return (b.violation_rate - a.violation_rate) / delta_pool
    return 0.0


def choose_next_pool_size(
    history: list[StepResult],
    *,
    initial_pool_size: int,
    max_pool_size: int,
    knee_slope_threshold: float,
    stop_violation_threshold: float,
    knee_zone_threshold: float = 0.30,
    min_bisect_gap: int = 8,
) -> int | None:
    """Pick the next pool size, or None to stop the ramp.

    Two distinct thresholds gate behaviour:

    * ``stop_violation_threshold`` — the upper bound. We never measure
      above any pool size that produced a violation rate ≥ this value
      (default 0.5).
    * ``knee_zone_threshold`` — the bisection trigger. Once any
      measurement crosses this (default 0.20, matching ``capacity_status
      == 'fail'``), the stepper switches from coarse-ramp to bisection
      mode. Without this, a measurement at 36% violation would not be
      treated as a "fail" by the bisector — we'd fall through to the
      coarse-ramp slope logic and could end up DOUBLING past the knee
      because a subsequent (noisy) low-violation measurement made the
      slope-based step think we were back in safe territory.
    """
    if not history:
        return initial_pool_size

    last = history[-1]
    measured: set[int] = {h.pool_size for h in history}
    # Hard fails (≥ stop_violation_threshold) — we won't measure higher.
    hard_fails = sorted(
        h.pool_size for h in history if h.violation_rate >= stop_violation_threshold
    )
    # Knee-zone brackets — anything past the knee_zone_threshold is in
    # or near the knee. Bisection bounds the knee from above using these.
    cliff_brackets = sorted(
        h.pool_size for h in history if h.violation_rate >= knee_zone_threshold
    )
    safe = sorted(
        h.pool_size for h in history if h.violation_rate < knee_zone_threshold
    )

    # ── Bisection phase ──────────────────────────────────────────────
    # Active once any measurement has crossed knee_zone_threshold.
    # Bisect between the largest safe pool size and the smallest cliff
    # bracket until the gap is below ``min_bisect_gap``.
    if cliff_brackets:
        # Don't propose pool sizes above hard_fails either — clamp.
        bracket_upper = (
            min(cliff_brackets[0], hard_fails[0])
            if hard_fails else cliff_brackets[0]
        )
        safe_below = [p for p in safe if p < bracket_upper]
        if safe_below:
            largest_safe = safe_below[-1]
            gap = bracket_upper - largest_safe
            if gap > min_bisect_gap:
                mid = (largest_safe + bracket_upper) // 2
                if mid not in measured and largest_safe < mid < bracket_upper:
                    return mid
        # No safe pool below the bracket (initial step itself was in the
        # knee zone) or gap is below resolution floor — stop.
        return None

    # ── Coarse-ramp phase ────────────────────────────────────────────
    # No fail measured yet. Use slope-based stepping.
    if last.pool_size >= max_pool_size:
        return None

    # Sub-cliff overshoot: a step that jumped >0.20 in violation rate
    # without crossing the stop threshold. Bisect once for resolution.
    if len(history) >= 2:
        prev = history[-2]
        if (
            last.pool_size > prev.pool_size
            and (last.violation_rate - prev.violation_rate) > 0.20
        ):
            mid = (prev.pool_size + last.pool_size) // 2
            if mid not in measured and prev.pool_size < mid < last.pool_size:
                return mid

    slope = _smoothed_slope(history)

    if last.violation_rate < 0.05:
        step = last.pool_size  # double
    elif slope >= knee_slope_threshold:
        step = max(1, min(4, int(0.05 / max(slope, 1e-6))))
    elif last.violation_rate > 0.30:
        step = max(8, last.pool_size // 6)
    else:
        step = max(4, last.pool_size // 8)

    nxt = min(max_pool_size, last.pool_size + step)
    if nxt <= last.pool_size or nxt in measured:
        return None
    return nxt
