"""Adaptive ramp-step selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    pool_size: int
    violation_rate: float


def _smoothed_slope(history: list[StepResult]) -> float:
    if len(history) < 2:
        return 0.0
    a, b = history[-2], history[-1]
    delta_pool = max(1, b.pool_size - a.pool_size)
    return (b.violation_rate - a.violation_rate) / delta_pool


def choose_next_pool_size(
    history: list[StepResult],
    *,
    initial_pool_size: int,
    max_pool_size: int,
    knee_slope_threshold: float,
    stop_violation_threshold: float,
) -> int | None:
    """Pick the next pool size, or None to stop the ramp."""
    if not history:
        return initial_pool_size

    last = history[-1]
    if last.violation_rate >= stop_violation_threshold:
        return None
    if last.pool_size >= max_pool_size:
        return None

    # Backtrack on overshoot: a single step jumped > 0.20 in violation rate.
    if len(history) >= 2:
        prev = history[-2]
        if (last.violation_rate - prev.violation_rate) > 0.20:
            mid = (last.pool_size + prev.pool_size) // 2
            if mid > prev.pool_size and mid < last.pool_size:
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
    if nxt <= last.pool_size:
        return None
    return nxt
