"""Queue-stability statistics for the open-loop methodology.

The open-loop capacity question is binary per arrival rate: is the
system *stationary* (queue depth fluctuates around an equilibrium —
Little's law holds, latency distribution is stable) or *divergent*
(arrivals exceed service rate, the queue grows without bound)? The
transition rate IS the capacity — it exists independent of any client
pool size, which is what the closed-loop methodology could never see.

Divergence is detected statistically on the per-second queue-depth
series sampled during a measurement window:

  * **Mann-Kendall** trend test (with tie correction — queue depths
    are small integers, ties are the norm) answers "is there a
    monotone upward trend at all, beyond noise?"
  * **Theil-Sen** slope (median of pairwise slopes — robust to the
    bursty spikes a batch scheduler produces) answers "how fast?"

Both are needed: MK alone flags a 0.1-request/min drift as confidently
as a collapse; slope alone can't distinguish a real trend from one
outlier-heavy oscillation. A window is *divergent* only when the
trend is statistically confident AND the projected growth over the
window is operationally meaningful (relative to the number of
requests actually being served).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STABLE = "stable"
DIVERGENT = "divergent"
INCONCLUSIVE = "inconclusive"

# Minimum samples for a verdict — below this, MK has essentially no
# power and any slope is noise.
MIN_SAMPLES = 30

# One-sided p-value bounds. Divergence requires strong evidence
# (p < 0.01); "no significant upward trend" (p ≥ 0.10) is enough to
# call stable when growth is also small.
P_DIVERGENT = 0.01
P_NO_TREND = 0.10


@dataclass
class StabilityVerdict:
    verdict: str            # stable | divergent | inconclusive
    slope_per_min: float    # Theil-Sen slope in queue-units per minute
    p_value: float          # one-sided MK p for an UPWARD trend
    mean_depth: float
    growth_over_window: float  # slope × window duration
    n: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "slope_per_min": round(self.slope_per_min, 3),
            "p_value": round(self.p_value, 5),
            "mean_depth": round(self.mean_depth, 2),
            "growth_over_window": round(self.growth_over_window, 1),
            "n": self.n,
            "reason": self.reason,
        }


def theil_sen_slope(series: list[float], interval_s: float = 1.0) -> float:
    """Median of pairwise slopes, in units per second.

    Robust to the spiky outliers a batch scheduler produces (a brief
    admission burst doesn't drag the slope the way it drags OLS).
    O(n²) pairs; for windows longer than ~800 samples the series is
    strided down to keep the pair count bounded.
    """
    n = len(series)
    if n < 2:
        return 0.0
    if n > 800:
        stride = n // 800 + 1
        series = series[::stride]
        interval_s *= stride
        n = len(series)
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = (j - i) * interval_s
            slopes.append((series[j] - series[i]) / dt)
    slopes.sort()
    m = len(slopes)
    mid = m // 2
    if m % 2:
        return slopes[mid]
    return (slopes[mid - 1] + slopes[mid]) / 2.0


def mann_kendall_upward_p(series: list[float]) -> float:
    """One-sided Mann-Kendall p-value for an UPWARD monotone trend,
    with tie correction (queue depths are small integers — ties are
    the norm, and the uncorrected variance overstates significance).

    Returns 1.0 when there's no evidence of an upward trend (or the
    series is too short / constant).
    """
    n = len(series)
    if n < 3:
        return 1.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = series[j] - series[i]
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    # Tie correction on the variance.
    counts: dict[float, int] = {}
    for v in series:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var <= 0:
        return 1.0  # constant series — no trend
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    # One-sided upper tail: P(Z >= z).
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def assess_queue_stability(
    series: list[float],
    *,
    interval_s: float = 1.0,
    served_mean: float | None = None,
) -> StabilityVerdict:
    """Stationary-or-divergent verdict for one measurement window.

    ``series`` is the per-second engine waiting-queue depth (summed
    across replicas), or the client's total in-flight count as a
    fallback when the engine exposes no queue gauge.

    ``served_mean`` — mean number of requests actively being served
    (engine running batch, or client in-flight). Scales the practical-
    significance floor: growing the backlog by half the active batch
    within one window is collapse; growing it by 2 requests is noise.
    """
    n = len(series)
    if n < MIN_SAMPLES:
        return StabilityVerdict(
            verdict=INCONCLUSIVE, slope_per_min=0.0, p_value=1.0,
            mean_depth=0.0, growth_over_window=0.0, n=n,
            reason=f"window too short ({n} samples < {MIN_SAMPLES})",
        )
    mean_depth = sum(series) / n
    # Empty-queue fast path: an engine keeping up produces a queue
    # that sits at ~0 with occasional admission blips. No statistics
    # needed — and MK on a nearly-all-zero series is degenerate.
    if mean_depth < 0.5 and max(series) < 5:
        return StabilityVerdict(
            verdict=STABLE, slope_per_min=0.0, p_value=1.0,
            mean_depth=mean_depth, growth_over_window=0.0, n=n,
            reason="queue essentially empty throughout the window",
        )

    slope_s = theil_sen_slope(series, interval_s)
    slope_per_min = slope_s * 60.0
    window_s = n * interval_s
    growth = slope_s * window_s
    p_up = mann_kendall_upward_p(series)

    # Practical-significance floor: the projected growth over one
    # window must be meaningful relative to the active batch. The
    # absolute floor of 10 keeps tiny-deployment noise (queue 0→2)
    # from ever reading as collapse.
    floor = max(10.0, 0.5 * served_mean) if served_mean else 10.0

    if p_up < P_DIVERGENT and growth >= floor:
        return StabilityVerdict(
            verdict=DIVERGENT, slope_per_min=slope_per_min, p_value=p_up,
            mean_depth=mean_depth, growth_over_window=growth, n=n,
            reason=(
                f"confident upward trend (p={p_up:.4f}) growing "
                f"{growth:.0f} requests over the window "
                f"(floor {floor:.0f})"
            ),
        )
    if p_up >= P_NO_TREND or growth < 0.25 * floor:
        return StabilityVerdict(
            verdict=STABLE, slope_per_min=slope_per_min, p_value=p_up,
            mean_depth=mean_depth, growth_over_window=growth, n=n,
            reason=(
                f"no meaningful growth (p={p_up:.4f}, "
                f"growth {growth:.1f} < {0.25 * floor:.1f})"
                if p_up < P_NO_TREND else
                f"no significant upward trend (p={p_up:.4f})"
            ),
        )
    return StabilityVerdict(
        verdict=INCONCLUSIVE, slope_per_min=slope_per_min, p_value=p_up,
        mean_depth=mean_depth, growth_over_window=growth, n=n,
        reason=(
            f"trend significant (p={p_up:.4f}) but growth "
            f"{growth:.1f} below the {floor:.0f} floor — "
            f"extend the window to distinguish drift from collapse"
        ),
    )
