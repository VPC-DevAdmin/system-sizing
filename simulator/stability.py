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

Both statistics assume independent samples, and per-second queue
depths are anything but: a batch scheduler's queue is a stationary
AR(1)-like process with lag-1 autocorrelation around 0.9. Fed raw, MK
read a flat-but-wandering queue as a confident trend in a third of
stable windows (about 5 % came out divergent, a quarter inconclusive).
Three things fix that, applied together:

  * the series is pre-aggregated into ``BIN_S``-second bin means
    before the trend statistics run (5 s bins cut the pair count 25×
    and the dependence between neighbours to ~0.6);
  * MK's variance is inflated by the AR(1) effective-sample-size
    factor (1+ρ_b)/(1−ρ_b) — the variance-correction approach of Yue &
    Wang (2004) — where ρ_b is the bin-level autocorrelation implied
    by the lag-1 autocorrelation of the DETRENDED raw series (Theil-
    Sen detrending, so a real ramp is not mistaken for dependence;
    the correction only engages when that autocorrelation clears the
    Anderson 10 % significance band);
  * the practical-significance rule uses the Sen slope's 90 % lower
    confidence bound (from the same corrected variance): growth whose
    lower bound sits inside the noise band is not evidence of drift.

A genuine ramp is as visible as ever; on the stationary AR(1) series
above, ≥ 95 % of windows now read stable. Reported slopes are in
per-minute units regardless of the binning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

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

# Pre-aggregation bin width (seconds) for the trend statistics — see
# the module docstring for why per-second samples cannot be tested
# directly.
BIN_S = 5.0

# z for the one-sided 90 % lower bound on the Sen slope.
Z_GROWTH_BOUND = 1.645


def bin_means(
    series: list[float], interval_s: float = 1.0, bin_s: float = BIN_S,
) -> tuple[list[float], float]:
    """Collapse ``series`` into consecutive ``bin_s``-second means.

    Returns ``(binned, bin_interval_s)``. Only full bins are kept (a
    trailing partial bin would be a noisier point than the rest);
    when the sampling interval is already at least ``bin_s`` the
    series is returned unchanged.
    """
    k = max(1, int(round(bin_s / interval_s)))
    if k == 1 or len(series) < k:
        return list(series), interval_s
    binned = [
        fmean(series[i:i + k]) for i in range(0, len(series) - k + 1, k)
    ]
    return binned, k * interval_s


@dataclass
class StabilityVerdict:
    verdict: str            # stable | divergent | inconclusive
    slope_per_min: float    # Theil-Sen slope in queue-units per minute
    p_value: float          # one-sided MK p for an UPWARD trend
    mean_depth: float
    growth_over_window: float  # slope × window duration
    n: int                  # raw (per-interval) samples in the window
    reason: str
    n_bins: int = 0         # BIN_S-second bins the statistics ran on
    growth_lower_bound: float = 0.0  # 90 % lower bound on the growth
    ar1_rho: float = 0.0    # lag-1 autocorrelation of the detrended series

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "slope_per_min": round(self.slope_per_min, 3),
            "p_value": round(self.p_value, 5),
            "mean_depth": round(self.mean_depth, 2),
            "growth_over_window": round(self.growth_over_window, 1),
            "growth_lower_bound": round(self.growth_lower_bound, 1),
            "n": self.n,
            "n_bins": self.n_bins,
            "ar1_rho": round(self.ar1_rho, 3),
            "reason": self.reason,
        }


def _pairwise_slopes(series: list[float], interval_s: float) -> list[float]:
    """Sorted pairwise slopes in units per second (Sen's estimator's
    raw material). O(n²) pairs; for windows longer than ~800 samples
    the series is strided down to keep the pair count bounded."""
    n = len(series)
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
    return slopes


def _median(sorted_values: list[float]) -> float:
    m = len(sorted_values)
    mid = m // 2
    if m % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def theil_sen_slope(series: list[float], interval_s: float = 1.0) -> float:
    """Median of pairwise slopes, in units per second.

    Robust to the spiky outliers a batch scheduler produces (a brief
    admission burst doesn't drag the slope the way it drags OLS).
    """
    if len(series) < 2:
        return 0.0
    return _median(_pairwise_slopes(series, interval_s))


def sen_slope_lower_bound(
    sorted_slopes: list[float], var_s: float, z: float = Z_GROWTH_BOUND,
) -> float:
    """Lower confidence bound on the Sen slope (Sen 1968): the pairwise
    slope at rank ``(m − z·√Var S)/2`` of the ``m`` sorted slopes.
    ``var_s`` is the (autocorrelation-corrected) variance of the MK
    statistic S for the same series."""
    m = len(sorted_slopes)
    if m == 0:
        return 0.0
    c = z * math.sqrt(max(0.0, var_s))
    idx = int((m - c) / 2.0)
    return sorted_slopes[max(0, min(m - 1, idx))]


def lag1_autocorrelation(series: list[float], slope_per_step: float = 0.0) -> float:
    """Lag-1 autocorrelation of ``series`` after removing the trend
    ``slope_per_step × index`` (so a genuine ramp does not read as
    dependence). 0.0 for series too short or constant."""
    n = len(series)
    if n < 4:
        return 0.0
    detrended = [v - slope_per_step * i for i, v in enumerate(series)]
    mean = fmean(detrended)
    dev = [d - mean for d in detrended]
    denom = sum(d * d for d in dev)
    if denom <= 0:
        return 0.0
    return sum(dev[i] * dev[i + 1] for i in range(n - 1)) / denom


def ar1_variance_factor(rho: float, n: int, bin_k: int = 1) -> float:
    """Variance-inflation factor n/n* for Mann-Kendall on an AR(1)
    series with lag-1 autocorrelation ``rho`` (estimated on ``n``
    samples) whose statistics run on ``bin_k``-sample bin means.

    Engages only when ``rho`` clears Anderson's (1942) one-sided 10 %
    significance band for a lag-1 autocorrelation estimated on ``n``
    samples; below it the samples are treated as independent (1.0).
    The bin-level autocorrelation is ``rho**bin_k`` (exact for the
    AR(1) process, close enough for bin means of one) and the factor
    is (1+ρ_b)/(1−ρ_b) — the variance-correction approach of Yue &
    Wang (2004).
    """
    if n < 4:
        return 1.0
    upper = (-1.0 + 1.645 * math.sqrt(n - 2)) / (n - 1)
    if rho <= upper:
        return 1.0
    rho_b = min(rho, 0.98) ** max(1, bin_k)
    return (1.0 + rho_b) / (1.0 - rho_b)


def _mann_kendall(series: list[float]) -> tuple[int, float]:
    """(S, Var S) with tie correction. Var S is 0 for a constant or
    too-short series."""
    n = len(series)
    if n < 3:
        return 0, 0.0
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
    return s, max(0.0, var)


def _mk_upward_p(s: int, var: float) -> float:
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


def mann_kendall_upward_p(
    series: list[float], *, variance_factor: float = 1.0,
) -> float:
    """One-sided Mann-Kendall p-value for an UPWARD monotone trend,
    with tie correction (queue depths are small integers — ties are
    the norm, and the uncorrected variance overstates significance).

    ``variance_factor`` inflates Var S for autocorrelated input (see
    ``ar1_variance_factor``); 1.0 is the classic independent-samples
    test.

    Returns 1.0 when there's no evidence of an upward trend (or the
    series is too short / constant).
    """
    s, var = _mann_kendall(series)
    return _mk_upward_p(s, var * max(1.0, variance_factor))


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

    The trend statistics run on ``BIN_S``-second bin means of the
    series with the AR(1) variance correction (see the module
    docstring); the empty-queue fast path, ``mean_depth`` and ``n``
    refer to the raw series and ``growth_over_window`` is the slope
    over the raw window.
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

    binned, bin_interval = bin_means(series, interval_s)
    bin_k = max(1, int(round(bin_interval / interval_s)))
    slopes = _pairwise_slopes(binned, bin_interval)
    slope_s = _median(slopes) if slopes else 0.0
    slope_per_min = slope_s * 60.0
    window_s = n * interval_s
    growth = slope_s * window_s
    # Dependence, measured on the detrended RAW series (n samples give
    # a far better ρ estimate than n/5 bins) and carried to the bins.
    rho = lag1_autocorrelation(
        series, theil_sen_slope(series, interval_s) * interval_s,
    )
    factor = ar1_variance_factor(rho, n, bin_k)
    s_stat, var_s = _mann_kendall(binned)
    var_s *= factor
    p_up = _mk_upward_p(s_stat, var_s)
    growth_lo = sen_slope_lower_bound(slopes, var_s) * window_s
    n_bins = len(binned)
    common = dict(
        slope_per_min=slope_per_min, p_value=p_up, mean_depth=mean_depth,
        growth_over_window=growth, n=n, n_bins=n_bins,
        growth_lower_bound=growth_lo, ar1_rho=rho,
    )

    # Practical-significance floor: the projected growth over one
    # window must be meaningful relative to the active batch. The
    # absolute floor of 10 keeps tiny-deployment noise (queue 0→2)
    # from ever reading as collapse.
    floor = max(10.0, 0.5 * served_mean) if served_mean else 10.0

    if p_up < P_DIVERGENT and growth >= floor:
        return StabilityVerdict(
            verdict=DIVERGENT, **common,
            reason=(
                f"confident upward trend (p={p_up:.4f}) growing "
                f"{growth:.0f} requests over the window "
                f"(floor {floor:.0f})"
            ),
        )
    # Stable when there is no significant trend, or when the growth —
    # judged by its lower confidence bound, so noise-scale drift does
    # not count — is below a quarter of the floor.
    if p_up >= P_NO_TREND or growth_lo < 0.25 * floor:
        return StabilityVerdict(
            verdict=STABLE, **common,
            reason=(
                f"no meaningful growth (p={p_up:.4f}, growth "
                f"{growth:.1f} with lower bound {growth_lo:.1f} < "
                f"{0.25 * floor:.1f})"
                if p_up < P_NO_TREND else
                f"no significant upward trend (p={p_up:.4f})"
            ),
        )
    return StabilityVerdict(
        verdict=INCONCLUSIVE, **common,
        reason=(
            f"trend significant (p={p_up:.4f}) but growth "
            f"{growth:.1f} below the {floor:.0f} floor — "
            f"extend the window to distinguish drift from collapse"
        ),
    )
