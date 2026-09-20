"""Queue-stability statistics (simulator/stability.py)."""

from __future__ import annotations

import random

from simulator.stability import (
    DIVERGENT,
    INCONCLUSIVE,
    STABLE,
    assess_queue_stability,
    mann_kendall_upward_p,
    theil_sen_slope,
)


def test_theil_sen_recovers_linear_slope():
    series = [2.0 * t for t in range(120)]
    assert abs(theil_sen_slope(series) - 2.0) < 1e-9


def test_theil_sen_robust_to_spikes():
    rng = random.Random(7)
    series = [10.0 + 0.5 * t for t in range(120)]
    # A batch scheduler's admission spikes shouldn't drag the slope.
    for i in rng.sample(range(120), 10):
        series[i] += 200.0
    assert abs(theil_sen_slope(series) - 0.5) < 0.1


def test_mann_kendall_flat_noise_not_significant():
    rng = random.Random(3)
    series = [50 + rng.gauss(0, 5) for _ in range(120)]
    assert mann_kendall_upward_p(series) > 0.05


def test_mann_kendall_ramp_significant():
    rng = random.Random(3)
    series = [t * 1.0 + rng.gauss(0, 3) for t in range(120)]
    assert mann_kendall_upward_p(series) < 0.001


def test_empty_queue_is_stable():
    series = [0.0] * 100
    series[40] = 2.0  # one admission blip
    v = assess_queue_stability(series)
    assert v.verdict == STABLE
    assert "empty" in v.reason


def test_stationary_busy_queue_is_stable():
    rng = random.Random(11)
    series = [40 + rng.gauss(0, 6) for _ in range(180)]
    v = assess_queue_stability(series, served_mean=100)
    assert v.verdict == STABLE


def test_unbounded_growth_is_divergent():
    rng = random.Random(11)
    # λ > μ: backlog grows ~2 requests/second on a 256-running batch.
    series = [2.0 * t + rng.gauss(0, 4) for t in range(180)]
    v = assess_queue_stability(series, served_mean=256)
    assert v.verdict == DIVERGENT
    assert v.growth_over_window > 128  # ≥ half the active batch


def test_short_window_inconclusive():
    v = assess_queue_stability([1.0, 2.0, 3.0])
    assert v.verdict == INCONCLUSIVE


def test_tiny_drift_reads_stable():
    # Statistically confident but operationally negligible growth
    # (~6 requests over 3 minutes against a 256-running batch) is
    # stable — drift below a quarter of the significance floor is
    # noise-scale, not collapse.
    rng = random.Random(5)
    series = [30 + 0.035 * t + rng.gauss(0, 0.4) for t in range(180)]
    v = assess_queue_stability(series, served_mean=256)
    assert v.verdict == STABLE


def test_intermediate_drift_inconclusive_asks_for_longer_window():
    # Growth in the ambiguous band (between 0.25× and 1× the floor):
    # confidently trending but not yet clearly collapse — the caller
    # should extend the window.
    rng = random.Random(5)
    series = [30 + 0.4 * t + rng.gauss(0, 1.0) for t in range(180)]
    v = assess_queue_stability(series, served_mean=256)
    assert v.verdict == INCONCLUSIVE
    assert "extend" in v.reason


# ── Autocorrelated input (A5) ──────────────────────────────────────


def _ar1(seed, n=120, rho=0.9, mean=40.0, sd_innov=3.0):
    """Stationary AR(1) queue-depth series (integer, non-negative),
    burnt in so it starts at equilibrium."""
    rng = random.Random(seed)
    x = mean
    out = []
    for _ in range(n + 50):
        x = mean + rho * (x - mean) + rng.gauss(0, sd_innov)
        out.append(max(0.0, float(round(x))))
    return out[50:]


def test_stationary_ar1_reads_stable_at_least_95_percent():
    """A flat-but-wandering queue (ρ=0.9 per second, n=120, stationary
    sd ≈ 7 around 40 with a 100-request running batch) must read
    stable in ≥ 95 % of windows. Fed raw to Mann-Kendall it did so
    in ~90 % (the rest inconclusive) — binning plus the AR(1)
    variance correction plus the growth lower bound fix that."""
    verdicts = [
        assess_queue_stability(_ar1(seed), served_mean=100).verdict
        for seed in range(200)
    ]
    assert verdicts.count(STABLE) >= 190, verdicts.count(STABLE)


def test_linear_ramp_on_ar1_noise_is_still_divergent():
    """The corrections must not blunt a real collapse: a 2/s ramp
    under the same AR(1) noise, against a 64-request batch."""
    for seed in range(20):
        base = _ar1(seed, mean=20.0, sd_innov=2.0)
        series = [v + 2.0 * t for t, v in enumerate(base)]
        v = assess_queue_stability(series, served_mean=64)
        assert v.verdict == DIVERGENT, (seed, v.reason)
        # Slope reported per minute, unaffected by the 5 s binning.
        assert 100 < v.slope_per_min < 140


def test_bin_means_keeps_full_bins_only():
    from simulator.stability import bin_means

    binned, dt = bin_means([1.0] * 5 + [3.0] * 5 + [9.0] * 3, 1.0, 5.0)
    assert binned == [1.0, 3.0] and dt == 5.0      # trailing partial dropped
    assert bin_means([1.0, 2.0], 1.0, 5.0) == ([1.0, 2.0], 1.0)  # too short
    assert bin_means([1.0, 2.0, 3.0], 5.0, 5.0) == ([1.0, 2.0, 3.0], 5.0)


def test_ar1_variance_factor_engages_only_on_significant_rho():
    from simulator.stability import ar1_variance_factor

    assert ar1_variance_factor(0.05, 120) == 1.0        # inside the band
    f = ar1_variance_factor(0.9, 120, bin_k=5)          # ρ_b = 0.9⁵ ≈ 0.59
    assert abs(f - (1 + 0.9 ** 5) / (1 - 0.9 ** 5)) < 1e-9
    assert ar1_variance_factor(0.9, 120, bin_k=1) > f   # raw is worse
