"""Core simulator unit tests — distributions, adaptive stepping, stats.

These cover the pure-logic pieces; runtime tests for the async pool /
virtual user need fakes for AsyncOpenAI and live in their own file.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from simulator.adaptive import StepResult, choose_next_pool_size
from simulator.distributions import Discrete, LogNormal, Constant
from simulator.measurement import (
    _percentile,
    _wait_for_throughput_convergence,
    _wilson_ci,
)
from simulator.personas import COHORTS, PERSONAS, get_cohort
from simulator.prefix_cache import (
    TurnRow,
    analyse_rows,
    analyse_sessions,
)
from simulator.virtual_user import SharedState


# ── Distributions ──────────────────────────────────────────────────────


def test_lognormal_sample_clamps_to_min() -> None:
    """``min_value`` exists so a fat-tail draw can't return zero token
    counts that would crash the corpus generator."""
    rng = random.Random(0)
    d = LogNormal.from_median(100, 0.5, min_value=10)
    samples = [d.sample(rng) for _ in range(2000)]
    assert min(samples) >= 10


def test_lognormal_median_is_close_to_target() -> None:
    """``LogNormal.from_median`` parameterises by the median (= exp(mu))
    so users don't have to think about log-space when picking values."""
    rng = random.Random(42)
    d = LogNormal.from_median(200, 0.4)
    samples = sorted(d.sample(rng) for _ in range(5000))
    median = samples[len(samples) // 2]
    assert 180 < median < 220, f"median was {median}"


def test_discrete_distribution_respects_weights() -> None:
    """The persona session-count weights are discrete; the sampler must
    match the requested distribution within sampling noise."""
    rng = random.Random(7)
    d = Discrete({1: 0.7, 2: 0.2, 3: 0.1})
    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(20000):
        counts[int(d.sample(rng))] += 1
    assert 0.65 < counts[1] / 20000 < 0.75
    assert 0.16 < counts[2] / 20000 < 0.24
    assert 0.07 < counts[3] / 20000 < 0.13


def test_constant_distribution_returns_value() -> None:
    rng = random.Random(0)
    assert Constant(value=42).sample(rng) == 42


# ── Adaptive stepping ─────────────────────────────────────────────────


def test_adaptive_starts_at_initial_size_when_history_empty() -> None:
    nxt = choose_next_pool_size(
        [], initial_pool_size=4, max_pool_size=512,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt == 4


def test_adaptive_doubles_when_far_from_knee() -> None:
    """Below 5% violation rate we double the pool size each step —
    cheap exploration in the safe regime."""
    hist = [StepResult(pool_size=4, violation_rate=0.0)]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=4, max_pool_size=512,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt == 8


def test_adaptive_stops_at_violation_threshold() -> None:
    """Past the configured stop threshold, ramping ends — extra steps
    only add noise once the system is plainly past its limit."""
    hist = [StepResult(pool_size=64, violation_rate=0.6)]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=4, max_pool_size=512,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt is None


def test_adaptive_backtracks_on_overshoot() -> None:
    """A single jump > 0.20 in violation rate means the previous step
    was too coarse near the knee. We bisect and re-measure between
    the last two pool sizes."""
    hist = [
        StepResult(pool_size=16, violation_rate=0.05),
        StepResult(pool_size=64, violation_rate=0.32),  # +0.27 jump
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=4, max_pool_size=512,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt is not None
    assert 16 < nxt < 64
    assert nxt == (16 + 64) // 2


def test_adaptive_uses_fine_steps_near_knee() -> None:
    """When slope is high (we're on the rising edge) we use small fixed
    steps so the curve has resolution where it matters."""
    hist = [
        StepResult(pool_size=32, violation_rate=0.10),
        StepResult(pool_size=36, violation_rate=0.13),  # slope = 0.0075/step
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=4, max_pool_size=512,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt is not None
    # Should be a small increment over 36 (≤ 4 from the formula).
    assert 36 < nxt <= 40


def test_adaptive_caps_at_max_pool_size() -> None:
    hist = [StepResult(pool_size=256, violation_rate=0.01)]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=4, max_pool_size=300,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt == 300


def test_adaptive_returns_none_at_max_when_already_there() -> None:
    hist = [StepResult(pool_size=300, violation_rate=0.01)]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=4, max_pool_size=300,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt is None


def test_adaptive_bisects_after_first_cliff_crossing() -> None:
    """Original bug: the run jumped 32 → 64 and pool=64 came in at
    66% violation. The stepper returned None instead of bisecting,
    leaving the knee unlocalised between 32 and 64. Fix: detect the
    failing measurement and bisect to the midpoint."""
    hist = [
        StepResult(pool_size=8, violation_rate=0.0),
        StepResult(pool_size=16, violation_rate=0.0),
        StepResult(pool_size=32, violation_rate=0.0),
        StepResult(pool_size=64, violation_rate=0.66),
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt == 48, f"expected midpoint of 32 and 64, got {nxt}"


def test_adaptive_iteratively_narrows_knee() -> None:
    """Bisect down from a fail-bracket to localise the knee further.

    32 passed, 64 failed → previous step bisected to 48. If 48 also
    fails, knee is between 32 and 48 — bisect again to 40.
    """
    hist = [
        StepResult(pool_size=32, violation_rate=0.0),
        StepResult(pool_size=64, violation_rate=0.66),
        StepResult(pool_size=48, violation_rate=0.55),
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    # smallest fail = 48; largest pass below = 32; gap = 16 > 8 → bisect to 40
    assert nxt == 40


def test_adaptive_bisects_upward_when_bisect_passed() -> None:
    """If the bisect itself passed, the knee is between the bisect and
    the original fail. Continue bisecting upward from there."""
    hist = [
        StepResult(pool_size=32, violation_rate=0.0),
        StepResult(pool_size=64, violation_rate=0.66),
        StepResult(pool_size=48, violation_rate=0.10),
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    # smallest fail = 64; largest pass below = 48; gap = 16 > 8 → bisect to 56
    assert nxt == 56


def test_adaptive_stops_when_bisect_gap_below_threshold() -> None:
    """Resolution floor: don't bisect when the pass→fail gap is
    already small enough that further refinement isn't useful."""
    hist = [
        StepResult(pool_size=32, violation_rate=0.0),
        StepResult(pool_size=40, violation_rate=0.55),
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
        min_bisect_gap=8,
    )
    # Gap is 8 — at the threshold (NOT > min_bisect_gap), stop.
    assert nxt is None


def test_adaptive_no_bisect_when_initial_step_fails() -> None:
    """If even the very first step crosses the threshold there's no
    pass to bisect against. Stop instead of looping."""
    hist = [StepResult(pool_size=8, violation_rate=0.7)]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    assert nxt is None


def test_adaptive_doesnt_double_past_marginal_step() -> None:
    """Regression: R470 first-light Xeon FP8 run produced this exact
    trace and the stepper jumped to pool=110 from pool=55, completely
    overshooting the knee that pool=64 had already revealed at 36%
    violation. The bug: stop_violation_threshold=0.5 gates the
    bisection state machine, so 36% (≥ 'fail' but < stop) didn't
    bracket the knee. A subsequent noisy 2% pass at pool=55 then made
    the slope-based logic 'double' to 110.

    Fix: knee_zone_threshold (default 0.20) is a separate trigger.
    Anything past it brackets the knee and the stepper bisects
    instead of doubling, regardless of whether stop has been hit."""
    hist = [
        StepResult(pool_size=8, violation_rate=0.00),
        StepResult(pool_size=16, violation_rate=0.00),
        StepResult(pool_size=32, violation_rate=0.00),
        StepResult(pool_size=64, violation_rate=0.36),  # cliff bracket (≥ 0.20)
        StepResult(pool_size=48, violation_rate=0.06),
        StepResult(pool_size=52, violation_rate=0.12),
        StepResult(pool_size=55, violation_rate=0.02),  # noise rebound
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
        knee_zone_threshold=0.20,
    )
    # smallest cliff bracket = 64; largest safe (viol < 0.20) below 64 = 55;
    # gap = 9 > 8 → bisect to (55+64)//2 = 59. Definitely NOT 110.
    assert nxt == 59, (
        f"expected bisection to 59 (between safe 55 and cliff 64), "
        f"got {nxt}"
    )


def test_adaptive_doesnt_repeat_already_measured_pool_size() -> None:
    """If bisection's chosen midpoint coincides with a pool size we've
    already measured (rare but possible), stop rather than measure it
    again."""
    hist = [
        StepResult(pool_size=32, violation_rate=0.0),
        StepResult(pool_size=48, violation_rate=0.10),
        StepResult(pool_size=64, violation_rate=0.66),
        StepResult(pool_size=56, violation_rate=0.55),
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    # smallest fail = 56; largest pass below = 48; gap = 8 → at threshold, stop
    assert nxt is None


# ── Statistics helpers ────────────────────────────────────────────────


def test_percentile_handles_empty_input() -> None:
    """Empty samples — return 0 rather than raise. The aggregation
    happens after a sample-size guard, so 0 is a safe sentinel."""
    assert _percentile([], 0.5) == 0.0


def test_percentile_p50_is_median() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_percentile_interpolates_between_samples() -> None:
    """Linear interpolation when k isn't an integer — the standard
    'C=1' percentile method."""
    # p75 of [1,2,3,4]: rank = 0.75 * 3 = 2.25 → 3 + 0.25*(4-3) = 3.25
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.75) == pytest.approx(3.25)


def test_wilson_ci_handles_extremes() -> None:
    """0/n and n/n shouldn't blow up the bound calculation."""
    lo, hi = _wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 <= hi <= 0.05
    lo, hi = _wilson_ci(100, 100)
    assert lo > 0.95 and hi == pytest.approx(1.0)


def test_wilson_ci_brackets_observed_proportion() -> None:
    lo, hi = _wilson_ci(50, 200)
    assert lo < 0.25 < hi
    # Wilson CI is symmetric around the centre — for n=200 a 95% CI
    # half-width near 0.06 is about right.
    assert (hi - lo) < 0.15


def test_wilson_ci_full_range_when_n_zero() -> None:
    """With no observations the CI is [0,1] — the export reader checks
    sample_size separately and won't draw a bar with zero samples."""
    assert _wilson_ci(0, 0) == (0.0, 1.0)


# ── Persona / cohort definitions ──────────────────────────────────────


def test_every_cohort_validates() -> None:
    """Cohort weights must sum to 1.0 and reference known persona ids
    — invariants the runner relies on at start-up."""
    for cid in COHORTS:
        get_cohort(cid)


def test_every_persona_has_sla_floors() -> None:
    """The aggregator reads SLA floors per-event; missing fields would
    silently classify every request as a violation."""
    for p in PERSONAS.values():
        assert p.ttft_floor_seconds > 0
        assert p.tpot_floor_ms > 0


# ── Prefix-cache analyser ─────────────────────────────────────────────


def _row(uid, sid, turn, ttft, persona="quick_lookup") -> TurnRow:
    return TurnRow(
        user_id=uid, session_id=sid, turn_index=turn, ttft_ms=ttft,
        persona_id=persona, input_tokens=300, history_tokens=turn * 200,
    )


def test_prefix_cache_classifies_session_as_hit_when_later_ttft_is_low() -> None:
    """A session whose later turns return much faster than turn-0 is
    the canonical 'prefix cache is working' signal."""
    rows = [
        _row("u1", "s1", 0, 1000.0),
        _row("u1", "s1", 1, 250.0),
        _row("u1", "s1", 2, 240.0),
    ]
    sessions, _ = analyse_sessions(rows)
    assert len(sessions) == 1
    assert sessions[0].hit is True
    assert sessions[0].ratio < 0.5


def test_prefix_cache_classifies_session_as_miss_when_later_ttft_high() -> None:
    """Engine NOT reusing the prefix → later TTFT close to turn-0."""
    rows = [
        _row("u1", "s1", 0, 1000.0),
        _row("u1", "s1", 1, 980.0),
    ]
    sessions, _ = analyse_sessions(rows)
    assert len(sessions) == 1
    assert sessions[0].hit is False
    assert sessions[0].ratio > 0.9


def test_prefix_cache_skips_single_turn_sessions() -> None:
    """Single-turn personas (summarizer, drafter) can't validate cache
    behaviour — the analyser must not pretend otherwise."""
    rows = [_row("u1", "s1", 0, 1000.0)]
    sessions, notes = analyse_sessions(rows)
    assert sessions == []
    assert any("single-turn" in n or "skipped" in n for n in notes)


def test_prefix_cache_overall_verdict_thresholds() -> None:
    """≥ 70% hit rate = effective; 30-70% = partial; < 30% = ineffective.
    These thresholds drive the buyer-page badge so they're load-bearing."""
    # Mostly hits → effective
    effective_rows = []
    for i in range(15):
        effective_rows.append(_row(f"u{i}", "s1", 0, 1000.0))
        effective_rows.append(_row(f"u{i}", "s1", 1, 200.0))
    rep = analyse_rows(effective_rows)
    assert rep.verdict == "prefix_cache_effective"
    assert rep.overall_hit_rate >= 0.7

    # Mostly misses → ineffective
    bad_rows = []
    for i in range(15):
        bad_rows.append(_row(f"u{i}", "s1", 0, 1000.0))
        bad_rows.append(_row(f"u{i}", "s1", 1, 950.0))
    rep = analyse_rows(bad_rows)
    assert rep.verdict == "prefix_cache_ineffective"
    assert rep.overall_hit_rate < 0.3


def test_prefix_cache_insufficient_data_when_too_few_sessions() -> None:
    """Don't hand a confident verdict for tiny samples."""
    rows = [_row("u1", "s1", 0, 1000.0), _row("u1", "s1", 1, 200.0)]
    rep = analyse_rows(rows)
    assert rep.verdict == "insufficient_data"


# ── Throughput convergence detector ──────────────────────────────────


def _drive_convergence(
    *, completions_per_second_schedule, monkeypatch,
    min_warmup_s=2, max_wait_s=30, window_s=3, threshold=0.20,
    min_completions_per_window=2,
):
    """Run the convergence detector against a fake state whose
    completion count advances on a schedule.

    ``completions_per_second_schedule`` is a list of ints — the number
    of completions to add at each 1s tick (after ``asyncio.sleep`` is
    monkey-patched to a no-op so the test is fast).
    """
    state = SharedState()
    schedule = iter(completions_per_second_schedule)

    async def fake_sleep(_seconds):
        try:
            inc = next(schedule)
        except StopIteration:
            inc = 0
        # Touch the protected counter directly — tests aren't in an
        # event loop where state.complete() can be awaited cheaply.
        state._completed += inc

    # Patch asyncio.sleep AND time.monotonic so the loop's elapsed
    # check advances without real wall-clock waiting.
    fake_now = [0.0]

    def fake_monotonic():
        fake_now[0] += 1.0
        return fake_now[0]

    import simulator.measurement as m
    monkeypatch.setattr(m.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(m.time, "monotonic", fake_monotonic)

    return asyncio.run(_wait_for_throughput_convergence(
        state,
        min_warmup_s=min_warmup_s,
        max_wait_s=max_wait_s,
        window_s=window_s,
        threshold=threshold,
        min_completions_per_window=min_completions_per_window,
    ))


def test_convergence_steady_throughput_is_detected(monkeypatch) -> None:
    """A flat completion stream (5/s every second) should converge once
    we have two full windows of history."""
    converged, reason = _drive_convergence(
        completions_per_second_schedule=[5] * 30,
        monkeypatch=monkeypatch,
    )
    assert converged is True
    assert "converged" in reason


def test_convergence_growing_throughput_is_not_detected(monkeypatch) -> None:
    """If the completion rate is still climbing (warmup phase), the
    detector should hold off until the rate flattens."""
    # Each second, completions/sec doubles: 1, 2, 4, 8, 16... never flat.
    # max_wait shorter than the schedule needs to flatten.
    converged, reason = _drive_convergence(
        completions_per_second_schedule=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        monkeypatch=monkeypatch,
        max_wait_s=10,
    )
    assert converged is False
    assert "timed out" in reason


def test_convergence_low_throughput_skipped_until_min(monkeypatch) -> None:
    """If neither window has min_completions_per_window completions,
    the detector should NOT declare convergence — comparing 1/s vs 0/s
    would be 100% relative diff but is meaningless."""
    converged, reason = _drive_convergence(
        completions_per_second_schedule=[0] * 50,  # nothing ever completes
        monkeypatch=monkeypatch,
        max_wait_s=20,
        min_completions_per_window=2,
    )
    assert converged is False
    assert "throughput too low" in reason or "timed out" in reason


def test_prefix_cache_per_persona_breakdown() -> None:
    """Different personas can have different cache effectiveness in the
    same run (long-context personas help cache more than chat ones).
    The breakdown surfaces this for the buyer page."""
    rows = []
    for i in range(15):
        rows.append(_row(f"a{i}", "s1", 0, 1000.0, persona="conversational"))
        rows.append(_row(f"a{i}", "s1", 1, 200.0, persona="conversational"))
        rows.append(_row(f"b{i}", "s1", 0, 1000.0, persona="quick_lookup"))
        rows.append(_row(f"b{i}", "s1", 1, 950.0, persona="quick_lookup"))
    rep = analyse_rows(rows)
    assert rep.per_persona["conversational"]["hit_rate"] >= 0.9
    assert rep.per_persona["quick_lookup"]["hit_rate"] <= 0.1
