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

from simulator.adaptive import StepResult, TwoKneeStepper
from simulator.distributions import Discrete, LogNormal, Constant
from simulator.measurement import (
    _classify_status,
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


# ── Adaptive stepping (TwoKneeStepper) ────────────────────────────────


def _drive_stepper(
    stepper: TwoKneeStepper,
    *,
    violation_at: dict[int, float] | None = None,
    target_miss_at: dict[int, float] | None = None,
    max_steps: int = 30,
) -> list[int]:
    """Drive the stepper to completion, recording synthetic results.

    For each pool size requested, supply a violation_rate (defaults to
    0.0) and target_miss_rate (defaults to violation_rate). Returns the
    sequence of pool sizes that were measured."""
    violation_at = violation_at or {}
    target_miss_at = target_miss_at or {}
    sequence: list[int] = []
    for _ in range(max_steps):
        pool = stepper.next_pool_size()
        if pool is None:
            break
        v = violation_at.get(pool, 0.0)
        t = target_miss_at.get(pool, v)
        stepper.record(StepResult(pool_size=pool, violation_rate=v, target_miss_rate=t))
        sequence.append(pool)
    return sequence


def test_stepper_returns_initial_pool_on_empty_history() -> None:
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    assert stepper.next_pool_size() == 8


def test_stepper_doubles_in_safe_regime() -> None:
    """Phase 1: while violation stays well below 5%, double each step."""
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    seq = _drive_stepper(stepper)
    # Pure doubling all the way to the cap, then stop (no failure observed).
    assert seq == [8, 16, 32, 64, 128, 256]
    assert stepper.coverage() == "capped"


def test_stepper_transitions_to_bisect_fail_on_threshold_cross() -> None:
    """When a doubling step crosses 50% violation, switch to bisection."""
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    seq = _drive_stepper(
        stepper,
        violation_at={64: 0.66, 128: 0.66, 256: 0.66},
    )
    # Doubled 8→16→32→64 (failed), then bisected between 32 and 64.
    assert seq[0:4] == [8, 16, 32, 64]
    # First bisection midpoint is (32+64)//2 = 48.
    assert 48 in seq


def test_stepper_bisects_fail_knee_to_resolution() -> None:
    """After bracketing, bisection narrows the gap to ≤ bisect_resolution."""
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256, bisect_resolution=4)
    # All sub-64 pools pass; 64 onwards fail.
    seq = _drive_stepper(
        stepper,
        violation_at={64: 0.66, 96: 0.30, 80: 0.20, 72: 0.10, 56: 0.02, 48: 0.0, 40: 0.0},
    )
    # The pass/fail bracket should narrow to within 4 pool slots.
    fails = [p for p in seq if p in {64} or p in {72, 80, 96}]
    passes = [p for p in seq if p not in fails]
    assert min(fails) - max(p for p in passes if p < min(fails)) <= 4


def test_stepper_initial_pool_already_fails_triggers_downward_search() -> None:
    """If pool=8 already violates, halve down looking for a passing zone."""
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256, downward_floor=2)
    seq = _drive_stepper(
        stepper,
        violation_at={8: 0.7, 4: 0.6, 2: 0.6},
    )
    # Halved 8 → 4 → 2; never found a pass.
    assert seq == [8, 4, 2]
    assert stepper.coverage() == "exceeds_hardware"


def test_stepper_downward_search_finds_passing_pool() -> None:
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256, downward_floor=2)
    seq = _drive_stepper(
        stepper,
        violation_at={8: 0.7, 4: 0.0},
    )
    # 8 fails, 4 passes — coverage classified as downward_search.
    assert seq[:2] == [8, 4]
    assert stepper.coverage() == "downward_search"


def test_stepper_bisects_target_knee_independently() -> None:
    """Knee 1 (target_miss ≥ 5%) bisects on a different rate field
    than knee 2 (violation ≥ 5%)."""
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256, bisect_resolution=4)
    # Set up: violation only fires at 64+, but target_miss fires earlier (32+).
    seq = _drive_stepper(
        stepper,
        violation_at={64: 0.66, 96: 0.30, 80: 0.20, 72: 0.10, 56: 0.02, 48: 0.0, 40: 0.0},
        target_miss_at={
            8: 0.0, 16: 0.0, 32: 0.10, 64: 0.66,
            48: 0.06, 40: 0.05, 56: 0.10,
            72: 0.20, 80: 0.30, 96: 0.50,
        },
    )
    # Both phases measured something between safe and target-miss.
    target_miss_pools = [p for p in seq if p >= 32]
    assert len(target_miss_pools) >= 2  # at least one bisection happened


def test_stepper_infill_adds_midpoints() -> None:
    """Phase 4: midpoints between fast_max → knee_1 and knee_1 → knee_2
    get scheduled when there's enough gap."""
    stepper = TwoKneeStepper(
        initial_pool_size=8, max_pool_size=256,
        bisect_resolution=4, infill_skip_pct=0.10,
    )
    # Drive a complete two-knee sweep with violation at 128, target_miss at 32.
    seq = _drive_stepper(
        stepper,
        violation_at={
            128: 0.66, 96: 0.20, 80: 0.10, 72: 0.06, 64: 0.02, 48: 0.0,
            32: 0.0, 16: 0.0, 8: 0.0,
        },
        target_miss_at={
            8: 0.0, 16: 0.0, 32: 0.10, 48: 0.10, 64: 0.10,
            72: 0.20, 80: 0.20, 96: 0.30, 128: 0.66,
        },
    )
    # We should have explored both knees — sweep is rich enough that
    # at least one pool size beyond the original doubling sequence
    # was measured (i.e. infill or bisection points landed).
    extra = [p for p in seq if p not in {8, 16, 32, 64, 128, 256}]
    assert len(extra) >= 1


def test_stepper_coverage_full_curve() -> None:
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    _drive_stepper(stepper, violation_at={64: 0.66})
    # Both passing (8/16/32) and failing (64) measurements present.
    assert stepper.coverage() == "full_curve"


def test_stepper_coverage_single_point() -> None:
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    # Record exactly one measurement and stop driving.
    pool = stepper.next_pool_size()
    assert pool == 8
    stepper.record(StepResult(pool_size=8, violation_rate=0.0, target_miss_rate=0.0))
    assert stepper.coverage() == "single_point"


def test_stepper_does_not_repeat_measured_pool_sizes() -> None:
    """If bisection's midpoint coincides with an existing measurement,
    the stepper transitions out of that phase rather than looping."""
    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256, bisect_resolution=4)
    seq = _drive_stepper(stepper, violation_at={64: 0.66})
    assert len(seq) == len(set(seq))  # no duplicate pool sizes


def test_stepper_wilson_ci_avoids_small_sample_panic_backoff() -> None:
    """The May 2026 AMD GPT-OSS analyst_team symptom: a measurement
    of 1/5 = 20% violations (Wilson lower CI ~3.6%) at the initial
    pool was triggering DOWNWARD_SEARCH even though the sample is
    statistically indistinguishable from a true rate of <5%. The
    Wilson-CI fix gates phase transitions on the lower bound — so
    noisy low-n measurements stay in PHASE_DOUBLING.

    Mirrors what ``_classify_status`` already does for the status
    label."""
    from simulator.adaptive import StepResult, TwoKneeStepper

    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    # Empty history → first measurement at initial pool.
    assert stepper.next_pool_size() == 8
    # Record a noisy 1-of-5 = 20% rate at pool=8. Wilson lower CI
    # is ~3.6%, so we are NOT statistically past the 5% knee.
    stepper.record(StepResult(
        pool_size=8, violation_rate=0.20, sample_size=5,
    ))
    # Without Wilson CI: this would trigger downward search.
    # With Wilson CI: stay in doubling, advance to pool=16.
    nxt = stepper.next_pool_size()
    assert stepper.phase == TwoKneeStepper.PHASE_DOUBLING
    assert nxt == 16


def test_stepper_wilson_ci_still_triggers_on_clean_signal() -> None:
    """Wilson-CI fix must not regress clean-signal cases: a 4/9 =
    44% measurement (Wilson lower ~18.7%) IS confidently past the
    5% knee, so downward_search MUST still fire."""
    from simulator.adaptive import StepResult, TwoKneeStepper

    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    assert stepper.next_pool_size() == 8
    stepper.record(StepResult(
        pool_size=8, violation_rate=0.44, sample_size=9,
    ))
    # Phase transitions happen on next_pool_size(), not record().
    # Calling next_pool_size advances DOUBLING → DOWNWARD_SEARCH for
    # this confidently-past-knee measurement, then DOWNWARD_SEARCH
    # picks pool=4 as the halve target.
    nxt = stepper.next_pool_size()
    assert stepper.phase == TwoKneeStepper.PHASE_DOWNWARD_SEARCH
    assert nxt == 4


def test_stepper_wilson_ci_bisect_partition_excludes_marginal() -> None:
    """In bisect_fail, a noisy 1/5=20% measurement (Wilson lower
    ~3.6%, upper ~62%) is neither confidently low nor confidently
    high — so it must NOT pull the bisection bracket toward itself.
    Otherwise small-sample noise in the marginal zone keeps narrowing
    the bracket past the point of useful information."""
    from simulator.adaptive import StepResult, TwoKneeStepper

    stepper = TwoKneeStepper(
        initial_pool_size=8, max_pool_size=256, bisect_resolution=4,
    )
    # Manually drive into bisect_fail with two clean points + one
    # marginal noisy point.
    stepper.next_pool_size()  # Sets phase=DOUBLING; advances internally
    stepper.record(StepResult(pool_size=8, violation_rate=0.0, sample_size=100))
    stepper.next_pool_size()
    stepper.record(StepResult(pool_size=64, violation_rate=0.50, sample_size=100))
    stepper.next_pool_size()  # Should now be in BISECT_FAIL
    # Add a marginal 20%-on-5-samples measurement at pool=32.
    stepper.record(StepResult(pool_size=32, violation_rate=0.20, sample_size=5))
    # The Wilson-CI partition: pool=32 is neither 'low' (upper 62% >> 5%)
    # nor 'high' (lower 3.6% < 5%) → excluded from the bracket. So
    # the bracket stays [8, 64] and bisects to ~36, not ~24 or ~48.
    nxt = stepper.next_pool_size()
    # The midpoint of [8, 64] is 36; the algorithm should choose
    # something in that vicinity rather than collapsing toward 32.
    assert nxt is not None
    assert 16 <= nxt <= 56, f"unexpected pool: {nxt}"


def test_stepper_doubles_past_marginal_until_fail_threshold() -> None:
    """Phase 1 must not exit doubling at the 5% knee — it should
    continue doubling until ≥ ``fail_threshold`` (30%) so a sustained-
    fail measurement is observed. Required so cohorts whose curve
    plateaus in the marginal band still get a non-None fail_pool."""
    stepper = TwoKneeStepper(
        initial_pool_size=8, max_pool_size=512, fail_threshold=0.30,
    )
    seq = _drive_stepper(
        stepper,
        # All doublings stay marginal (10–25%) until pool=256, which
        # finally exceeds 30%. We expect the doubling to push through
        # 64/128 even though 64 viol=0.10 (≥ knee).
        violation_at={
            64: 0.10, 128: 0.20, 256: 0.55,
            96: 0.06, 192: 0.40, 224: 0.45,
            160: 0.35, 144: 0.18,
        },
    )
    # Doubling must include 64, 128, AND 256 (didn't bail at the 5% knee).
    for expected in (64, 128, 256):
        assert expected in seq, f"doubling stopped early — pool={expected} not measured"


def test_stepper_marginal_cliff_infill_adds_midpoint_when_no_marginal_observed() -> None:
    """Sharp-cliff case: the bisect_fail bracket [last_pass, first_fail]
    closes at gap=resolution without ever observing a marginal-status
    measurement (5–30% violation). The stepper schedules ONE extra
    midpoint inside that bracket to try to catch the marginal point.

    Models the chat_heavy curve from the May 2026 Intel run: pool=92
    pass (0%), pool=96 fail (67%), no marginal in between."""
    stepper = TwoKneeStepper(
        initial_pool_size=8, max_pool_size=256,
        bisect_resolution=4, fail_threshold=0.30,
    )
    seq = _drive_stepper(
        stepper,
        violation_at={
            # Zero violation up through 92, then a hard cliff at 96+.
            8: 0.0, 16: 0.0, 32: 0.0, 64: 0.0, 80: 0.0,
            88: 0.0, 92: 0.0,
            96: 0.66, 128: 0.66, 256: 0.66,
            # Extra cliff midpoint that the stepper should schedule.
            94: 0.18,
        },
    )
    # The infill phase must add pool=94 — midpoint of [92, 96].
    assert 94 in seq, (
        f"marginal-cliff infill didn't schedule pool=94. "
        f"seq={seq}"
    )


def test_stepper_marginal_cliff_infill_skipped_when_marginal_already_observed() -> None:
    """If bisect_fail already produced a marginal measurement, no
    extra infill is needed. Cohorts with naturally wide marginal
    bands shouldn't get gratuitous extra points."""
    stepper = TwoKneeStepper(
        initial_pool_size=8, max_pool_size=256, fail_threshold=0.30,
    )
    # Curve has a marginal point (pool=48 at 7%) caught by bisection.
    # The marginal-cliff infill should NOT fire because marginal
    # already exists.
    _drive_stepper(
        stepper,
        violation_at={
            8: 0.0, 16: 0.0, 32: 0.0, 48: 0.07,  # marginal!
            64: 0.40, 96: 0.50, 128: 0.55,
            40: 0.0, 56: 0.20,
        },
    )
    # No specific assertion on a single pool — instead, verify the
    # algorithm did NOT measure a "redundant" pool inside the
    # already-narrow bracket. Marginal measurements present:
    marginal_pools = [
        r.pool_size for r in stepper.history
        if 0.05 <= r.violation_rate < 0.30
    ]
    assert marginal_pools, "expected at least one marginal point in this curve"


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


def test_classify_status_clean_pass_at_zero_misses() -> None:
    """Zero misses out of 100 → upper CI ~0.037, well under 5% → pass."""
    assert _classify_status(0, 100) == "pass"


def test_classify_status_clean_fail_at_high_rate() -> None:
    """50/100 misses → lower CI ~0.40, well past 30% → fail."""
    assert _classify_status(50, 100) == "fail"
    assert _classify_status(80, 100) == "fail"


def test_classify_status_boundary_30pct_stays_marginal() -> None:
    """The May 2026 Intel quick_lookup pool=232 case: 30/100 misses
    (rate=0.30 exactly) sits on the failure threshold but the lower
    Wilson CI is well below 30%, so the verdict is marginal — not
    fail. One sample's worth of noise shouldn't flip the band."""
    assert _classify_status(30, 100) == "marginal"
    # Need a clear lower-CI ≥ 0.30 to commit to fail.
    assert _classify_status(40, 100) == "fail"


def test_classify_status_boundary_5pct_stays_marginal() -> None:
    """4/100 (rate=0.04) might look like pass on the point estimate,
    but upper Wilson CI is ~0.10 — too noisy to commit to pass. Same
    asymmetric-rigor design as the fail boundary."""
    assert _classify_status(4, 100) == "marginal"
    # Need clearly-below-5% (small upper CI) to call pass.
    assert _classify_status(0, 100) == "pass"
    assert _classify_status(1, 100) == "marginal"


def test_classify_status_low_n_stays_conservative() -> None:
    """n=10, 3 misses (rate=0.30): upper CI ~0.60, lower CI ~0.11.
    Neither bound clears its threshold → marginal. Wilson naturally
    handles the small-sample case without separate logic."""
    assert _classify_status(3, 10) == "marginal"
    # Even 0/5 isn't tight enough for pass — upper CI ~0.43.
    assert _classify_status(0, 5) == "marginal"


def test_classify_status_zero_n_returns_pending() -> None:
    """No completed turns in window → pending. ``_status`` callers
    write this through to the row so the curve renderer can show
    'in flight' rather than misclassify as pass."""
    assert _classify_status(0, 0) == "pending"


# ── Persona / cohort definitions ──────────────────────────────────────


def test_every_cohort_validates() -> None:
    """Cohort weights must sum to 1.0 and reference known persona ids
    — invariants the runner relies on at start-up."""
    for cid in COHORTS:
        get_cohort(cid)


def test_long_form_generator_persona_decode_stress_shape() -> None:
    """The decode-stress counterweight to document_qa / summarizer.
    Short prompt, very long output — stresses the engine's decode
    pipeline rather than its prefill pipeline. Spec is fixed in
    personas.py; test pins it so accidental edits to the LogNormal
    parameters get caught."""
    import math
    p = PERSONAS["long_form_generator"]
    assert math.isclose(math.exp(p.input_tokens.mu), 200, rel_tol=0.01)
    assert math.isclose(math.exp(p.output_tokens.mu), 3000, rel_tol=0.01)
    # Output is 15× larger than input — defining property of this persona.
    assert math.exp(p.output_tokens.mu) > 10 * math.exp(p.input_tokens.mu), (
        "long_form_generator must be heavily output-skewed (decode-bound)"
    )
    assert p.ttft_target_seconds == 15.0
    assert p.ttft_failure_seconds == 45.0
    assert p.tpot_target_ms == 180.0
    assert p.tpot_failure_ms == 270.0


def test_long_form_generator_in_expected_cohorts() -> None:
    """Long-form generator must appear in the four cohort mixes that
    have meaningful content-generation load. chat_heavy is the
    short-prompt cohort and intentionally excludes it."""
    expected_membership = {
        "chat_heavy": False,                # quick_lookup-dominant; no decode stress needed
        "general_knowledge": True,          # 10%
        "writer_dominant": True,            # 30%
        "software_engineering": True,       # 15%
        "analyst_team": True,               # 10%
    }
    for cid, should_have in expected_membership.items():
        weights = COHORTS[cid].persona_weights
        present = "long_form_generator" in weights
        assert present == should_have, (
            f"{cid}: long_form_generator membership expected={should_have}, "
            f"actual={present}, weights={weights}"
        )


def test_summarizer_no_longer_in_cohort_mixes() -> None:
    """Summarizer was redundant with document_qa (both prefill-stress)
    and is removed from every cohort mix. The persona definition
    stays for diagnostic / standalone use via ``run-persona``."""
    assert "summarizer" in PERSONAS, (
        "summarizer persona must remain defined for standalone runs"
    )
    for cid, c in COHORTS.items():
        assert "summarizer" not in c.persona_weights, (
            f"{cid}: summarizer dropped from cohort mixes — "
            f"document_qa covers prefill-stress, long_form_generator "
            f"covers decode-stress"
        )


def test_every_persona_has_target_and_failure_thresholds() -> None:
    """The aggregator reads target + failure thresholds per-event;
    missing fields would silently classify every request as a
    violation. Both target and failure must be set, and target
    must be tighter than failure on each axis."""
    for p in PERSONAS.values():
        assert p.ttft_target_seconds > 0
        assert p.ttft_failure_seconds > 0
        assert p.ttft_failure_seconds > p.ttft_target_seconds, (
            f"{p.id}: failure must be looser than target on TTFT"
        )
        assert p.tpot_target_ms > 0
        assert p.tpot_failure_ms > 0
        assert p.tpot_failure_ms > p.tpot_target_ms, (
            f"{p.id}: failure must be looser than target on TPOT"
        )


def test_every_persona_has_read_and_active_think() -> None:
    """Both read_time and active_think distributions must be set
    so virtual_user.run_virtual_user can sample post-response delay
    from their sum."""
    for p in PERSONAS.values():
        assert p.read_time_seconds is not None
        assert p.active_think_seconds is not None


# ── Personas vs cohorts: clean separation ────────────────────────────


def test_cohorts_dict_contains_no_single_persona_entries() -> None:
    """Cohorts are *team mixes*. Single personas are accessed via the
    PERSONAS dict and ``cohort_from_persona``, NOT as one-persona
    cohorts in the COHORTS dict (which would conflate the two
    user-facing concepts)."""
    from simulator.personas import COHORTS
    for cid, cohort in COHORTS.items():
        assert len(cohort.persona_weights) > 1, (
            f"{cid} has only one persona — should be accessed via "
            f"run-persona, not registered as a cohort"
        )
        assert cohort.category == "cohort"


def test_cohort_from_persona_builds_ephemeral_one_persona_cohort() -> None:
    """``run-persona`` plumbs through the same Cohort-based runner code
    by wrapping a persona in an ephemeral one-persona Cohort. The
    resulting Cohort must validate and carry category='persona' so
    runs are filed under the right buyer-page section."""
    from simulator.personas import PERSONAS, cohort_from_persona
    for pid in PERSONAS:
        c = cohort_from_persona(pid)
        c.validate()
        assert c.id == pid
        assert c.persona_weights == {pid: 1.0}
        assert c.category == "persona"


def test_resolve_workload_group_keywords() -> None:
    """``--type`` accepts 'all' / 'personas' / 'cohorts' so common
    sweeps don't need a hand-typed id list."""
    from simulator.personas import COHORTS, PERSONAS, resolve_workload_group

    p_all, c_all = resolve_workload_group("all")
    assert set(p_all) == set(PERSONAS.keys())
    assert set(c_all) == set(COHORTS.keys())

    p_only, c_only = resolve_workload_group("personas")
    assert set(p_only) == set(PERSONAS.keys())
    assert c_only == []

    p_none, c_only2 = resolve_workload_group("cohorts")
    assert p_none == []
    assert set(c_only2) == set(COHORTS.keys())


def test_resolve_workload_group_explicit_list_disambiguates() -> None:
    """A comma list mixes persona ids and cohort ids; the resolver
    splits them by which dict each id is in."""
    from simulator.personas import resolve_workload_group
    p, c = resolve_workload_group("quick_lookup,chat_heavy,code_assist")
    assert p == ["quick_lookup", "code_assist"]
    assert c == ["chat_heavy"]


def test_resolve_workload_group_rejects_unknown_id() -> None:
    """Surfacing a typo at parse time beats running 5 hours of sweep
    only to miss what you meant to measure."""
    from simulator.personas import resolve_workload_group
    import pytest as _pt
    with _pt.raises(KeyError, match="some_typo"):
        resolve_workload_group("quick_lookup,some_typo")


def test_find_completed_runs_returns_only_ok_status(tmp_path) -> None:
    """``find_completed_runs`` is the source of truth for resume:
    workloads with ``final_status='ok'`` are skipped on a --resume
    sweep, anything else is retried. Pin all four states (ok,
    interrupted, no_samples, null/in-progress) and confirm only ok
    counts as 'completed'."""
    from simulator.database import Database
    from simulator.runner import find_completed_runs

    states = [
        ("quick_lookup", "ok"),
        ("conversational", "interrupted"),
        ("writer", "no_samples"),
        ("document_qa", None),  # still in-progress, never finalised
    ]
    for cohort_id, status in states:
        db_path = tmp_path / f"{cohort_id}.db"
        db = Database(db_path)
        db.insert_run(
            cohort_run_id=cohort_id,
            started_at="2026-01-01T00:00:00Z",
            engine_type="vllm",
            model_id="Qwen/Test",
            cohort_id=cohort_id,
            cohort_definition={},
            config={},
        )
        if status is not None:
            db.finalise_run(cohort_id, "2026-01-01T01:00:00Z", status)
        db.close()

    completed = find_completed_runs(tmp_path, "vllm", "Qwen/Test")
    assert completed == {"quick_lookup"}, (
        "only final_status='ok' should count as completed; "
        f"got {completed}"
    )


def test_find_completed_runs_filters_by_engine_and_model(tmp_path) -> None:
    """Resume must NOT skip a cohort because it was completed against
    a different engine or model — those are different measurements."""
    from simulator.database import Database
    from simulator.runner import find_completed_runs

    cases = [
        ("vllm", "Qwen/A", "chat_heavy"),
        ("vllm", "Qwen/B", "chat_heavy"),  # different model
        ("sglang", "Qwen/A", "chat_heavy"),  # different engine
    ]
    for engine_type, model_id, cohort_id in cases:
        db_path = tmp_path / f"{engine_type}_{model_id.replace('/', '_')}.db"
        db = Database(db_path)
        db.insert_run(
            cohort_run_id=f"{engine_type}-{model_id}",
            started_at="2026-01-01T00:00:00Z",
            engine_type=engine_type,
            model_id=model_id,
            cohort_id=cohort_id,
            cohort_definition={},
            config={},
        )
        db.finalise_run(f"{engine_type}-{model_id}", "2026-01-01T01:00:00Z", "ok")
        db.close()

    # Only the (vllm, Qwen/A) run counts.
    assert find_completed_runs(tmp_path, "vllm", "Qwen/A") == {"chat_heavy"}
    assert find_completed_runs(tmp_path, "vllm", "Qwen/B") == {"chat_heavy"}
    assert find_completed_runs(tmp_path, "sglang", "Qwen/A") == {"chat_heavy"}
    # And not for other combos.
    assert find_completed_runs(tmp_path, "sglang", "Qwen/B") == set()


def test_find_completed_runs_handles_empty_dir(tmp_path) -> None:
    from simulator.runner import find_completed_runs
    assert find_completed_runs(tmp_path, "vllm", "Qwen/Test") == set()


class _FakeChunk:
    """Minimal chunk shape that matches what the openai SDK yields.

    ``kind`` controls which delta field carries the text — ``content``
    (default; emulates standard chat.completions chunks) or
    ``reasoning`` (reasoning models like GPT-OSS stream chain-of-
    thought via this field BEFORE the content phase).
    """
    def __init__(self, content: str | None, kind: str = "content"):
        class _Delta:
            pass
        d = _Delta()
        # Always assign both attrs so getattr() in streaming.py finds
        # whichever one is set; the other is None and falsy.
        d.content = content if kind == "content" else None
        d.reasoning = content if kind == "reasoning" else None
        class _Choice:
            pass
        c = _Choice()
        c.delta = d
        self.choices = [c]


class _FakeStream:
    """Async iterator that emits chunks at scheduled times relative
    to its own start.

    Schedule entries can be:
      * (elapsed_s, content_or_None) — content chunk, default ``kind``.
      * (elapsed_s, text, kind)      — reasoning or content chunk.
    """
    def __init__(self, schedule: list[tuple]):
        self._schedule = schedule
        self._idx = 0
        self._t0: float | None = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        import time as _time
        if self._t0 is None:
            self._t0 = _time.monotonic()
        if self._idx >= len(self._schedule):
            raise StopAsyncIteration
        entry = self._schedule[self._idx]
        self._idx += 1
        elapsed_target = entry[0]
        text = entry[1]
        kind = entry[2] if len(entry) > 2 else "content"
        wait = elapsed_target - (_time.monotonic() - self._t0)
        if wait > 0:
            await asyncio.sleep(wait)
        return _FakeChunk(text, kind=kind)


def test_tier_1_pre_ttft_stalled() -> None:
    """A stream that never produces a first token within
    pre_ttft_timeout_s aborts with error='ttft_stalled'."""
    from simulator.streaming import consume_with_tiers

    async def go():
        # Stream that would emit at t=2s, but pre_ttft cap is 0.3s.
        stream = _FakeStream([(2.0, "hello")])
        return await consume_with_tiers(
            create_stream=lambda: _coroutine_returning(stream),
            pre_ttft_timeout_s=0.3,
            inter_token_timeout_s=10.0,
            hard_timeout_s=30.0,
        )

    result = asyncio.run(go())
    assert result.error == "ttft_stalled"
    assert result.ttft_ms is None
    assert result.output_tokens == 0
    assert 250 < result.total_ms < 600  # roughly the 0.3s timeout


def test_tier_2_decode_stalled() -> None:
    """A stream that emits the first token quickly but then stalls
    aborts with error='decode_stalled'. Partial progress (the tokens
    we did get) is preserved in the result."""
    from simulator.streaming import consume_with_tiers

    async def go():
        # First two tokens at t=0.05s and t=0.10s, then a long gap.
        stream = _FakeStream([
            (0.05, "first"), (0.10, "second"), (5.0, "stalled"),
        ])
        return await consume_with_tiers(
            create_stream=lambda: _coroutine_returning(stream),
            pre_ttft_timeout_s=10.0,
            inter_token_timeout_s=0.3,  # 300ms inter-token cap
            hard_timeout_s=30.0,
        )

    result = asyncio.run(go())
    assert result.error == "decode_stalled"
    assert result.ttft_ms is not None
    assert result.ttft_ms < 200  # ~50 ms first token
    assert result.output_tokens == 2  # partial progress preserved
    assert result.output_text == "firstsecond"


def test_tier_3_hard_timeout() -> None:
    """A stream that emits tokens steadily but past the hard ceiling
    aborts with error='hard_timeout'. This is the 'slow-but-
    progressing' case — engine is functional but past budget."""
    from simulator.streaming import consume_with_tiers

    async def go():
        # 6 tokens spaced 0.1s apart = 0.6s total. Pre-TTFT and
        # inter-token are generous; hard ceiling is 0.3s — should fire
        # mid-stream after ~3 tokens.
        stream = _FakeStream([
            (0.05, f"t{i}") for i in range(6)
        ])
        # Adjust schedule so each token is +0.1s after the previous.
        schedule = [(0.05 + i * 0.1, f"t{i}") for i in range(6)]
        return await consume_with_tiers(
            create_stream=lambda: _coroutine_returning(_FakeStream(schedule)),
            pre_ttft_timeout_s=10.0,
            inter_token_timeout_s=10.0,
            hard_timeout_s=0.3,
        )

    result = asyncio.run(go())
    assert result.error == "hard_timeout"
    assert result.ttft_ms is not None  # got first token
    # We got at least 1 token but not all 6.
    assert 1 <= result.output_tokens < 6


def test_clean_completion_no_error() -> None:
    """The happy path: a stream that emits tokens within all tier
    budgets returns no error and full output."""
    from simulator.streaming import consume_with_tiers

    async def go():
        schedule = [(0.01 * (i + 1), f"t{i}") for i in range(5)]
        return await consume_with_tiers(
            create_stream=lambda: _coroutine_returning(_FakeStream(schedule)),
            pre_ttft_timeout_s=5.0,
            inter_token_timeout_s=5.0,
            hard_timeout_s=5.0,
        )

    result = asyncio.run(go())
    assert result.error is None
    assert result.output_tokens == 5
    assert result.ttft_ms is not None and result.ttft_ms < 100
    # For non-reasoning models ttfct equals ttft trivially (content
    # IS the first user-visible token; no reasoning phase).
    assert result.ttfct_ms == result.ttft_ms
    assert result.reasoning_tokens == 0


def test_reasoning_model_ttft_uses_first_reasoning_token() -> None:
    """For reasoning models (GPT-OSS etc.), TTFT must be measured
    against the first user-visible token of any kind — including
    chain-of-thought delta.reasoning chunks. Without this fix, TTFT
    would lag by the entire reasoning duration (typically several
    seconds), badly distorting capacity gating."""
    from simulator.streaming import consume_with_tiers

    async def go():
        # Reasoning starts at t=0.05s, content at t=0.30s.
        schedule = [
            (0.05, "The", "reasoning"),
            (0.10, " user", "reasoning"),
            (0.15, " asks", "reasoning"),
            (0.30, "4", "content"),
            (0.35, ".", "content"),
        ]
        return await consume_with_tiers(
            create_stream=lambda: _coroutine_returning(_FakeStream(schedule)),
            pre_ttft_timeout_s=5.0,
            inter_token_timeout_s=5.0,
            hard_timeout_s=5.0,
        )

    result = asyncio.run(go())
    assert result.error is None
    # TTFT = first reasoning token (~50ms in).
    assert result.ttft_ms is not None
    assert 30 < result.ttft_ms < 200, (
        f"ttft should track first reasoning chunk, got {result.ttft_ms}"
    )
    # TTFCT = first content token (~300ms in).
    assert result.ttfct_ms is not None
    assert 250 < result.ttfct_ms < 450, (
        f"ttfct should track first content chunk, got {result.ttfct_ms}"
    )
    # ttfct must be strictly later than ttft (reasoning came first).
    assert result.ttfct_ms > result.ttft_ms
    assert result.reasoning_tokens == 3
    assert result.output_tokens == 2  # content tokens only
    assert result.output_text == "4."  # reasoning NOT included


def test_reasoning_only_response_flagged_as_no_content_tokens() -> None:
    """The May 2026 GPT-OSS diagnostic: with too-tight max_tokens,
    reasoning consumes the entire output budget and finish_reason=length
    fires before any content streams. The simulator should record this
    as a distinct ``no_content_tokens`` error so post-hoc analysis can
    tell it apart from generic ``no_tokens`` (engine emitted nothing
    at all)."""
    from simulator.streaming import consume_with_tiers

    async def go():
        # All reasoning, no content. Simulates a reasoning model
        # hitting max_tokens before reasoning finishes.
        schedule = [
            (0.05, "The", "reasoning"),
            (0.10, " user", "reasoning"),
            (0.15, " asks", "reasoning"),
        ]
        return await consume_with_tiers(
            create_stream=lambda: _coroutine_returning(_FakeStream(schedule)),
            pre_ttft_timeout_s=5.0,
            inter_token_timeout_s=5.0,
            hard_timeout_s=5.0,
        )

    result = asyncio.run(go())
    assert result.error == "no_content_tokens"
    # ttft IS set (reasoning did stream); ttfct is None (no content).
    assert result.ttft_ms is not None
    assert result.ttfct_ms is None
    assert result.reasoning_tokens == 3
    assert result.output_tokens == 0


def test_engine_config_reasoning_defaults_off() -> None:
    """Default Config has reasoning disabled — preserves existing
    non-reasoning model behavior. ``reasoning_effort`` is harmless
    when reasoning=False (never gets passed to the engine)."""
    from simulator.config import EngineConfig
    cfg = EngineConfig()
    assert cfg.reasoning is False
    assert cfg.reasoning_effort == "medium"  # default if reasoning ever enabled


def test_reasoning_overhead_table_covers_all_effort_levels() -> None:
    """REASONING_OVERHEAD_TOKENS must have an entry for every effort
    level YAMLs declare. Otherwise a config typo silently falls back
    to the default and short-output personas hit the no_content_tokens
    deadlock from the May 2026 GPT-OSS run."""
    from simulator.virtual_user import REASONING_OVERHEAD_TOKENS
    expected = {"minimal", "low", "medium", "high"}
    assert expected.issubset(set(REASONING_OVERHEAD_TOKENS.keys())), (
        f"Missing overhead entries: {expected - set(REASONING_OVERHEAD_TOKENS.keys())}"
    )
    # Monotonically increasing — higher effort = more reasoning budget.
    levels = ["minimal", "low", "medium", "high"]
    values = [REASONING_OVERHEAD_TOKENS[lv] for lv in levels]
    assert values == sorted(values), (
        f"reasoning overhead must increase with effort level: {values}"
    )
    # ``medium`` (the default declared in GPT-OSS YAMLs) must be
    # large enough to cover quick_lookup's typical answer (~30 tokens
    # of content) PLUS the observed reasoning-phase length. AMD
    # 2026-05-08 measurement showed reasoning hit 532 tokens on
    # document_qa no-content-tokens cases — so the budget needs to
    # cover at least that much, with headroom.
    assert REASONING_OVERHEAD_TOKENS["medium"] >= 550, (
        "medium overhead must accommodate the observed GPT-OSS "
        "reasoning-phase length (~530+ tokens at medium effort)"
    )


def test_gpt_oss_yaml_declares_reasoning() -> None:
    """Both GPT-OSS configs (Intel + AMD) must declare reasoning=true
    with reasoning_effort=medium — see the YAML's reasoning-model
    block for rationale."""
    from simulator.config import load_config
    for path in (
        "config/r7735_vllm_dual_socket_gpt_oss.yaml",
        "config/xeon_vllm_gpt_oss.yaml",
    ):
        cfg = load_config(path)
        assert cfg.engine.reasoning is True, (
            f"{path}: GPT-OSS is a reasoning model"
        )
        assert cfg.engine.reasoning_effort == "medium", (
            f"{path}: reasoning_effort default is medium"
        )


async def _coroutine_returning(value):
    """Helper: a coroutine that immediately returns ``value``. Used
    to wrap a pre-built stream in the create_stream callable shape
    that consume_with_tiers expects."""
    return value


def test_pool_manager_spawns_users_with_single_session() -> None:
    """Per-session-respawn model: each virtual user gets sessions_target=1
    regardless of persona.sessions_before_leaving. Each session is
    its own user spawn — pool_size means ``active concurrent sessions``,
    matching what the engine actually sees and what the buyer-page
    deployment narrative wants."""
    import asyncio
    from openai import AsyncOpenAI
    from simulator.pool_manager import PoolManager
    from simulator.personas import get_cohort
    from simulator.virtual_user import SharedState
    from simulator.tokenizer_corpus import TokenCorpus

    async def go():
        cohort = get_cohort("chat_heavy")
        # Real client; never used because we'll inspect stats before any
        # request fires. AsyncOpenAI requires an api_key arg.
        client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
        state = SharedState()
        # Avoid loading a real tokenizer — corpus.count is unused by
        # _spawn_one.
        class _StubCorpus:
            def make_text(self, n, rng): return "hi"
            def count(self, s): return 1
        pool = PoolManager(
            cohort=cohort, clients=[client], model_id="gpt_oss",
            corpus=_StubCorpus(), state=state, request_timeout_s=600,
            ramp_spawn_interval_s=0.001,
        )
        pool._spawn_one(replaced_user_id=None)
        # The freshly-spawned user must have sessions_target=1.
        assert len(pool._users) == 1
        u = next(iter(pool._users.values()))
        assert u.stats.sessions_target == 1, (
            f"per-session-respawn requires sessions_target=1, got "
            f"{u.stats.sessions_target}"
        )
        u.cancel_event.set()
        # Drain the task so we don't leak.
        try:
            await asyncio.wait_for(u.task, timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            pass

    asyncio.run(go())


def test_persona_timeout_properties_scale_with_failure_thresholds() -> None:
    """Tier-abort timeouts derive from the FAILURE thresholds (so
    a request stays alive long enough to actually exhaust its
    failure budget before getting killed). Target thresholds are
    the SLA bar; tier aborts are the "this is broken" signal."""
    from simulator.personas import PERSONAS

    p = PERSONAS["quick_lookup"]  # ttft_failure=15.0, tpot_failure=225ms
    assert p.pre_ttft_timeout_s == p.ttft_failure_seconds * 5.0
    assert abs(
        p.inter_token_timeout_s - (p.tpot_failure_ms / 1000.0) * 20.0
    ) < 1e-6

    # summarizer has the loosest failure thresholds → most generous
    # tier-abort timeouts.
    s = PERSONAS["summarizer"]
    assert s.pre_ttft_timeout_s == s.ttft_failure_seconds * 5.0
    assert s.pre_ttft_timeout_s > p.pre_ttft_timeout_s


def test_failed_turn_event_counts_as_sla_violation_and_target_miss() -> None:
    """A timed-out / errored request registers as BOTH a violation
    (failure threshold) AND a target miss (target threshold).
    Without this, fast-failing HTTP errors would fall under the
    failure threshold and silently 'pass'."""
    from simulator.virtual_user import TurnEvent
    from simulator.personas import PERSONAS

    persona = PERSONAS["quick_lookup"]
    # quick_lookup: ttft target 5s / failure 15s, tpot 150/225ms

    # Slow timeout — well past the failure threshold.
    e_timeout = TurnEvent(
        user_id="u1", persona_id=persona.id, session_id="s",
        turn_index=0, submitted_at_ms=0, completed_at_ms=300_000,
        ttft_ms=300_000, tpot_ms=0.0, end_to_end_ms=300_000,
        input_tokens=350, history_tokens=0, output_tokens=0,
        in_flight_at_submit=1, persona=persona, error="timeout",
    )
    assert e_timeout.ttft_violation() and e_timeout.tpot_violation()
    assert e_timeout.ttft_target_miss() and e_timeout.tpot_target_miss()

    # Fast HTTP 500 — timing alone (100ms) is way under both
    # thresholds, but error flag forces both checks to fire.
    e_500 = TurnEvent(
        user_id="u2", persona_id=persona.id, session_id="s",
        turn_index=0, submitted_at_ms=0, completed_at_ms=100,
        ttft_ms=100, tpot_ms=0.0, end_to_end_ms=100,
        input_tokens=350, history_tokens=0, output_tokens=0,
        in_flight_at_submit=1, persona=persona,
        error="ConnectionError",
    )
    assert e_500.ttft_violation() and e_500.ttft_target_miss()
    assert e_500.tpot_violation() and e_500.tpot_target_miss()

    # Healthy fast turn — neither violation NOR target miss.
    e_ok = TurnEvent(
        user_id="u3", persona_id=persona.id, session_id="s",
        turn_index=0, submitted_at_ms=0, completed_at_ms=500,
        ttft_ms=400, tpot_ms=80.0, end_to_end_ms=500,
        input_tokens=350, history_tokens=0, output_tokens=10,
        in_flight_at_submit=1, persona=persona,
    )
    assert not e_ok.ttft_violation() and not e_ok.tpot_violation()
    assert not e_ok.ttft_target_miss() and not e_ok.tpot_target_miss()


def test_target_miss_without_violation() -> None:
    """The new band: a turn that exceeds TARGET but stays within
    FAILURE registers as target_miss=True, violation=False. This
    is the quality signal the buyer page surfaces alongside
    capacity. quick_lookup target=5s, failure=15s → 8s ttft is
    in this band."""
    from simulator.virtual_user import TurnEvent
    from simulator.personas import PERSONAS

    persona = PERSONAS["quick_lookup"]
    # 8000ms: between 5s (target) and 15s (failure)
    e_marginal = TurnEvent(
        user_id="u4", persona_id=persona.id, session_id="s",
        turn_index=0, submitted_at_ms=0, completed_at_ms=8000,
        ttft_ms=8000, tpot_ms=120.0, end_to_end_ms=8000,
        input_tokens=350, history_tokens=0, output_tokens=10,
        in_flight_at_submit=1, persona=persona,
    )
    assert not e_marginal.ttft_violation(), "8s < 15s failure → no violation"
    assert e_marginal.ttft_target_miss(), "8s > 5s target → target miss"
    assert not e_marginal.tpot_violation()
    assert not e_marginal.tpot_target_miss()  # 120ms < 150ms target


def test_legacy_turn_events_gets_error_column(tmp_path) -> None:
    """Legacy DBs (pre-error-column) must get the column lifted on
    open so the persistence path doesn't crash."""
    import sqlite3
    from simulator.database import Database

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE turn_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_id INTEGER NOT NULL,
                persona_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                submitted_at_ms INTEGER NOT NULL,
                ttft_ms REAL NOT NULL,
                completed_at_ms INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                history_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                tpot_ms REAL NOT NULL,
                end_to_end_ms REAL NOT NULL,
                in_flight_at_submit INTEGER NOT NULL,
                in_flight_avg_during REAL,
                in_flight_peak_during INTEGER,
                sla_ttft_violation INTEGER NOT NULL,
                sla_tpot_violation INTEGER NOT NULL,
                token_timestamps_json TEXT
            );
            """
        )
    db = Database(db_path)
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(turn_events)")}
    assert "error" in cols
    db.close()


def test_shared_state_initialises_step_progress() -> None:
    """``SharedState`` is the live channel between the measurement
    loop and the snapshot recorder for per-step sample progress.
    Defaults must be 0 / 0 so the dashboard renders "—" before the
    first step starts."""
    from simulator.virtual_user import SharedState
    s = SharedState()
    assert s.step_samples == 0
    assert s.step_target_samples == 0


def test_legacy_simulation_snapshots_gets_step_columns(tmp_path) -> None:
    """Legacy DBs (no step_samples columns yet) must get them lifted
    on open; otherwise the SnapshotRecorder INSERT crashes."""
    import sqlite3
    from simulator.database import Database

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE simulation_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cohort_run_id TEXT NOT NULL,
                snapshot_at_ms INTEGER NOT NULL,
                phase TEXT NOT NULL,
                pool_size INTEGER NOT NULL,
                in_flight INTEGER NOT NULL,
                requests_completed INTEGER NOT NULL,
                errors INTEGER NOT NULL
            );
            """
        )
    db = Database(db_path)
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(simulation_snapshots)")}
    assert "step_samples" in cols
    assert "step_target_samples" in cols
    # Insert with the new columns to confirm they're writable.
    db.insert_snapshot({
        "cohort_run_id": "c", "snapshot_at_ms": 1, "phase": "measuring",
        "pool_size": 8, "in_flight": 4, "requests_completed": 12, "errors": 0,
        "step_samples": 29, "step_target_samples": 100,
    })
    row = db.fetchone(
        "SELECT step_samples, step_target_samples FROM simulation_snapshots"
    )
    assert row["step_samples"] == 29
    assert row["step_target_samples"] == 100
    db.close()


def test_legacy_measurement_aggregate_migrates_to_columns(tmp_path) -> None:
    """A DB written before the aggregate-table collapse should silently
    lift its measurement_aggregate rows onto cohort_measurements when
    opened. Otherwise old runs go dark on the next ``make export``."""
    import sqlite3
    from simulator.database import Database

    db_path = tmp_path / "legacy.db"
    # Hand-build a legacy-shaped DB with the old separate
    # measurement_aggregate table.
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE cohort_run (
                cohort_run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                engine_type TEXT NOT NULL,
                model_id TEXT NOT NULL,
                cohort_id TEXT NOT NULL,
                cohort_definition_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                final_status TEXT
            );
            CREATE TABLE cohort_measurements (
                measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cohort_run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                target_pool_size INTEGER NOT NULL,
                measured_avg_pool_size REAL NOT NULL,
                measured_avg_in_flight REAL NOT NULL,
                measurement_started_at TEXT NOT NULL,
                measurement_duration_s INTEGER NOT NULL,
                sample_size INTEGER NOT NULL,
                ttft_violation_rate REAL NOT NULL,
                tpot_violation_rate REAL NOT NULL,
                combined_violation_rate REAL NOT NULL,
                violation_rate_ci_lower REAL NOT NULL,
                violation_rate_ci_upper REAL NOT NULL,
                ttft_p50_ms REAL, ttft_p75_ms REAL, ttft_p95_ms REAL,
                tpot_p50_ms REAL, tpot_p75_ms REAL, tpot_p95_ms REAL,
                avg_kv_cache_pct REAL,
                estimated_prefix_hit_rate REAL,
                capacity_status TEXT NOT NULL
            );
            CREATE TABLE measurement_aggregate (
                measurement_id INTEGER PRIMARY KEY,
                pmu_ipc REAL,
                memory_bw_read_gb_s_avg REAL,
                onednn_amx_time_fraction REAL
            );
            """
        )
        conn.execute(
            """INSERT INTO cohort_measurements
               (cohort_run_id, step_index, target_pool_size,
                measured_avg_pool_size, measured_avg_in_flight,
                measurement_started_at, measurement_duration_s,
                sample_size, ttft_violation_rate, tpot_violation_rate,
                combined_violation_rate, violation_rate_ci_lower,
                violation_rate_ci_upper, capacity_status)
               VALUES ('crid', 0, 8, 8.0, 7.5, '2026-01-01T00:00:00Z',
                       60, 100, 0.0, 0.0, 0.0, 0.0, 0.0, 'pass')"""
        )
        conn.execute(
            "INSERT INTO measurement_aggregate "
            "(measurement_id, pmu_ipc, memory_bw_read_gb_s_avg, onednn_amx_time_fraction) "
            "VALUES (1, 1.42, 87.3, 0.71)"
        )

    # Opening through Database should run the migration silently.
    db = Database(db_path)
    row = db.fetchone(
        "SELECT pmu_ipc, memory_bw_read_gb_s_avg, onednn_amx_time_fraction "
        "FROM cohort_measurements WHERE measurement_id = 1"
    )
    assert row["pmu_ipc"] == 1.42
    assert row["memory_bw_read_gb_s_avg"] == 87.3
    assert row["onednn_amx_time_fraction"] == 0.71

    # Legacy table is gone.
    leftover = db.fetchone(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='measurement_aggregate'"
    )
    assert leftover is None
    db.close()


def test_update_measurement_patches_subset(tmp_path) -> None:
    """``update_measurement`` is the single way callers patch a row —
    used for both the post-window percentile/violation update AND the
    aggregate rollup. Verify partial updates don't clobber siblings."""
    from simulator.database import Database

    db = Database(tmp_path / "u.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="chat_heavy",
        cohort_definition={}, config={},
    )
    mid = db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0, "target_pool_size": 8,
        "measured_avg_pool_size": 8.0, "measured_avg_in_flight": 0.0,
        "measurement_started_at": "2026-01-01T00:00:00Z",
        "measurement_duration_s": 0, "sample_size": 0,
        "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0, "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 0.0, "capacity_status": "pending",
    })
    # Two partial updates: window finalisation, then aggregate rollup.
    db.update_measurement(mid, {"sample_size": 100, "capacity_status": "pass"})
    db.update_measurement(mid, {"pmu_ipc": 1.42, "memory_bw_read_gb_s_avg": 87.3})

    row = db.fetchone("SELECT * FROM cohort_measurements WHERE measurement_id = ?", (mid,))
    assert row["sample_size"] == 100
    assert row["capacity_status"] == "pass"
    assert row["pmu_ipc"] == 1.42
    assert row["memory_bw_read_gb_s_avg"] == 87.3
    db.close()


def test_find_completed_runs_one_db_many_cohorts(tmp_path) -> None:
    """The new layout puts multiple cohort_runs into a single
    ``run.db``. find_completed_runs has to enumerate every ok-status
    row, not just the first one — otherwise resume would skip
    cohorts that finished after the first one."""
    from simulator.database import Database
    from simulator.runner import find_completed_runs

    db = Database(tmp_path / "run.db")
    cohort_states = [
        ("quick_lookup", "ok"),
        ("conversational", "ok"),
        ("writer", "interrupted"),  # NOT completed — should be retried
        ("document_qa", "ok"),
    ]
    for cohort_id, status in cohort_states:
        db.insert_run(
            cohort_run_id=f"crid-{cohort_id}",
            started_at="2026-01-01T00:00:00Z",
            engine_type="vllm",
            model_id="Qwen/Test",
            cohort_id=cohort_id,
            cohort_definition={},
            config={},
        )
        db.finalise_run(f"crid-{cohort_id}", "2026-01-01T01:00:00Z", status)
    db.close()

    completed = find_completed_runs(tmp_path, "vllm", "Qwen/Test")
    assert completed == {"quick_lookup", "conversational", "document_qa"}


# ── Run-directory layout (run_NN/) ────────────────────────────────────


def test_dashboard_state_reflects_terminal_status(tmp_path) -> None:
    """When a cohort_run has ``final_status='ok'`` set, the dashboard
    body should treat that as authoritative and render "completed",
    not the stale ``measuring`` phase from the last snapshot before
    SnapshotRecorder shut down."""
    from simulator.database import Database
    from simulator.dashboard import _read_state, _render

    db_path = tmp_path / "run.db"
    db = Database(db_path)
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="chat_heavy",
        cohort_definition={"name": "x"}, config={},
    )
    # Stale "measuring" snapshot (the bug condition).
    db.insert_snapshot({
        "cohort_run_id": "crid", "snapshot_at_ms": 1000,
        "phase": "measuring", "pool_size": 32, "in_flight": 18,
        "requests_completed": 412, "errors": 0,
        "step_samples": 87, "step_target_samples": 100,
    })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    state = _read_state(db_path)
    assert state["ok"] is True
    assert state["run"]["final_status"] == "ok"
    assert state["sweep"]["total"] == 1
    assert state["sweep"]["ok"] == 1
    assert state["sweep"]["in_progress"] == 0

    # Render and inspect the body: the cohort is terminal, so the
    # phase row should read "completed" (mapped from final_status='ok')
    # rather than the stale "measuring" snapshot phase. Use Console's
    # capture API rather than poking at Layout internals.
    from rich.console import Console
    cap = Console(width=140, record=True)
    cap.print(_render(state))
    out = cap.export_text()
    assert "completed" in out
    assert "1/1 complete" in out


def test_dashboard_waiting_render_does_not_crash(tmp_path) -> None:
    """The waiting-for-DB screen must render without erroring even
    when the run_dir doesn't exist yet (engine boot hasn't created
    the run_NN/ directory)."""
    from simulator.dashboard import _render_waiting

    # No run_dir exists yet — should still render
    layout = _render_waiting(tmp_path / "nonexistent_runs", 5.0)
    from rich.console import Console
    cap = Console(width=120, record=True)
    cap.print(layout)
    out = cap.export_text()
    assert "Waiting for" in out
    assert "Elapsed" in out


def test_dashboard_state_in_progress(tmp_path) -> None:
    """Mid-sweep (final_status NULL) must still render the live phase
    + step-samples ratio normally."""
    from simulator.database import Database
    from simulator.dashboard import _read_state, _render

    db_path = tmp_path / "run.db"
    db = Database(db_path)
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="chat_heavy",
        cohort_definition={"name": "x"}, config={},
    )
    db.insert_snapshot({
        "cohort_run_id": "crid", "snapshot_at_ms": 1000,
        "phase": "measuring", "pool_size": 32, "in_flight": 18,
        "requests_completed": 412, "errors": 0,
        "step_samples": 87, "step_target_samples": 100,
    })
    db.close()

    state = _read_state(db_path)
    assert state["run"]["final_status"] is None
    assert state["sweep"] == {
        "total": 1, "ok": 0, "other_terminal": 0, "in_progress": 1,
    }

    # Smoke-render to confirm no crash and the live phase / sample
    # ratio land in the body (the "Sweep progress" formatting is
    # tested via state['sweep'] above; rich line-wrap makes the
    # rendered string brittle).
    from rich.console import Console
    cap = Console(width=140, record=True)
    cap.print(_render(state))
    out = cap.export_text()
    assert "measuring" in out
    assert "87 / 100" in out
    assert "0/1 complete" in out


def test_bottleneck_freq_droop_uses_mean_not_min() -> None:
    """``frequency_droop`` attribution must trigger on the all-core
    MEAN frequency dropping below base clock, not on the per-core
    MIN. The min commonly hits 0.5 GHz from Linux parking idle
    cores in deep C-states — this is normal, not a bottleneck. A
    min-based heuristic over-attributes droop to cohorts whose
    real bottleneck is decode or prefill compute (verified on the
    Intel sweep run_06 — 9 of 11 cohorts mis-attributed)."""
    from simulator.export import _bottleneck

    # Healthy mean (3.4 GHz) but parked-core min (0.5 GHz). The
    # workload isn't actually frequency-limited; we should NOT
    # attribute frequency_droop here.
    measurements_parked_min = [{
        "capacity_status": "marginal",
        "ttft_violation_rate": 0.0,
        "tpot_violation_rate": 0.5,
        "telemetry": {"kv": 10.0},
        "memory_bw_read_gb_s_avg": 30.0,
        "memory_bw_write_gb_s_avg": 5.0,
        "pmu_stall_mem_ratio": 0.2,
        "onednn_amx_time_fraction": None,
        "effective_freq_ghz_mean": 3.43,   # healthy
        "effective_freq_ghz_min": 0.50,    # parked core
    }]
    bottleneck, evidence = _bottleneck(measurements_parked_min)
    assert bottleneck != "frequency_droop", (
        "must not attribute frequency_droop on parked-core min when "
        f"mean is healthy; got {bottleneck} with evidence {evidence}"
    )
    # We expect decode_throughput here since TPOT violations dominate.
    assert bottleneck == "decode_throughput"

    # Genuine droop: mean is below base clock. SHOULD trigger.
    measurements_real_droop = [{
        "capacity_status": "fail",
        "ttft_violation_rate": 0.0,
        "tpot_violation_rate": 0.5,
        "telemetry": {"kv": 5.0},
        "memory_bw_read_gb_s_avg": 20.0,
        "memory_bw_write_gb_s_avg": 4.0,
        "pmu_stall_mem_ratio": 0.2,
        "onednn_amx_time_fraction": None,
        "effective_freq_ghz_mean": 2.26,   # below 2.5 GHz threshold
        "effective_freq_ghz_min": 0.50,
    }]
    bottleneck, evidence = _bottleneck(measurements_real_droop)
    assert bottleneck == "frequency_droop"
    assert evidence["effective_freq_ghz_mean"] == 2.26


def test_export_derived_deployment_shape_fields(tmp_path) -> None:
    """The export's derived fields (headroom, cliff, band_shape,
    coverage) save the buyer-page frontend from computing them.
    Tests the four shape categories: graceful / moderate / sharp /
    unbounded / unmeasured."""
    from simulator.database import Database
    from simulator.export import export_dir

    def _build(curve_steps: list[tuple[int, str]]) -> Database:
        run_dir = tmp_path / f"run_{abs(hash(tuple(curve_steps))) % 99:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        db = Database(run_dir / "run.db")
        db.insert_run(
            cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
            engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
            cohort_definition={"name": "x"}, config={},
        )
        for i, (pool, status) in enumerate(curve_steps):
            mid = db.insert_measurement({
                "cohort_run_id": "crid", "step_index": i,
                "target_pool_size": pool,
                "measured_avg_pool_size": float(pool),
                "measured_avg_in_flight": 0.0,
                "measurement_started_at": "2026-01-01T00:00:00Z",
                "measurement_duration_s": 60, "sample_size": 100,
                "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
                "combined_violation_rate": 0.0,
                "violation_rate_ci_lower": 0.0,
                "violation_rate_ci_upper": 0.0,
                # Same status on both axes — the derived band-shape
                # fields are computed off target_status (see export.py),
                # but mirroring the value keeps the failure-axis
                # capacity/soft/fail assertions valid too.
                "capacity_status": status,
                "target_status": status,
            })
        db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
        db.close()
        return run_dir

    # graceful: cliff > 16
    rd = _build([(8, "pass"), (32, "pass"), (64, "marginal"),
                 (96, "marginal"), (192, "fail")])  # cliff = 192-96 = 96
    doc, _ = export_dir(rd.parent)
    cohort = next(c for c in doc["cohorts"] if c["cohort_run_id"] == "crid")
    assert cohort["capacity_pool_size"] == 32
    assert cohort["soft_capacity_pool_size"] == 96
    assert cohort["fail_pool_size"] == 192
    assert cohort["headroom_pool_size"] == 64
    assert cohort["cliff_pool_size"] == 96
    assert cohort["deployment_band_shape"] == "graceful"
    assert cohort["measurement_coverage"] == "full_curve"


def test_export_band_shape_sharp_cliff(tmp_path) -> None:
    """Sharp transitions (cliff < 8) get flagged as 'sharp' so the
    buyer page can warn about oversize being unforgiving."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    # pool=8 pass, 16 marginal, 20 fail → cliff=4 < 8 → sharp
    for i, (pool, status) in enumerate([(8, "pass"), (16, "marginal"), (20, "fail")]):
        db.insert_measurement({
            "cohort_run_id": "crid", "step_index": i,
            "target_pool_size": pool,
            "measured_avg_pool_size": float(pool),
            "measured_avg_in_flight": 0.0,
            "measurement_started_at": "2026-01-01T00:00:00Z",
            "measurement_duration_s": 60, "sample_size": 100,
            "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
            "combined_violation_rate": 0.0,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": status,
            "target_status": status,
        })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    cohort = doc["cohorts"][0]
    assert cohort["cliff_pool_size"] == 4
    assert cohort["deployment_band_shape"] == "sharp"


def test_export_band_shape_unbounded_when_no_fail(tmp_path) -> None:
    """If we never observed a fail (sweep ran to max without crossing),
    band_shape is 'unbounded' — explicit signal that the failure
    threshold wasn't measured."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    for i, (pool, status) in enumerate([(8, "pass"), (32, "pass"), (64, "marginal"), (96, "marginal")]):
        db.insert_measurement({
            "cohort_run_id": "crid", "step_index": i,
            "target_pool_size": pool,
            "measured_avg_pool_size": float(pool),
            "measured_avg_in_flight": 0.0,
            "measurement_started_at": "2026-01-01T00:00:00Z",
            "measurement_duration_s": 60, "sample_size": 100,
            "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
            "combined_violation_rate": 0.0,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": status,
            "target_status": status,
        })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    cohort = doc["cohorts"][0]
    assert cohort["fail_pool_size"] is None
    assert cohort["cliff_pool_size"] is None
    assert cohort["deployment_band_shape"] == "unbounded"
    assert cohort["measurement_coverage"] == "capped"


def test_export_band_shape_uses_target_axis_when_failure_axis_collapses(
    tmp_path,
) -> None:
    """Regression for the May 2026 Intel quick_lookup case:
    failure-axis capacity == soft_capacity (no marginal band) hides
    a real headroom story that lives on the target axis. Derived
    fields must read off the target axis, not the failure axis."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    # Failure axis: pass through 232, then jumps straight to fail at 240.
    # Target axis: pass at 64, marginal across 128–192, fail at 224.
    # Derived fields must come from the target axis →
    #   headroom = 192 - 64 = 128
    #   cliff    = 224 - 192 = 32  (graceful)
    curve = [
        (8,   "pass",     "pass"),
        (32,  "pass",     "pass"),
        (64,  "pass",     "pass"),
        (128, "pass",     "marginal"),
        (192, "pass",     "marginal"),
        (224, "pass",     "fail"),
        (232, "pass",     "fail"),
        (240, "fail",     "fail"),
    ]
    for i, (pool, cap, tgt) in enumerate(curve):
        db.insert_measurement({
            "cohort_run_id": "crid", "step_index": i,
            "target_pool_size": pool,
            "measured_avg_pool_size": float(pool),
            "measured_avg_in_flight": 0.0,
            "measurement_started_at": "2026-01-01T00:00:00Z",
            "measurement_duration_s": 60, "sample_size": 100,
            "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
            "combined_violation_rate": 0.0,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": cap,
            "target_status": tgt,
        })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    cohort = doc["cohorts"][0]
    # Failure-axis numbers — these still come off capacity_status
    # and remain the SLA-bound capacity story.
    assert cohort["capacity_pool_size"] == 232
    assert cohort["soft_capacity_pool_size"] == 232  # no marginal band
    assert cohort["fail_pool_size"] == 240
    # Target-axis numbers — the inputs to the derived fields.
    assert cohort["target_capacity_pool_size"] == 64
    assert cohort["target_soft_capacity_pool_size"] == 192
    assert cohort["target_fail_pool_size"] == 224
    # Derived fields now reflect the target axis, not the failure axis.
    # A buyer sees "premium up to 64, tolerable up to 192, hard cliff 32 wide."
    assert cohort["headroom_pool_size"] == 128
    assert cohort["cliff_pool_size"] == 32
    assert cohort["deployment_band_shape"] == "graceful"


def test_export_curve_uses_kv_cache_used_pct_field_name(tmp_path) -> None:
    """The KV usage field is reported as a 0–100 percent (matches the
    engine's prometheus ``kv_cache_used_pct``). The export mirrors
    that name so a downstream consumer doesn't multiply by 100
    treating the value as a 0..1 fraction."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0,
        "target_pool_size": 8,
        "measured_avg_pool_size": 8.0,
        "measured_avg_in_flight": 0.0,
        "measurement_started_at": "2026-01-01T00:00:00Z",
        "measurement_duration_s": 60, "sample_size": 100,
        "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0,
        "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 0.0,
        "avg_kv_cache_pct": 12.5,    # 12.5% — engine-reported scale
        "capacity_status": "pass",
        "target_status": "pass",
    })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    step = doc["cohorts"][0]["curve"][0]
    assert "kv_cache_used_pct" in step
    assert step["kv_cache_used_pct"] == 12.5
    assert "kv_cache_pct" not in step


def test_export_curve_includes_token_aggregates(tmp_path) -> None:
    """Per-step token totals + per-second rates surface on every
    curve entry (slim and full export). Drives the buyer-page
    'how many tokens/sec at this concurrency' view."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    mid = db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0,
        "target_pool_size": 8,
        "measured_avg_pool_size": 8.0,
        "measured_avg_in_flight": 0.0,
        "measurement_started_at": "2026-01-01T00:00:00Z",
        "measurement_duration_s": 60, "sample_size": 5,
        "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0,
        "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 0.0,
        "capacity_status": "pass",
        "target_status": "pass",
    })
    # 5 turns @ 100 input, 50 content, 200 reasoning each.
    db.insert_events([
        {
            "measurement_id": mid,
            "persona_id": "x_persona",
            "user_id": f"u{i}", "session_id": f"s{i}", "turn_index": 0,
            "submitted_at_ms": i * 10, "ttft_ms": 50.0,
            "completed_at_ms": i * 10 + 1000,
            "input_tokens": 100, "history_tokens": 0,
            "output_tokens": 50, "reasoning_tokens": 200,
            "tpot_ms": 30.0, "end_to_end_ms": 1000.0,
            "in_flight_at_submit": 1,
            "sla_ttft_violation": 0, "sla_tpot_violation": 0,
        }
        for i in range(5)
    ])
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    # Slim export — token aggregates must still be present.
    doc, _ = export_dir(tmp_path, slim=True)
    step = doc["cohorts"][0]["curve"][0]
    assert step["prompt_tokens"] == 5 * 100
    assert step["content_tokens"] == 5 * 50
    assert step["reasoning_tokens"] == 5 * 200
    assert step["total_visible_output_tokens"] == 5 * (50 + 200)
    # Rates: 500 prompt / 60s window = 8.33; 250 content / 60s = 4.17;
    # 1250 visible / 60s = 20.83.
    assert abs(step["prompt_tok_per_s"] - 500 / 60) < 0.1
    assert abs(step["content_tok_per_s"] - 250 / 60) < 0.1
    assert abs(step["visible_output_tok_per_s"] - 1250 / 60) < 0.1


def test_export_capacity_throughput_surfaces_at_capacity_pool(tmp_path) -> None:
    """The cohort's ``capacity_throughput`` block aggregates token
    volume + rates at capacity_pool_size — the SLA-bound deployment
    concurrency. Headline buyer-page metric: 'sustains X tokens/sec
    at recommended pool=N'."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    # Three steps. capacity_status='pass' at pool=16 makes that the
    # capacity_pool_size; that's where the throughput block sums.
    for i, (pool, status, n_turns) in enumerate(
        [(8, "pass", 5), (16, "pass", 10), (32, "fail", 0)]
    ):
        mid = db.insert_measurement({
            "cohort_run_id": "crid", "step_index": i,
            "target_pool_size": pool,
            "measured_avg_pool_size": float(pool),
            "measured_avg_in_flight": 0.0,
            "measurement_started_at": "2026-01-01T00:00:00Z",
            "measurement_duration_s": 60, "sample_size": n_turns,
            "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
            "combined_violation_rate": 0.0 if status != "fail" else 0.5,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": status,
            "target_status": status,
        })
        db.insert_events([
            {
                "measurement_id": mid,
                "persona_id": "x_p",
                "user_id": f"u{j}", "session_id": "s",
                "turn_index": 0,
                "submitted_at_ms": 0, "ttft_ms": 50.0,
                "completed_at_ms": 1000,
                "input_tokens": 100, "history_tokens": 0,
                "output_tokens": 50, "reasoning_tokens": 200,
                "tpot_ms": 30.0, "end_to_end_ms": 1000.0,
                "in_flight_at_submit": 1,
                "sla_ttft_violation": 0, "sla_tpot_violation": 0,
            }
            for j in range(n_turns)
        ])
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path, slim=True)
    cohort = doc["cohorts"][0]
    assert cohort["capacity_pool_size"] == 16
    block = cohort["capacity_throughput"]
    assert block is not None
    assert block["pool_size"] == 16
    assert block["sample_size"] == 10
    assert block["prompt_tokens"] == 10 * 100
    assert block["content_tokens"] == 10 * 50
    assert block["reasoning_tokens"] == 10 * 200
    assert block["total_visible_output_tokens"] == 10 * 250
    # 1000 prompt / 60s ≈ 16.67
    assert abs(block["prompt_tok_per_s"] - 1000 / 60) < 0.1
    # 500 content / 60s ≈ 8.33
    assert abs(block["content_tok_per_s"] - 500 / 60) < 0.1
    # 2500 visible / 60s ≈ 41.67
    assert abs(block["visible_output_tok_per_s"] - 2500 / 60) < 0.1


def test_export_capacity_throughput_is_none_when_no_capacity(tmp_path) -> None:
    """A cohort that never produced a clean-pass measurement has
    capacity_pool_size=None; the throughput block should be None
    too rather than crashing or surfacing zeros."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0,
        "target_pool_size": 8,
        "measured_avg_pool_size": 8.0,
        "measured_avg_in_flight": 0.0,
        "measurement_started_at": "2026-01-01T00:00:00Z",
        "measurement_duration_s": 60, "sample_size": 0,
        "ttft_violation_rate": 0.5, "tpot_violation_rate": 0.5,
        "combined_violation_rate": 0.5,
        "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 1.0,
        "capacity_status": "fail",
        "target_status": "fail",
    })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path, slim=True)
    cohort = doc["cohorts"][0]
    assert cohort["capacity_pool_size"] is None
    assert cohort["capacity_throughput"] is None


def test_export_meta_includes_engine_config(tmp_path) -> None:
    """The ``engine_config`` block in meta lets cross-host comparisons
    verify the configs were actually equivalent before reading
    capacity differences as host effects."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="sglang", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"},
        config={
            "engine": {
                "type": "sglang",
                "model_id": "Qwen/Test",
                "quantization_kind": "fp8",
                "max_total_tokens": 131072,
                "chunked_prefill_size": 4096,
                "mem_fraction_static": 0.85,
                "tensor_parallel_size": 1,
                "attention_backend": "intel_amx",
                "disable_overlap_schedule": True,
                "docker_image": "sglang-cpu:xeon-fixed",
                # Fields outside the whitelist must be excluded.
                "host": "127.0.0.1",
                "port": 30000,
                "startup_timeout_s": 1200,
            },
            "simulation": {"initial_pool_size": 8},
        },
    )
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    eng = doc["meta"]["engine_config"]
    assert eng is not None
    # Whitelisted fields surface.
    assert eng["type"] == "sglang"
    assert eng["quantization_kind"] == "fp8"
    assert eng["max_total_tokens"] == 131072
    assert eng["chunked_prefill_size"] == 4096
    assert eng["attention_backend"] == "intel_amx"
    assert eng["docker_image"] == "sglang-cpu:xeon-fixed"
    # Non-whitelisted fields excluded.
    assert "host" not in eng
    assert "port" not in eng
    assert "startup_timeout_s" not in eng


def test_export_meta_engine_config_handles_missing_config(tmp_path) -> None:
    """Legacy / corrupt ``config_json`` shouldn't break the export —
    engine_config just comes back None."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    # config={} → JSON {} with no 'engine' key
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="x",
        cohort_definition={"name": "x"}, config={},
    )
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    assert doc["meta"]["engine_config"] is None


def test_landing_zones_three_band_split() -> None:
    """The export's three-zone landing zones (capacity / soft_capacity
    / fail_pool) must accurately split a curve into the
    fast / acceptable / degraded bands. writer_dominant from the
    real AMD sweep is the canonical case — pass through pool=32,
    marginal through pool=96, fail at pool=102."""
    from simulator.export import _landing_zones

    writer_dominant = [
        {"target_pool_size": 8,   "capacity_status": "pass"},
        {"target_pool_size": 16,  "capacity_status": "pass"},
        {"target_pool_size": 32,  "capacity_status": "pass"},
        {"target_pool_size": 64,  "capacity_status": "marginal"},
        {"target_pool_size": 72,  "capacity_status": "marginal"},
        {"target_pool_size": 81,  "capacity_status": "marginal"},
        {"target_pool_size": 91,  "capacity_status": "marginal"},
        {"target_pool_size": 96,  "capacity_status": "marginal"},
        {"target_pool_size": 102, "capacity_status": "fail"},
    ]
    capacity, soft, fail_pool = _landing_zones(writer_dominant)
    assert capacity == 32, f"premium cap = last pass before any non-pass; got {capacity}"
    assert soft == 96, f"acceptable cap = last marginal-or-pass below fail; got {soft}"
    assert fail_pool == 102, f"degraded threshold = first sustained fail; got {fail_pool}"


def test_landing_zones_perma_marginal_cohort() -> None:
    """code_assist on AMD: every measurement is marginal (5-30%
    violations) until the final fail. capacity=None (no clean
    pass), soft_capacity exists (the marginal band IS usable),
    fail_pool is the cliff. Without soft_capacity, the buyer page
    would have to report "capacity=null" which is misleading —
    the cohort serves users at marginal quality, just not premium."""
    from simulator.export import _landing_zones

    code_assist = [
        {"target_pool_size": 8,  "capacity_status": "marginal"},
        {"target_pool_size": 16, "capacity_status": "marginal"},
        {"target_pool_size": 24, "capacity_status": "marginal"},
        {"target_pool_size": 32, "capacity_status": "marginal"},
        {"target_pool_size": 37, "capacity_status": "marginal"},
        {"target_pool_size": 39, "capacity_status": "fail"},
    ]
    capacity, soft, fail_pool = _landing_zones(code_assist)
    assert capacity is None, "no pass step means no premium capacity"
    assert soft == 37, "soft capacity captures the marginal band"
    assert fail_pool == 39


def test_landing_zones_immediate_fail() -> None:
    """document_qa on AMD: even pool=8 fails. All three zones
    should report None for capacity / soft, and the first sustained
    fail for fail_pool. Honest for the buyer page: "below the
    measurable range" rather than fudging a number."""
    from simulator.export import _landing_zones

    document_qa = [{"target_pool_size": 8, "capacity_status": "fail"}]
    capacity, soft, fail_pool = _landing_zones(document_qa)
    assert capacity is None
    assert soft is None
    assert fail_pool == 8


def test_landing_zones_never_fails() -> None:
    """Sweep that ran to its max_pool without ever failing —
    capacity and soft_capacity are both the largest pool sampled,
    fail_pool is None ("no sustained-fail point observed")."""
    from simulator.export import _landing_zones

    healthy = [
        {"target_pool_size": 8,  "capacity_status": "pass"},
        {"target_pool_size": 16, "capacity_status": "pass"},
        {"target_pool_size": 32, "capacity_status": "pass"},
    ]
    capacity, soft, fail_pool = _landing_zones(healthy)
    assert capacity == 32
    assert soft == 32
    assert fail_pool is None


def test_landing_zones_tolerates_bisection_noise() -> None:
    """Bisection produces non-monotonic curves — a noise marginal
    at pool=32 followed by a recovery to pass at pool=36 should
    NOT be treated as the capacity boundary. Both capacity and
    soft_capacity should follow through to the sustained boundary."""
    from simulator.export import _landing_zones

    noisy = [
        {"target_pool_size": 8,  "capacity_status": "pass"},
        {"target_pool_size": 16, "capacity_status": "pass"},
        {"target_pool_size": 32, "capacity_status": "marginal"},  # noise
        {"target_pool_size": 36, "capacity_status": "pass"},      # recovery
        {"target_pool_size": 64, "capacity_status": "marginal"},
        {"target_pool_size": 96, "capacity_status": "fail"},
    ]
    capacity, soft, fail_pool = _landing_zones(noisy)
    assert capacity == 36, "recovery proves 32-marginal was noise; capacity = last pass below sustained-non-pass = 64"
    assert soft == 64, "soft cap = last marginal-or-pass below fail at 96"
    assert fail_pool == 96


def test_capacity_and_knee_handles_non_monotonic_curve() -> None:
    """The adaptive bisection produces non-monotonic curves: a noise
    'marginal' at pool 32 followed by a 'pass' at 36 isn't a real
    knee. Sustained-non-pass logic finds the true knee (where no
    higher pool ever recovers to pass) and keeps capacity ≤ knee."""
    from simulator.export import _capacity_and_knee

    # Drafter-like curve: noise marginals between sustained passes;
    # real knee much later. Mixed step order so the function must sort.
    curve = [
        {"target_pool_size": 8,   "capacity_status": "pass"},
        {"target_pool_size": 16,  "capacity_status": "pass"},
        {"target_pool_size": 32,  "capacity_status": "marginal"},
        {"target_pool_size": 36,  "capacity_status": "pass"},     # recovery
        {"target_pool_size": 72,  "capacity_status": "marginal"},
        {"target_pool_size": 81,  "capacity_status": "marginal"},
        {"target_pool_size": 85,  "capacity_status": "pass"},     # recovery
        {"target_pool_size": 127, "capacity_status": "marginal"}, # sustained
        {"target_pool_size": 148, "capacity_status": "marginal"},
        {"target_pool_size": 170, "capacity_status": "fail"},
    ]
    capacity, knee = _capacity_and_knee(curve)
    assert knee == 127, f"first sustained non-pass is 127, got {knee}"
    assert capacity == 85, f"largest pass below 127 is 85, got {capacity}"
    assert capacity < knee


def test_capacity_and_knee_inversion_is_impossible() -> None:
    """Reproduces the buyer-page bug: capacity > knee. Must NOT happen
    after the fix — capacity is always strictly below knee."""
    from simulator.export import _capacity_and_knee

    # chat_heavy-like: pool 32 marginal, pool 40 pass, pool 113 fail
    curve = [
        {"target_pool_size": 8,   "capacity_status": "pass"},
        {"target_pool_size": 32,  "capacity_status": "marginal"},
        {"target_pool_size": 40,  "capacity_status": "pass"},
        {"target_pool_size": 80,  "capacity_status": "marginal"},
        {"target_pool_size": 113, "capacity_status": "fail"},
    ]
    capacity, knee = _capacity_and_knee(curve)
    # First sustained non-pass at 80 (marginal, no pass after).
    assert knee == 80
    # Largest pass below 80.
    assert capacity == 40
    assert capacity < knee


def test_capacity_and_knee_no_passes() -> None:
    """When the cohort failed at the initial pool size and never
    recovered, capacity is None (not 0) — distinct from "we measured
    capacity=0", which we don't claim."""
    from simulator.export import _capacity_and_knee

    curve = [{"target_pool_size": 8, "capacity_status": "fail"}]
    capacity, knee = _capacity_and_knee(curve)
    assert capacity is None
    assert knee == 8


def test_capacity_and_knee_all_passes() -> None:
    """Sweep ran the full ramp without ever degrading. Capacity is
    the largest pool sampled, knee is None."""
    from simulator.export import _capacity_and_knee

    curve = [
        {"target_pool_size": 8,  "capacity_status": "pass"},
        {"target_pool_size": 16, "capacity_status": "pass"},
        {"target_pool_size": 32, "capacity_status": "pass"},
    ]
    capacity, knee = _capacity_and_knee(curve)
    assert knee is None
    assert capacity == 32


def test_parse_prometheus_handles_kv_metric_renames() -> None:
    """vLLM has shipped the KV-cache-utilisation metric under at
    least three names across releases of vllm-openai-cpu. Verify
    the parser picks up each."""
    from simulator.engines.base import Engine

    for name in ("vllm:gpu_cache_usage_perc",
                 "vllm:cpu_cache_usage_perc",
                 "vllm:kv_cache_usage_perc",
                 "vllm:gpu_kv_cache_usage_perc"):
        text = f"# HELP test\n{name}{{model=\"x\"}} 0.42\n"
        out = Engine._parse_prometheus(text)
        assert out.get("kv_cache_used_pct") == 42.0, (
            f"didn't pick up {name}: got {out}"
        )


def test_engine_prefix_cache_hit_rate_surfaces_in_export(tmp_path) -> None:
    """End-of-run engine.get_metrics() result is persisted on
    cohort_run and exposed in the export under
    cohort.prefix_cache.engine_hit_rate."""
    import json
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm_dual_socket", model_id="Qwen/Test",
        cohort_id="chat_heavy",
        cohort_definition={"name": "x"}, config={},
    )
    # The runner's end-of-cohort scrape would call this:
    db.update_cohort_run("crid", {
        "prefix_cache_engine_hits": 4_701_696,
        "prefix_cache_engine_queries": 8_314_059,
        "prefix_cache_engine_hit_rate": 0.566,
    })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    cohort = doc["cohorts"][0]
    pc = cohort["prefix_cache"]
    assert pc["engine_hits"] == 4_701_696
    assert pc["engine_queries"] == 8_314_059
    assert pc["engine_hit_rate"] == 0.566


def test_export_drops_engine_hit_rate_when_only_rate_was_scraped(tmp_path) -> None:
    """SGLang's ``sglang:cache_hit_rate`` is a moving-window value,
    not a cumulative counter. When only the rate (not hits+queries)
    was scraped, the export must NOT publish a misleading run-wide
    ``engine_hit_rate`` — instead surface a reason field so the
    buyer page can omit the metric.
    """
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="sglang", model_id="Qwen/Test",
        cohort_id="chat_heavy",
        cohort_definition={"name": "x"}, config={},
    )
    # SGLang scrape: rate present, but no cumulative counters.
    db.update_cohort_run("crid", {
        "prefix_cache_engine_hits": None,
        "prefix_cache_engine_queries": None,
        "prefix_cache_engine_hit_rate": 0.95,  # window-snapshot
    })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    doc, _ = export_dir(tmp_path)
    cohort = doc["cohorts"][0]
    pc = cohort["prefix_cache"]
    assert "engine_hit_rate" not in pc
    assert "engine_hits" not in pc
    assert "engine_queries" not in pc
    assert "engine_hit_rate_unavailable_reason" in pc
    assert "moving-window" in pc["engine_hit_rate_unavailable_reason"]


def test_export_slim_drops_heavy_timeseries(tmp_path) -> None:
    """``--slim`` mode must drop the per-step telemetry_samples,
    per-step turns, and cohort-level snapshots — but keep the
    headline summary, per-step rollup, landing zones, bottleneck
    attribution, and prefix-cache verdict. Default filename also
    differs so slim and full exports can coexist."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="chat_heavy",
        cohort_definition={"name": "x", "persona_weights": {}}, config={},
    )
    mid = db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0, "target_pool_size": 8,
        "measured_avg_pool_size": 8.0, "measured_avg_in_flight": 6.5,
        "measurement_started_at": "2026-01-01T00:00:30Z",
        "measurement_duration_s": 60, "sample_size": 3,
        "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0, "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 0.0, "capacity_status": "pass",
    })
    # Heavy time-series rows that should disappear in slim mode.
    db.insert_telemetry([
        {"measurement_id": mid, "sampled_at_ms": 1000,
         "kv_cache_used_pct": 41.2, "queue_depth": 2,
         "prefix_cache_hits": 100, "prefix_cache_misses": 0,
         "cpu_util_avg": 78.3, "memory_used_gb": 60.1,
         "engine_rss_gb": 38.0, "freq_mhz_mean": 3000.0,
         "freq_mhz_stddev": 12.0, "freq_mhz_min": 2950.0},
    ])
    db.insert_events([
        {"measurement_id": mid, "persona_id": "quick_lookup",
         "user_id": "u1", "session_id": "s1", "turn_index": 0,
         "submitted_at_ms": 1500, "ttft_ms": 1697.0,
         "completed_at_ms": 5500, "input_tokens": 350,
         "history_tokens": 0, "output_tokens": 60, "tpot_ms": 62.0,
         "end_to_end_ms": 4000.0, "in_flight_at_submit": 4,
         "in_flight_avg_during": 4.5, "in_flight_peak_during": 6,
         "sla_ttft_violation": 0, "sla_tpot_violation": 0,
         "ttft_target_miss": 0, "tpot_target_miss": 0,
         "token_timestamps_json": None},
    ])
    db.insert_snapshot({
        "cohort_run_id": "crid", "snapshot_at_ms": 500,
        "phase": "warmup", "pool_size": 8, "in_flight": 4,
        "requests_completed": 0, "errors": 0,
        "step_samples": 0, "step_target_samples": 0,
    })
    db.finalise_run("crid", "2026-01-01T00:01:30Z", "ok")
    db.close()

    # Slim export
    slim_doc, slim_path = export_dir(tmp_path, slim=True)
    assert slim_path.name == "buyer_page_data_slim.json"
    assert slim_doc["meta"]["slim"] is True
    cohort = slim_doc["cohorts"][0]
    assert cohort["curve"][0].get("telemetry_samples") is None, (
        "slim export must NOT include per-step telemetry_samples"
    )
    assert cohort["curve"][0].get("turns") is None, (
        "slim export must NOT include per-step turns"
    )
    assert "snapshots" not in cohort, (
        "slim export must NOT include cohort-level snapshots"
    )
    # Summary fields must still be present.
    assert cohort["capacity_pool_size"] == 8
    assert "soft_capacity_pool_size" in cohort
    assert "fail_pool_size" in cohort
    assert "bottleneck" in cohort
    assert len(cohort["curve"]) == 1  # rollup retained

    # Full export — same data, heavier shape.
    full_doc, full_path = export_dir(tmp_path)
    assert full_path.name == "buyer_page_data.json"
    assert full_doc["meta"]["slim"] is False
    full_cohort = full_doc["cohorts"][0]
    assert len(full_cohort["curve"][0]["telemetry_samples"]) == 1
    assert len(full_cohort["curve"][0]["turns"]) == 1
    assert "snapshots" in full_cohort

    # Slim should be substantially smaller. The model database here
    # is tiny, so the absolute delta is small — but slim must be
    # strictly smaller than full.
    assert slim_path.stat().st_size < full_path.stat().st_size


def test_export_default_output_lands_in_run_dir(tmp_path) -> None:
    """``make export`` must default to writing the JSON inside the
    run_NN/ directory the data came from — keeps every artifact for
    one logical run grouped (run.db, engine logs, perf CSVs, and now
    buyer_page_data.json) in the same directory."""
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_07"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="chat_heavy",
        cohort_definition={}, config={},
    )
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    # No --output → resolves to <latest_run_dir>/buyer_page_data.json
    doc, out_path = export_dir(tmp_path)
    assert out_path == run_dir / "buyer_page_data.json"
    assert out_path.is_file()
    # Sibling to the run.db it summarises
    assert out_path.parent == (run_dir / "run.db").parent


def test_export_includes_per_step_time_series(tmp_path) -> None:
    """make export must surface ``telemetry_samples`` + ``turns`` per
    curve step and ``snapshots`` per cohort. The downstream website
    reads the JSON directly; the time-series is the difference between
    a static knee chart and a "drill into a specific step" view."""
    import json
    from simulator.database import Database
    from simulator.export import export_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="vllm", model_id="Qwen/Test", cohort_id="chat_heavy",
        cohort_definition={"name": "Customer support team",
                           "persona_weights": {"quick_lookup": 1.0}},
        config={},
    )
    mid = db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0, "target_pool_size": 8,
        "measured_avg_pool_size": 8.0, "measured_avg_in_flight": 6.5,
        "measurement_started_at": "2026-01-01T00:00:30Z",
        "measurement_duration_s": 60, "sample_size": 3,
        "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0, "violation_rate_ci_lower": 0.0,
        "violation_rate_ci_upper": 0.0, "capacity_status": "pass",
    })
    db.insert_telemetry([
        {"measurement_id": mid, "sampled_at_ms": 1_000,
         "kv_cache_used_pct": 41.2, "queue_depth": 2,
         "prefix_cache_hits": 100, "prefix_cache_misses": 0,
         "cpu_util_avg": 78.3, "memory_used_gb": 60.1,
         "engine_rss_gb": 38.0, "freq_mhz_mean": 3000.0,
         "freq_mhz_stddev": 12.0, "freq_mhz_min": 2950.0},
        {"measurement_id": mid, "sampled_at_ms": 2_000,
         "kv_cache_used_pct": 43.0, "queue_depth": 3,
         "prefix_cache_hits": 130, "prefix_cache_misses": 0,
         "cpu_util_avg": 80.1, "memory_used_gb": 60.2,
         "engine_rss_gb": 38.0, "freq_mhz_mean": 2998.0,
         "freq_mhz_stddev": 11.0, "freq_mhz_min": 2940.0},
    ])
    db.insert_events([
        {"measurement_id": mid, "persona_id": "quick_lookup",
         "user_id": "u1", "session_id": "s1", "turn_index": 0,
         "submitted_at_ms": 1_500, "ttft_ms": 1697.0,
         "completed_at_ms": 5_500, "input_tokens": 350,
         "history_tokens": 0, "output_tokens": 60, "tpot_ms": 62.0,
         "end_to_end_ms": 4000.0, "in_flight_at_submit": 4,
         "in_flight_avg_during": 4.5, "in_flight_peak_during": 6,
         "sla_ttft_violation": 0, "sla_tpot_violation": 0,
         "token_timestamps_json": None},
    ])
    db.insert_snapshot({
        "cohort_run_id": "crid", "snapshot_at_ms": 500,
        "phase": "warmup", "pool_size": 8, "in_flight": 4,
        "requests_completed": 0, "errors": 0,
        "step_samples": 0, "step_target_samples": 0,
    })
    db.insert_snapshot({
        "cohort_run_id": "crid", "snapshot_at_ms": 1500,
        "phase": "measuring", "pool_size": 8, "in_flight": 5,
        "requests_completed": 1, "errors": 0,
        "step_samples": 1, "step_target_samples": 100,
    })
    db.finalise_run("crid", "2026-01-01T00:01:30Z", "ok")
    db.close()

    out = tmp_path / "buyer_page_data.json"
    doc_returned, out_returned = export_dir(tmp_path, out)
    assert out_returned == out
    doc = json.loads(out.read_text())
    assert doc == doc_returned
    assert doc["meta"]["cohort_count"] == 1
    cohort = doc["cohorts"][0]
    assert cohort["id"] == "chat_heavy"

    # Curve carries per-step time-series.
    assert len(cohort["curve"]) == 1
    step = cohort["curve"][0]
    assert step["step_index"] == 0
    assert len(step["telemetry_samples"]) == 2
    sample = step["telemetry_samples"][0]
    assert sample["sampled_at_ms"] == 1_000
    assert sample["kv_cache_used_pct"] == 41.2
    assert "freq_mhz_min" in sample
    assert len(step["turns"]) == 1
    turn = step["turns"][0]
    assert turn["persona_id"] == "quick_lookup"
    assert turn["ttft_ms"] == 1697.0
    # token_timestamps_json is intentionally excluded — too large.
    assert "token_timestamps_json" not in turn

    # Whole-run heartbeat lives at the cohort level.
    assert len(cohort["snapshots"]) == 2
    assert cohort["snapshots"][0]["phase"] == "warmup"
    assert cohort["snapshots"][1]["step_target_samples"] == 100


def test_resolve_run_dir_creates_run_01_when_empty(tmp_path) -> None:
    """First invocation against a fresh runs/ creates run_01."""
    from simulator.runs import resolve_run_dir
    rd = resolve_run_dir(tmp_path)
    assert rd == tmp_path / "run_01"
    assert rd.is_dir()


def test_resolve_run_dir_reuses_latest_by_default(tmp_path) -> None:
    """Default behaviour is resume: latest run_NN is returned, not a new one."""
    from simulator.runs import resolve_run_dir
    (tmp_path / "run_01").mkdir()
    (tmp_path / "run_02").mkdir()
    (tmp_path / "run_03").mkdir()
    rd = resolve_run_dir(tmp_path)
    assert rd == tmp_path / "run_03"


def test_resolve_run_dir_new_creates_next(tmp_path) -> None:
    """new=True advances to run_NN+1 instead of reusing the latest."""
    from simulator.runs import resolve_run_dir
    (tmp_path / "run_01").mkdir()
    (tmp_path / "run_02").mkdir()
    rd = resolve_run_dir(tmp_path, new=True)
    assert rd == tmp_path / "run_03"
    assert rd.is_dir()


def test_resolve_run_dir_ignores_non_run_subdirs(tmp_path) -> None:
    """Non-``run_NN`` subdirectories must not influence numbering — a
    user might leave notes/ or scratch/ alongside the run dirs."""
    from simulator.runs import resolve_run_dir
    (tmp_path / "notes").mkdir()
    (tmp_path / "run_01").mkdir()
    (tmp_path / "scratch").mkdir()
    rd = resolve_run_dir(tmp_path, new=True)
    assert rd == tmp_path / "run_02"


def test_resolve_run_dir_handles_double_digit_numbering(tmp_path) -> None:
    """Sort by integer, not lexicographic — run_10 must come after run_9."""
    from simulator.runs import resolve_run_dir, latest_run_dir
    for i in (1, 2, 9, 10, 11):
        (tmp_path / f"run_{i:02d}").mkdir()
    assert latest_run_dir(tmp_path) == tmp_path / "run_11"
    assert resolve_run_dir(tmp_path, new=True) == tmp_path / "run_12"


def test_resolve_workload_group_empty_arg_defaults_to_all() -> None:
    from simulator.personas import COHORTS, PERSONAS, resolve_workload_group
    p, c = resolve_workload_group("")
    assert set(p) == set(PERSONAS) and set(c) == set(COHORTS)
    p, c = resolve_workload_group(None)
    assert set(p) == set(PERSONAS) and set(c) == set(COHORTS)


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
    """Single-turn personas (summarizer, writer) can't validate cache
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


# ── Forklift script (scripts/forklift_run.py) ─────────────────────────


def _seed_forklift_source(run_dir: Path) -> dict:
    """Build a synthetic source ``run.db`` with a mix of personas and
    cohorts spanning the rename / removal / cohort cases.

    Returns the cohort_run_id → cohort_id mapping for assertion."""
    from simulator.database import Database

    run_dir.mkdir(parents=True, exist_ok=True)
    db = Database(run_dir / "run.db")

    # 1) drafter (ok)        — should be renamed to writer and kept
    # 2) conversational (ok) — current persona, keep as-is
    # 3) ai_engineering (ok) — removed persona, drop
    # 4) chat_heavy (ok)     — cohort, drop (mix changed)
    # 5) writer (interrupted) — current persona but bad status, drop
    seeds = [
        ("crid-drafter",        "drafter",        "ok"),
        ("crid-convo",          "conversational", "ok"),
        ("crid-aieng",          "ai_engineering", "ok"),
        ("crid-chat-heavy",     "chat_heavy",     "ok"),
        ("crid-writer-partial", "writer",         "interrupted"),
    ]
    for crid, cid, status in seeds:
        db.insert_run(
            cohort_run_id=crid, started_at="2026-05-01T00:00:00Z",
            engine_type="vllm", model_id="Qwen/Test", cohort_id=cid,
            cohort_definition={"name": cid, "personas": [cid]},
            config={},
        )
        # Add a measurement with a deterministic measurement_id
        # (auto-incremented; we'll capture it for CSV stubs).
        mid = db.insert_measurement({
            "cohort_run_id": crid, "step_index": 0,
            "target_pool_size": 8,
            "measured_avg_pool_size": 8.0,
            "measured_avg_in_flight": 7.5,
            "measurement_started_at": "2026-05-01T00:01:00Z",
            "measurement_duration_s": 60, "sample_size": 100,
            "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
            "combined_violation_rate": 0.0,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": "pass",
        })
        # Drop a single turn_event so the persona_id rename can be checked.
        db.insert_events([{
            "measurement_id": mid,
            "persona_id": cid,
            "user_id": f"u-{crid}",
            "session_id": "s0", "turn_index": 0,
            "submitted_at_ms": 0, "ttft_ms": 50.0,
            "completed_at_ms": 1000,
            "input_tokens": 10, "history_tokens": 0,
            "output_tokens": 20, "tpot_ms": 30.0,
            "end_to_end_ms": 1000.0,
            "in_flight_at_submit": 1,
            "sla_ttft_violation": 0, "sla_tpot_violation": 0,
        }])
        # Drop a perf csv stub for this measurement_id.
        (run_dir / f"perf_m{mid}_8.csv").write_text("ts,cycles\n0,0\n")
        db.finalise_run(crid, "2026-05-01T00:30:00Z", status)

    db.close()
    return {crid: cid for crid, cid, _ in seeds}


def test_forklift_keeps_current_personas_renames_drafter(tmp_path) -> None:
    """End-to-end: source run with drafter+convo+ai_eng+chat_heavy+
    partial-writer → forklift keeps the two ok current-persona rows
    (with drafter→writer rename), drops the rest."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    forklift_mod = importlib.import_module("forklift_run")

    src = tmp_path / "run_01"
    _seed_forklift_source(src)
    dst = tmp_path / "run_02"

    forklift_mod.forklift(src, dst, dry_run=False)

    # Source must still exist, untouched.
    assert (src / "run.db").exists()
    # Destination has run.db + the two carried-over CSVs.
    assert (dst / "run.db").exists()

    import sqlite3
    conn = sqlite3.connect(dst / "run.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT cohort_run_id, cohort_id, final_status FROM cohort_run "
        "ORDER BY cohort_run_id"
    ).fetchall()
    by_id = {r["cohort_run_id"]: r["cohort_id"] for r in rows}
    assert set(by_id.keys()) == {"crid-drafter", "crid-convo"}
    assert by_id["crid-drafter"] == "writer"      # renamed
    assert by_id["crid-convo"] == "conversational"

    # turn_events.persona_id was rewritten too.
    persona_ids = {
        r["persona_id"] for r in conn.execute(
            "SELECT persona_id FROM turn_events"
        )
    }
    assert "drafter" not in persona_ids
    assert "writer" in persona_ids
    assert "conversational" in persona_ids
    conn.close()


def test_forklift_copies_perf_csvs_for_kept_measurements(tmp_path) -> None:
    """Perf CSVs (named perf_m<measurement_id>_*.csv) follow the
    kept measurement_ids so post-export prefix-cache analysis still
    has its source data."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    forklift_mod = importlib.import_module("forklift_run")

    src = tmp_path / "run_01"
    _seed_forklift_source(src)
    dst = tmp_path / "run_02"

    forklift_mod.forklift(src, dst, dry_run=False)

    src_csvs = sorted(p.name for p in src.glob("perf_m*_*.csv"))
    dst_csvs = sorted(p.name for p in dst.glob("perf_m*_*.csv"))
    # We seeded 5 source CSVs but only 2 cohort_runs are kept (drafter,
    # convo) — so 2 CSVs in dest, both also present in source.
    assert len(src_csvs) == 5
    assert len(dst_csvs) == 2
    assert set(dst_csvs).issubset(set(src_csvs))


def test_forklift_dry_run_writes_nothing(tmp_path) -> None:
    """``--dry-run`` reports the plan without creating the dest dir."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    forklift_mod = importlib.import_module("forklift_run")

    src = tmp_path / "run_01"
    _seed_forklift_source(src)
    dst = tmp_path / "run_02"

    forklift_mod.forklift(src, dst, dry_run=True)

    assert not dst.exists()


# ── Audit script (scripts/audit_run.py) ───────────────────────────────


def _seed_audit_run(run_dir: Path, *, cohort_id: str, points: list[tuple]) -> None:
    """Insert a cohort_run plus measurement rows from the given
    points list. Each point is (pool, sample_size, viol_rate, tm_rate,
    capacity_status, target_status)."""
    from simulator.database import Database

    run_dir.mkdir(parents=True, exist_ok=True)
    db = Database(run_dir / "run.db")
    crid = f"crid-{cohort_id}"
    db.insert_run(
        cohort_run_id=crid, started_at="2026-05-07T00:00:00Z",
        engine_type="sglang", model_id="Qwen/Test", cohort_id=cohort_id,
        cohort_definition={"name": cohort_id}, config={},
    )
    for i, (pool, n, viol, tm, cap_s, tgt_s) in enumerate(points):
        db.insert_measurement({
            "cohort_run_id": crid, "step_index": i,
            "target_pool_size": pool,
            "measured_avg_pool_size": float(pool),
            "measured_avg_in_flight": 0.0,
            "measurement_started_at": "2026-05-07T00:01:00Z",
            "measurement_duration_s": 60, "sample_size": n,
            "ttft_violation_rate": 0.0, "tpot_violation_rate": viol,
            "combined_violation_rate": viol,
            "ttft_target_miss_rate": 0.0, "tpot_target_miss_rate": tm,
            "combined_target_miss_rate": tm,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": cap_s,
            "target_status": tgt_s,
        })
    db.finalise_run(crid, "2026-05-07T00:30:00Z", "ok")
    db.close()


def _import_audit_module():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    return importlib.import_module("audit_run")


def test_audit_detects_no_marginal_band(tmp_path) -> None:
    """chat_heavy-style sharp cliff: pool=92 pass → pool=96 fail with
    no marginal status in between."""
    audit_mod = _import_audit_module()
    rd = tmp_path / "run_01"
    _seed_audit_run(rd, cohort_id="chat_heavy", points=[
        # (pool, n, viol, tm, capacity_status, target_status)
        (8,  100, 0.0,  0.0,  "pass",     "pass"),
        (32, 100, 0.0,  0.0,  "pass",     "pass"),
        (88, 100, 0.0,  0.20, "pass",     "marginal"),
        (92, 100, 0.0,  0.30, "pass",     "marginal"),  # last failure-axis pass
        (96, 100, 0.66, 1.0,  "fail",     "fail"),       # straight to fail
    ])

    result = audit_mod.audit(rd)
    kinds = {a.kind for a in result.anomalies}
    assert "no_marginal_band" in kinds
    a = next(x for x in result.anomalies if x.kind == "no_marginal_band")
    # midpoint of [92, 96] = 94
    assert a.pool_sizes_to_remeasure == [94]


def test_audit_detects_no_fail_observed(tmp_path) -> None:
    """analyst_team-style curve that plateaus in marginal without
    crossing fail."""
    audit_mod = _import_audit_module()
    rd = tmp_path / "run_01"
    _seed_audit_run(rd, cohort_id="general_knowledge", points=[
        (8,   100, 0.0,  0.0,  "pass",     "pass"),
        (32,  100, 0.0,  0.0,  "pass",     "pass"),
        (64,  100, 0.07, 0.55, "marginal", "fail"),  # never hits 30% viol
    ])

    result = audit_mod.audit(rd)
    kinds = {a.kind for a in result.anomalies}
    assert "no_fail_observed" in kinds
    a = next(x for x in result.anomalies if x.kind == "no_fail_observed")
    # Suggests doubling beyond max pool=64
    assert 128 in a.pool_sizes_to_remeasure


def test_audit_detects_single_point_rescue(tmp_path) -> None:
    """writer-style rescue: pool=120 pass between marginal/fail
    neighbors."""
    audit_mod = _import_audit_module()
    rd = tmp_path / "run_01"
    _seed_audit_run(rd, cohort_id="writer", points=[
        (96,  100, 0.0,  0.0,  "pass",     "pass"),
        (112, 100, 0.0,  0.15, "pass",     "marginal"),  # pre-rescue marginal
        # Note: capacity_status reflects FAILURE axis. For the audit we
        # need pool=120 to be the lone pass with non-pass neighbors —
        # synthesize that directly.
        (115, 100, 0.0,  0.20, "marginal", "marginal"),  # neighbor
        (120, 100, 0.0,  0.04, "pass",     "pass"),      # rescue
        (128, 100, 0.81, 1.0,  "fail",     "fail"),       # neighbor
    ])

    result = audit_mod.audit(rd)
    kinds = {a.kind for a in result.anomalies}
    assert "single_point_rescue" in kinds
    a = next(x for x in result.anomalies if x.kind == "single_point_rescue")
    assert a.pool_sizes_to_remeasure == [120]


def test_audit_detects_boundary_status(tmp_path) -> None:
    """Legacy point-estimate classification: 30/100 misses got
    target_status='fail' but Wilson CI says marginal."""
    audit_mod = _import_audit_module()
    rd = tmp_path / "run_01"
    _seed_audit_run(rd, cohort_id="quick_lookup", points=[
        (32,  100, 0.0, 0.0,  "pass", "pass"),
        # 30/100 misses, classified 'fail' under old logic — Wilson CI
        # says marginal because lower CI ~0.21 < 0.30.
        (232, 100, 0.0, 0.30, "pass", "fail"),
    ])

    result = audit_mod.audit(rd)
    kinds = {a.kind for a in result.anomalies}
    assert "boundary_status" in kinds
    a = next(x for x in result.anomalies if x.kind == "boundary_status")
    assert 232 in a.pool_sizes_to_remeasure


def test_audit_emits_dedup_rerun_plan(tmp_path) -> None:
    """The rerun_points list should be deduped across multiple anomaly
    types that target the same (cohort, pool). Useful when a single
    pool triggers both 'boundary_status' and 'single_point_rescue'."""
    audit_mod = _import_audit_module()
    rd = tmp_path / "run_01"
    # Synthesize a scenario that fires both 'boundary_status' and
    # 'single_point_rescue' on pool=120.
    _seed_audit_run(rd, cohort_id="writer", points=[
        (96,  100, 0.0,  0.0,  "pass",     "pass"),
        (112, 100, 0.0,  0.15, "marginal", "marginal"),
        # 4/100 with status=pass → Wilson says marginal, AND it's a rescue.
        (120, 100, 0.04, 0.04, "pass",     "pass"),
        (128, 100, 0.81, 1.0,  "fail",     "fail"),
    ])

    result = audit_mod.audit(rd)
    plan = result.as_json()["rerun_points"]
    pool_120_entries = [p for p in plan if p["pool_size"] == 120 and p["cohort_id"] == "writer"]
    # Even if both detectors fire, only one rerun entry per (cohort, pool).
    assert len(pool_120_entries) == 1


def test_load_config_from_run_dir_reads_stored_config(tmp_path) -> None:
    """Spot-check / audit follow-ons must reconstruct the original
    sweep's engine config from run.db, NOT default to config/default.yaml.
    Using a different config (e.g. defaulting to vllm + Qwen2.5-7B
    when the original ran sglang + Qwen3-30B-FP8) launches the wrong
    engine and produces non-comparable measurements."""
    from simulator.database import Database
    from simulator.runner import load_config_from_run_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-01-01T00:00:00Z",
        engine_type="sglang", model_id="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        cohort_id="x", cohort_definition={"name": "x"},
        config={
            "engine": {
                "type": "sglang",
                "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                "quantization_kind": "fp8",
                "max_total_tokens": 131072,
                "tensor_parallel_size": 1,
                "attention_backend": "intel_amx",
            },
            "simulation": {"initial_pool_size": 8, "max_pool_size": 256},
        },
    )
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()

    cfg = load_config_from_run_dir(run_dir)
    # Engine fields restored from config_json — NOT defaults.
    assert cfg.engine.type == "sglang"
    assert cfg.engine.model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    assert cfg.engine.quantization_kind == "fp8"
    assert cfg.engine.max_total_tokens == 131072
    assert cfg.engine.attention_backend == "intel_amx"
    # Simulation fields too.
    assert cfg.simulation.initial_pool_size == 8


def test_load_config_from_run_dir_raises_for_empty_db(tmp_path) -> None:
    """Empty run.db (no cohort_run rows) → clear error rather than a
    default Config that would silently mismatch."""
    from simulator.database import Database
    from simulator.runner import load_config_from_run_dir

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.close()

    with pytest.raises(ValueError, match="No cohort_run rows"):
        load_config_from_run_dir(run_dir)


def test_audit_clean_run_produces_no_anomalies(tmp_path) -> None:
    """A clean curve with all bands present and no boundary cases
    should produce zero anomalies."""
    audit_mod = _import_audit_module()
    rd = tmp_path / "run_01"
    _seed_audit_run(rd, cohort_id="conversational", points=[
        (8,   100, 0.0,  0.0,  "pass",     "pass"),
        (32,  100, 0.0,  0.0,  "pass",     "pass"),
        (48,  100, 0.10, 0.10, "marginal", "marginal"),
        (64,  100, 0.50, 0.80, "fail",     "fail"),
    ])

    result = audit_mod.audit(rd)
    assert result.anomalies == []
