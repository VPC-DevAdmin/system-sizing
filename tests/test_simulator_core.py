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


def test_adaptive_default_threshold_tolerates_low_pool_noise() -> None:
    """At n=100, a true violation rate of 10-15% can spike up to ~25%
    just from sampling noise. The default knee_zone_threshold of 0.30
    leaves a buffer so this kind of noise doesn't latch the algorithm
    into a terminal stop.

    Concrete scenario: an early step at pool=16 measures 22%
    violations (true rate maybe ~12%, n=100 noise spike). With a
    0.20 threshold this entered bisection mode and stopped because
    the gap from pool=8 was already at the resolution floor; the
    algorithm produced a single fail point and quit. With 0.30,
    the algorithm continues — either bisecting once for confirmation
    (sub-cliff overshoot) or ramping further — but does NOT terminate.
    """
    hist = [
        StepResult(pool_size=8, violation_rate=0.0),
        StepResult(pool_size=16, violation_rate=0.22),  # noisy
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=512,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
        # Default knee_zone_threshold (0.30) — explicitly omit the
        # arg so this test breaks if someone accidentally lowers it.
    )
    # The algorithm must not stop here. It may pick:
    #   * 12 (sub-cliff overshoot bisect — confirmation step)
    #   * 17+ (slope-based fine step — continued ramp)
    # Both are acceptable; the ESSENTIAL property is that we keep
    # measuring and don't declare the run over on a single noisy step.
    assert nxt is not None, (
        "expected continued measurement after 22% noisy step, got None "
        "(algorithm prematurely terminated)"
    )


def test_adaptive_default_threshold_still_catches_real_cliff() -> None:
    """Confirm the noise-tolerant 0.30 default still triggers
    bisection on a clear cliff crossing (35%+)."""
    hist = [
        StepResult(pool_size=32, violation_rate=0.0),
        StepResult(pool_size=64, violation_rate=0.36),
    ]
    nxt = choose_next_pool_size(
        hist, initial_pool_size=8, max_pool_size=256,
        knee_slope_threshold=0.005, stop_violation_threshold=0.5,
    )
    # 36% > 30% threshold → bisection. mid = (32+64)/2 = 48.
    assert nxt == 48


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
        ("drafter", "no_samples"),
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
    """Minimal chunk shape that matches what the openai SDK yields."""
    def __init__(self, content: str | None):
        class _Delta:
            pass
        d = _Delta()
        d.content = content
        class _Choice:
            pass
        c = _Choice()
        c.delta = d
        self.choices = [c]


class _FakeStream:
    """Async iterator that emits chunks at scheduled times relative
    to its own start. ``schedule`` is [(elapsed_s, content_or_None), …]
    where content=None means "no token, just a scheduling tick"."""
    def __init__(self, schedule: list[tuple[float, str | None]]):
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
        elapsed_target, content = self._schedule[self._idx]
        self._idx += 1
        wait = elapsed_target - (_time.monotonic() - self._t0)
        if wait > 0:
            await asyncio.sleep(wait)
        return _FakeChunk(content)


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


async def _coroutine_returning(value):
    """Helper: a coroutine that immediately returns ``value``. Used
    to wrap a pre-built stream in the create_stream callable shape
    that consume_with_tiers expects."""
    return value


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
        ("drafter", "interrupted"),  # NOT completed — should be retried
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


def test_landing_zones_three_band_split() -> None:
    """The export's three-zone landing zones (capacity / soft_capacity
    / fail_pool) must accurately split a curve into the
    fast / acceptable / degraded bands. Drafter-dominant from the
    real AMD sweep is the canonical case — pass through pool=32,
    marginal through pool=96, fail at pool=102."""
    from simulator.export import _landing_zones

    drafter_dominant = [
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
    capacity, soft, fail_pool = _landing_zones(drafter_dominant)
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
