"""Two-knee adaptive stepper.

Drives the simulator's measurement loop through five phases, each
designed to extract a specific piece of the violation curve:

    Phase 1 (DOUBLING):
        Start at ``initial_pool_size`` (typically 8), double until
        violation_rate ≥ ``stop_violation_threshold`` (0.50) OR pool
        reaches ``max_pool_size``. Establishes a coarse bracket for
        the failure knee.

    Phase 1b (DOWNWARD_SEARCH):
        Triggered if the initial pool size already fails (violation
        ≥ 0.05). Halves down 8 → 4 → 2 looking for an acceptable
        zone. Coverage is recorded as ``downward_search`` for
        post-hoc analysis.

    Phase 2 (BISECT_FAIL):
        Bisect between the largest passing pool and the smallest
        failing pool until the gap is ≤ ``bisect_resolution`` (4).
        Locates ``fail_pool_size`` (knee 2: violation_rate ≥ 5%).

    Phase 3 (BISECT_TARGET):
        Bisect between the largest target-passing pool and the
        smallest target-missing pool. Locates
        ``soft_capacity_pool_size`` (knee 1: target_miss_rate ≥ 5%).

    Phase 4 (INFILL):
        Add midpoint measurements between fast_max → knee 1 and
        knee 1 → knee 2. Skipped if existing measurements are
        already within ±10% of the target. Yields ~2 extra points
        for curve density on the buyer page.

The 5%/5% knee thresholds are deliberate: stochastic single-sample
noise is bounded by the measurement window (n=100); 5% is well above
that floor and represents real degradation, not measurement jitter.

Use:

    stepper = TwoKneeStepper(initial_pool_size=8, max_pool_size=256)
    while (pool := stepper.next_pool_size()) is not None:
        result = run_measurement(pool)
        stepper.record(StepResult(
            pool_size=pool,
            violation_rate=result.combined_violation_rate,
            target_miss_rate=result.combined_target_miss_rate,
        ))
    coverage = stepper.coverage()  # for export
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StepResult:
    pool_size: int
    violation_rate: float
    # Target-miss rate (looser-bar quality signal). Default 0.0 for
    # back-compat with callers that haven't been updated to pass it
    # — they get the old behavior (knee 1 == knee 2 since target
    # never gets missed).
    target_miss_rate: float = 0.0


# Phase labels — exposed as class constants on TwoKneeStepper too.
PHASE_DOUBLING = "doubling"
PHASE_DOWNWARD_SEARCH = "downward_search"
PHASE_BISECT_FAIL = "bisect_fail"
PHASE_BISECT_TARGET = "bisect_target"
PHASE_INFILL = "infill"
PHASE_DONE = "done"


class TwoKneeStepper:
    """Phase-based stepper that locates both knees plus infill."""

    # Class-level access
    PHASE_DOUBLING = PHASE_DOUBLING
    PHASE_DOWNWARD_SEARCH = PHASE_DOWNWARD_SEARCH
    PHASE_BISECT_FAIL = PHASE_BISECT_FAIL
    PHASE_BISECT_TARGET = PHASE_BISECT_TARGET
    PHASE_INFILL = PHASE_INFILL
    PHASE_DONE = PHASE_DONE

    def __init__(
        self,
        *,
        initial_pool_size: int = 8,
        max_pool_size: int = 256,
        bisect_resolution: int = 4,
        knee_threshold: float = 0.05,
        # Doubling continues until violation crosses this threshold
        # (or max_pool_size is hit). Default 0.30 matches the
        # ``capacity_status='fail'`` threshold so every cohort is
        # guaranteed at least one fail measurement when the host is
        # capable of producing one — needed so ``fail_pool_size``
        # can be reported instead of None for cohorts whose curve
        # plateaus in the marginal band (5–30% violation).
        fail_threshold: float = 0.30,
        stop_violation_threshold: float = 0.50,
        infill_skip_pct: float = 0.10,
        downward_floor: int = 2,
    ):
        self.initial_pool_size = initial_pool_size
        self.max_pool_size = max_pool_size
        self.bisect_resolution = bisect_resolution
        self.knee_threshold = knee_threshold
        self.fail_threshold = fail_threshold
        self.stop_violation_threshold = stop_violation_threshold
        self.infill_skip_pct = infill_skip_pct
        self.downward_floor = downward_floor

        self.history: list[StepResult] = []
        self.phase: str = PHASE_DOUBLING
        # Populated when we transition to PHASE_INFILL.
        self._infill_queue: list[int] = []
        # Whether any downward-search measurement happened (used by
        # ``coverage()``).
        self._used_downward_search = False

    # ── Public API ───────────────────────────────────────────────────

    def record(self, result: StepResult) -> None:
        self.history.append(result)

    def next_pool_size(self) -> Optional[int]:
        """Return the next pool size to measure, or None when done.

        Phase transitions happen lazily inside this call — each
        helper either returns a pool size, or sets ``self.phase`` to
        the next phase and re-dispatches."""
        # Empty history → first measurement at initial_pool_size.
        if not self.history:
            return self.initial_pool_size

        for _ in range(6):  # bounded — at most 5 phase transitions
            if self.phase == PHASE_DOUBLING:
                nxt = self._next_doubling()
            elif self.phase == PHASE_DOWNWARD_SEARCH:
                nxt = self._next_downward()
            elif self.phase == PHASE_BISECT_FAIL:
                nxt = self._next_bisect_fail()
            elif self.phase == PHASE_BISECT_TARGET:
                nxt = self._next_bisect_target()
            elif self.phase == PHASE_INFILL:
                nxt = self._next_infill()
            else:  # PHASE_DONE
                return None
            if nxt is not None:
                return nxt
            # Phase is exhausted — _next_* updated self.phase, loop
            # to dispatch the new one.
        return None

    def coverage(self) -> str:
        """Return ``measurement_coverage`` label for the export.

        Reflects how informative the cohort's curve is:

          full_curve      — sweep located both knees AND has at least
                            one passing measurement.
          capped          — sweep ran to max_pool_size without
                            crossing knee 2 (no failure observed).
          downward_search — initial pool already failed; downward
                            phase ran and found at least one passing
                            sub-initial pool.
          exceeds_hardware — initial AND every sub-initial pool failed
                            (workload exceeds host capability at any
                            measurable concurrency).
          single_point    — only one measurement total (the
                            sweep terminated very early).
        """
        if not self.history:
            return "single_point"
        if len(self.history) == 1:
            return "single_point"

        any_pass = any(
            r.violation_rate < self.knee_threshold for r in self.history
        )
        any_fail = any(
            r.violation_rate >= self.knee_threshold for r in self.history
        )
        if self._used_downward_search:
            return "downward_search" if any_pass else "exceeds_hardware"
        if any_pass and any_fail:
            return "full_curve"
        if any_pass and not any_fail:
            return "capped"
        # No pass, no downward search — pathological; treat as
        # exceeds_hardware (initial pool failed but downward never
        # ran for some reason — shouldn't happen with this stepper).
        return "exceeds_hardware"

    # ── Phase implementations ────────────────────────────────────────

    def _measured(self) -> set[int]:
        return {r.pool_size for r in self.history}

    def _next_doubling(self) -> Optional[int]:
        """Phase 1 — double until violation crosses ``fail_threshold``
        (or stop_violation_threshold or max_pool_size). If the initial
        pool ALREADY fails (≥ knee_threshold), transition to downward
        search.

        Doubling now continues past the 5% knee threshold up to the
        ``fail_threshold`` (30%) — required so a sustained-fail point
        is observed even when the curve plateaus in the marginal
        band. The 5%-knee bracket is still located naturally during
        bisect_fail because the doubling sequence visited at least
        one sub-knee point.
        """
        last = self.history[-1]

        # Initial pool failed — switch to downward search.
        if (
            last.pool_size == self.initial_pool_size
            and last.violation_rate >= self.knee_threshold
        ):
            self.phase = PHASE_DOWNWARD_SEARCH
            return None

        # Crossed the stop threshold — bracket established, bisect.
        if last.violation_rate >= self.stop_violation_threshold:
            self.phase = PHASE_BISECT_FAIL
            return None

        # Crossed the fail threshold (30% violation) — bracket the
        # sustained-fail point, then proceed to bisect knee 2.
        if last.violation_rate >= self.fail_threshold:
            self.phase = PHASE_BISECT_FAIL
            return None

        # Hit the cap — go bisect knees with what we have. (Will
        # no-op gracefully if no fail-side bracket exists.)
        if last.pool_size >= self.max_pool_size:
            self.phase = PHASE_BISECT_FAIL
            return None

        # Continue doubling.
        nxt = min(self.max_pool_size, last.pool_size * 2)
        if nxt <= last.pool_size or nxt in self._measured():
            self.phase = PHASE_BISECT_FAIL
            return None
        return nxt

    def _next_downward(self) -> Optional[int]:
        """Phase 1b — halve from current minimum measured pool down
        to ``downward_floor``. Triggered when initial pool already
        failed."""
        self._used_downward_search = True
        smallest = min(r.pool_size for r in self.history)
        smallest_record = next(
            r for r in self.history if r.pool_size == smallest
        )

        # Found a passing sub-initial pool — proceed to bisect.
        if smallest_record.violation_rate < self.knee_threshold:
            self.phase = PHASE_BISECT_FAIL
            return None

        # Already at floor and still failing — give up.
        if smallest <= self.downward_floor:
            self.phase = PHASE_DONE
            return None

        nxt = max(self.downward_floor, smallest // 2)
        if nxt in self._measured() or nxt >= smallest:
            self.phase = PHASE_DONE
            return None
        return nxt

    def _next_bisect_fail(self) -> Optional[int]:
        """Phase 2 — bisect between largest violation-passing pool
        and smallest violation-failing pool until gap ≤ resolution.
        Pins knee 2 (5% violation rate)."""
        lows = sorted(
            r.pool_size for r in self.history
            if r.violation_rate < self.knee_threshold
        )
        highs = sorted(
            r.pool_size for r in self.history
            if r.violation_rate >= self.knee_threshold
        )
        if not lows or not highs:
            # Can't bisect without both sides — go to knee 1.
            self.phase = PHASE_BISECT_TARGET
            return None

        smallest_high = highs[0]
        candidates = [p for p in lows if p < smallest_high]
        if not candidates:
            self.phase = PHASE_BISECT_TARGET
            return None
        largest_low = max(candidates)
        gap = smallest_high - largest_low
        if gap <= self.bisect_resolution:
            self.phase = PHASE_BISECT_TARGET
            return None
        mid = (largest_low + smallest_high) // 2
        if mid in self._measured() or not (largest_low < mid < smallest_high):
            self.phase = PHASE_BISECT_TARGET
            return None
        return mid

    def _next_bisect_target(self) -> Optional[int]:
        """Phase 3 — bisect knee 1 (5% target_miss_rate) using the
        same logic as phase 2 but on a different rate field."""
        lows = sorted(
            r.pool_size for r in self.history
            if r.target_miss_rate < self.knee_threshold
        )
        highs = sorted(
            r.pool_size for r in self.history
            if r.target_miss_rate >= self.knee_threshold
        )
        if not lows or not highs:
            self.phase = PHASE_INFILL
            self._build_infill_queue()
            return None

        smallest_high = highs[0]
        candidates = [p for p in lows if p < smallest_high]
        if not candidates:
            self.phase = PHASE_INFILL
            self._build_infill_queue()
            return None
        largest_low = max(candidates)
        gap = smallest_high - largest_low
        if gap <= self.bisect_resolution:
            self.phase = PHASE_INFILL
            self._build_infill_queue()
            return None
        mid = (largest_low + smallest_high) // 2
        if mid in self._measured() or not (largest_low < mid < smallest_high):
            self.phase = PHASE_INFILL
            self._build_infill_queue()
            return None
        return mid

    def _build_infill_queue(self) -> None:
        """Compute midpoints for phase 4 (called once on entry).

        Three midpoint cases:
          * fast_max → knee 1   (between premium-quality and target-miss)
          * knee 1 → knee 2     (between target-miss and SLA-fail)
          * marginal-cliff      — when bisect_fail's bracket
            ``[last_pass, first_violation]`` contains no measurement
            with status='marginal' (5–30% violation), schedule one
            extra midpoint inside the bracket. Catches cohorts whose
            failure transition is sharp enough that the regular
            bisection skipped over the marginal band entirely.

        Skipped when an existing measurement is already within ±10%
        of the midpoint."""
        fast_pools = [
            r.pool_size for r in self.history
            if r.target_miss_rate < self.knee_threshold
            and r.violation_rate == 0.0
        ]
        fast_max = max(fast_pools) if fast_pools else None

        knee_1_pools = [
            r.pool_size for r in self.history
            if r.target_miss_rate >= self.knee_threshold
        ]
        knee_1 = min(knee_1_pools) if knee_1_pools else None

        knee_2_pools = [
            r.pool_size for r in self.history
            if r.violation_rate >= self.knee_threshold
        ]
        knee_2 = min(knee_2_pools) if knee_2_pools else None

        midpoints: list[int] = []
        if fast_max is not None and knee_1 is not None and fast_max < knee_1:
            midpoints.append((fast_max + knee_1) // 2)
        if knee_1 is not None and knee_2 is not None and knee_1 < knee_2:
            midpoints.append((knee_1 + knee_2) // 2)

        # Marginal-cliff case: bisect_fail's bracket may have narrowed
        # to gap=resolution without ever observing a marginal-status
        # point (≥5%/<30% violation). Schedule one midpoint inside
        # that bracket to try to catch one. ONE attempt only —
        # marginal might genuinely not exist (very sharp cliff), and
        # we don't want to bisect indefinitely. The bracket itself
        # has gap ≤ resolution, so the midpoint is at most
        # ``resolution//2`` away from a measured point — bypass the
        # ±10% skip below for this case via direct append.
        marginal_cliff_mp = self._marginal_cliff_midpoint()
        if marginal_cliff_mp is not None:
            self._infill_queue.append(marginal_cliff_mp)

        measured = self._measured()
        for mp in midpoints:
            # Skip if mp is already measured or within ±10% of an existing point.
            if mp in measured:
                continue
            too_close = any(
                abs(p - mp) / max(mp, 1) <= self.infill_skip_pct
                for p in measured
            )
            if too_close:
                continue
            self._infill_queue.append(mp)

    def _marginal_cliff_midpoint(self) -> Optional[int]:
        """If bisect_fail closed without any marginal-status point in
        the [last_pass, first_violation] bracket, return the bracket
        midpoint to schedule one extra measurement. Otherwise None.
        """
        # Failure-axis bracket — the [last_pass, first_violation]
        # slice we already located via bisect_fail.
        lows = [
            r for r in self.history if r.violation_rate < self.knee_threshold
        ]
        highs = [
            r for r in self.history if r.violation_rate >= self.knee_threshold
        ]
        if not lows or not highs:
            return None
        smallest_high = min(r.pool_size for r in highs)
        candidates = [r for r in lows if r.pool_size < smallest_high]
        if not candidates:
            return None
        largest_low = max(r.pool_size for r in candidates)

        # Already a marginal-status measurement somewhere? (Anything
        # with viol in [knee_threshold, fail_threshold).) If yes, the
        # cliff has been characterised and we can skip this infill.
        any_marginal = any(
            self.knee_threshold <= r.violation_rate < self.fail_threshold
            for r in self.history
        )
        if any_marginal:
            return None

        # Sharp-cliff case: bracket has no marginal point. Add one
        # midpoint measurement IF the bracket is wide enough to fit
        # one (gap ≥ 2). Skip if the midpoint is already measured.
        gap = smallest_high - largest_low
        if gap < 2:
            return None
        mp = (largest_low + smallest_high) // 2
        if mp in self._measured() or mp == largest_low or mp == smallest_high:
            return None
        return mp

    def _next_infill(self) -> Optional[int]:
        """Phase 4 — pop infill targets one at a time."""
        while self._infill_queue:
            nxt = self._infill_queue.pop(0)
            if nxt in self._measured():
                continue
            return nxt
        self.phase = PHASE_DONE
        return None
