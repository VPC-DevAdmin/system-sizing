"""Arrival-rate search for the open-loop methodology.

Searches over λ (session arrivals per second) for two knees:

  * **λ_max** — the stability boundary: highest rate at which the
    queue is stationary; above it the backlog grows without bound.
    This is the box's capacity, independent of any client pool.
  * **λ_sla** — the quality boundary: highest *stable* rate whose
    steady-state turns also pass the SLA gate. The gate is the
    window's ``capacity_status == "pass"``, i.e. the Wilson 95 %
    UPPER bound of the combined violation rate is below 5 % — so a
    ``marginal`` window (point estimate under 5 % but the upper
    bound not) counts as an SLA FAIL and bounds the bisection from
    above. λ_sla is therefore conservative by construction: it is
    the highest rate at which the SLA is demonstrably met, not the
    highest at which it was not demonstrably missed. λ_sla ≤ λ_max;
    the band between them is "surviving but degraded".

Search shape mirrors the coarse-to-fine philosophy used elsewhere in
this repo: geometric doubling to bracket the stability boundary,
geometric bisection (midpoint in log space — rates are ratio-scaled
quantities) until the bracket ratio is tight, then a second bisection
on the SLA axis inside the stable region. A ``client_limited`` step
acts as a ceiling exactly like a divergent one for search purposes,
but is reported distinctly — it means the *generator* gave out, not
the engine, and the resulting numbers are lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STABLE = "stable"
DIVERGENT = "divergent"
CLIENT_LIMITED = "client_limited"

PHASE_DOUBLING = "doubling"
PHASE_DOWNWARD = "downward"
PHASE_BISECT_STABILITY = "bisect_stability"
PHASE_BISECT_SLA = "bisect_sla"
PHASE_BACKFILL = "backfill"
PHASE_DONE = "done"


@dataclass
class RateStep:
    rate_per_s: float
    stability: str                 # stable | divergent | client_limited
    # The SLA gate verdict for a stable window: True iff the window's
    # capacity_status is "pass" (Wilson upper bound of the violation
    # rate < 5 %); "marginal" is False. None: divergent / limited /
    # no samples. Resolved by the orchestrator, not re-derived here.
    sla_pass: Optional[bool] = None
    violation_rate: float = 0.0
    target_miss_rate: float = 0.0
    sample_size: int = 0
    # Share of the window's turns that errored or timed out. The
    # ceiling signal when the SLA gate is off.
    error_rate: float = 0.0


# A stable window whose violation rate is at least this is a ceiling,
# not a stable point: requests are timing out (ttft_stalled) faster
# than they queue, so the queue looks stationary while almost nothing
# is served. The XE7740's 1-GPU quick_lookup sweep read "stable" at
# 3,840, 7,680 and 15,360 sessions/min with TTFT p95 at the 75 s
# timeout, doubled straight past the knee, and buried the engine.
HOPELESS_VIOLATION = 0.5


def _hopeless(step: "RateStep") -> bool:
    return step.stability == STABLE and step.sla_pass is False \
        and step.violation_rate >= HOPELESS_VIOLATION


def _round_rate(rate: float) -> float:
    """Round to 3 significant digits — keeps rates human-readable and
    prevents float dust from generating near-duplicate probe points."""
    if rate <= 0:
        return 0.0
    from math import floor, log10
    digits = 2 - int(floor(log10(rate)))
    return round(rate, digits)


class RateStepper:
    """Two-knee search over arrival rate. Same ``next``/``record``
    driving contract as the pool steppers so the orchestrator loop
    stays uniform."""

    def __init__(
        self,
        *,
        initial_rate_per_s: float = 1.0,
        max_rate_per_s: float = 256.0,
        min_rate_per_s: float = 0.02,
        # The search is done when the stable/divergent bracket is
        # within this ratio — 1.05 = the capacity is pinned to 5%.
        # Each bisection halves the bracket in log space, so tightening
        # from 30% to 5% costs ~3 extra windows.
        resolution_ratio: float = 1.05,
        # Planner campaign mode: no SLA bisection, ceilings from errors
        # and timeouts instead of SLA violations, configurable growth
        # per step, and a backfill pass of stable rates below the knee.
        sla_gate: bool = True,
        step_factor: float = 2.0,
        error_ceiling: float = 0.05,
        backfill_factor: Optional[float] = None,
        backfill_floor_ratio: float = 8.0,
    ):
        self.initial_rate = _round_rate(initial_rate_per_s)
        self.max_rate = max_rate_per_s
        self.min_rate = min_rate_per_s
        self.resolution_ratio = resolution_ratio
        self.sla_gate = sla_gate
        self.step_factor = max(1.05, float(step_factor))
        self.error_ceiling = error_ceiling
        self.backfill_factor = backfill_factor
        self.backfill_floor_ratio = backfill_floor_ratio
        self.history: list[RateStep] = []
        self.phase = PHASE_DOUBLING

    # ── Driving contract ─────────────────────────────────────────────

    def record(self, step: RateStep) -> None:
        self.history.append(step)

    def next_rate(self) -> Optional[float]:
        if not self.history:
            return self.initial_rate
        for _ in range(5):  # bounded phase transitions
            if self.phase == PHASE_DOUBLING:
                nxt = self._next_doubling()
            elif self.phase == PHASE_DOWNWARD:
                nxt = self._next_downward()
            elif self.phase == PHASE_BISECT_STABILITY:
                nxt = self._next_bisect_stability()
            elif self.phase == PHASE_BISECT_SLA:
                nxt = self._next_bisect_sla()
            elif self.phase == PHASE_BACKFILL:
                nxt = self._next_backfill()
            else:
                return None
            if nxt is not None:
                return _round_rate(nxt)
        return None

    @property
    def in_refinement(self) -> bool:
        """True once the coarse bracket exists — the orchestrator uses
        longer measurement windows near the boundary, where slow
        divergence needs more samples to distinguish from noise."""
        return self.phase in (PHASE_BISECT_STABILITY, PHASE_BISECT_SLA)

    # ── Phase logic ──────────────────────────────────────────────────

    def is_ceiling_step(self, step: RateStep) -> bool:
        """A "stable" window that still bounds the search from above:
        timeouts are draining the queue. Without the SLA gate the
        evidence is the error and timeout share, not SLA violations."""
        if self.sla_gate:
            return _hopeless(step)
        return step.stability == STABLE and step.error_rate >= self.error_ceiling

    def _after_stability(self) -> str:
        return PHASE_BISECT_SLA if self.sla_gate else PHASE_BACKFILL

    def _measured(self) -> set[float]:
        return {_round_rate(s.rate_per_s) for s in self.history}

    def _stable(self) -> list[RateStep]:
        return [s for s in self.history
                if s.stability == STABLE and not self.is_ceiling_step(s)]

    def _ceilings(self) -> list[RateStep]:
        """Steps that bound the search from above: divergent (the
        engine gave out), client_limited (the generator did), or a
        hopeless "stable" window (timeouts draining the queue)."""
        return [
            s for s in self.history
            if s.stability in (DIVERGENT, CLIENT_LIMITED) or self.is_ceiling_step(s)
        ]

    def _next_doubling(self) -> Optional[float]:
        last = self.history[-1]
        if last.stability != STABLE or self.is_ceiling_step(last):
            if len(self.history) == 1:
                self.phase = PHASE_DOWNWARD
            else:
                self.phase = PHASE_BISECT_STABILITY
            return None
        if last.rate_per_s >= self.max_rate:
            self.phase = self._after_stability()  # capped — no upper bracket
            return None
        nxt = min(self.max_rate, last.rate_per_s * self.step_factor)
        if _round_rate(nxt) in self._measured():
            self.phase = PHASE_BISECT_STABILITY
            return None
        return nxt

    def _next_downward(self) -> Optional[float]:
        smallest = min(s.rate_per_s for s in self.history)
        smallest_step = next(
            s for s in self.history
            if _round_rate(s.rate_per_s) == _round_rate(smallest)
        )
        if smallest_step.stability == STABLE and not self.is_ceiling_step(smallest_step):
            self.phase = PHASE_BISECT_STABILITY
            return None
        if smallest <= self.min_rate:
            self.phase = PHASE_DONE  # even a trickle diverges
            return None
        return max(self.min_rate, smallest / 2.0)

    def _bracket(
        self, lows: list[float], highs: list[float],
    ) -> Optional[float]:
        """Geometric midpoint of the tightest (low, high) bracket, or
        None when the bracket is already tighter than resolution."""
        if not lows or not highs:
            return None
        lo = max(p for p in lows)
        hi_candidates = [p for p in highs if p > lo]
        if not hi_candidates:
            return None
        hi = min(hi_candidates)
        if hi / lo <= self.resolution_ratio:
            return None
        mid = (lo * hi) ** 0.5
        if _round_rate(mid) in self._measured():
            return None
        return mid

    def _next_bisect_stability(self) -> Optional[float]:
        mid = self._bracket(
            [s.rate_per_s for s in self._stable()],
            [s.rate_per_s for s in self._ceilings()],
        )
        if mid is None:
            self.phase = self._after_stability()
            return None
        return mid

    def _next_backfill(self) -> Optional[float]:
        """Lowest missing point of the geometric grid below the knee:
        knee / f, knee / f^2, ... down to knee / floor_ratio. A grid
        point counts as measured when a stable rate lies within half a
        grid step of it. Ascending, so each window adds sessions."""
        f = self.backfill_factor
        stable = self._stable()
        if not f or f <= 1.0 or not stable:
            self.phase = PHASE_DONE
            return None
        knee = max(s.rate_per_s for s in stable)
        floor = knee / max(1.0, self.backfill_floor_ratio)
        half = f ** 0.5
        grid = []
        r = knee / f
        while r >= floor * 0.999 and r >= self.min_rate:
            grid.append(r)
            r /= f
        rates = [s.rate_per_s for s in self.history]
        missing = [g for g in sorted(grid)
                   if not any(g / half <= x <= g * half for x in rates)]
        if not missing:
            self.phase = PHASE_DONE
            return None
        return missing[0]

    def _next_bisect_sla(self) -> Optional[float]:
        stable = self._stable()
        passes = [s.rate_per_s for s in stable if s.sla_pass]
        # The SLA ceiling: a stable rate that did not pass the gate
        # (marginal or fail), or any stability ceiling (a divergent
        # queue means unbounded latency — SLA fails there by
        # construction).
        fails = [s.rate_per_s for s in stable if s.sla_pass is False]
        fails += [s.rate_per_s for s in self._ceilings()]
        mid = self._bracket(passes, fails)
        if mid is None:
            self.phase = PHASE_DONE
            return None
        return mid

    # ── Results ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        stable = self._stable()
        ceilings = self._ceilings()
        divergent = [s for s in self.history if s.stability == DIVERGENT]
        limited = [s for s in self.history if s.stability == CLIENT_LIMITED]

        rate_max = max((s.rate_per_s for s in stable), default=None)
        passes = [s.rate_per_s for s in stable if s.sla_pass]
        rate_sla = max(passes, default=None) if self.sla_gate else None

        # Coverage semantics (mirrors the closed-loop labels):
        #   full_curve      — stability boundary bracketed by real
        #                     engine divergence.
        #   client_limited  — the generator hit its ceiling first
        #                     (even after scaling out) — rates are
        #                     lower bounds.
        #   capped          — the rate rail was reached while still
        #                     stable — rates are lower bounds.
        #   exceeds_hardware— even the minimum probed rate diverges.
        #   single_point    — one measurement only.
        if len(self.history) <= 1 and not divergent:
            coverage = "single_point"
        elif not stable:
            coverage = "exceeds_hardware"
        elif divergent and any(
            d.rate_per_s > (rate_max or 0) for d in divergent
        ):
            coverage = "full_curve"
        elif limited:
            coverage = "client_limited"
        elif rate_max is not None and rate_max >= self.max_rate:
            coverage = "capped"
        else:
            coverage = "full_curve" if divergent else "capped"

        lower_bound = coverage in ("capped", "client_limited")
        # The boundary bracket, for reporting precision honestly.
        ceiling_above = min(
            (c.rate_per_s for c in ceilings
             if rate_max is None or c.rate_per_s > rate_max),
            default=None,
        )
        return {
            "rate_max_per_s": rate_max,
            "rate_sla_per_s": rate_sla,
            "rate_ceiling_per_s": ceiling_above,
            "coverage": coverage,
            "rates_are_lower_bounds": lower_bound,
            "steps": len(self.history),
        }
