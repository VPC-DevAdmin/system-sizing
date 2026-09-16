"""Arrival-rate search for the open-loop methodology.

Searches over λ (session arrivals per second) for two knees:

  * **λ_max** — the stability boundary: highest rate at which the
    queue is stationary; above it the backlog grows without bound.
    This is the box's capacity, independent of any client pool.
  * **λ_sla** — the quality boundary: highest *stable* rate whose
    steady-state turns also meet the SLA (violation rate < 5%).
    λ_sla ≤ λ_max; the band between them is "surviving but degraded".

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
PHASE_DONE = "done"


@dataclass
class RateStep:
    rate_per_s: float
    stability: str                 # stable | divergent | client_limited
    sla_pass: Optional[bool] = None  # None: divergent/limited or no samples
    violation_rate: float = 0.0
    target_miss_rate: float = 0.0
    sample_size: int = 0


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
        resolution_ratio: float = 1.3,
        sla_threshold: float = 0.05,
    ):
        self.initial_rate = _round_rate(initial_rate_per_s)
        self.max_rate = max_rate_per_s
        self.min_rate = min_rate_per_s
        self.resolution_ratio = resolution_ratio
        self.sla_threshold = sla_threshold
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

    def _measured(self) -> set[float]:
        return {_round_rate(s.rate_per_s) for s in self.history}

    def _stable(self) -> list[RateStep]:
        return [s for s in self.history if s.stability == STABLE]

    def _ceilings(self) -> list[RateStep]:
        """Steps that bound the search from above: divergent (the
        engine gave out) or client_limited (the generator did)."""
        return [
            s for s in self.history
            if s.stability in (DIVERGENT, CLIENT_LIMITED)
        ]

    def _next_doubling(self) -> Optional[float]:
        last = self.history[-1]
        if last.stability != STABLE:
            if len(self.history) == 1:
                self.phase = PHASE_DOWNWARD
            else:
                self.phase = PHASE_BISECT_STABILITY
            return None
        if last.rate_per_s >= self.max_rate:
            self.phase = PHASE_BISECT_SLA  # capped — no upper bracket
            return None
        nxt = min(self.max_rate, last.rate_per_s * 2.0)
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
        if smallest_step.stability == STABLE:
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
            self.phase = PHASE_BISECT_SLA
            return None
        return mid

    def _next_bisect_sla(self) -> Optional[float]:
        stable = self._stable()
        passes = [s.rate_per_s for s in stable if s.sla_pass]
        # The SLA ceiling: a stable-but-violating rate, or any
        # stability ceiling (a divergent queue means unbounded
        # latency — SLA fails there by construction).
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
        rate_sla = max(passes, default=None)

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
