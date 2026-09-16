"""Arrival-rate two-knee search (simulator/rate_search.py)."""

from __future__ import annotations

from simulator.rate_search import (
    CLIENT_LIMITED,
    DIVERGENT,
    STABLE,
    RateStep,
    RateStepper,
)


def _drive(stepper: RateStepper, oracle) -> list[float]:
    """Run the search against an oracle(rate) -> RateStep factory."""
    probed = []
    while (rate := stepper.next_rate()) is not None:
        probed.append(rate)
        stepper.record(oracle(rate))
        assert len(probed) < 60, "search did not terminate"
    return probed


def _oracle(stability_boundary: float, sla_boundary: float):
    """Stable below the boundary, divergent above; SLA passes below
    sla_boundary."""
    def f(rate: float) -> RateStep:
        if rate > stability_boundary:
            return RateStep(rate_per_s=rate, stability=DIVERGENT,
                            sla_pass=False, violation_rate=1.0)
        return RateStep(
            rate_per_s=rate, stability=STABLE,
            sla_pass=rate <= sla_boundary,
            violation_rate=0.0 if rate <= sla_boundary else 0.2,
            sample_size=200,
        )
    return f


def test_doubles_then_bisects_to_boundary():
    stepper = RateStepper(initial_rate_per_s=1.0, max_rate_per_s=256.0)
    _drive(stepper, _oracle(stability_boundary=10.0, sla_boundary=10.0))
    s = stepper.summary()
    assert s["coverage"] == "full_curve"
    # Bracket around the true boundary within the default 5%
    # resolution — the run only completes once λ_max is pinned tight.
    assert s["rate_max_per_s"] is not None
    assert s["rate_ceiling_per_s"] is not None
    assert s["rate_max_per_s"] <= 10.0 < s["rate_ceiling_per_s"]
    assert s["rate_ceiling_per_s"] / s["rate_max_per_s"] <= 1.06
    assert not s["rates_are_lower_bounds"]


def test_sla_knee_below_stability_knee():
    stepper = RateStepper(initial_rate_per_s=1.0, max_rate_per_s=256.0)
    _drive(stepper, _oracle(stability_boundary=40.0, sla_boundary=10.0))
    s = stepper.summary()
    assert s["rate_sla_per_s"] is not None
    assert s["rate_sla_per_s"] <= 10.0
    assert s["rate_max_per_s"] > s["rate_sla_per_s"]
    # SLA bracket refined too: the found λ_sla is within resolution of
    # the true 10/s boundary.
    assert s["rate_sla_per_s"] >= 10.0 / 1.06


def test_capped_at_rail_reports_lower_bound():
    stepper = RateStepper(initial_rate_per_s=1.0, max_rate_per_s=8.0)
    _drive(stepper, _oracle(stability_boundary=1e9, sla_boundary=1e9))
    s = stepper.summary()
    assert s["coverage"] == "capped"
    assert s["rates_are_lower_bounds"]
    assert s["rate_max_per_s"] == 8.0


def test_client_limited_is_a_ceiling_and_marks_coverage():
    def oracle(rate: float) -> RateStep:
        if rate > 16.0:  # generator gives out before the engine
            return RateStep(rate_per_s=rate, stability=CLIENT_LIMITED)
        return RateStep(rate_per_s=rate, stability=STABLE, sla_pass=True,
                        sample_size=200)
    stepper = RateStepper(initial_rate_per_s=1.0, max_rate_per_s=256.0)
    _drive(stepper, oracle)
    s = stepper.summary()
    assert s["coverage"] == "client_limited"
    assert s["rates_are_lower_bounds"]
    assert s["rate_max_per_s"] <= 16.0


def test_initial_rate_divergent_searches_downward():
    stepper = RateStepper(initial_rate_per_s=8.0, max_rate_per_s=256.0)
    probed = _drive(stepper, _oracle(stability_boundary=1.5, sla_boundary=1.5))
    assert any(r < 8.0 for r in probed)
    s = stepper.summary()
    assert s["coverage"] == "full_curve"
    assert s["rate_max_per_s"] <= 1.5


def test_everything_divergent_is_exceeds_hardware():
    def oracle(rate: float) -> RateStep:
        return RateStep(rate_per_s=rate, stability=DIVERGENT, sla_pass=False)
    stepper = RateStepper(initial_rate_per_s=1.0, max_rate_per_s=256.0)
    _drive(stepper, oracle)
    assert stepper.summary()["coverage"] == "exceeds_hardware"


def test_refinement_flag_gates_longer_windows():
    stepper = RateStepper(initial_rate_per_s=1.0, max_rate_per_s=256.0)
    assert not stepper.in_refinement
    stepper.record(RateStep(rate_per_s=1.0, stability=STABLE, sla_pass=True))
    stepper.next_rate()
    stepper.record(RateStep(rate_per_s=2.0, stability=DIVERGENT))
    stepper.next_rate()
    assert stepper.in_refinement
