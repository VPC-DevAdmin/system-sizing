"""Headline shape search — climb logic, objective, cell personas."""

from __future__ import annotations

from simulator.headline_search import (
    LATTICE_IN,
    LATTICE_OUT,
    CellResult,
    ShapeClimb,
    _cell_persona_spec,
    _neighbors,
    objective,
)


def test_objective_is_symmetric_and_honest():
    assert objective(100, 10000) == (100 * 10000) ** 0.5
    # A cell that trades everything for one axis scores worse than a
    # balanced one with the same product... and no boundary = zero.
    assert objective(None, 5000) == 0.0
    assert objective(0, 5000) == 0.0


def test_neighbors_respect_lattice_bounds():
    corner = (LATTICE_IN[0], LATTICE_OUT[0])
    ns = _neighbors(corner)
    assert (LATTICE_IN[1], LATTICE_OUT[0]) in ns
    assert (LATTICE_IN[0], LATTICE_OUT[1]) in ns
    assert len(ns) == 2  # two edges clipped
    mid = (256, 512)
    assert len(_neighbors(mid)) == 4


def _drive(climb: ShapeClimb, score_fn) -> None:
    while (cell := climb.propose()) is not None:
        climb.record(cell, score_fn(cell))


def test_climb_finds_concave_peak_within_budget():
    # Synthetic concave surface peaking at (256, 1024).
    peak = (256, 1024)

    def score(cell):
        di = abs(LATTICE_IN.index(cell[0]) - LATTICE_IN.index(peak[0]))
        do = abs(LATTICE_OUT.index(cell[1]) - LATTICE_OUT.index(peak[1]))
        return 1000 - 100 * (di + do)

    climb = ShapeClimb(start=(128, 512), budget=12)
    _drive(climb, score)
    assert climb.best() == peak
    assert len(climb.scores) <= 12


def test_climb_stops_at_local_optimum_without_burning_budget():
    # Flat surface: start is immediately a local optimum once its
    # neighborhood is scored.
    climb = ShapeClimb(start=(128, 512), budget=12)
    _drive(climb, lambda c: 1.0)
    assert len(climb.scores) == 1 + 4  # start + its four neighbors


def test_climb_respects_budget():
    # Monotone surface keeps improving toward a far corner — the
    # budget must cut it off.
    def score(cell):
        return LATTICE_IN.index(cell[0]) + LATTICE_OUT.index(cell[1])
    climb = ShapeClimb(start=(32, 64), budget=6)
    _drive(climb, score)
    assert len(climb.scores) == 6


def test_cell_persona_spec_parses():
    from simulator.persona_loader import parse_persona
    spec = _cell_persona_spec(256, 512, name="n", description="d")
    p = parse_persona("headline_cell", spec)
    assert p.ignore_eos is True
    import random
    rng = random.Random(0)
    assert p.input_tokens.sample_int(rng) == 256
    assert p.output_tokens.sample_int(rng) == 512
    assert p.turns_per_session.sample_int(rng) == 1


def test_cell_result_objective_roundtrip():
    r = CellResult(input_tokens=128, output_tokens=512,
                   rate_max_per_min=600.0, out_tok_s=20000.0,
                   in_flight=800.0, objective=objective(800.0, 20000.0))
    assert r.objective == (800.0 * 20000.0) ** 0.5
