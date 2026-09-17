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
                   out_tok_s=20000.0, prompt_tok_s=1200.0,
                   in_flight=800.0, queue_depth=40.0,
                   objective=objective(800.0, 20000.0))
    assert r.objective == (800.0 * 20000.0) ** 0.5


def test_model_family_strips_org_and_quantization():
    from simulator.headline_shapes import model_family
    assert model_family("Qwen/Qwen3-30B-A3B-Instruct-2507") \
        == model_family("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8") \
        == "qwen3-30b-a3b-instruct-2507"
    assert model_family("meta-llama/Llama-3.1-70B-Instruct-AWQ") \
        == model_family("other-org/Llama-3.1-70B-Instruct")
    # Different sizes are different families.
    assert model_family("Qwen/Qwen3-30B-A3B") != model_family("Qwen/Qwen3-4B")


def test_shape_store_roundtrip(tmp_path):
    from simulator.headline_shapes import save_shape, shape_for
    path = tmp_path / "headline_shapes.json"
    fam = save_shape(path, "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                     {"input_tokens": 256, "output_tokens": 1024,
                      "objective": 4000.0})
    assert fam == "qwen3-30b-a3b-instruct-2507"
    # The auto sibling resolves to the same stored optimum.
    got = shape_for(path, "Qwen/Qwen3-30B-A3B-Instruct-2507")
    assert got["input_tokens"] == 256 and got["output_tokens"] == 1024
    assert got["model_id"].endswith("-FP8")
    assert shape_for(path, "Qwen/Qwen3-4B") is None


def test_apply_shape_updates_generation_persona(tmp_path):
    """The winner lands IN Headline: Generation — same id, new
    token distributions, everything else preserved."""
    import random

    from simulator.headline_shapes import (
        apply_shape_to_generation,
        generation_shape,
    )
    from simulator.personas import PERSONAS, reload_personas
    try:
        apply_shape_to_generation(tmp_path, 512, 2048)
        p = PERSONAS["headline_generation"]
        rng = random.Random(0)
        assert p.input_tokens.sample_int(rng) == 512
        assert p.output_tokens.sample_int(rng) == 2048
        assert p.ignore_eos is True                  # preserved
        assert p.name == "Headline: Generation"      # preserved
        assert generation_shape() == (512, 2048)
    finally:
        reload_personas()


def test_outstanding_mode_holds_and_respawns(monkeypatch):
    """Saturation mode: the launcher keeps exactly N sessions active,
    respawning as each finishes — the fast shape search's pressure
    source."""
    import asyncio

    import simulator.arrivals as arrivals_mod
    from simulator.arrivals import SessionArrivalLauncher
    from simulator.virtual_user import SharedState

    spawned = []

    async def fake_run_virtual_user(**kw):
        spawned.append(kw["stats"].persona_id)
        await asyncio.sleep(0.02)

    monkeypatch.setattr(arrivals_mod, "run_virtual_user",
                        fake_run_virtual_user)

    async def main():
        launcher = SessionArrivalLauncher(
            persona_weights={"quick_lookup": 1.0},
            clients=[object()], model_id="m", corpus=None,
            state=SharedState(), request_timeout_s=5,
        )
        launcher.start()
        launcher.set_outstanding(8)
        await asyncio.sleep(0.01)
        held = launcher.stats.sessions_active
        await asyncio.sleep(0.15)   # several respawn generations
        held2 = launcher.stats.sessions_active
        done = launcher.stats.sessions_done
        await launcher.stop()
        return held, held2, done

    held, held2, done = asyncio.run(main())
    assert held == 8 and held2 == 8
    assert done >= 16          # respawned repeatedly
    assert len(spawned) >= 24


def test_chunks_converged_guards_young_kv_bias():
    """A long-output cell's running batch keeps sliding while KV
    fills — consecutive chunks must AGREE before a cell may score."""
    from simulator.headline_search import Chunk, chunks_converged

    def ch(running, rate):
        return Chunk(running=running, queue=0.0,
                     out_rate=rate, prompt_rate=None)

    # Batch still sagging 4096 → 3600 → not converged.
    assert not chunks_converged(ch(4096, 30000), ch(3600, 30000))
    # Rate still moving > 5% → not converged.
    assert not chunks_converged(ch(800, 30000), ch(800, 26000))
    # Both settled within tolerance → converged.
    assert chunks_converged(ch(812, 21000), ch(805, 20800))
    # Metrics missing entirely → never "converged" by default.
    assert not chunks_converged(ch(None, None), ch(None, None))
    # One signal present and steady is enough.
    assert chunks_converged(ch(None, 21000), ch(None, 21000))


def test_restart_respawns_with_fresh_population(monkeypatch):
    """The instant shape swap: cancelling every session in saturation
    mode leaves the outstanding target intact — the population comes
    back immediately (with freshly-resolved personas)."""
    import asyncio

    import simulator.arrivals as arrivals_mod
    from simulator.arrivals import SessionArrivalLauncher
    from simulator.virtual_user import SharedState

    async def fake_run_virtual_user(**kw):
        # Long-running unless cancelled — models a 4096-token
        # straggler. Honors cancel_event like the real virtual user.
        try:
            await asyncio.wait_for(kw["cancel_event"].wait(), timeout=30)
        except asyncio.TimeoutError:
            pass

    monkeypatch.setattr(arrivals_mod, "run_virtual_user",
                        fake_run_virtual_user)

    async def main():
        launcher = SessionArrivalLauncher(
            persona_weights={"quick_lookup": 1.0},
            clients=[object()], model_id="m", corpus=None,
            state=SharedState(), request_timeout_s=5,
        )
        launcher.start()
        launcher.set_outstanding(6)
        await asyncio.sleep(0.05)
        before = launcher.stats.arrivals_total
        launcher.cancel_active_sessions()   # the "restart" worker cmd
        await asyncio.sleep(0.1)
        held = launcher.stats.sessions_active
        respawned = launcher.stats.arrivals_total - before
        await launcher.stop()
        return held, respawned

    held, respawned = asyncio.run(main())
    assert held == 6            # population fully restored
    assert respawned >= 6       # every cancelled session was replaced
