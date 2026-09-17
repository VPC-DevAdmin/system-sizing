"""Joint engine + shape search: grid, ranking guards, orchestration."""

from __future__ import annotations

import json

from simulator.headline_optimize import (
    PRESETS,
    SEARCH_LADDER,
    Candidate,
    estimate_minutes,
    grid,
    rank,
)


def test_grid_is_the_product_of_engine_and_shape():
    g = grid("quick")
    assert len(g) == len(PRESETS["quick"]["max_num_seqs"]) * len(
        PRESETS["quick"]["output_tokens"])
    # Cheapest engines first, so an interrupted search still ranks.
    assert g == sorted(g)
    # Explicit lists override the preset.
    assert grid("quick", max_num_seqs=[512], output_tokens=[1024, 2048]) == [
        (512, 1024), (512, 2048)]


def test_estimate_scales_with_the_grid():
    quick, thorough = grid("quick"), grid("thorough")
    assert estimate_minutes(quick) < estimate_minutes(thorough)
    # Every pair costs an engine launch — vLLM cannot change
    # max_num_seqs in place.
    assert estimate_minutes(quick) >= len(quick) * 5


def test_ranking_discards_untrustworthy_candidates():
    """The whole point of the honesty flags: a bigger number measured
    before the engine settled is an artifact, not a win."""
    best_real = Candidate(512, 2048, 128, out_tok_s=54867.0, in_flight=4095)
    second = Candidate(1024, 1024, 128, out_tok_s=54717.0, in_flight=6514)
    unsettled = Candidate(2048, 512, 128, out_tok_s=99999.0,
                          in_flight=8000, steady_state=False)
    failed = Candidate(256, 4096, 128, error="launch failed")
    empty = Candidate(512, 512, 128, out_tok_s=None)

    ranked = rank([unsettled, second, best_real, failed, empty])
    assert [c.out_tok_s for c in ranked] == [54867.0, 54717.0]
    assert unsettled.usable is False      # despite the biggest number
    assert failed.usable is False
    assert empty.usable is False
    assert best_real.usable is True


def test_search_ladder_is_coarse_but_reaches_the_top():
    # Ranking only needs the peak; the low rungs cost time and teach
    # nothing. The winner earns the full ladder afterwards.
    from simulator.headline_sweep import DEFAULT_LADDER
    assert len(SEARCH_LADDER) < len(DEFAULT_LADDER)
    assert max(SEARCH_LADDER) == max(DEFAULT_LADDER)


def test_optimize_survives_a_failing_candidate(tmp_path, monkeypatch):
    """One engine shape that will not launch must not abandon the
    search — it is recorded and the walk continues."""
    import asyncio

    import simulator.headline_optimize as ho

    calls = []

    async def fake_sweep(cfg, cohort, **kw):
        calls.append(kw.get("ladder_override"))
        # Second candidate blows up; the rest measure fine.
        if len(calls) == 2:
            raise RuntimeError("replica 0 container exited during startup")
        d = tmp_path / f"run_{len(calls)}"
        d.mkdir(exist_ok=True)
        p = d / "headline_sweep.json"
        p.write_text(json.dumps({
            "status": "ok",
            "peak": {"out_tok_s": 1000.0 * len(calls), "in_flight": 100.0,
                     "total_tok_s": 1100.0, "steady_state": True,
                     "held": True},
        }))
        return p

    monkeypatch.setattr(ho, "run_headline_sweep", fake_sweep, raising=False)
    import simulator.headline_sweep as hs
    monkeypatch.setattr(hs, "run_headline_sweep", fake_sweep)
    monkeypatch.setattr(
        "simulator.headline_shapes.apply_shape_to_generation",
        lambda *a, **k: None)

    from simulator.config import Config
    from simulator.personas import cohort_from_persona

    def build_config(overrides):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "engine:\n  type: mock\n  model_id: mock-model\n"
            f"  mock_capacity_inflight: {overrides.get('max_num_seqs', 8)}\n"
            "simulation:\n  request_timeout_s: 30\n"
            f"output:\n  db_directory: {tmp_path}\n")
        return path

    cfg = Config()
    progress: dict = {}
    out = asyncio.run(ho.run_headline_optimize(
        cfg, cohort_from_persona("headline_generation"),
        preset="quick", build_config=build_config,
        runs_base=tmp_path, progress=progress))

    doc = json.loads(out.read_text())
    assert len(doc["candidates"]) == len(grid("quick"))
    failed = [c for c in doc["candidates"] if c["error"]]
    assert len(failed) == 1
    assert "container exited" in failed[0]["error"]
    # The search still produced a winner and reported progress.
    assert doc["winner"] is not None
    assert progress["done"] is True


def test_explicit_grid_overrides_the_preset():
    """A model's KV cost decides which shapes are reachable at all.
    Llama-3.3-70B costs 16x Qwen3.6's per token, so its grid belongs
    at short outputs — a fixed preset cannot know that."""
    llama = grid("standard", max_num_seqs=[512, 1024, 2048],
                 output_tokens=[128, 256, 512])
    assert len(llama) == 9
    assert max(o for _, o in llama) == 512
    # The preset's own (longer) shapes are not smuggled in.
    assert 2048 not in {o for _, o in llama}


def test_request_model_accepts_an_explicit_grid():
    from simulator.service import StartRunRequest

    req = StartRunRequest(
        workload={"kind": "headline_optimize", "id": "headline_generation"},
        search_max_num_seqs=[512, 1024],
        search_output_tokens=[128, 256],
    )
    assert req.search_max_num_seqs == [512, 1024]
    assert req.search_output_tokens == [128, 256]
    # Absent by default, so the preset still governs.
    plain = StartRunRequest(workload={"kind": "cohort", "id": "x"})
    assert plain.search_max_num_seqs is None
