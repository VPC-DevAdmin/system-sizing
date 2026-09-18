"""Roofline autopilot: ranking, planning, persistence, resume.

The thing this module has to get right is not speed, it is SURVIVING.
A roofline run takes hours, the operator closes a laptop, and the page
they come back to over a VPN has to be correct without having watched
anything happen.
"""

from __future__ import annotations

import json

import pytest

from simulator.roofline import (
    Candidate,
    State,
    cell_key,
    cells,
    estimate_minutes,
    load_state,
    save_state,
    score_models,
    summarize,
)

CATALOG = [
    {"id": "org/moe-nvfp4", "quant": "nvfp4", "params_b": 35.0, "moe": True,
     "approx_size_gb": 21, "min_vram_gb": 24},
    {"id": "org/dense-bf16", "quant": "bf16", "params_b": 70.6, "moe": False,
     "approx_size_gb": 140, "min_vram_gb": 160},
    {"id": "org/huge", "quant": "fp8", "params_b": 685.0, "moe": True,
     "approx_size_gb": 687, "min_vram_gb": 5000},
]


def test_ranking_prefers_cheap_kv_and_says_why():
    ranked = score_models(CATALOG, vram_per_gpu_gb=95.6)
    assert ranked[0].id == "org/moe-nvfp4"
    # Every candidate explains itself; a bare score is not a reason.
    assert all(c.why for c in ranked)
    assert "MoE" in ranked[0].why


def test_a_model_that_cannot_fit_is_excluded_not_ranked_last():
    ranked = {c.id: c for c in score_models(CATALOG, vram_per_gpu_gb=95.6)}
    assert ranked["org/huge"].fits is False
    assert ranked["org/huge"].score == 0.0
    assert "does not fit" in ranked["org/huge"].why


def test_unstaged_models_say_their_kv_cost_was_estimated():
    """KV bytes per token decides a headline, and it can only be read
    from a config on disk. Guessing it silently is how a dense 70B
    gets mistaken for a cheap model."""
    ranked = score_models(CATALOG, vram_per_gpu_gb=95.6)
    assert any("estimated" in c.why for c in ranked)


def test_plan_is_model_major():
    """Switching model costs a full weight load; switching engine does
    not. Model-major also means a partial run holds a COMPLETE answer
    for the models it reached."""
    c = cells(["m1", "m2"], ["e1", "e2"], {"max_num_seqs": [1, 2],
                                           "output_tokens": [8, 16]})
    assert len(c) == 16
    assert [x["model"] for x in c[:8]] == ["m1"] * 8
    assert estimate_minutes(len(c), n_models=2) > 0


def test_state_round_trips_through_disk():
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    st = State(status="searching")
    st.results = [{"model": "m", "engine": "e", "max_num_seqs": 1,
                   "output_tokens": 8, "out_tok_s": 100.0,
                   "steady_state": True}]
    save_state(d / "roofline.json", st)
    back = load_state(d / "roofline.json")
    assert back.status == "searching"
    assert back.results[0]["out_tok_s"] == 100.0
    # The written document is what the browser reads.
    doc = json.loads((d / "roofline.json").read_text())
    assert doc["kind"] == "roofline"
    assert doc["summary"]["best"]["out_tok_s"] == 100.0
    assert doc["done"] is False


def test_state_is_written_atomically():
    """A browser reconnecting mid-write must not read half a file."""
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    p = d / "roofline.json"
    for i in range(20):
        st = State(status="searching")
        st.results = [{"model": f"m{i}", "engine": "e", "max_num_seqs": 1,
                       "output_tokens": 8, "out_tok_s": float(i),
                       "steady_state": True}]
        save_state(p, st)
        json.loads(p.read_text())          # always parseable
    assert not list(d.glob("*.tmp"))       # no debris left behind


def test_summary_excludes_unsettled_cells():
    """An unsettled rung is routinely the biggest number in a sweep,
    which is exactly why it cannot win a roofline."""
    rows = [
        {"model": "a", "engine": "x", "out_tok_s": 100.0, "steady_state": True},
        {"model": "a", "engine": "y", "out_tok_s": 999.0, "steady_state": False},
    ]
    s = summarize(rows)
    assert s["best"]["out_tok_s"] == 100.0
    assert s["measured"] == 1 and s["attempted"] == 2


def test_summary_reports_the_matrix_not_just_a_winner():
    """Which engine wins for which model is what tells you what to do
    with the NEXT model; a single number does not."""
    rows = [
        {"model": "a", "engine": "vllm", "out_tok_s": 100.0, "steady_state": True},
        {"model": "a", "engine": "trt", "out_tok_s": 90.0, "steady_state": True},
        {"model": "b", "engine": "trt", "out_tok_s": 120.0, "steady_state": True},
    ]
    s = summarize(rows)
    assert s["best"]["model"] == "b"
    assert set(s["best_per_model"]) == {"a", "b"}
    assert s["best_per_engine"]["trt"]["out_tok_s"] == 120.0
    assert s["best_per_engine"]["vllm"]["out_tok_s"] == 100.0


def test_resume_skips_cells_already_measured():
    """Each repeated cell costs minutes of engine launch."""
    rows = [{"model": "m1", "engine": "e1", "max_num_seqs": 1,
             "output_tokens": 8, "out_tok_s": 1.0, "steady_state": True}]
    done = {cell_key(r) for r in rows}
    plan = cells(["m1"], ["e1"], {"max_num_seqs": [1], "output_tokens": [8]})
    assert cell_key(plan[0]) in done


def test_failed_cells_do_not_abort_the_matrix():
    rows = [
        {"model": "a", "engine": "x", "error": "launch failed"},
        {"model": "a", "engine": "y", "out_tok_s": 50.0, "steady_state": True},
    ]
    s = summarize(rows)
    assert s["best"]["out_tok_s"] == 50.0
    assert len(s["failed"]) == 1


def test_endpoint_reports_nothing_gracefully_before_any_run():
    from fastapi.testclient import TestClient

    from simulator.service import create_app

    c = TestClient(create_app())
    d = c.get("/api/roofline").json()
    assert d["status"] == "none" and d["done"] is True
    assert d["summary"]["best"] is None


def test_candidates_endpoint_ranks_and_explains():
    from fastapi.testclient import TestClient

    from simulator.service import create_app

    c = TestClient(create_app())
    d = c.get("/api/roofline/candidates?limit=3").json()
    assert len(d["candidates"]) <= 3
    assert all(x["why"] for x in d["candidates"])


def test_measured_kv_outranks_a_guess(monkeypatch):
    """A staged model whose KV cost was read from its config is a known
    quantity. An unstaged one is a guess from parameter count that
    flatters small models -- and acting on it costs a multi-gigabyte
    download before anyone finds out it was wrong."""
    import simulator.roofline as rf

    cat = [
        {"id": "org/tiny-unstaged", "quant": "nvfp4", "params_b": 8.0,
         "moe": True, "approx_size_gb": 5, "min_vram_gb": 8},
        {"id": "org/staged", "quant": "nvfp4", "params_b": 35.0,
         "moe": True, "approx_size_gb": 21, "min_vram_gb": 24},
    ]
    monkeypatch.setattr(rf, "kv_bytes_per_token",
                        lambda mid, cache=None: 40960 if "staged" in mid
                        and "unstaged" not in mid else None)
    ranked = rf.score_models(cat, vram_per_gpu_gb=95.6)
    assert ranked[0].id == "org/staged"
    assert ranked[0].measured_kv is True
    assert "own config" in ranked[0].why
    assert ranked[1].measured_kv is False
