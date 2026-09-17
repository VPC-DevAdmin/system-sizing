"""Concurrency-ladder scoring: each candidate is scored at ITS OWN
best SLA-passing rung, not at an arbitrary fixed concurrency."""

from __future__ import annotations

import textwrap

import pytest

from simulator.search import (
    Objective,
    SearchSpaceError,
    best_rung,
    load_space,
    rung_sla_ok,
    score_ladder,
)

OBJ = Objective(kind="sla_throughput", ttft_p95_cap_ms=10_000,
                tpot_p95_cap_ms=100, penalty=0.25)


def _rung(name, tps, ttft=1000.0, tpot=50.0, samples=32, errors=0,
          timeouts=0):
    return {"cell_name": name, "throughput_out_tok_s": tps,
            "ttft_p95_ms": ttft, "tpot_p95_ms": tpot,
            "samples": samples, "errors": errors, "timeouts": timeouts}


def test_score_is_best_rung_not_sum() -> None:
    """dp-wide configs keep climbing to their real capacity; the sum
    would reward ladder length instead of demonstrated throughput."""
    cells = [_rung("ladder_c0008", 500), _rung("ladder_c0032", 1800),
             _rung("ladder_c0128", 3900)]
    assert score_ladder(cells, OBJ) == 3900
    r = best_rung(cells, OBJ)
    assert r["cell_name"] == "ladder_c0128" and r["sla_ok"] is True


def test_sla_blown_rung_is_penalized_not_erased() -> None:
    """A rung over the caps contributes ×penalty — 'fast but slightly
    over' still beats 'slow', but an SLA-passing lower rung wins when
    the raw gain is under 4×."""
    cells = [_rung("ladder_c0032", 2000),
             _rung("ladder_c0128", 3000, tpot=180.0)]   # blows TPOT cap
    assert not rung_sla_ok(cells[1], OBJ)
    # 3000 × 0.25 = 750 < 2000 → the SLA-passing rung wins.
    assert score_ladder(cells, OBJ) == 2000
    assert best_rung(cells, OBJ)["cell_name"] == "ladder_c0032"

    # But a config that NEVER passes SLA keeps a gradient.
    cells = [_rung("ladder_c0008", 1000, tpot=150.0)]
    assert score_ladder(cells, OBJ) == 250
    assert best_rung(cells, OBJ)["sla_ok"] is False


def test_incomplete_requests_bleed_score() -> None:
    cells = [_rung("ladder_c0032", 2000, samples=16, errors=8, timeouts=8)]
    assert score_ladder(cells, OBJ) == 2000 * 0.5


def test_latency_objective_uses_gentlest_rung() -> None:
    obj = Objective(kind="latency")
    cells = [_rung("ladder_c0008", 500, ttft=800.0),
             _rung("ladder_c0032", 1800, ttft=2500.0)]
    assert score_ladder(cells, obj) == -800.0


def test_measurement_parses_and_fingerprints(tmp_path) -> None:
    base = textwrap.dedent("""\
        name: m
        engine: vllm_cuda
        device_groups: [[0, 1]]
        model_variants:
          v: {model: org/M}
        dimensions:
          tp: [1, 2]
    """)
    p = tmp_path / "s.yaml"
    p.write_text(base)
    space = load_space(p)
    # Defaults: chat-shaped workload, 8→512 ladder.
    assert space.measurement.ladder == [8, 32, 128, 512]
    assert space.measurement.input_tokens == 512
    h_default = space.space_hash()

    p.write_text(base + "measurement: {ladder: [16, 64]}\n")
    space2 = load_space(p)
    assert space2.measurement.ladder == [16, 64]
    # A different measurement means scores aren't comparable — the
    # fingerprint changes so a stale state refuses to resume.
    assert space2.space_hash() != h_default

    p.write_text(base + "measurement: {ladder: [64, 16]}\n")
    with pytest.raises(SearchSpaceError, match="ascending"):
        load_space(p)


def test_summarize_carries_best_rung(tmp_path) -> None:
    from simulator.search import SearchState, record_evaluation, summarize
    p = tmp_path / "s.yaml"
    p.write_text(textwrap.dedent("""\
        name: m
        engine: vllm_cuda
        device_groups: [[0, 1]]
        model_variants:
          v: {model: org/M}
        dimensions:
          tp: [1, 2]
    """))
    space = load_space(p)
    state = SearchState(space_hash=space.space_hash())
    cells = [_rung("ladder_c0032", 2000), _rung("ladder_c0128", 3100)]
    record_evaluation(state, space, {"tp": 1}, status="ok",
                      score=score_ladder(cells, space.objective),
                      config_name="c1", cells=cells, iteration=0)
    s = summarize(state, space)
    assert s["best"]["best_rung"]["cell_name"] == "ladder_c0128"
    assert s["top"][0]["best_rung"]["throughput_out_tok_s"] == 3100
    # Without a space (older callers) the field is simply absent-null.
    assert summarize(state)["best"]["best_rung"] is None


def test_arena_space_declares_measurement(tmp_path, monkeypatch) -> None:
    from simulator import arena as arena_mod
    from simulator.arena import build_space_doc, summarize_space_doc
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)
    catalog = [{"id": "org/M-30B", "family": "m-30b", "quant": "bf16",
                "min_vram_gb": 70, "approx_size_gb": 61, "moe": False,
                "gated": False, "engine_args": [], "notes": ""}]
    doc = build_space_doc({}, catalog, budget=12)
    assert doc["measurement"]["ladder"] == [8, 32, 128, 512]
    s = summarize_space_doc(doc)
    assert s["ladder"] == [8, 32, 128, 512]
    assert s["measurement_tokens"] == [512, 256]


def test_extend_ladder_reaches_candidate_capacity() -> None:
    """The measurement ladder must climb to each candidate's own
    box-wide sequence capacity, or batch-width/KV dims are never
    probed where they differ."""
    from simulator.search import extend_ladder
    base = [8, 32, 128, 512]
    # mns 512 × dp8 = 4096: two more ×4 rungs land exactly on it.
    assert extend_ladder(base, 512, 8) == [8, 32, 128, 512, 2048, 4096]
    # mns 128 × dp8 = 1024: a final partial rung tops out at capacity.
    assert extend_ladder(base, 128, 8) == [8, 32, 128, 512, 1024]
    # Capacity at/below the base top: unchanged.
    assert extend_ladder(base, 64, 8) == base
    assert extend_ladder(base, 64, 2) == base
    # "default" / missing mns: unchanged, never an exception.
    assert extend_ladder(base, "default", 8) == base
    assert extend_ladder(base, None, 8) == base
    # Ladders stay strictly ascending (search-space invariant).
    for mns, dp in ((512, 8), (256, 4), (128, 8)):
        lad = extend_ladder(base, mns, dp)
        assert lad == sorted(set(lad))


def _ev(status, score, **params):
    from simulator.search import Evaluation
    return Evaluation(params=params, iteration=0, status=status,
                      score=score, config_name="c", cells=[])


def test_blocked_values_stops_burning_budget_on_a_rejected_flag() -> None:
    """nvfp4 KV burned 7 of 27 evaluations before the search gave up
    on it. Two failures with no success anywhere is enough to stop
    proposing that value."""
    from simulator.search import SearchState, blocked_values, is_blocked

    st = SearchState(space_hash="h")
    st.evaluated = {
        "a": _ev("ok", 8000.0, kv_cache_dtype="auto", dp=8),
        "b": _ev("launch_failed", None, kv_cache_dtype="nvfp4", dp=8),
        "c": _ev("launch_failed", None, kv_cache_dtype="nvfp4", dp=4),
    }
    blocked = blocked_values(st)
    assert blocked == {"kv_cache_dtype": {"nvfp4"}}
    assert is_blocked({"kv_cache_dtype": "nvfp4", "dp": 2}, blocked)
    assert not is_blocked({"kv_cache_dtype": "auto", "dp": 8}, blocked)
    # dp=8 failed once but also succeeded — never blocked.
    assert "dp" not in blocked


def test_blocked_values_needs_two_failures_and_a_success_somewhere() -> None:
    from simulator.search import SearchState, blocked_values

    st = SearchState(space_hash="h")
    # A single failure is not yet evidence.
    st.evaluated = {
        "a": _ev("ok", 1.0, kv_cache_dtype="auto"),
        "b": _ev("launch_failed", None, kv_cache_dtype="nvfp4"),
    }
    assert blocked_values(st) == {}
    # Nothing has succeeded at all: report that, don't block the world.
    st.evaluated = {
        "b": _ev("launch_failed", None, kv_cache_dtype="nvfp4"),
        "c": _ev("launch_failed", None, kv_cache_dtype="nvfp4"),
    }
    assert blocked_values(st) == {}


def test_optimizer_prompts_defeat_prefix_caching() -> None:
    """Identical prompts + vLLM's prefix caching made prefill nearly
    free, so the prefill-chunk dimension measured nothing. Every
    prompt must now differ from its FIRST characters."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from engine_optimizer import TestCell, make_prompts

    cells = [TestCell("ladder_c0008", input_tokens=128,
                      output_tokens=64, concurrency=8),
             TestCell("ladder_c0032", input_tokens=128,
                      output_tokens=64, concurrency=32)]
    prompts = make_prompts(cells)
    a = prompts["ladder_c0008"]
    assert len(set(a)) == len(a)                  # all unique
    # Divergent in the FIRST token — a shared leading block would
    # still be a prefix-cache hit.
    assert len({p[:8] for p in a}) == len(a)
    # Different rungs don't re-serve each other's cached prompts.
    assert not set(a) & set(prompts["ladder_c0032"])
