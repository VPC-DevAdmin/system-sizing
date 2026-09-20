"""Guided-search core (simulator/search.py): space loading, device
placement, coverage sampling, neighbor proposal, SLA-aware scoring,
and the resumable coarse-to-fine state machine."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from simulator.search import (
    Objective,
    SearchSpaceError,
    SearchState,
    assign_devices,
    candidate_summary,
    canonical_key,
    load_space,
    next_batch,
    propose_initial,
    propose_neighbors,
    record_evaluation,
    score_cells,
    summarize,
    validate_candidate,
)

SPACE_PATH = Path(__file__).parent.parent / "config" / "search" / "xe7740-qwen3.yaml"


@pytest.fixture()
def space():
    return load_space(SPACE_PATH)


# ── space loading ─────────────────────────────────────────────────────


def test_load_space(space) -> None:
    assert space.name == "xe7740-qwen3"
    assert space.total_devices == 8
    assert list(space.dimensions)[0] == "model_variant"
    assert space.objective.kind == "sla_throughput"
    assert space.search.budget == 40


def test_load_space_rejects_bad(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("engine: vllm_cuda\ndevice_groups: [[0]]\n"
                   "model_variants: {a: {model: m}}\n"
                   "dimensions: {warp_speed: [1, 2]}\n")
    with pytest.raises(SearchSpaceError, match="warp_speed"):
        load_space(bad)
    bad.write_text("engine: sglang\ndevice_groups: [[0]]\n"
                   "model_variants: {a: {model: m}}\ndimensions: {tp: [1]}\n")
    with pytest.raises(SearchSpaceError, match="not supported"):
        load_space(bad)


# ── placement ─────────────────────────────────────────────────────────


def test_assign_devices_pack_and_spread() -> None:
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # TP=2 DP=2 pack: both replicas inside group 0.
    assert assign_devices(2, 2, "pack", groups) == [[0, 1], [2, 3]]
    # spread: one replica per domain.
    assert assign_devices(2, 2, "spread", groups) == [[0, 1], [4, 5]]
    # TP=4 DP=2 fills one group per replica either way.
    assert assign_devices(4, 2, "pack", groups) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    # Doesn't fit: more devices than exist.
    assert assign_devices(4, 4, "pack", groups) is None
    # A TP set may not span groups: tp=8 impossible on 4+4.
    assert assign_devices(8, 1, "pack", groups) is None


def test_validate_candidate(space) -> None:
    ok, _ = validate_candidate(
        {"model_variant": "bf16", "tp": 2, "dp": 4,
         "gpu_memory_utilization": 0.90, "max_num_seqs": 256,
         "max_num_batched_tokens": "default", "placement": "pack"}, space)
    assert ok
    bad, reason = validate_candidate(
        {"model_variant": "bf16", "tp": 4, "dp": 4,
         "gpu_memory_utilization": 0.90, "max_num_seqs": 256,
         "max_num_batched_tokens": "default", "placement": "pack"}, space)
    assert not bad and "does not fit" in reason


def test_placement_normalized_on_single_device(space) -> None:
    a = {"model_variant": "bf16", "tp": 1, "dp": 1,
         "gpu_memory_utilization": 0.90, "max_num_seqs": 256,
         "max_num_batched_tokens": "default", "placement": "pack"}
    b = {**a, "placement": "spread"}
    assert canonical_key(a, space) == canonical_key(b, space)


# ── sampling and refinement ───────────────────────────────────────────


def test_initial_sample_is_valid_deduped_and_covering(space) -> None:
    batch = propose_initial(space, random.Random(42))
    assert len(batch) == space.search.initial_samples
    keys = {canonical_key(c, space) for c in batch}
    assert len(keys) == len(batch)
    for c in batch:
        ok, reason = validate_candidate(c, space)
        assert ok, reason
    # Coverage: every value of every dimension appears somewhere
    # (feasible here: 14 samples, max dimension size 4).
    for dim, vals in space.dimensions.items():
        seen = {str(c[dim]) for c in batch if dim in c}
        missing = {str(v) for v in vals} - seen
        # placement can be normalized away on 1-device candidates;
        # everything else must be fully covered.
        assert not missing or dim == "placement", (dim, missing)
    # Deterministic under the seed.
    again = propose_initial(space, random.Random(42))
    assert [canonical_key(c, space) for c in again] == \
           [canonical_key(c, space) for c in batch]


def test_neighbors_change_exactly_one_dimension(space) -> None:
    parent = {"model_variant": "bf16", "tp": 2, "dp": 2,
              "gpu_memory_utilization": 0.90, "max_num_seqs": 256,
              "max_num_batched_tokens": "default", "placement": "pack"}
    out = propose_neighbors(space, [parent], set(), limit=20)
    assert out
    for cand in out:
        diffs = [d for d in cand if str(cand[d]) != str(parent[d])]
        assert len(diffs) == 1, (cand, diffs)
        ok, reason = validate_candidate(cand, space)
        assert ok, reason
    # Ordinal dims step to ADJACENT values only: tp neighbors of 2 are 1 and 4.
    tps = {c["tp"] for c in out if c["tp"] != 2}
    assert tps <= {1, 4}
    # Dedup against already-measured keys.
    evaluated = {canonical_key(c, space) for c in out}
    again = propose_neighbors(space, [parent], evaluated, limit=20)
    assert not ({canonical_key(c, space) for c in again} & evaluated)


# ── scoring ───────────────────────────────────────────────────────────


def _cell(tps, ttft=1000.0, tpot=20.0, samples=8, errors=0, timeouts=0):
    return {"cell_name": "c", "samples": samples, "errors": errors,
            "timeouts": timeouts, "ttft_p95_ms": ttft, "tpot_p95_ms": tpot,
            "throughput_out_tok_s": tps}


def test_score_sla_throughput_penalizes_cap_violations() -> None:
    obj = Objective(kind="sla_throughput", ttft_p95_cap_ms=10000,
                    tpot_p95_cap_ms=100, penalty=0.25)
    clean = score_cells([_cell(1000), _cell(2000)], obj)
    assert clean == 3000
    slow = score_cells([_cell(1000), _cell(2000, ttft=60000)], obj)
    assert slow == 1000 + 2000 * 0.25
    # Failed requests bleed score via the completion fraction.
    flaky = score_cells([_cell(1000, samples=6, errors=2)], obj)
    assert flaky == 1000 * 6 / 8
    # Pure throughput ignores caps.
    assert score_cells([_cell(2000, ttft=60000)],
                       Objective(kind="throughput")) == 2000
    # Latency objective: higher (less negative) is better.
    fast = score_cells([_cell(1000, ttft=500)], Objective(kind="latency"))
    slow_l = score_cells([_cell(1000, ttft=5000)], Objective(kind="latency"))
    assert fast > slow_l
    assert score_cells([], obj) is None


# ── state machine ─────────────────────────────────────────────────────


def _drive(space, scores_by_key, max_rounds=20):
    """Run the state machine with a fake evaluator that scores each
    candidate via ``scores_by_key`` (default 100)."""
    state = SearchState(space_hash=space.space_hash())
    rng = random.Random(space.search.seed)
    rounds = []
    for _ in range(max_rounds):
        kind, batch = next_batch(state, space, rng)
        if kind.startswith("done"):
            return state, rounds, kind
        rounds.append((kind, len(batch)))
        for params in batch:
            key = canonical_key(params, space)
            record_evaluation(
                state, space, params, status="ok",
                score=scores_by_key.get(key, 100.0),
                config_name=key, cells=[_cell(100)],
                iteration=state.iterations[-1]["index"],
            )
    raise AssertionError("state machine did not terminate")


def test_state_machine_converges_when_flat(space) -> None:
    # Every candidate scores the same → refine once, see no
    # improvement, stop.
    state, rounds, done = _drive(space, {})
    assert rounds[0] == ("initial", space.search.initial_samples)
    assert rounds[1][0] == "refine"
    assert done.startswith("done:converged")
    assert summarize(state)["best"]["score"] == 100.0


def test_state_machine_respects_budget(space) -> None:
    # Escalating scores → never converges; must stop on budget or
    # max_iterations, never exceeding budget.
    counter = {"n": 0}

    class Escalating(dict):
        def get(self, key, default=None):
            counter["n"] += 1
            return float(counter["n"])
    state, _rounds, done = _drive(space, Escalating())
    assert done.startswith("done:budget") or done.startswith("done:max_iterations")
    assert len(state.evaluated) <= space.search.budget


def test_state_machine_resumes_pending(space) -> None:
    state = SearchState(space_hash=space.space_hash())
    rng = random.Random(1)
    kind, batch = next_batch(state, space, rng)
    assert kind == "initial"
    # Evaluate only half, then ask again: the same pending candidates
    # come back with the same kind, no new iteration is opened.
    for params in batch[: len(batch) // 2]:
        record_evaluation(state, space, params, status="ok", score=1.0,
                          config_name="x", cells=[], iteration=0)
    kind2, batch2 = next_batch(state, space, rng)
    assert kind2 == "initial"
    assert len(batch2) == len(batch) - len(batch) // 2
    assert len(state.iterations) == 1


def test_state_machine_refuses_changed_space(space) -> None:
    state = SearchState(space_hash="deadbeef00000000")
    kind, batch = next_batch(state, space, random.Random(1))
    assert kind.startswith("done:space_changed") and not batch


def test_state_roundtrip_and_summary(space) -> None:
    state, _r, _d = _drive(space, {})
    again = SearchState.from_dict(state.to_dict())
    assert summarize(again) == summarize(state)


# ── candidate → engine view ───────────────────────────────────────────


def test_candidate_summary(space) -> None:
    view = candidate_summary(
        {"model_variant": "fp8", "tp": 2, "dp": 2,
         "gpu_memory_utilization": 0.95, "max_num_seqs": 128,
         "max_num_batched_tokens": 2048, "placement": "spread"}, space)
    assert view["model"].endswith("-FP8")
    assert view["served_name"] == "qwen3_30b_a3b"
    assert view["replica_devices"] == [[0, 1], [4, 5]]
    joined = " ".join(view["engine_args"])
    assert "--tensor-parallel-size 2" in joined
    assert "--max-num-seqs 128" in joined
    assert "--max-num-batched-tokens 2048" in joined
    assert "--gpu-memory-utilization 0.95" in joined
    # 'default' sentinel emits no flag.
    view2 = candidate_summary(
        {"model_variant": "bf16", "tp": 1, "dp": 1,
         "gpu_memory_utilization": 0.90, "max_num_seqs": 256,
         "max_num_batched_tokens": "default", "placement": "pack"}, space)
    assert "--max-num-batched-tokens" not in " ".join(view2["engine_args"])
    assert "--tensor-parallel-size" not in " ".join(view2["engine_args"])


def test_engine_foreign_levers_collapse_to_their_defaults(tmp_path) -> None:
    """A TensorRT-LLM lever on a vLLM candidate changes nothing about
    the launch, so two candidates differing only in it are the SAME
    candidate. Without this the coverage sampler chased four copies of
    every vLLM shape (improvement plan A7)."""
    import random

    from simulator.search import normalize

    p = tmp_path / "space.yaml"
    p.write_text("""
name: multi
engine: vllm_cuda
device_groups: [[0, 1, 2, 3]]
model_variants:
  m: {model: org/model, served_name: m}
dimensions:
  engine: [vllm_cuda_multi, trtllm]
  tp: [1]
  dp: [4]
  trtllm_moe_backend: [CUTLASS, auto]
  trtllm_chunked_prefill: ['on', 'off']
search: {budget: 8, initial_samples: 4}
""")
    space = load_space(p)
    base = {"model_variant": "m", "tp": 1, "dp": 4}
    a = canonical_key({**base, "engine": "vllm_cuda_multi",
                       "trtllm_moe_backend": "CUTLASS",
                       "trtllm_chunked_prefill": "on"}, space)
    b = canonical_key({**base, "engine": "vllm_cuda_multi",
                       "trtllm_moe_backend": "auto",
                       "trtllm_chunked_prefill": "off"}, space)
    assert a == b
    # Collapsed to the engine's own default, not to whichever value
    # happened to be listed first.
    n = normalize({**base, "engine": "vllm_cuda_multi",
                   "trtllm_moe_backend": "CUTLASS",
                   "trtllm_chunked_prefill": "on"}, space)
    assert n["trtllm_moe_backend"] == "auto"
    assert n["trtllm_chunked_prefill"] == "off"
    # On the owning engine the lever is real and the keys differ.
    c = canonical_key({**base, "engine": "trtllm",
                       "trtllm_moe_backend": "CUTLASS"}, space)
    d = canonical_key({**base, "engine": "trtllm",
                       "trtllm_moe_backend": "auto"}, space)
    assert c != d
    # The coverage sample never proposes two copies of one launch.
    picked = propose_initial(space, random.Random(1))
    keys = [canonical_key(x, space) for x in picked]
    assert len(keys) == len(set(keys))


def test_fixed_values_pin_a_dimension_for_every_candidate(tmp_path) -> None:
    """``fixed:`` carries the arena's gpu_memory_utilization into the
    space document, and the built-in default IS the arena's constant
    rather than a second number kept in the search."""
    from simulator.arena import FIXED_GMU

    p = tmp_path / "space.yaml"
    p.write_text("""
name: f
engine: vllm_cuda
device_groups: [[0, 1]]
model_variants:
  m: {model: org/model, served_name: m}
dimensions:
  tp: [1]
fixed:
  gpu_memory_utilization: 0.8
search: {budget: 2, initial_samples: 1}
""")
    space = load_space(p)
    view = candidate_summary({"model_variant": "m", "tp": 1}, space)
    assert view["gpu_memory_utilization"] == 0.8
    # The pin is part of the fingerprint -- a different fixed value
    # is a different space.
    p.write_text(p.read_text().replace("0.8", "0.7"))
    assert load_space(p).space_hash() != space.space_hash()
    # Without a pin the default is the arena's advertised constant.
    p.write_text(p.read_text().replace(
        "fixed:\n  gpu_memory_utilization: 0.7\n", ""))
    view = candidate_summary({"model_variant": "m", "tp": 1}, load_space(p))
    assert view["gpu_memory_utilization"] == FIXED_GMU
    # A value cannot be both searched and pinned.
    p.write_text(p.read_text().replace(
        "dimensions:\n  tp: [1]\n",
        "dimensions:\n  tp: [1]\n  gpu_memory_utilization: [0.9]\n"
        "fixed:\n  gpu_memory_utilization: 0.8\n"))
    with pytest.raises(SearchSpaceError):
        load_space(p)
