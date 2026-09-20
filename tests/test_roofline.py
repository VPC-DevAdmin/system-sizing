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
    State,
    cell_key,
    cells,
    estimate_minutes,
    load_state,
    pick_models,
    save_state,
    score_models,
    summarize,
    vendor_of,
)

CATALOG = [
    {"id": "org/moe-nvfp4", "quant": "nvfp4", "params_b": 35.0, "moe": True,
     "approx_size_gb": 21, "min_vram_gb": 24},
    {"id": "org/dense-bf16", "quant": "bf16", "params_b": 70.6, "moe": False,
     "approx_size_gb": 140, "min_vram_gb": 160},
    {"id": "org/huge", "quant": "fp8", "params_b": 685.0, "moe": True,
     "approx_size_gb": 687, "min_vram_gb": 5000},
]


# Four Qwen entries (two precisions of one family, two families) plus
# one each of three other vendor lines. The pure ranking puts every
# Qwen ahead of everything else: exactly the shortlist the XE7740 got.
VENDORS = [
    {"id": "Qwen/Qwen3-30B-A3B-FP8", "family": "qwen3-30b-a3b", "series": "Qwen3",
     "quant": "fp8", "params_b": 30.5, "moe": True, "approx_size_gb": 32,
     "min_vram_gb": 40},
    {"id": "Qwen/Qwen3-30B-A3B", "family": "qwen3-30b-a3b", "series": "Qwen3",
     "quant": "bf16", "params_b": 30.5, "moe": True, "approx_size_gb": 61,
     "min_vram_gb": 80},
    {"id": "Qwen/Qwen3.6-35B-NVFP4", "family": "qwen3.6-35b", "series": "Qwen3.6",
     "quant": "nvfp4", "params_b": 35.0, "moe": True, "approx_size_gb": 21,
     "min_vram_gb": 24},
    {"id": "Qwen/Qwen3-32B-FP8", "family": "qwen3-32b", "series": "Qwen3",
     "quant": "fp8", "params_b": 32.8, "moe": False, "approx_size_gb": 34,
     "min_vram_gb": 40},
    {"id": "openai/gpt-oss-120b", "family": "gpt-oss-120b", "series": "gpt-oss",
     "quant": "mxfp4", "params_b": 117.0, "moe": True, "approx_size_gb": 65,
     "min_vram_gb": 80},
    {"id": "zai-org/GLM-4.7-FP8", "family": "glm-4.7", "series": "GLM-4.7",
     "quant": "fp8", "params_b": 355.0, "moe": True, "approx_size_gb": 360,
     "min_vram_gb": 400},
    {"id": "meta-llama/Llama-3.3-70B-FP8", "family": "llama-3.3-70b",
     "series": "Llama-3.3", "quant": "fp8", "params_b": 70.6, "moe": False,
     "approx_size_gb": 70, "min_vram_gb": 80},
]


def test_candidates_carry_family_and_series():
    by_id = {c.id: c for c in score_models(VENDORS, vram_per_gpu_gb=96)}
    assert by_id["Qwen/Qwen3-30B-A3B-FP8"].series == "Qwen3"
    assert by_id["Qwen/Qwen3-30B-A3B-FP8"].family == "qwen3-30b-a3b"
    # Inferred when the catalog entry has neither.
    inferred = score_models([{"id": "org/Foo-9B-FP8", "params_b": 9}],
                            vram_per_gpu_gb=96)[0]
    assert inferred.family == "foo-9b" and inferred.series == "foo"


def test_pure_ranking_fills_the_shortlist_with_one_vendor():
    old = pick_models(VENDORS, vram_per_gpu_gb=96, limit=4, diverse=False)
    assert {vendor_of(c.series) for c in old} == {"Qwen"}
    assert [c.id for c in old] == [
        c.id for c in score_models(VENDORS, vram_per_gpu_gb=96)[:4]]
    assert all(c.pick_round == 0 for c in old)


def test_diverse_pick_takes_the_best_of_every_series_first():
    picks = pick_models(VENDORS, vram_per_gpu_gb=96, limit=4)
    # Qwen3 and Qwen3.6 are separate series but ONE vendor: Qwen gets
    # one round-one slot, not two.
    assert [c.series for c in picks] == ["Qwen3.6", "gpt-oss", "Llama-3.3", "GLM-4.7"]
    # The Qwen pick is still the ranker's favourite, and its round is
    # written into the reason the operator reads.
    assert picks[0].id == "Qwen/Qwen3.6-35B-NVFP4"
    assert picks[0].pick_round == 1
    assert picks[0].why.startswith("best of the Qwen line;")
    assert picks[3].why.startswith("best of the GLM line;")


def test_diverse_pick_returns_to_a_series_only_after_every_series_has_one():
    picks = pick_models(VENDORS, vram_per_gpu_gb=96, limit=6)
    assert [c.series for c in picks[:4]] == [
        "Qwen3.6", "gpt-oss", "Llama-3.3", "GLM-4.7"]
    assert [c.series for c in picks[4:]] == ["Qwen3", "Qwen3"]
    # Second and third Qwen picks are an unmeasured series first, then
    # different weights, before a second precision of weights already
    # on the list.
    assert picks[4].family != picks[0].family
    assert picks[5].family not in {picks[0].family, picks[4].family}
    assert picks[4].pick_round == 2 and picks[5].pick_round == 3
    assert picks[4].why.startswith("second pick from Qwen;")
    assert picks[5].why.startswith("third pick from Qwen;")


def test_diverse_pick_falls_back_to_a_precision_twin_last():
    picks = pick_models(VENDORS, vram_per_gpu_gb=96, limit=7)
    last = picks[-1]
    assert last.series == "Qwen3" and last.pick_round == 4
    assert last.family == "qwen3-30b-a3b"
    assert "another precision of qwen3-30b-a3b" in last.why
    assert len({c.id for c in picks}) == 7


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
    """Each repeated cell costs minutes of engine launch. A measured
    row carries the shape it ran with (input_tokens at least), and a
    planned cell of the same shape reuses it."""
    rows = [{"model": "m1", "engine": "e1", "max_num_seqs": 1,
             "output_tokens": 8, "input_tokens": 128,
             "out_tok_s": 1.0, "steady_state": True}]
    done = {cell_key(r) for r in rows}
    plan = cells(["m1"], ["e1"], {"max_num_seqs": [1], "output_tokens": [8]},
                 input_tokens=128)
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
    # Pick order: the tab's auto mode plans the first N of this list,
    # so each row says which round of the series round-robin chose it.
    assert d["diverse"] is True
    assert all(x["pick_round"] >= 1 for x in d["candidates"])
    assert all(x["series"] and x["family"] for x in d["candidates"])
    plain = c.get("/api/roofline/candidates?limit=3&diverse=false").json()
    assert plain["diverse"] is False
    assert all(x["pick_round"] == 0 for x in plain["candidates"])


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


def test_state_endpoint_reports_liveness_from_the_run_registry():
    """A state file left at 'searching' by a killed process must not
    read as running, or the page waits forever on a dead run."""
    from fastapi.testclient import TestClient

    from simulator.service import create_app

    app = create_app()
    c = TestClient(app)
    # No active run at all: the endpoint must still answer.
    d = c.get("/api/roofline").json()
    assert d.get("live") in (False, None)
    assert "status" in d


def test_a_cell_that_fails_the_same_way_twice_is_written_off():
    """An engine that cannot load a model architecture, or a kernel
    that refuses the GPU, will not succeed on the fourth attempt
    either. Retrying costs an engine launch every pass and never
    produces a number -- an autopilot left alone would spin on it."""
    from simulator.roofline import GIVE_UP_AFTER, permanently_failed

    base = {"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
            "output_tokens": 256}
    rows = [
        {**base, "error": "RuntimeError: DeepGEMM only supports Hopper "
                          "(SM90) (full log: runs/run_12/x.log)"},
        {**base, "error": "RuntimeError: DeepGEMM only supports Hopper "
                          "(SM90) (full log: runs/run_99/y.log)"},
    ]
    assert GIVE_UP_AFTER == 2
    out = permanently_failed(rows)
    assert len(out) == 1
    assert "DeepGEMM" in list(out.values())[0]


def test_a_cell_that_failed_once_is_still_retried():
    """A transient failure -- a port collision, a stale container --
    must not be mistaken for an impossibility."""
    from simulator.roofline import permanently_failed

    rows = [{"model": "m", "engine": "sglang_cuda", "max_num_seqs": 1024,
             "output_tokens": 256, "error": "EADDRINUSE"}]
    assert permanently_failed(rows) == {}


def test_two_different_failures_are_not_the_same_failure():
    """Failing twice for unrelated reasons is not evidence of
    impossibility."""
    from simulator.roofline import permanently_failed

    base = {"model": "m", "engine": "e", "max_num_seqs": 1,
            "output_tokens": 8}
    rows = [{**base, "error": "EADDRINUSE port 42128"},
            {**base, "error": "ValueError: unknown architecture"}]
    assert permanently_failed(rows) == {}


def test_signature_ignores_run_ids_and_ports():
    """The same failure reported with a different run directory is the
    same failure."""
    from simulator.roofline import error_signature

    a = error_signature("boom (full log: runs/run_12/engine_abc12345.log)")
    b = error_signature("boom (full log: runs/run_99/engine_def67890.log)")
    assert a == b


# ── Resume identity and write-off rules (improvement plan A8) ─────────


def test_cell_key_carries_the_full_launch_shape():
    """Changing the prompt length, the memory share, the KV precision
    or a lever and pressing Start must measure NEW cells. The old key
    was (model, engine, max_num_seqs, output_tokens) and reused cells
    measured under a different shape."""
    base = {"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
            "output_tokens": 256, "input_tokens": 128,
            "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8"}
    k = cell_key(base)
    assert cell_key({**base, "input_tokens": 512}) != k
    assert cell_key({**base, "gpu_memory_utilization": 0.9}) != k
    assert cell_key({**base, "kv_cache_dtype": "auto"}) != k
    assert cell_key({**base, "trtllm_moe_backend": "CUTLASS"}) != k
    assert cell_key({**base, "replicas": 4}) != k
    # Result fields are not identity.
    assert cell_key({**base, "out_tok_s": 1.0, "run_dir": "x"}) == k
    # A row from before the shape fields existed does not match a cell
    # planned now: nobody knows what shape it ran with.
    old = {"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
           "output_tokens": 256}
    assert cell_key(old) != k


def test_plan_cells_carry_the_shape_and_dedupe_identical_launches():
    plan = cells(["m"], ["trtllm"], {"max_num_seqs": [1024],
                                     "output_tokens": [128]},
                 input_tokens=512,
                 engine_shape={"gpu_memory_utilization": 0.95,
                               "kv_cache_dtype": "fp8", "replicas": 8,
                               "trtllm_moe_backend": "CUTLASS",
                               "model_id": "ignored", "device": "gpu"})
    assert len(plan) == 1
    c = plan[0]
    assert c["input_tokens"] == 512
    assert c["gpu_memory_utilization"] == 0.95
    assert c["kv_cache_dtype"] == "fp8"
    assert c["trtllm_moe_backend"] == "CUTLASS"
    assert "model_id" not in c and "device" not in c
    # ...and every one of them reaches the config builder.
    from simulator.roofline import cell_overrides
    ov = cell_overrides(c)
    assert ov["model_id"] == "m" and ov["engine"] == "trtllm"
    assert ov["max_num_seqs"] == 1024
    assert ov["kv_cache_dtype"] == "fp8" and ov["replicas"] == 8
    assert ov["trtllm_moe_backend"] == "CUTLASS"
    assert "input_tokens" not in ov          # a request property


def test_resume_reuses_only_cells_of_the_same_shape():
    rows = [{"model": "m1", "engine": "e1", "max_num_seqs": 1,
             "output_tokens": 8, "input_tokens": 128,
             "kv_cache_dtype": "fp8", "out_tok_s": 1.0,
             "steady_state": True}]
    done = {cell_key(r) for r in rows}
    same = cells(["m1"], ["e1"], {"max_num_seqs": [1], "output_tokens": [8]},
                 input_tokens=128, engine_shape={"kv_cache_dtype": "fp8"})
    assert cell_key(same[0]) in done
    longer = cells(["m1"], ["e1"], {"max_num_seqs": [1], "output_tokens": [8]},
                   input_tokens=2048, engine_shape={"kv_cache_dtype": "fp8"})
    assert cell_key(longer[0]) not in done


def test_startup_timeouts_and_smoke_failures_are_never_written_off():
    """Two slow launches are two slow launches, not an impossibility.
    Only a failure that says the shape cannot run here -- an unknown
    architecture, a kernel that refuses the GPU -- earns a write-off."""
    from simulator.roofline import is_transient, permanently_failed

    base = {"model": "m", "engine": "sglang_cuda", "max_num_seqs": 1024,
            "output_tokens": 256}
    timeouts = [
        {**base, "error": "TimeoutError: replica 0 not healthy in 1800s "
                          "— see runs/a/engine_sglang_cuda_1.log"},
        {**base, "error": "TimeoutError: replica 0 not healthy in 1800s "
                          "— see runs/b/engine_sglang_cuda_2.log"},
        {**base, "error": "TimeoutError: replica 0 not healthy in 1800s "
                          "— see runs/c/engine_sglang_cuda_3.log"},
    ]
    assert permanently_failed(timeouts) == {}
    smoke = [
        {**base, "error": "EngineBrokenError: smoke request to "
                          "http://127.0.0.1:9100/v1/chat/completions failed "
                          "before any load was offered: ReadTimeout"},
    ] * 3
    assert permanently_failed(smoke) == {}
    no_peak = [{**base, "error": "sweep produced no peak"}] * 3
    assert permanently_failed(no_peak) == {}
    ports = [{**base, "error": "RuntimeError: EADDRINUSE port 42128"}] * 3
    assert permanently_failed(ports) == {}
    for rows in (timeouts, smoke, no_peak, ports):
        assert is_transient(rows[0]["error"])
    # An architectural failure still is written off.
    arch = [{**base, "error": "RuntimeError: replica 2 container exited "
                              "during startup: ValueError: unknown "
                              "architecture Qwen3NextForCausalLM"}] * 2
    assert not is_transient(arch[0]["error"])
    assert len(permanently_failed(arch)) == 1


def test_signature_ignores_replica_index_and_timing():
    """The same failure from replica 3 and from replica 5 is one
    failure; so is one reported after 12.3 s and after 40.1 s."""
    from simulator.roofline import error_signature

    a = error_signature("[r3] RuntimeError: replica 3 container exited "
                        "during startup after 12.3s: DeepGEMM only "
                        "supports Hopper (SM90) (full log: runs/a/x.log)")
    b = error_signature("[r5] RuntimeError: replica 5 container exited "
                        "during startup after 40.1s: DeepGEMM only "
                        "supports Hopper (SM90) (full log: runs/b/y.log)")
    assert a == b
    assert "DeepGEMM" in a
    assert "replica 3" not in a and "12.3" not in a


def test_ktransformers_cells_get_their_own_defaults():
    """The GPU-engine defaults (eight replicas, fp8 KV) are refused by
    KTransformers at config time -- every cell failed and the matrix
    showed a blank for the one engine that can serve a model larger
    than VRAM. Its cells run one replica, no KV precision knob, and
    its documented batch width."""
    from simulator.engines.ktransformers import DOCUMENTED_MAX_BATCH
    from simulator.roofline import cell_overrides, engine_defaults

    assert engine_defaults("ktransformers") == {"replicas": 1,
                                                "kv_cache_dtype": "auto"}
    assert engine_defaults("trtllm") == {}
    plan = cells(["m"], ["ktransformers", "trtllm"],
                 {"max_num_seqs": [1024, 2048], "output_tokens": [128]},
                 engine_shape={"replicas": 8, "kv_cache_dtype": "fp8",
                               "gpu_memory_utilization": 0.95})
    kt = [c for c in plan if c["engine"] == "ktransformers"]
    trt = [c for c in plan if c["engine"] == "trtllm"]
    # Two batch widths clamp to one documented width -> one cell.
    assert len(kt) == 1
    assert kt[0]["max_num_seqs"] == DOCUMENTED_MAX_BATCH
    assert kt[0]["replicas"] == 1
    assert kt[0]["kv_cache_dtype"] == "auto"
    assert kt[0]["gpu_memory_utilization"] == 0.95     # untouched
    ov = cell_overrides(kt[0])
    assert ov["replicas"] == 1 and ov["kv_cache_dtype"] == "auto"
    # The GPU engines keep the GPU defaults.
    assert len(trt) == 2
    assert all(c["replicas"] == 8 and c["kv_cache_dtype"] == "fp8"
               for c in trt)
    # And the KTransformers cell passes the builder's refusal gate once
    # GGUF weights are staged (without them it is refused in
    # milliseconds rather than after a 30-minute health timeout).
    import tempfile

    from simulator.engines.custom import ShapeError, custom_engine
    hw = {"count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
          "vram_per_gpu_gb": 96.0}
    with pytest.raises(ShapeError, match="GGUF"):
        custom_engine({**ov, "model_id": "org/M", "device": "gpu", "tp": 1},
                      hw=hw)
    eng = custom_engine(
        {**ov, "model_id": "org/M", "device": "gpu", "tp": 1,
         "ktransformers_gguf_path": tempfile.mkdtemp(prefix="kt-gguf-")},
        hw=hw)
    assert eng["type"] == "ktransformers"
    assert eng["replica_devices"] == [[0]]
    assert eng["max_num_seqs"] == DOCUMENTED_MAX_BATCH


def test_start_honours_new_run_and_records_the_shape(tmp_path, monkeypatch):
    """The UI could never start a fresh roofline: new_run was ignored
    and the spec's resume defaulted true."""
    from fastapi.testclient import TestClient

    import simulator.arena as arena
    import simulator.roofline as rf
    from simulator.service import create_app

    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 96.0})
    captured: list[dict] = []

    async def fake_run_roofline(**kw):
        captured.append(kw)
        return tmp_path / "roofline.json"
    monkeypatch.setattr(rf, "run_roofline", fake_run_roofline)

    body = {"workload": {"kind": "roofline",
                         "spec": {"models": ["org/M"], "engines": ["trtllm"],
                                  "input_tokens": 512}},
            "custom": {"engine": "trtllm", "trtllm_moe_backend": "CUTLASS"}}
    with TestClient(create_app(tmp_path / "runs")) as c:
        r = c.post("/api/runs", json={**body, "new_run": True})
        assert r.status_code == 202, r.text
        import time
        for _ in range(50):
            if captured:
                break
            time.sleep(0.05)
        c.post("/api/runs/stop")
    assert captured and captured[0]["resume"] is False
    assert captured[0]["input_tokens"] == 512
    shape = captured[0]["engine_shape"]
    assert shape["kv_cache_dtype"] == "fp8" and shape["replicas"] == 8
    assert shape["trtllm_moe_backend"] == "CUTLASS"
    assert "model_id" not in shape

    captured.clear()
    with TestClient(create_app(tmp_path / "runs2")) as c:
        assert c.post("/api/runs", json=body).status_code == 202
        for _ in range(50):
            if captured:
                break
            time.sleep(0.05)
        c.post("/api/runs/stop")
    assert captured and captured[0]["resume"] is True
    # resume:false in the spec is the same as new_run.
    captured.clear()
    spec_off = {**body, "workload": {"kind": "roofline", "spec": {
        **body["workload"]["spec"], "resume": False}}}
    with TestClient(create_app(tmp_path / "runs3")) as c:
        assert c.post("/api/runs", json=spec_off).status_code == 202
        for _ in range(50):
            if captured:
                break
            time.sleep(0.05)
        c.post("/api/runs/stop")
    assert captured and captured[0]["resume"] is False


def test_start_without_a_template_defaults_engine_and_model(tmp_path, monkeypatch):
    """The Roofline tab sends no `custom` and lets the ranker pick the
    models; the service used to answer 422 'pass profile or config'
    (and, with a template but no model, 'custom.model_id must be an
    org/name id'). The template now defaults to the first planned
    engine and borrows the first picked model for the pre-flight."""
    from fastapi.testclient import TestClient

    import simulator.arena as arena
    import simulator.engine_runtimes as runtimes
    import simulator.roofline as rf
    from simulator.service import create_app

    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 96.0})
    monkeypatch.setattr(runtimes, "available_engines",
                        lambda: ["vllm_cuda_multi", "trtllm"])

    class _Pick:
        id = "org/Auto"
    monkeypatch.setattr(rf, "pick_models", lambda *a, **kw: [_Pick()])
    captured: list[dict] = []

    async def fake_run_roofline(**kw):
        captured.append(kw)
        return tmp_path / "roofline.json"
    monkeypatch.setattr(rf, "run_roofline", fake_run_roofline)

    body = {"workload": {"kind": "roofline",
                         "spec": {"models": None, "engines": None,
                                  "max_num_seqs": [1024], "output_tokens": [128],
                                  "input_tokens": 128}},
            "new_run": True}
    with TestClient(create_app(tmp_path / "runs")) as c:
        r = c.post("/api/runs", json=body)
        assert r.status_code == 202, r.text
        import time
        for _ in range(50):
            if captured:
                break
            time.sleep(0.05)
        c.post("/api/runs/stop")
    assert captured, "run_roofline was never reached"
    assert captured[0]["models"] == ["org/Auto"]
    assert captured[0]["engines"] == ["vllm_cuda_multi", "trtllm"]
    assert captured[0]["resume"] is False
    assert "model_id" not in captured[0]["engine_shape"]
