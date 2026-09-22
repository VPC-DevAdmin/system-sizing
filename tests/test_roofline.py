"""Roofline autopilot: ranking, planning, persistence, resume.

The thing this module has to get right is not speed, it is SURVIVING.
A roofline run takes hours, the operator closes a laptop, and the page
they come back to over a VPN has to be correct without having watched
anything happen.
"""

from __future__ import annotations

import asyncio
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
    old = pick_models(VENDORS, vram_per_gpu_gb=96, limit=4, diverse=False,
                      spectrum=False)
    assert {vendor_of(c.series) for c in old} == {"Qwen"}
    assert [c.id for c in old] == [
        c.id for c in score_models(VENDORS, vram_per_gpu_gb=96)[:4]]
    assert all(c.pick_round == 0 for c in old)


def test_diverse_pick_takes_the_best_of_every_series_first():
    picks = pick_models(VENDORS, vram_per_gpu_gb=96, limit=4, spectrum=False)
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
    picks = pick_models(VENDORS, vram_per_gpu_gb=96, limit=6, spectrum=False)
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
    picks = pick_models(VENDORS, vram_per_gpu_gb=96, limit=7, spectrum=False)
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
    assert d["spectrum"] is True
    # Every pick says its tier; the FAST ones say which round of the
    # vendor round-robin chose them.
    assert all(x["tier"] in ("fast", "large", "beyond_vram")
               for x in d["candidates"])
    assert all(x["pick_round"] >= 1 for x in d["candidates"]
               if x["tier"] == "fast")
    assert all(x["series"] and x["family"] for x in d["candidates"])
    # And carries what decides its engines and shape.
    assert all({"fits_gpu", "kt_eligible", "fits_ram", "tp", "replicas"}
               <= set(x) for x in d["candidates"])
    assert "host_ram_gb" in d["hardware"]
    plain = c.get("/api/roofline/candidates?limit=3&diverse=false"
                  "&spectrum=false").json()
    assert plain["diverse"] is False
    assert all(x["pick_round"] == 0 for x in plain["candidates"])
    assert all(x["tier"] == "fast" for x in plain["candidates"])
    # ``extra`` appends unpicked models (tier "") for the greyed rows.
    more = c.get("/api/roofline/candidates?limit=2&extra=3").json()
    assert 2 < len(more["candidates"]) <= 5
    assert [x["tier"] for x in more["candidates"]][2:] == [""] * (
        len(more["candidates"]) - 2)


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
                                                "kv_cache_dtype": "auto",
                                                "max_model_len": 4096}
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


# ── Spectrum: fastest, largest the GPUs hold, beyond VRAM ─────────────

# VENDORS plus the two ends of the spectrum: a 322 GB model that needs
# four cards, and a 700 GB one no card set holds but KTransformers can
# serve from RAM because a GGUF companion is catalogued.
SPECTRUM = VENDORS + [
    # Ranked below GLM-4.7 (more parameters, no better precision) so
    # the GLM vendor's FAST pick is settled and this one is free for
    # the LARGE tier.
    {"id": "zai-org/GLM-5.3-Flash", "family": "glm-5.3-flash", "series": "GLM-5",
     "quant": "fp8", "params_b": 744.0, "moe": True, "approx_size_gb": 322,
     "min_vram_gb": 350},
    {"id": "deepseek-ai/DeepSeek-V4", "family": "deepseek-v4",
     "series": "DeepSeek V4", "quant": "fp8", "params_b": 1000.0, "moe": True,
     "approx_size_gb": 700, "min_vram_gb": 5000,
     "gguf": {"repo": "u/DeepSeek-V4-GGUF", "file": "q4.gguf", "size_gb": 380}},
    {"id": "org/huge-no-gguf", "family": "huge", "series": "Huge",
     "quant": "fp8", "params_b": 1200.0, "moe": True, "approx_size_gb": 900,
     "min_vram_gb": 5000},
]


def test_candidates_know_how_they_fit():
    by = {c.id: c for c in score_models(SPECTRUM, vram_per_gpu_gb=96,
                                        host_ram_gb=2048)}
    small = by["Qwen/Qwen3-30B-A3B-FP8"]
    assert small.fits_gpu and small.tp == 1 and small.replicas == 8
    big = by["zai-org/GLM-5.3-Flash"]
    assert big.fits_gpu and big.tp == 4 and big.replicas == 2
    beyond = by["deepseek-ai/DeepSeek-V4"]
    assert not beyond.fits_gpu and beyond.tp is None
    assert beyond.kt_eligible and beyond.fits_ram and beyond.fits
    assert beyond.score == 0.0 and "KTransformers" in beyond.why
    nope = by["org/huge-no-gguf"]
    assert not nope.kt_eligible and not nope.fits
    assert "no GGUF" in nope.why


def test_ram_budget_decides_beyond_vram():
    """700 GB of weights fit 85% of 2 TB but not of 512 GB; and with
    no RAM figure at all nothing is beyond VRAM, only beyond reach."""
    def dsv4(**kw):
        return {c.id: c for c in score_models(SPECTRUM, vram_per_gpu_gb=96,
                                              **kw)}["deepseek-ai/DeepSeek-V4"]
    assert dsv4(host_ram_gb=2048).fits_ram
    assert not dsv4(host_ram_gb=512).fits_ram
    assert "exceed 85%" in dsv4(host_ram_gb=512).why
    assert not dsv4().fits and "RAM is unknown" in dsv4().why


def test_spectrum_pick_fills_three_tiers():
    picks = pick_models(SPECTRUM, max_tp=4, vram_per_gpu_gb=96, host_ram_gb=2048,
                        limit=8, large_limit=2, beyond_limit=1)
    tiers = {c.id: c.tier for c in picks}
    assert tiers["deepseek-ai/DeepSeek-V4"] == "beyond_vram"
    assert "org/huge-no-gguf" not in tiers
    # LARGE and BEYOND are settled before FAST, so a vendor's largest
    # model is never consumed as its "fastest" pick. With TP capped at
    # the 4-card domain the 360 GB GLM-4.7-FP8 (min 400 GB) does not
    # fit at all; the 322 GB GLM-5.3 is the largest, then another
    # vendor's largest (the 70 GB Llama).
    large = [c for c in picks if c.tier == "large"]
    assert [c.id for c in large][0] == "zai-org/GLM-5.3-Flash"
    assert large[0].tp == 4 and large[0].replicas == 2
    assert len({vendor_of(c.series) for c in large}) == 2
    assert "zai-org/GLM-4.7-FP8" not in tiers
    fast = [c for c in picks if c.tier == "fast"]
    # FAST fills the remaining 8 - 2 - 1 = 5 slots, vendor round-robin
    # over what LARGE left (Qwen and gpt-oss): both in round one, then
    # Qwen's second, third and fourth.
    assert len(fast) == 5
    assert {vendor_of(c.series) for c in fast} == {"Qwen", "gpt-oss"}
    assert sorted(c.pick_round for c in fast) == [1, 1, 2, 3, 4]
    # The order is fast, large, beyond -- the tiers are labelled.
    assert [c.tier for c in picks] == ["fast"] * 5 + ["large"] * 2 + ["beyond_vram"]
    dsv4 = picks[-1]
    assert dsv4.why == ("beyond VRAM — KTransformers / llama.cpp only, "
                        "700 GB of weights in 2.048 TB of RAM")
    assert large[0].why.startswith("largest that fits the GPUs: 322 GB at tp4")
    big = large[1]
    assert big.why.startswith(f"largest that fits the GPUs: {big.size_gb} GB "
                              f"at tp{big.tp}")


def test_large_tier_reports_tp_and_replicas():
    picks = pick_models(SPECTRUM, max_tp=4, vram_per_gpu_gb=96, host_ram_gb=2048,
                        limit=6, large_limit=1, beyond_limit=1)
    assert [c.tier for c in picks] == ["fast"] * 4 + ["large", "beyond_vram"]
    glm = picks[4]
    assert glm.id == "zai-org/GLM-5.3-Flash"
    assert glm.tp == 4 and glm.replicas == 2
    assert "322 GB at tp4" in glm.why


def test_without_ram_the_beyond_tier_is_empty_and_fast_takes_the_slot():
    picks = pick_models(SPECTRUM, vram_per_gpu_gb=96, limit=8,
                        large_limit=2, beyond_limit=1)
    assert "deepseek-ai/DeepSeek-V4" not in {c.id for c in picks}
    assert len(picks) == 8
    assert sum(c.tier == "fast" for c in picks) == 6
    # The top-up continues the round-robin rather than restarting it:
    # a vendor's second pick says so.
    assert any(c.pick_round == 2 for c in picks if c.tier == "fast")


def test_spectrum_false_is_the_fast_tier_alone():
    picks = pick_models(SPECTRUM, vram_per_gpu_gb=96, host_ram_gb=2048,
                        limit=8, spectrum=False)
    assert all(c.tier == "fast" for c in picks)
    assert "deepseek-ai/DeepSeek-V4" not in {c.id for c in picks}


def test_cells_choose_engines_per_model():
    """GPU engines only where the weights fit the cards; KTransformers
    only where a GGUF companion exists; a beyond-VRAM model gets
    KTransformers cells alone; a model needing four cards launches at
    tp4 x 2 replicas rather than the tp1 x 8 default."""
    from simulator.roofline import KT_MAX_MODEL_LEN
    picks = pick_models(SPECTRUM, vram_per_gpu_gb=96, host_ram_gb=2048,
                        limit=8, large_limit=2, beyond_limit=1)
    info = {c.id: c.info() for c in picks}
    notes: list[str] = []
    plan = cells([c.id for c in picks], ["vllm_cuda_multi", "ktransformers"],
                 {"max_num_seqs": [1024], "output_tokens": [128]},
                 engine_shape={"replicas": 8, "tp": 1, "kv_cache_dtype": "fp8"},
                 model_info=info, notes=notes)
    by = {}
    for c in plan:
        by.setdefault(c["model"], {}).setdefault(c["engine"], []).append(c)
    dsv4 = by["deepseek-ai/DeepSeek-V4"]
    assert set(dsv4) == {"ktransformers"}
    kt = dsv4["ktransformers"][0]
    assert kt["replicas"] == 1 and kt["max_model_len"] == KT_MAX_MODEL_LEN
    assert kt["max_num_seqs"] == 4
    assert "tp" not in kt or kt["tp"] == 1
    # A GPU-only model without a companion: no KTransformers cell, and
    # the plan says why.
    gpt = by["openai/gpt-oss-120b"]
    assert set(gpt) == {"vllm_cuda_multi"}
    assert any("gpt-oss-120b" in n and "no GGUF companion" in n for n in notes)
    assert any("DeepSeek-V4: no vllm_cuda_multi cells" in n for n in notes)
    # The 322 GB GLM needs four cards: tp4, two replicas, in the key.
    glm = by["zai-org/GLM-5.3-Flash"]["vllm_cuda_multi"][0]
    assert glm["tp"] == 4 and glm["replicas"] == 2
    assert "tp=4" in cell_key(glm) and "replicas=2" in cell_key(glm)
    from simulator.roofline import cell_overrides
    assert cell_overrides(glm)["tp"] == 4
    # And the 360 GB one needs every card: tp8, one replica.
    glm47 = by["zai-org/GLM-4.7-FP8"]["vllm_cuda_multi"][0]
    assert glm47["tp"] == 8 and glm47["replicas"] == 1
    # A one-card model keeps the shared shape.
    small = by["Qwen/Qwen3-30B-A3B-FP8"]["vllm_cuda_multi"][0]
    assert small["tp"] == 1 and small["replicas"] == 8
    # Without model_info the old behaviour holds: every engine.
    old = cells(["x"], ["vllm_cuda_multi", "ktransformers"],
                {"max_num_seqs": [1024], "output_tokens": [128]})
    assert {c["engine"] for c in old} == {"vllm_cuda_multi", "ktransformers"}


def test_a_tp8_model_gets_one_replica():
    from simulator.roofline import tp_for
    assert tp_for(730, 96) == 8
    assert tp_for(350, 96) == 4
    assert tp_for(97, 96) == 2
    assert tp_for(96, 96) == 1
    assert tp_for(800, 96) is None
    by = {c.id: c for c in score_models(
        [{"id": "o/tp8", "approx_size_gb": 687, "min_vram_gb": 730}],
        vram_per_gpu_gb=96)}
    assert by["o/tp8"].tp == 8 and by["o/tp8"].replicas == 1


def test_summary_names_fastest_and_largest_and_draws_the_spectrum():
    info = {
        "o/small": {"tier": "fast", "approx_size_gb": 30, "params_b": 30,
                    "vendor": "O"},
        "o/big": {"tier": "large", "approx_size_gb": 322, "params_b": 355,
                  "vendor": "O"},
        "o/beyond": {"tier": "beyond_vram", "approx_size_gb": 700,
                     "params_b": 1000, "vendor": "O"},
        "o/pending": {"tier": "fast", "approx_size_gb": 60, "params_b": 60,
                      "vendor": "O"},
    }
    rows = [
        {"model": "o/small", "engine": "vllm", "out_tok_s": 5000.0,
         "total_tok_s": 9000.0, "concurrency": 2048, "kv_cache_tokens": 1e6,
         "ttft_p95_ms": 900.0, "steady_state": True},
        {"model": "o/small", "engine": "trt", "out_tok_s": 5200.0,
         "total_tok_s": 8000.0, "steady_state": True},
        {"model": "o/big", "engine": "vllm", "out_tok_s": 800.0,
         "total_tok_s": 1500.0, "steady_state": True},
        {"model": "o/beyond", "engine": "ktransformers",
         "error": "RuntimeError: boom"},
    ]
    s = summarize(rows, model_info=info,
                  models=["o/small", "o/big", "o/beyond", "o/pending"])
    # Fastest is by TOTAL tok/s (the vllm cell), while ``best`` keeps
    # its output-rate meaning (the trt cell).
    # Ranked on GENERATION tokens: trt generated more even though vllm
    # moved more total (prompt + generation) tokens.
    assert s["fastest"]["engine"] == "trt"
    assert s["best"]["engine"] == "trt"
    assert s["largest_served"]["model"] == "o/big"
    assert s["largest_served"]["best_engine"] == "vllm"
    assert s["largest_served"]["total_tok_s"] == 1500.0
    assert s["largest_attempted"]["model"] == "o/beyond"
    assert s["largest_attempted"]["status"] == "failed"
    spec = s["spectrum"]
    assert [r["model"] for r in spec] == ["o/small", "o/pending", "o/big",
                                          "o/beyond"]
    assert [r["status"] for r in spec] == ["served", "pending", "served",
                                           "failed"]
    small = spec[0]
    assert small["tier"] == "fast" and small["best_engine"] == "trt"
    assert small["out_tok_s"] == 5200.0 and small["total_tok_s"] == 8000.0
    # The kv/ttft detail came from the vllm cell; trt won on generation
    # and carries none, so the row reports what its best cell measured.
    assert small["kv_capacity_tokens"] is None and small["ttft_p95_ms"] is None
    assert {"model", "vendor", "params_b", "approx_size_gb", "tier",
            "best_engine", "out_tok_s", "total_tok_s", "concurrency",
            "kv_capacity_tokens", "ttft_p95_ms", "status"} == set(small)
    # A model that could not be staged says so.
    s2 = summarize(rows, model_info=info, models=["o/pending"],
                   staging={"o/pending": "unavailable"})
    assert next(r for r in s2["spectrum"] if r["model"] == "o/pending"
                )["status"] == "unavailable"
    # Without a plan record the spectrum still lists what was measured.
    bare = summarize(rows)
    assert {r["model"] for r in bare["spectrum"]} == {"o/small", "o/big",
                                                       "o/beyond"}
    assert bare["largest_served"]["model"] in ("o/small", "o/big")


def test_state_carries_the_plan_record_into_the_summary(tmp_path):
    st = State(plan={"models": ["a", "b"], "engines": ["e"],
                     "model_info": {"a": {"tier": "fast", "approx_size_gb": 10},
                                    "b": {"tier": "large",
                                          "approx_size_gb": 300}},
                     "notes": ["b: no ktransformers cells — no GGUF companion "
                               "staged"]},
               models=[{"id": "a", "status": "cached"},
                       {"id": "b", "status": "unavailable"}],
               results=[{"model": "a", "engine": "e", "max_num_seqs": 1,
                         "output_tokens": 8, "out_tok_s": 10.0,
                         "total_tok_s": 20.0}])
    save_state(tmp_path / "r.json", st)
    d = json.loads((tmp_path / "r.json").read_text())
    assert d["summary"]["fastest"]["model"] == "a"
    assert d["summary"]["largest_served"]["model"] == "a"
    assert [r["status"] for r in d["summary"]["spectrum"]] == [
        "served", "unavailable"]
    assert d["plan"]["notes"]


def test_roofline_state_endpoint_carries_the_spectrum_fields():
    from fastapi.testclient import TestClient

    from simulator.service import create_app

    d = TestClient(create_app()).get("/api/roofline").json()
    assert d["status"] == "none"
    for k in ("fastest", "largest_served", "largest_attempted"):
        assert d["summary"][k] is None
    assert d["summary"]["spectrum"] == []


def test_roofline_request_defaults_to_the_spectrum():
    from simulator.service.schemas import RooflineRequest
    r = RooflineRequest()
    assert (r.model_limit, r.large_limit, r.beyond_limit) == (8, 3, 2)
    assert r.spectrum is True and r.diverse is True


def test_plan_carries_model_info_and_picks_engines(tmp_path, monkeypatch):
    """The auto plan hands run_roofline the per-model record (tier,
    fits_gpu, kt_eligible, tp) it needs to choose engines, and the
    beyond-VRAM tier is only requested when KTransformers is among the
    engines that will run."""
    from fastapi.testclient import TestClient

    import simulator.arena as arena
    import simulator.engine_runtimes as runtimes
    import simulator.model_catalog as mc
    import simulator.roofline as rf
    from simulator.service import create_app

    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 96.0, "host_ram_gb": 2048.0})
    monkeypatch.setattr(runtimes, "available_engines",
                        lambda: ["vllm_cuda_multi", "ktransformers"])
    monkeypatch.setattr(mc, "load_model_catalog", lambda: SPECTRUM)
    captured: list[dict] = []

    async def fake_run_roofline(**kw):
        captured.append(kw)
        return tmp_path / "roofline.json"
    monkeypatch.setattr(rf, "run_roofline", fake_run_roofline)

    body = {"workload": {"kind": "roofline",
                         "spec": {"models": None, "engines": None,
                                  "model_limit": 8, "large_limit": 2,
                                  "beyond_limit": 1,
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
    kw = captured[0]
    info = kw["model_info"]
    assert set(info) == set(kw["models"])
    assert info["deepseek-ai/DeepSeek-V4"]["tier"] == "beyond_vram"
    assert info["deepseek-ai/DeepSeek-V4"]["fits_gpu"] is False
    assert info["zai-org/GLM-5.3-Flash"]["tp"] == 4
    assert info["zai-org/GLM-5.3-Flash"]["tier"] == "large"
    # The shared shape stays tp1 x 8 even though the pre-flight had to
    # validate a tp4 model.
    assert kw["engine_shape"]["tp"] == 1 and kw["engine_shape"]["replicas"] == 8

    # Without KTransformers in the run, nothing beyond VRAM is picked.
    captured.clear()
    body["workload"]["spec"]["engines"] = ["vllm_cuda_multi"]
    with TestClient(create_app(tmp_path / "runs2")) as c:
        assert c.post("/api/runs", json=body).status_code == 202
        import time
        for _ in range(50):
            if captured:
                break
            time.sleep(0.05)
        c.post("/api/runs/stop")
    assert captured
    assert "deepseek-ai/DeepSeek-V4" not in captured[0]["models"]
    assert all(v["fits_gpu"] for v in captured[0]["model_info"].values())


def test_tensor_parallel_never_spans_a_device_group():
    """TP peers stay inside one PCIe/NUMA domain (cross-domain
    all-reduce is never the answer), so a 687 GB model on a box with
    two groups of four 96 GB cards does NOT fit at tp8 -- it fits
    only KTransformers, and only with a GGUF companion."""
    from simulator.roofline import max_tp_of, tp_for

    hw = {"count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
          "vram_per_gpu_gb": 96.0}
    assert max_tp_of(hw) == 4
    assert tp_for(322, 96, 8, max_tp=4) == 4
    assert tp_for(687, 96, 8) == 8
    assert tp_for(687, 96, 8, max_tp=4) is None
    big = [{"id": "org/Huge", "family": "huge", "series": "Huge", "quant": "fp8",
            "params_b": 671, "moe": True, "approx_size_gb": 687, "min_vram_gb": 730}]
    c = score_models(big, vram_per_gpu_gb=96, gpu_count=8, max_tp=4)[0]
    assert c.fits_gpu is False and c.fits is False
    c = score_models(big, vram_per_gpu_gb=96, gpu_count=8)[0]
    assert c.fits_gpu is True and c.tp == 8


def test_cross_domain_tp_is_an_explicit_opt_in():
    """tp8 across both PCIe domains is off by default (slow all-reduce)
    and on request holds the NVFP4 giants on the GPUs, saying so."""
    giants = [{"id": "nvidia/Kimi-K2-Thinking-NVFP4", "family": "kimi-k2-thinking",
               "series": "Kimi K2", "quant": "nvfp4", "params_b": 1026, "moe": True,
               "approx_size_gb": 594, "min_vram_gb": 620}]
    capped = score_models(giants, vram_per_gpu_gb=96, gpu_count=8, max_tp=4)[0]
    assert capped.fits_gpu is False
    lifted = score_models(giants, vram_per_gpu_gb=96, gpu_count=8, max_tp=None)[0]
    assert lifted.fits_gpu is True and lifted.tp == 8 and lifted.replicas == 1
    assert "spans both PCIe domains" in lifted.why


def test_ktransformers_is_never_scheduled_for_a_dense_model():
    """KeyError: 'LlamaForCausalLM' on the XE7740: the v0.3 image is an
    MoE engine. A dense model with a GGUF keeps llama.cpp only."""
    dense = [{"id": "m/Llama-70B", "family": "llama-70b", "series": "Llama",
              "quant": "fp8", "params_b": 70, "moe": False, "approx_size_gb": 70,
              "min_vram_gb": 80, "gguf": {"repo": "u/L-GGUF", "file": "L.gguf"}}]
    c = score_models(dense, vram_per_gpu_gb=96, gpu_count=8, host_ram_gb=2048)[0]
    assert c.kt_eligible is True
    assert "ktransformers" not in c.gguf_engines and "llamacpp" in c.gguf_engines
    from simulator.roofline import is_transient
    assert is_transient("RuntimeError: replica 0 container exited during startup: "
                        "FileNotFoundError: [Errno 2] No such file or directory: '/gguf/x'")


def test_resume_can_retry_an_engines_failed_cells(tmp_path):
    """After a launcher fix the operator asks for the failed TensorRT-LLM
    cells back; measured cells and other engines' failures stay."""
    from simulator.roofline import State, save_state

    st = State()
    st.plan = {"cells": [{"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
                          "output_tokens": 128}]}
    st.results = [
        {"model": "m", "engine": "trtllm", "max_num_seqs": 1024, "output_tokens": 128,
         "error": "RuntimeError: Executor creation failed due to insufficient GPU memory."},
        {"model": "m", "engine": "trtllm", "max_num_seqs": 2048, "output_tokens": 128,
         "out_tok_s": 100.0},
        {"model": "m", "engine": "sglang_cuda", "max_num_seqs": 1024, "output_tokens": 128,
         "error": "boom"},
    ]
    save_state(tmp_path / "roofline.json", st)
    kept = [r for r in st.results if not (r.get("error") and r["engine"] in ["trtllm"])]
    assert len(kept) == 2 and all(r["engine"] != "trtllm" or r.get("out_tok_s") for r in kept)


# ── tp escalation on memory failures ─────────────────────────────────
# The user's rule (2026-09-21): if a model needs tp=N, run it at N,
# 2N, 4N... rather than eliminating it and reporting a blank. The model
# card's size is a guess about how an engine holds the weights.


def test_memory_failures_are_recognised_and_shared_memory_is_not():
    from simulator.roofline import is_memory_failure
    assert is_memory_failure(
        "RuntimeError: replica 0 container exited during startup: "
        "RuntimeError: Executor creation failed due to insufficient GPU memory.")
    assert is_memory_failure(
        "RuntimeError: replica 3 ...: torch.OutOfMemoryError: CUDA out of memory.")
    assert is_memory_failure(
        "ValueError: max_num_seqs (1024) exceeds available KV cache capacity")
    # Triton's OutOfResources is shared memory per SM; no tp buys that.
    assert not is_memory_failure(
        "triton.runtime.errors.OutOfResources: out of resource: shared memory")
    assert not is_memory_failure("ValueError: unknown architecture qwen3_5_moe")
    assert not is_memory_failure("")


def test_escalate_cell_doubles_tp_and_halves_replicas_up_to_the_cap():
    from simulator.roofline import escalate_cell
    cell = {"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
            "output_tokens": 128, "tp": 1, "replicas": 8,
            "kv_cache_dtype": "fp8", "error": "CUDA out of memory"}
    nxt = escalate_cell(cell, gpu_count=8, max_tp=4)
    assert nxt["tp"] == 2 and nxt["replicas"] == 4
    assert nxt["escalated_from"] == 1 and "error" not in nxt
    assert nxt["kv_cache_dtype"] == "fp8"           # the rest of the shape stays
    nxt2 = escalate_cell(nxt, gpu_count=8, max_tp=4)
    assert nxt2["tp"] == 4 and nxt2["replicas"] == 2 and nxt2["escalated_from"] == 2
    assert escalate_cell(nxt2, gpu_count=8, max_tp=4) is None      # domain cap
    assert escalate_cell(nxt2, gpu_count=8, max_tp=None)["tp"] == 8  # cross-domain
    assert escalate_cell({**nxt2, "tp": 8, "replicas": 1}, gpu_count=8) is None
    # GGUF engines have no tp lever.
    assert escalate_cell({**cell, "engine": "ktransformers"}, gpu_count=8) is None


def test_escalated_cells_key_apart_from_their_origin():
    from simulator.roofline import cell_key, escalate_cell
    cell = {"model": "m", "engine": "vllm_cuda_multi", "max_num_seqs": 2048,
            "output_tokens": 256, "tp": 1, "replicas": 8}
    nxt = escalate_cell(cell, gpu_count=8)
    assert cell_key(nxt) != cell_key(cell)
    assert "escalated_from" not in cell_key(nxt)   # not a shape field


def test_a_memory_failure_is_written_off_after_one_attempt():
    """Out of memory at a fixed shape is deterministic and the run has
    already planned the cell wider; a second launch would only spend
    the launch. Other failures still need GIVE_UP_AFTER."""
    from simulator.roofline import permanently_failed
    oom = {"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
           "output_tokens": 128, "tp": 1, "replicas": 8,
           "error": "RuntimeError: Executor creation failed due to insufficient GPU memory."}
    arch = {**oom, "engine": "sglang_cuda", "error": "ValueError: unknown architecture"}
    assert len(permanently_failed([oom])) == 1
    assert len(permanently_failed([arch])) == 0
    assert len(permanently_failed([arch, arch])) == 1


def test_resume_owes_escalations_to_recorded_memory_failures():
    """The plan is recomputed from the model list on resume; the results
    remember which cells ran out of memory, so the wider cells come
    back without the operator asking."""
    from simulator.roofline import cell_key, escalations
    plan = [{"model": "m", "engine": "trtllm", "max_num_seqs": 1024,
             "output_tokens": 128, "tp": 1, "replicas": 8, "input_tokens": 128}]
    results = [{**plan[0], "error": "torch.OutOfMemoryError: CUDA out of memory."},
               {**plan[0], "output_tokens": 256, "out_tok_s": 10.0},
               {**plan[0], "engine": "sglang_cuda", "error": "ValueError: arch"}]
    owed = escalations(plan, results, gpu_count=8, max_tp=4)
    assert len(owed) == 1
    assert owed[0]["tp"] == 2 and owed[0]["replicas"] == 4
    assert owed[0]["escalated_from"] == 1 and "error" not in owed[0]
    # Already in the plan: nothing owed twice.
    assert escalations(plan + owed, results, gpu_count=8, max_tp=4) == []
    # The wider cell failing for memory again owes the next step.
    results.append({**owed[0], "error": "CUDA out of memory"})
    owed2 = escalations(plan + owed, results, gpu_count=8, max_tp=4)
    assert [c["tp"] for c in owed2] == [4]
    assert cell_key(owed2[0]) != cell_key(owed[0])


def test_run_escalates_tp_in_place_until_the_model_fits(tmp_path, monkeypatch):
    """End to end on a fake engine: tp1 and tp2 die for memory, tp4
    measures. The tp4 cell runs right after the failures, before the
    next model, and the plan records it."""
    import simulator.roofline as rf

    launched: list[tuple[str, int, int]] = []

    def fake_build(overrides):
        launched.append((overrides["model_id"], overrides.get("tp") or 1,
                         overrides.get("replicas") or 8))
        cfg = tmp_path / f"cfg_{len(launched)}.yaml"
        cfg.write_text("x: 1")
        return cfg

    class _Out:
        db_directory = ""

    class _Cfg:
        output = _Out()

    async def fake_sweep(cfg, cohort, *, new_run, ladder_override=None):
        model, tp, _ = launched[-1]
        if model == "big" and tp < 4:
            raise RuntimeError(
                f"replica 0 container exited during startup: "
                f"torch.OutOfMemoryError: CUDA out of memory at tp{tp}")
        out = tmp_path / f"sweep_{len(launched)}.json"
        out.write_text(json.dumps({"peak": {"out_tok_s": 1000.0 * tp,
                                            "total_tok_s": 1500.0 * tp,
                                            "concurrency": 64}}))
        return out

    async def fake_staged(models, st, path, **kw):
        return set(models)

    monkeypatch.setattr("simulator.config.load_config", lambda p: _Cfg())
    monkeypatch.setattr("simulator.headline_sweep.run_headline_sweep", fake_sweep)
    monkeypatch.setattr(rf, "ensure_staged", fake_staged)
    monkeypatch.setattr("simulator.personas.cohort_from_persona", lambda name: object())
    monkeypatch.setattr("simulator.headline_shapes.apply_shape_to_generation",
                        lambda *a, **k: None)

    path = asyncio.run(rf.run_roofline(
        models=["big", "small"], engines=["trtllm"],
        shapes={"max_num_seqs": [1024], "output_tokens": [128]},
        build_config=fake_build, runs_base=tmp_path, resume=False,
        confirm_winners=False,
        engine_shape={"tp": 1, "replicas": 8, "gpu_memory_utilization": 0.95},
        gpu_count=8, max_tp=4))
    st = rf.load_state(path)
    assert [(m, tp, r) for m, tp, r in launched] == [
        ("big", 1, 8), ("big", 2, 4), ("big", 4, 2), ("small", 1, 8)]
    rows = [(r["model"], r["tp"], bool(r.get("error"))) for r in st.results]
    assert rows == [("big", 1, True), ("big", 2, True), ("big", 4, False),
                    ("small", 1, False)]
    winner = [r for r in st.results if r["model"] == "big" and not r.get("error")][0]
    assert winner["replicas"] == 2 and winner["escalated_from"] == 2
    assert winner["out_tok_s"] == 4000.0
    planned = [(c["model"], c.get("tp") or 1) for c in st.plan["cells"]]
    assert planned == [("big", 1), ("small", 1), ("big", 2), ("big", 4)]


def test_run_notes_when_no_wider_tp_exists(tmp_path, monkeypatch):
    import simulator.roofline as rf

    class _Out:
        db_directory = ""

    class _Cfg:
        output = _Out()

    async def fake_sweep(cfg, cohort, *, new_run, ladder_override=None):
        raise RuntimeError("Executor creation failed due to insufficient GPU memory.")

    async def fake_staged(models, st, path, **kw):
        return set(models)

    def fake_build(overrides):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("x: 1")
        return cfg

    monkeypatch.setattr("simulator.config.load_config", lambda p: _Cfg())
    monkeypatch.setattr("simulator.headline_sweep.run_headline_sweep", fake_sweep)
    monkeypatch.setattr(rf, "ensure_staged", fake_staged)
    monkeypatch.setattr("simulator.personas.cohort_from_persona", lambda name: object())
    monkeypatch.setattr("simulator.headline_shapes.apply_shape_to_generation",
                        lambda *a, **k: None)
    path = asyncio.run(rf.run_roofline(
        models=["big"], engines=["vllm_cuda_multi"],
        shapes={"max_num_seqs": [1024], "output_tokens": [128]},
        build_config=fake_build, runs_base=tmp_path, resume=False,
        confirm_winners=False, engine_shape={"tp": 4, "replicas": 2},
        gpu_count=8, max_tp=4))
    st = rf.load_state(path)
    assert len(st.results) == 1
    assert "no wider tp" in st.results[0]["note"]


def test_a_non_weight_bound_oom_steps_the_memory_share_down_first():
    """gpt-oss-20b (14 GB) at 2048 seqs died for memory at tp1, tp2 AND
    tp4: the weights were never the problem, the 0.95 share was (the
    sampler's first softmax had nowhere to go). Small weights -> lean
    the share first; big weights -> widen tp first; each falls back to
    the other when spent."""
    from simulator.roofline import SHARE_FLOOR, escalate_cell
    small = {"model": "s", "engine": "vllm_cuda_multi", "max_num_seqs": 2048,
             "output_tokens": 128, "tp": 1, "replicas": 8,
             "gpu_memory_utilization": 0.95}
    nxt = escalate_cell(small, gpu_count=8, max_tp=4, weight_gb=14, vram_gb=96)
    assert nxt["tp"] == 1 and nxt["gpu_memory_utilization"] == 0.90
    assert nxt["escalated_from_share"] == 0.95 and "escalated_from" not in nxt
    # Walks down to the floor, then widens tp.
    c = nxt
    shares = []
    while c["tp"] == 1:
        shares.append(c["gpu_memory_utilization"])
        c = escalate_cell(c, gpu_count=8, max_tp=4, weight_gb=14, vram_gb=96)
    assert shares[-1] == SHARE_FLOOR and c["tp"] == 2 and c["escalated_from"] == 1
    # Big weights: tp first, share only once tp is capped.
    big = {**small, "model": "b"}
    w = escalate_cell(big, gpu_count=8, max_tp=4, weight_gb=63, vram_gb=96)
    assert w["tp"] == 2 and w["gpu_memory_utilization"] == 0.95
    capped = {**big, "tp": 4, "replicas": 2}
    lean = escalate_cell(capped, gpu_count=8, max_tp=4, weight_gb=63, vram_gb=96)
    assert lean["tp"] == 4 and lean["gpu_memory_utilization"] == 0.90
    # Unknown weights keep the old rule (tp first).
    assert escalate_cell(small, gpu_count=8, max_tp=4)["tp"] == 2
    # Nothing left: floor share at the tp cap.
    spent = {**capped, "gpu_memory_utilization": SHARE_FLOOR}
    assert escalate_cell(spent, gpu_count=8, max_tp=4, weight_gb=63, vram_gb=96) is None


def test_share_escalations_are_owed_on_resume_with_model_weights():
    from simulator.roofline import cell_key, escalations
    plan = [{"model": "openai/gpt-oss-20b", "engine": "vllm_cuda_multi",
             "max_num_seqs": 2048, "output_tokens": 128, "tp": 1, "replicas": 8,
             "gpu_memory_utilization": 0.95, "input_tokens": 128}]
    results = [{**plan[0], "error": "torch.OutOfMemoryError: CUDA out of memory."}]
    info = {"openai/gpt-oss-20b": {"approx_size_gb": 14}}
    owed = escalations(plan, results, gpu_count=8, max_tp=4, model_info=info,
                       vram_gb=96)
    assert len(owed) == 1 and owed[0]["gpu_memory_utilization"] == 0.90
    assert owed[0]["tp"] == 1 and owed[0]["escalated_from_share"] == 0.95
    assert cell_key(owed[0]) != cell_key(plan[0])


def test_resume_does_not_confirm_a_winner_twice(tmp_path, monkeypatch):
    import simulator.roofline as rf

    sweeps: list[str] = []

    def fake_build(overrides):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("x: 1")
        return cfg

    class _Out:
        db_directory = ""

    class _Cfg:
        output = _Out()

    async def fake_sweep(cfg, cohort, *, new_run, ladder_override=None):
        sweeps.append("confirm" if ladder_override is None else "search")
        out = tmp_path / f"sweep_{len(sweeps)}.json"
        out.write_text(json.dumps({"peak": {"out_tok_s": 100.0, "concurrency": 8}}))
        return out

    async def fake_staged(models, st, path, **kw):
        return set(models)

    monkeypatch.setattr("simulator.config.load_config", lambda p: _Cfg())
    monkeypatch.setattr("simulator.headline_sweep.run_headline_sweep", fake_sweep)
    monkeypatch.setattr(rf, "ensure_staged", fake_staged)
    monkeypatch.setattr("simulator.personas.cohort_from_persona", lambda name: object())
    monkeypatch.setattr("simulator.headline_shapes.apply_shape_to_generation",
                        lambda *a, **k: None)
    kw = dict(models=["m"], engines=["trtllm"],
              shapes={"max_num_seqs": [1024], "output_tokens": [128]},
              build_config=fake_build, runs_base=tmp_path, confirm_winners=True,
              engine_shape={"tp": 1, "replicas": 8})
    asyncio.run(rf.run_roofline(resume=False, **kw))
    assert sweeps == ["search", "confirm"]
    asyncio.run(rf.run_roofline(resume=True, **kw))
    assert sweeps == ["search", "confirm"]          # nothing re-run


def test_a_staged_native_checkpoint_opens_ktransformers(tmp_path, monkeypatch):
    """Kimi-K2-Thinking's GGUF companion is llama.cpp-only (the v0.3
    KTransformers line cannot read it), but its native compressed-
    tensors INT4 checkpoint is exactly what the v0.7 line loads. Once
    those safetensors are staged the model gets a KTransformers cell;
    config-only staging does not open it."""
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    cat = [{"id": "moonshotai/Kimi-K2-Thinking", "family": "kimi-k2",
            "series": "Kimi K2", "params_b": 1026, "moe": True,
            "approx_size_gb": 594, "min_vram_gb": 620, "kt_only": True,
            "gguf": {"repo": "unsloth/Kimi-K2-Thinking-GGUF", "file": "UD-Q4_K_XL",
                     "size_gb": 646.2, "engines": ["llamacpp"]}}]
    snap = tmp_path / "hub" / "models--moonshotai--Kimi-K2-Thinking" / "snapshots" / "a"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV3ForCausalLM"],
        "quantization_config": {"quant_method": "compressed-tensors",
                                "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int"}}}}}))
    (snap / "tiktoken.model").write_text("x")

    def info():
        c = score_models(cat, vram_per_gpu_gb=96, host_ram_gb=2015, gpu_count=8,
                         max_tp=4, cache=tmp_path)[0]
        return c.info()
    before = info()
    assert before["kt_native"] is False
    assert before["gguf_engines"] == ["llamacpp"]
    (snap / "model-00001-of-00062.safetensors").write_bytes(b"0")
    after = info()
    assert after["kt_native"] is True and after["fits_gpu"] is False
    assert after["gguf_engines"] == ["llamacpp", "ktransformers"]
    from simulator.roofline import engines_for
    assert engines_for("ktransformers", after) and engines_for("llamacpp", after)
    assert not engines_for("vllm_cuda_multi", after)


def test_gguf_engine_cells_climb_a_ladder_sized_to_their_slots():
    """A 512-stream first rung against a 32-slot llama-server measured
    a queue (XE7740 giants pass). GGUF engines climb to their slots and
    one rung past; GPU engines keep the caller's ladder."""
    from simulator.roofline import ladder_for
    assert ladder_for("llamacpp", 32, [512, 2048]) == [8, 16, 32, 64]
    assert ladder_for("ktransformers", 4, [512, 2048]) == [1, 2, 4, 8]
    assert ladder_for("ktransformers", 4) == [1, 2, 4, 8]
    assert ladder_for("vllm_cuda_multi", 1024, [512, 2048]) == [512, 2048]
    assert ladder_for("vllm_cuda_multi", 1024) is None       # sweep default


def test_resume_can_redo_an_engines_measured_cells(tmp_path, monkeypatch):
    """Unlike retry_engines (failures only), redo_engines forgets every
    row of the engine so a wrong measurement is replaced."""
    import simulator.roofline as rf

    sweeps: list[tuple[str, list | None]] = []

    def fake_build(overrides):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("x: 1")
        return cfg

    class _Out:
        db_directory = ""

    class _Cfg:
        output = _Out()

    async def fake_sweep(cfg, cohort, *, new_run, ladder_override=None):
        sweeps.append(("sweep", ladder_override))
        out = tmp_path / f"sweep_{len(sweeps)}.json"
        out.write_text(json.dumps({"peak": {"out_tok_s": 5.0, "concurrency": 32}}))
        return out

    async def fake_staged(models, st, path, **kw):
        return set(models)

    monkeypatch.setattr("simulator.config.load_config", lambda p: _Cfg())
    monkeypatch.setattr("simulator.headline_sweep.run_headline_sweep", fake_sweep)
    monkeypatch.setattr(rf, "ensure_staged", fake_staged)
    monkeypatch.setattr("simulator.personas.cohort_from_persona", lambda name: object())
    monkeypatch.setattr("simulator.headline_shapes.apply_shape_to_generation",
                        lambda *a, **k: None)
    kw = dict(models=["m"], engines=["llamacpp"],
              shapes={"max_num_seqs": [32], "output_tokens": [128]},
              build_config=fake_build, runs_base=tmp_path, confirm_winners=False,
              engine_shape={"tp": 1, "replicas": 1})
    asyncio.run(rf.run_roofline(resume=False, **kw))
    assert sweeps == [("sweep", [8, 16, 32, 64])]
    asyncio.run(rf.run_roofline(resume=True, **kw))
    assert len(sweeps) == 1                                   # measured: skipped
    asyncio.run(rf.run_roofline(resume=True, redo_engines=["llamacpp"], **kw))
    assert len(sweeps) == 2                                   # forgotten: re-run
    st = rf.load_state(tmp_path / rf.STATE_NAME)
    assert len([r for r in st.results if r["engine"] == "llamacpp"]) == 1


def test_cross_domain_cells_launch_with_placement_span():
    """Every tp8 cell of the giants pass failed at config time: the
    planner allowed cross-domain tp but the device assigner still
    refused a TP set spanning the two PCIe domains. Such cells carry
    placement "span" (a shape key, so they key apart from the refused
    rows), and so does an escalation that crosses the domain."""
    from simulator.roofline import cell_key, cells, escalate_cell
    from simulator.search import assign_devices
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert assign_devices(8, 1, "pack", groups) is None
    assert assign_devices(8, 1, "span", groups) == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert assign_devices(4, 2, "span", groups) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    giant = {"tier": "large", "fits_gpu": True, "tp": 8, "replicas": 1,
             "cross_domain": True, "kt_eligible": False}
    c = cells(["nvidia/Kimi-K2-Thinking-NVFP4"], ["vllm_cuda_multi"],
              {"max_num_seqs": [1024], "output_tokens": [128]},
              engine_shape={"tp": 1, "replicas": 8},
              model_info={"nvidia/Kimi-K2-Thinking-NVFP4": giant})
    assert c[0]["tp"] == 8 and c[0]["placement"] == "span"
    assert "placement=span" in cell_key(c[0])
    plain = {**c[0]}
    plain.pop("placement")
    assert cell_key(plain) != cell_key(c[0])
    # An escalation from tp4 (the domain) to tp8 crosses it.
    cell = {"model": "m", "engine": "vllm_cuda_multi", "max_num_seqs": 2048,
            "output_tokens": 128, "tp": 4, "replicas": 2, "gpu_memory_utilization": 0.95}
    nxt = escalate_cell(cell, gpu_count=8, max_tp=None, weight_gb=235, vram_gb=96,
                        domain_tp=4)
    assert nxt["tp"] == 8 and nxt["placement"] == "span"
    within = escalate_cell({**cell, "tp": 2, "replicas": 4}, gpu_count=8,
                           max_tp=None, weight_gb=235, vram_gb=96, domain_tp=4)
    assert within["tp"] == 4 and "placement" not in within


def test_scored_giants_are_marked_cross_domain():
    from simulator.roofline import score_models
    cat = [{"id": "nvidia/Kimi-K2-Thinking-NVFP4", "family": "kimi-k2", "series": "Kimi K2",
            "params_b": 1026, "moe": True, "approx_size_gb": 594, "min_vram_gb": 620}]
    c = score_models(cat, vram_per_gpu_gb=96, host_ram_gb=2015, gpu_count=8, max_tp=None)[0]
    assert c.tp == 8 and c.cross_domain and c.info()["cross_domain"]
    capped = score_models(cat, vram_per_gpu_gb=96, host_ram_gb=2015, gpu_count=8, max_tp=4)[0]
    assert not capped.fits_gpu and not capped.cross_domain


def test_a_modelopt_nvfp4_checkpoint_does_not_open_ktransformers(tmp_path, monkeypatch):
    """nvidia/DeepSeek-V3.1-NVFP4 declares nothing in config.json (dtype
    bfloat16) and keeps its quantisation in hf_quant_config.json; the
    resolver read it as a native BF16 checkpoint and the giants pass
    planned two KTransformers cells that could only fail."""
    from simulator.roofline import _native_kt_checkpoint
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    snap = tmp_path / "hub" / "models--nvidia--DeepSeek-V3.1-NVFP4" / "snapshots" / "a"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    (snap / "model-00001-of-00163.safetensors").write_bytes(b"0")
    assert _native_kt_checkpoint("nvidia/DeepSeek-V3.1-NVFP4", tmp_path)
    (snap / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "NVFP4"}}))
    assert not _native_kt_checkpoint("nvidia/DeepSeek-V3.1-NVFP4", tmp_path)
