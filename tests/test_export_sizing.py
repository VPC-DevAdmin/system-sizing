"""The sizing-tool export: the tool's shape kept field for field, with
saturation-benchmark facts added beside it rather than folded in."""

from __future__ import annotations

import json

from simulator import export_sizing as ex

SYSTEM = {"vendor": "Dell", "platform": "PowerEdge XE7740", "cpuKey": "Intel:6787P",
          "sockets": 2, "memoryGb": 2016,
          "accelerators": [{"vendor": "NVIDIA", "model": "RTX PRO 6000 Blackwell Server Edition",
                            "count": 8, "memoryGbEach": 96}]}

RUNGS = [
    {"concurrency": 64, "in_flight": 62.6, "queue_depth": 0.0, "out_tok_s": 6666.1,
     "prompt_tok_s": 3624.4, "total_tok_s": 10290.5, "ttft_p50_ms": 103.4, "ttft_p95_ms": 156.7,
     "tpot_p50_ms": 9.22, "tpot_p95_ms": 9.94, "samples": 3266, "errors": 0,
     "kv_cache_pct": 0.31, "gpu_power_w": 2276.6, "steady_state": True, "measure_s": 124,
     "held": True},
    {"concurrency": 8192, "in_flight": 7004.5, "queue_depth": 0.0, "out_tok_s": 127349.7,
     "prompt_tok_s": 71699.4, "samples": 47840, "errors": 0, "steady_state": True,
     "measure_s": 98, "held": True, "gpu_power_w": 3609.9},
    {"concurrency": 16384, "in_flight": 7100, "queue_depth": 9000, "out_tok_s": 120000,
     "samples": 100, "errors": 900, "steady_state": True, "measure_s": 90, "held": False},
]
SWEEP = {"cohort_id": "headline_generation", "cohort_name": "Persona: Headline: Generation",
         "kv_cache_tokens": 28913792.0, "rungs": RUNGS, "peak": RUNGS[1],
         "shape": {"input_tokens": 128.0, "output_tokens": 256.0, "ignore_eos": True},
         "stop_reason": "ladder exhausted"}
ROW = {"model": "nvidia/Gemma-4-26B-A4B-NVFP4", "engine": "vllm_cuda_multi",
       "max_num_seqs": 1024, "output_tokens": 256, "input_tokens": 128, "replicas": 8,
       "tp": 1, "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8",
       "max_model_len": 2048, "out_tok_s": 127349.7, "concurrency": 8192,
       "run_dir": "runs/run_326", "confirmed": True, "samples": 47840, "errors": 0,
       "success_rate": 1.0, "tokens_per_watt": 35.28}


def test_cpu_key():
    assert ex.cpu_key("Intel(R) Xeon(R) 6787P") == "Intel:6787P"
    assert ex.cpu_key("AMD EPYC 9655P 96-Core Processor") == "AMD:9655P"
    assert ex.cpu_key(None) is None


def test_row_document_keeps_the_tool_shape_and_adds_beside_it():
    d = ex.row_document(ROW, info={"quant": "NVFP4", "params_b": 26}, sweep=SWEEP,
                        system=SYSTEM, generated_at="t")
    m, c = d["meta"], d["cohorts"][0]
    for k in ("generated_at", "models", "engines", "source_dir", "engine_config", "system"):
        assert k in m
    ec = m["engine_config"]
    assert ec["quantization_kind"] == "nvfp4" and ec["tensor_parallel_size"] == 1
    assert ec["max_model_len"] == 2048 and ec["replicas"] == 8
    topo = m["system"]["topology"]
    assert topo == {"gpusUsed": 8, "replicas": 8, "tensorParallelSize": 1,
                    "placement": "pack", "cpuExperts": False}
    assert m["phase"] == "confirmation" and m["workload"]["sla_enforced"] is False
    for k in ("id", "category", "final_status", "target_capacity_pool_size",
              "soft_capacity_pool_size", "fail_pool_size", "capacity_throughput", "curve"):
        assert k in c
    assert c["final_status"] == "ok"
    assert c["target_capacity_pool_size"] == 8192
    assert c["soft_capacity_pool_size"] == 8192
    assert c["fail_pool_size"] == 16384
    cap = c["capacity_throughput"]
    assert cap["visible_output_tok_per_s"] == cap["generated_tok_per_s"] == 127349.7
    p0 = c["curve"][0]
    for k in ("pool_size", "sample_size", "status", "target_status", "violation_rate",
              "target_miss_rate", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms",
              "prompt_tok_per_s", "visible_output_tok_per_s", "kv_cache_used_pct",
              "measurement_duration_s"):
        assert k in p0
    assert p0["status"] == "pass" and p0["target_status"] == "not_evaluated"
    assert p0["tokens_per_watt"] == round(6666.1 / 2276.6, 3)
    last = c["curve"][-1]
    assert last["status"] == "fail" and last["violation_rate"] == 0.9


def test_failed_configuration_is_exported_with_its_cause():
    row = {"model": "MiniMaxAI/MiniMax-M2.7", "engine": "vllm_cuda_multi", "tp": 8,
           "replicas": 1, "max_num_seqs": 2048, "output_tokens": 128,
           "error": "RuntimeError: container exited"}
    d = ex.row_document(row, info={}, sweep=None, system=SYSTEM, generated_at="t")
    c = d["cohorts"][0]
    assert c["final_status"] == "failed" and c["curve"] == []
    assert c["capacity_throughput"] is None and "container exited" in c["error"]
    assert d["meta"]["engine_config"]["placement"] == "span"
    assert "reasoning" in d["meta"]["output_accounting"]


def test_ktransformers_quant_is_the_checkpoint_or_the_gguf():
    kt = {"model": "moonshotai/Kimi-K2-Thinking", "engine": "ktransformers", "kt_native": True}
    assert ex.quantization_kind(kt, {"quant": "INT4"}) == "int4"
    assert ex.quantization_kind({**kt, "kt_native": False}, {}) == "gguf"
    assert ex.quantization_kind({"model": "x/y", "engine": "llamacpp"}, {"quant": "FP8"}) == "gguf"


def test_placement_windows_become_one_cohort_per_mix():
    res = {"plan": {"max_tokens": 256, "configs": [{"name": "dp2_frequency", "custom": {
        "engine": "ktransformers", "model_id": "moonshotai/Kimi-K2-Thinking", "tp": 4,
        "replicas": 2, "ktransformers_numa_pin": True,
        "ktransformers_expert_placement": "frequency", "max_model_len": 2048}}]},
        "calibration": {"answers": 851, "freq_path": "/x.pt"},
        "windows": [
            {"config": "dp2_frequency", "mix": "balanced", "concurrency": 64, "tp": 4,
             "replicas": 2, "gpu_experts": 167, "stream_tok_s": 102.9, "succeeded": 144,
             "success_rate": 1.0, "measure_s": 420, "engine": {"gauge_tok_s": 100.1},
             "experts": {"cpu_share": 0.343}},
            {"config": "dp2_frequency", "mix": "balanced", "concurrency": 128, "tp": 4,
             "replicas": 2, "gpu_experts": 167, "stream_tok_s": 132.6, "succeeded": 180,
             "success_rate": 1.0, "measure_s": 420, "engine": {}, "experts": {"cpu_share": 0.34}}]}
    docs = ex.placement_documents(res, system=SYSTEM, generated_at="t")
    assert len(docs) == 1
    d = docs[0]
    assert d["meta"]["run_kind"] == "kt_expert_placement"
    assert d["meta"]["engine_config"]["ktransformers_gpu_experts"] == 167
    assert "freq_path" not in d["meta"]["calibration"]
    c = d["cohorts"][0]
    assert c["id"] == "mix:balanced" and c["target_capacity_pool_size"] == 128
    assert c["capacity_throughput"]["generated_tok_per_s"] == 132.6
    assert [p["pool_size"] for p in c["curve"]] == [64, 128]


def test_build_reads_sweeps_and_write_lays_out_files(tmp_path):
    runs = tmp_path / "runs"
    (runs / "run_326").mkdir(parents=True)
    (runs / "run_326" / "headline_sweep.json").write_text(json.dumps(SWEEP))
    state = {"plan": {"model_info": {ROW["model"]: {"quant": "NVFP4"}}},
             "results": [ROW, {**ROW, "run_dir": None, "error": "boom", "confirmed": False}]}
    docs = ex.build(state, runs, system=SYSTEM)
    assert len(docs[0]["cohorts"][0]["curve"]) == 3
    out = tmp_path / "out"
    ex.write(docs, out)
    assert len(json.loads((out / "all.json").read_text())) == 2
    idx = json.loads((out / "index.json").read_text())
    assert idx[0]["generated_tok_per_s"] == 127349.7 and idx[1]["status"] == "failed"
    assert len(list((out / "runs").glob("*.json"))) == 2
