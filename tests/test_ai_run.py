"""Planner GPU runs: configs per GPU count, the finalized file, and the
importer's rules applied before hand-over."""

from __future__ import annotations

import yaml

from simulator import ai_run

SYSTEM = {"vendor": "Dell", "platform": "PowerEdge XE7740", "cpuKey": "Intel:6787P",
          "sockets": 2, "memoryGb": 2048,
          "accelerators": [{"vendor": "NVIDIA", "model": "RTX PRO 6000 Blackwell Server Edition",
                            "count": 8, "memoryGbEach": 96}]}


def _cohort(pid, cap=256, above=True, fail=512):
    curve = [{"pool_size": 128, "status": "pass"}, {"pool_size": cap, "status": "pass"}]
    if above:
        curve.append({"pool_size": cap * 2, "status": "fail"})
    return {"id": pid, "category": "persona", "capacity_throughput": {"pool_size": cap},
            "fail_pool_size": fail if above else None, "curve": curve}


def _export(models=None):
    return {"meta": {"models": models or [ai_run.PLANNER_MODEL], "engines": ["vllm_cuda"],
                     "engine_config": {"tensor_parallel_size": 1, "max_model_len": 8192}},
            "cohorts": [_cohort(p) for p in ai_run.PERSONAS]}


def test_configs_are_tp1_one_replica_per_gpu_spread_across_domains(tmp_path, monkeypatch):
    from simulator import arena
    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]], "vram_per_gpu_gb": 96})
    paths = ai_run.write_configs(ai_run.PLANNER_MODEL, tmp_path)
    assert [p.name for p in paths] == [f"ai-qwen3-30b-a3b-instruct-2507-fp8-{n}gpu.yaml"
                                       for n in (1, 2, 4, 8)]
    docs = [yaml.safe_load(p.read_text()) for p in paths]
    assert docs[0]["engine"]["type"] == "vllm_cuda"
    two = docs[1]["engine"]
    assert two["type"] == "vllm_cuda_multi" and two["tensor_parallel_size"] == 1
    # spread: the two replicas sit in different PCIe/NUMA domains
    assert {d // 4 for g in two["replica_devices"] for d in g} == {0, 1}
    assert len(docs[3]["engine"]["replica_devices"]) == 8
    assert all(d["engine"]["max_model_len"] == 8192 for d in docs)
    assert [d["simulation"]["open_loop_max_workers"] for d in docs] == [16, 16, 32, 64]
    assert all(d["simulation"]["mode"] == "open" for d in docs)
    assert [d["simulation"]["open_loop_min_workers"] for d in docs] == [4, 8, 16, 32]
    assert all(d["output"]["db_directory"] == "runs_ai" for d in docs)


def test_finalize_adds_system_topology_and_persona_models():
    doc = ai_run.finalize(_export(), system=SYSTEM, gpus=4)
    s = doc["meta"]["system"]
    assert s["memoryGb"] == 2048 and s["accelerators"][0]["count"] == 8
    assert s["topology"] == {"gpusUsed": 4, "replicas": 4, "tensorParallelSize": 1,
                             "placement": "spread"}
    p = doc["meta"]["personas"]
    assert set(p) == set(ai_run.PERSONAS)
    assert "sla" in p["quick_lookup"] and "active_think_seconds" in p["writer"]
    assert ai_run.validate(doc) == []


def test_validate_applies_the_importers_rules():
    doc = ai_run.finalize(_export(models=["nvidia/Qwen3.6-35B-A3B-NVFP4"]), system=SYSTEM, gpus=2)
    assert any("models must be exactly" in x for x in ai_run.validate(doc))
    assert ai_run.validate(doc, require_planner_model=False) == []
    doc["meta"]["system"]["topology"]["replicas"] = 3
    assert any("replicas x" in x for x in ai_run.validate(doc, require_planner_model=False))


def test_floors_are_named():
    exp = _export()
    exp["cohorts"][0] = _cohort("quick_lookup", above=False)
    exp["cohorts"][1] = {**_cohort("conversational"), "capacity_throughput": None}
    exp["cohorts"] = exp["cohorts"][:-1]
    probs = ai_run.validate(ai_run.finalize(exp, system=SYSTEM, gpus=1))
    assert "floor: quick_lookup has no tested step above its capacity 256" in probs
    assert "floor: conversational has no passing step (capacity null)" in probs
    assert "persona long_form_generator missing" in probs
