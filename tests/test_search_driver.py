"""Search driver in scripts/engine_optimizer.py: candidate → docker
config translation, per-candidate model rebinding, state persistence
and resume — with run_config stubbed (no Docker)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


@pytest.fixture()
def optimizer():
    import sys
    spec = importlib.util.spec_from_file_location(
        "engine_optimizer", REPO / "scripts" / "engine_optimizer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Must be registered before exec: dataclass introspection resolves
    # type hints via sys.modules[cls.__module__].
    sys.modules["engine_optimizer"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("engine_optimizer", None)


@pytest.fixture()
def space_file(tmp_path) -> Path:
    p = tmp_path / "space.yaml"
    p.write_text("""
name: tiny
engine: vllm_cuda
device_groups: [[0, 1], [2, 3]]
model_variants:
  bf16: {model: org/model-bf16, served_name: m}
  fp8:  {model: org/model-fp8,  served_name: m}
dimensions:
  model_variant: [bf16, fp8]
  tp: [1, 2]
  dp: [1, 2]
  max_num_seqs: [128, 256]
search:
  seed: 7
  initial_samples: 6
  top_k: 2
  neighbors_per_iteration: 4
  max_iterations: 2
  budget: 14
""")
    return p


def _stub_run_config(optimizer, seen):
    """Fake evaluator: fp8 + tp2 is the sweet spot; records what the
    driver bound per candidate."""
    async def fake_run_config(cfg, state, prompts, save,
                              cells=None, early_stop=None):
        # The driver now passes its ladder cells + SLA early-stop.
        assert cells and cells[0].name.startswith("ladder_c")
        assert callable(early_stop)
        seen.append({
            "name": cfg.name,
            "model": optimizer.MODEL_PATH,
            "replicas": [(r.name, r.port, r.gpus) for r in cfg.replicas],
            "args": list(cfg.replica_args),
        })
        tps = 1000.0
        if "fp8" in optimizer.MODEL_PATH:
            tps *= 1.5
        if "--tensor-parallel-size" in cfg.replica_args:
            tps *= 1.3
        tps *= len(cfg.replicas)
        cell = optimizer.CellResult(
            cell_name="ladder_c0032", samples=8, errors=0, timeouts=0,
            ttft_p50_ms=100.0, ttft_p95_ms=200.0,
            tpot_p50_ms=8.0, tpot_p95_ms=10.0,
            throughput_out_tok_s=tps,
        )
        return optimizer.ConfigResult(
            name=cfg.name, description=cfg.description, status="ok",
            launch_seconds=1.0, cells=[cell],
        )
    return fake_run_config


def test_search_driver_end_to_end_and_resume(
    optimizer, space_file, tmp_path, monkeypatch,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(optimizer, "run_config", _stub_run_config(optimizer, seen))
    monkeypatch.setattr(optimizer, "make_prompts", lambda cells: {})
    out = tmp_path / "search.json"

    asyncio.run(optimizer.run_search(space_file, out, new_run=False))

    doc = json.loads(out.read_text())
    assert doc["kind"] == "search"
    # The search path must bind the GPU image — the CPU-image module
    # default once made every candidate run vLLM on the host CPU.
    assert optimizer.IMAGE == "vllm/vllm-openai:latest"
    summary = doc["summary"]
    assert summary["evaluated"] <= 14
    assert summary["evaluated"] == len(seen)
    # The planted optimum (fp8 + tp2, dp maxed) wins.
    best = summary["best"]["params"]
    assert best["model_variant"] == "fp8"
    assert best["tp"] == 2 and best["dp"] == 2

    # The driver bound the right model per candidate and built real
    # GPU replicas.
    models = {c["model"] for c in seen}
    assert models == {"org/model-bf16", "org/model-fp8"}
    for c in seen:
        for _name, port, gpus in c["replicas"]:
            assert gpus.startswith("device=")
            assert 8000 <= port < 8010

    # Resume: everything evaluated → immediate done, no new evals.
    evaluated_before = summary["evaluated"]
    seen.clear()
    asyncio.run(optimizer.run_search(space_file, out, new_run=False))
    assert seen == []
    doc2 = json.loads(out.read_text())
    assert doc2["summary"]["evaluated"] == evaluated_before

    # --new-run wipes and re-searches deterministically.
    asyncio.run(optimizer.run_search(space_file, out, new_run=True))
    assert len(seen) == evaluated_before


def test_search_driver_survives_launch_failures(
    optimizer, space_file, tmp_path, monkeypatch,
) -> None:
    async def failing_run_config(cfg, state, prompts, save,
                                 cells=None, early_stop=None):
        return optimizer.ConfigResult(
            name=cfg.name, description=cfg.description,
            status="launch_failed", failure_reason="no CUDA here",
        )
    monkeypatch.setattr(optimizer, "run_config", failing_run_config)
    monkeypatch.setattr(optimizer, "make_prompts", lambda cells: {})
    out = tmp_path / "search.json"
    asyncio.run(optimizer.run_search(space_file, out, new_run=False))
    doc = json.loads(out.read_text())
    assert doc["summary"]["best"] is None
    assert doc["state"]["done_reason"] == "no_successful_candidates"


def test_search_seeding_skips_prior_results(
    optimizer, space_file, tmp_path, monkeypatch,
) -> None:
    """--seed-results: a fresh run over a grown space records the
    group's prior ok evaluations up front and never re-runs them —
    reopening an investigation ADDS instead of re-measuring."""
    from simulator.search import canonical_key, load_space

    space = load_space(space_file)
    seeded_params = {"model_variant": "bf16", "tp": 1, "dp": 1,
                     "max_num_seqs": 128}
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"evaluated": {
        canonical_key(seeded_params, space): {
            "status": "ok", "score": 4321.0, "iteration": 0,
            "config_name": "prior", "params": seeded_params,
            "cells": [{"cell_name": "ladder_c0032", "samples": 32,
                       "errors": 0, "timeouts": 0,
                       "throughput_out_tok_s": 4321.0,
                       "ttft_p95_ms": 100.0, "tpot_p95_ms": 10.0}]},
        # Invalid in this space (tp=4 not a dim value) — must be
        # dropped, not crash.
        "model_variant=bf16|tp=4": {
            "status": "ok", "score": 9.0, "iteration": 0,
            "params": {"model_variant": "bf16", "tp": 4}, "cells": []},
        # Failures never seed — they deserve a retry.
        "model_variant=fp8|tp=2|dp=2|max_num_seqs=256": {
            "status": "launch_failed", "score": None, "iteration": 0,
            "params": {"model_variant": "fp8", "tp": 2, "dp": 2,
                       "max_num_seqs": 256}, "cells": []},
    }}))

    seen: list[dict] = []
    monkeypatch.setattr(optimizer, "run_config",
                        _stub_run_config(optimizer, seen))
    monkeypatch.setattr(optimizer, "make_prompts", lambda cells: {})
    out = tmp_path / "search.json"
    asyncio.run(optimizer.run_search(space_file, out, new_run=False,
                                     seed_results=seed))

    doc = json.loads(out.read_text())
    evaluated = doc["state"]["evaluated"]
    key = canonical_key(seeded_params, space)
    assert evaluated[key]["score"] == 4321.0
    assert evaluated[key]["config_name"] == "prior"
    # The driver never re-ran the seeded candidate...
    ran_models = {(s["name"]) for s in seen}
    assert not any("prior" in n for n in ran_models)
    assert all(evaluated[canonical_key(p, space)]["config_name"] != "prior"
               or canonical_key(p, space) == key
               for p in [seeded_params])
    # ...the invalid seed was dropped, and new candidates were measured.
    assert "model_variant=bf16|tp=4" not in evaluated
    assert len(evaluated) > 1
    assert len(seen) == len(evaluated) - 1     # everything else ran
