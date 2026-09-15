"""Winner → benchmark profile promotion (simulator/promote.py +
POST /api/optimizer/promote)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from simulator.promote import (
    PromoteError,
    parse_engine_args,
    promote_registry_winner,
    promote_search_winner,
    space_overview,
)
from simulator.search import load_space

SPACE_YAML = textwrap.dedent("""\
    name: promo-test
    engine: vllm_cuda
    device_groups: [[0, 1], [2, 3]]
    model_variants:
      bf16:
        model: org/Model-30B
        served_name: m
      fp8:
        model: org/Model-30B-FP8
        served_name: m
    dimensions:
      model_variant: [bf16, fp8]
      tp: [1, 2]
      dp: [1, 2]
      gpu_memory_utilization: [0.85, 0.95]
      max_num_seqs: [64, 256]
      placement: [pack, spread]
""")


def _search_doc(space_path: Path, params: dict, score: float = 123.4) -> dict:
    space = load_space(space_path)
    return {
        "kind": "search",
        "space": space.name,
        "space_file": str(space_path),
        "space_hash": space.space_hash(),
        "generated_at": "2026-09-15T00:00:00+00:00",
        "summary": {"best": {"key": "k", "params": params, "score": score}},
    }


def test_parse_engine_args_lifts_known_flags() -> None:
    fields, leftover = parse_engine_args([
        "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", "2",
        "--max-num-seqs", "128",
        "--enable-prefix-caching",
        "--quantization", "fp8",
    ])
    assert fields == {"gpu_memory_utilization": 0.95,
                      "tensor_parallel_size": 2,
                      "quantization_kind": "fp8"}
    assert leftover == ["--max-num-seqs", "128", "--enable-prefix-caching"]


def test_promote_search_winner_writes_valid_profile(tmp_path) -> None:
    space_path = tmp_path / "space.yaml"
    space_path.write_text(SPACE_YAML)
    doc = _search_doc(space_path, {
        "model_variant": "fp8", "tp": 2, "dp": 1,
        "gpu_memory_utilization": 0.95, "max_num_seqs": 256,
        "placement": "pack",
    })
    out = promote_search_winner(doc, out_dir=tmp_path / "profiles")
    assert out["profile"] == "optimized-promo-test"
    assert out["warnings"] == []
    saved = yaml.safe_load(Path(out["path"]).read_text())
    eng = saved["engine"]
    assert eng["type"] == "vllm_cuda"
    assert eng["model_id"] == "org/Model-30B-FP8"
    assert eng["tensor_parallel_size"] == 2
    assert eng["gpu_memory_utilization"] == 0.95
    assert eng["gpu_device_ids"] == [0, 1]          # pack: one domain
    assert "--max-num-seqs" in eng["vllm_extra_flags"]
    assert saved["telemetry"]["enable_gpu"] is True
    # The generated file loads through the real config loader.
    from simulator.config import load_config
    cfg = load_config(out["path"])
    assert cfg.engine.tensor_parallel_size == 2


def test_promote_search_dp_winner_warns_and_pins_replica0(tmp_path) -> None:
    space_path = tmp_path / "space.yaml"
    space_path.write_text(SPACE_YAML)
    doc = _search_doc(space_path, {
        "model_variant": "bf16", "tp": 2, "dp": 2,
        "gpu_memory_utilization": 0.85, "max_num_seqs": 64,
        "placement": "spread",
    })
    out = promote_search_winner(doc, out_dir=tmp_path / "profiles")
    assert out["warnings"] and "dp=2" in out["warnings"][0]
    eng = yaml.safe_load(Path(out["path"]).read_text())["engine"]
    assert eng["gpu_device_ids"] == [0, 1]          # replica 0 only


def test_promote_search_refuses_stale_or_empty(tmp_path) -> None:
    space_path = tmp_path / "space.yaml"
    space_path.write_text(SPACE_YAML)
    doc = _search_doc(space_path, {"model_variant": "bf16"})
    doc["summary"]["best"] = None
    with pytest.raises(PromoteError, match="no successful candidate"):
        promote_search_winner(doc, out_dir=tmp_path / "p")
    doc = _search_doc(space_path, {"model_variant": "bf16"})
    doc["space_hash"] = "different"
    with pytest.raises(PromoteError, match="space changed"):
        promote_search_winner(doc, out_dir=tmp_path / "p")


CATALOG = {
    "profiles": {"nvidia_test": [
        {"name": "tp2", "description": "TP=2 across two GPUs",
         "replica_args": ["--gpu-memory-utilization", "0.90",
                          "--tensor-parallel-size", "2"],
         "replica_gpus": ["device=0,1"], "shm_size": "8g"},
        {"name": "dp2", "description": "two replicas",
         "replica_args": ["--gpu-memory-utilization", "0.90"],
         "replica_gpus": ["device=0", "device=1"], "shm_size": "8g"},
    ]},
    "profile_defaults": {"nvidia_test": {"model": "org/Default-M"}},
}

RESULTS = {
    "profile": "nvidia_test",
    "model": "org/Swept-M",
    "generated_at": "2026-09-15T01:00:00+00:00",
    "configs": [
        {"name": "tp2", "status": "ok", "cells": [{}]},
        {"name": "dp2", "status": "ok", "cells": [{}]},
        {"name": "broken", "status": "launch_failed", "cells": []},
    ],
}


def test_promote_registry_winner(tmp_path) -> None:
    out = promote_registry_winner(CATALOG, RESULTS, "tp2",
                                  out_dir=tmp_path / "profiles")
    eng = yaml.safe_load(Path(out["path"]).read_text())["engine"]
    assert eng["model_id"] == "org/Swept-M"
    assert eng["tensor_parallel_size"] == 2
    assert eng["gpu_device_ids"] == [0, 1]
    assert out["warnings"] == []

    # Multi-replica config warns and pins replica 0.
    out = promote_registry_winner(CATALOG, RESULTS, "dp2",
                                  out_dir=tmp_path / "profiles")
    assert out["warnings"] and "replica 0" in out["warnings"][0]
    eng = yaml.safe_load(Path(out["path"]).read_text())["engine"]
    assert eng["gpu_device_ids"] == [0]

    with pytest.raises(PromoteError, match="did not complete"):
        promote_registry_winner(CATALOG, RESULTS, "broken",
                                out_dir=tmp_path / "p")
    with pytest.raises(PromoteError, match="not in the sweep"):
        promote_registry_winner(CATALOG, RESULTS, "nope",
                                out_dir=tmp_path / "p")


def test_space_overview(tmp_path) -> None:
    space_path = tmp_path / "space.yaml"
    space_path.write_text(SPACE_YAML)
    ov = space_overview(load_space(space_path))
    assert ov["models"] == {"bf16": "org/Model-30B",
                            "fp8": "org/Model-30B-FP8"}
    assert ov["devices"] == 4 and len(ov["device_groups"]) == 2
    assert ov["dimensions"]["tp"] == 2
    assert ov["objective"] == "sla_throughput"


def test_promote_api(tmp_path, monkeypatch) -> None:
    """End-to-end: search.json on disk → POST promote → profile file
    the /api/profiles listing then exposes."""
    from fastapi.testclient import TestClient

    from simulator.service import create_app
    monkeypatch.chdir(tmp_path)
    space_path = tmp_path / "space.yaml"
    space_path.write_text(SPACE_YAML)
    runs = tmp_path / "runs"
    (runs / "engine_optimizer").mkdir(parents=True)
    doc = _search_doc(space_path, {
        "model_variant": "fp8", "tp": 1, "dp": 1,
        "gpu_memory_utilization": 0.85, "max_num_seqs": 64,
        "placement": "pack",
    })
    (runs / "engine_optimizer" / "search.json").write_text(json.dumps(doc))

    with TestClient(create_app(runs)) as client:
        r = client.post("/api/optimizer/promote", json={"source": "search"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["profile"] == "optimized-promo-test"
        assert Path(body["path"]).exists()
        profiles = client.get("/api/profiles").json()
        assert "optimized-promo-test" in profiles

        # No registry results → 404; bad source → 422.
        assert client.post("/api/optimizer/promote",
                           json={"source": "registry",
                                 "config_name": "x"}).status_code == 404
        assert client.post("/api/optimizer/promote",
                           json={"source": "?"}).status_code == 422
