"""Every offered engine must be the engine that actually runs.

This file exists because it did not. The benchmark config builder
special-cased TensorRT-LLM and let everything else fall through to
vLLM, so adding SGLang and KTransformers to the picker silently routed
both to vllm_cuda_multi. The runs launched, measured and reported as
though the requested engine had been used -- an engine comparison in
which two of the arms were the same engine.

Nothing about that failed loudly. That is the point: the guard has to
be a test, because the symptom is a plausible number.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simulator.engines.knobs import ENGINE_LABELS, GPU_ENGINES


@pytest.fixture()
def hw(monkeypatch):
    import simulator.arena as arena
    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 95.6})


def _build(engine: str, tmp_path: Path, **extra) -> dict:
    from simulator.service import _build_custom_config
    custom = {"model_id": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
              "device": "gpu", "engine": engine, "replicas": 8, "tp": 1,
              "gpu_memory_utilization": 0.95, "max_model_len": 8192}
    custom.update(extra)
    path = _build_custom_config(custom, tmp_path)
    return yaml.safe_load(path.read_text())["engine"]


@pytest.mark.parametrize("engine", sorted(GPU_ENGINES))
def test_requested_engine_is_the_engine_configured(engine, hw, tmp_path):
    """The bug: sglang_cuda and ktransformers both came back as
    vllm_cuda_multi, and every downstream artifact agreed."""
    assert _build(engine, tmp_path)["type"] == engine


@pytest.mark.parametrize("engine", sorted(GPU_ENGINES))
def test_every_offered_engine_can_actually_be_built(engine, hw, tmp_path):
    """An engine in the picker must construct. A name offered in the
    UI that no factory knows is a failure the operator only discovers
    minutes into a run."""
    from simulator.config import EngineConfig
    from simulator.engines import make_engine

    doc = _build(engine, tmp_path)
    eng = make_engine(doc["type"], EngineConfig(**doc))
    assert eng is not None
    # Whole-box engines describe themselves by replica, so the routing
    # must have carried the device assignment through.
    assert len(eng.replica_urls) == 8


@pytest.mark.parametrize("engine", sorted(GPU_ENGINES))
def test_each_engine_launches_its_own_binary(engine, hw, tmp_path):
    """Two engines that produce the same argv are the same engine
    wearing a different label."""
    from simulator.config import EngineConfig
    from simulator.engines import make_engine

    doc = _build(engine, tmp_path)
    eng = make_engine(doc["type"], EngineConfig(**doc))
    cmd = " ".join(eng.build_replica_command(0, [0], f"{engine}-r0-x"))
    marker = {
        "vllm_cuda_multi": "vllm/vllm-openai",
        "trtllm": "trtllm-serve",
        "sglang_cuda": "sglang.launch_server",
        "ktransformers": "ktransformers.server.main",
    }[engine]
    assert marker in cmd


def test_every_engine_has_a_label():
    """The picker renders these; a missing one shows a raw key."""
    for engine in GPU_ENGINES:
        assert ENGINE_LABELS.get(engine)


def test_an_unknown_engine_is_refused_not_defaulted():
    """Falling back to vLLM is exactly how the original defect
    produced confident, wrong numbers."""
    from fastapi import HTTPException
    from simulator.service import _build_custom_config

    with pytest.raises(HTTPException) as e:
        _build_custom_config({"model_id": "org/M", "device": "gpu",
                              "engine": "not_an_engine", "replicas": 1,
                              "tp": 1}, Path("/tmp"))
    assert e.value.status_code == 422
