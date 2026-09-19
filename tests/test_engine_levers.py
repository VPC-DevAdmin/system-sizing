"""Tuning levers carry their evidence, and default to doing nothing.

Every lever in engine_notes was measured on real hardware, and the
ones marked "harm" each made throughput markedly worse. They stay
offered because a smaller model may trade differently -- but a knob
offered WITHOUT what it did last time invites the same afternoon to
be spent discovering its cost twice.
"""

from __future__ import annotations

import pytest

from simulator.engine_notes import (
    ENGINE_NOTES,
    LEVERS,
    as_dicts,
    levers_for,
    searchable_dimensions,
)


def test_every_lever_carries_evidence():
    for lv in LEVERS:
        assert lv.text, lv.key
        assert lv.verdict in ("help", "harm", "required", "untested"), lv.key
        # A verdict that is not "untested" is a claim, and a claim needs
        # its numbers attached.
        if lv.verdict != "untested":
            assert lv.measured, f"{lv.key} claims '{lv.verdict}' with no evidence"


def test_harmful_levers_default_to_off():
    """The default must be the behaviour we measured as fastest."""
    for lv in LEVERS:
        if lv.verdict == "harm":
            assert lv.default == lv.values[0], lv.key
            # All spellings of "leave the engine alone".
            assert lv.default in ("off", "0", "default", "auto"), lv.key


def test_levers_only_appear_for_staged_engines():
    """A dimension for an engine that is not on the box is budget spent
    discovering a download it cannot do."""
    assert searchable_dimensions([]) == {}
    only_vllm = searchable_dimensions(["vllm_cuda_multi"])
    assert only_vllm == {}
    with_trt = searchable_dimensions(["trtllm"])
    assert "trtllm_chunked_prefill" in with_trt
    assert all(k.startswith("trtllm_") for k in with_trt)


def test_engine_config_defaults_change_nothing():
    """With every lever at its default the options document must be
    identical to the untuned engine's -- otherwise the 'default' is a
    silent third configuration nobody measured."""
    from simulator.config import EngineConfig
    from simulator.engines.trtllm import llm_api_options

    base = dict(type="trtllm", model_id="m", max_num_seqs=1024,
                gpu_memory_utilization=0.95, vram_per_gpu_gb=95.6,
                model_weights_gb=42.0)
    opts = llm_api_options(EngineConfig(**base))
    for key in ("enable_chunked_prefill", "num_postprocess_workers",
                "cuda_graph_config"):
        assert key not in opts, key


@pytest.mark.parametrize("lever,field,expected", [
    ("trtllm_chunked_prefill", "enable_chunked_prefill", True),
    ("trtllm_postprocess_workers", "num_postprocess_workers", 4),
])
def test_turning_a_lever_on_reaches_the_engine(lever, field, expected):
    from simulator.config import EngineConfig
    from simulator.engines.trtllm import llm_api_options

    kw = {lever: (True if isinstance(expected, bool) else expected)}
    opts = llm_api_options(EngineConfig(
        type="trtllm", model_id="m", max_num_seqs=1024, **kw))
    assert opts[field] == expected


def test_search_space_knows_the_lever_defaults():
    """An absent dimension must fall back to the measured-best value,
    not to whatever the engine would otherwise do."""
    from simulator.search import KNOWN_DIMENSIONS

    for lv in LEVERS:
        if lv.searchable:
            assert lv.key in KNOWN_DIMENSIONS, lv.key


def test_every_staged_engine_has_a_narrative():
    from simulator.engines.knobs import GPU_ENGINES

    for engine in GPU_ENGINES:
        assert ENGINE_NOTES.get(engine), engine


def test_api_view_is_serialisable():
    rows = as_dicts()
    assert rows and all(isinstance(r["values"], list) for r in rows)
    assert {r["engine"] for r in rows} <= {"trtllm", "sglang_cuda",
                                           "vllm_cuda_multi", "ktransformers"}
    assert levers_for("trtllm")


def test_levers_reach_the_engine_from_a_benchmark_request():
    """They were wired into the arena search and NOT into the direct
    config path -- which is where an operator is most likely to reach
    for them. A roofline passing trtllm_moe_backend had it silently
    dropped, and the run failed exactly as it had without it."""
    import tempfile
    from pathlib import Path

    import yaml

    import simulator.arena as arena
    from simulator.service import _build_custom_config

    real = arena.hardware
    arena.hardware = lambda: {"count": 8, "vram_per_gpu_gb": 95.6,
                              "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]]}
    try:
        with tempfile.TemporaryDirectory() as td:
            p = _build_custom_config({
                "model_id": "org/M", "device": "gpu", "engine": "trtllm",
                "replicas": 8, "tp": 1,
                "trtllm_moe_backend": "CUTLASS",
                "trtllm_chunked_prefill": True,
            }, Path(td))
            eng = yaml.safe_load(p.read_text())["engine"]
    finally:
        arena.hardware = real

    assert eng["trtllm_moe_backend"] == "CUTLASS"
    assert eng["trtllm_chunked_prefill"] is True

    # ...and it actually shapes the options document the server reads.
    from simulator.config import EngineConfig
    from simulator.engines.trtllm import llm_api_options
    opts = llm_api_options(EngineConfig(**eng))
    assert opts["moe_config"]["backend"] == "CUTLASS"


def test_an_unknown_key_cannot_inject_a_setting():
    """A typo must not become a silent engine option."""
    import tempfile
    from pathlib import Path

    import yaml

    import simulator.arena as arena
    from simulator.service import _build_custom_config

    real = arena.hardware
    arena.hardware = lambda: {"count": 8, "vram_per_gpu_gb": 95.6,
                              "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]]}
    try:
        with tempfile.TemporaryDirectory() as td:
            p = _build_custom_config({
                "model_id": "org/M", "device": "gpu", "engine": "trtllm",
                "replicas": 8, "tp": 1,
                "trtllm_not_a_real_knob": "boom",
            }, Path(td))
            eng = yaml.safe_load(p.read_text())["engine"]
    finally:
        arena.hardware = real
    assert "trtllm_not_a_real_knob" not in eng
