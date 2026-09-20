"""One memory knob, translated per engine — and checked afterwards.

The operator sets a share of TOTAL VRAM. vLLM and SGLang mean exactly
that; TensorRT-LLM's flag sizes the KV pool against memory left FREE
AFTER weights. Passing the same digits to both is two different
experiments reported as one comparison.
"""

from __future__ import annotations

import pytest

from simulator.engines.vram import (
    to_engine_fraction,
    weights_per_gpu_gb,
)

T = 95.6          # RTX PRO 6000 Blackwell
W = 21.0          # Qwen3.6-35B-A3B-NVFP4 weights, one GPU


def _footprint_gb(engine: str, f: float) -> float:
    """Total VRAM the engine ends up occupying, given its own flag.

    This -- not the KV pool -- is what the operator's knob means, and
    it is the only quantity all three engines can be held to. vLLM's
    fraction already includes activation workspace; SGLang's excludes
    it, so its held-back reserve counts toward the footprint; and
    TensorRT-LLM's governs KV on top of weights.
    """
    value, _ = to_engine_fraction(engine, f, total_vram_gb=T, weights_gb=W)
    assert value is not None
    if engine == "trtllm":
        return W + value * (T - W)
    if engine == "sglang_cuda":
        from simulator.engines.vram import SGLANG_ACTIVATION_RESERVE
        return value * T + SGLANG_ACTIVATION_RESERVE * T
    return value * T


def test_the_same_request_yields_the_same_total_footprint():
    """One number in, the same share of the card out.

    Equal KV pools would be the stronger claim, and it is the one this
    test used to make -- but it was only true while SGLang's fraction
    was believed to include activations. It does not, and asserting
    equal KV is what let a configuration through that could not launch
    at all. Equal footprint is what the knob actually promises; whether
    the pools really landed level is checked against the engines' own
    kv_cache_tokens, not against arithmetic."""
    for f in (0.85, 0.90, 0.95):
        base = _footprint_gb("vllm_cuda_multi", f)
        assert _footprint_gb("trtllm", f) == pytest.approx(base, rel=1e-6)
        assert _footprint_gb("sglang_cuda", f) == pytest.approx(base,
                                                                rel=1e-6)


def test_engines_that_already_mean_total_vram_pass_through():
    """No translation where none is needed — every result capsim has
    recorded keeps its meaning."""
    for engine in ("vllm_cuda_multi", "vllm_cuda"):
        value, why = to_engine_fraction(engine, 0.95, total_vram_gb=T,
                                        weights_gb=W)
        assert value == 0.95
        assert "total VRAM" in why


def test_translation_differs_from_passing_it_through():
    """If the two were interchangeable this module would be pointless.
    The gap widens with the weight footprint, which is exactly when a
    comparison is most likely to be run."""
    small, _ = to_engine_fraction("trtllm", 0.95, total_vram_gb=T,
                                  weights_gb=10.0)
    large, _ = to_engine_fraction("trtllm", 0.95, total_vram_gb=T,
                                  weights_gb=60.0)
    assert small != pytest.approx(0.95, abs=1e-3)
    assert large < small                       # heavier weights, less free
    assert large == pytest.approx((0.95 * T - 60.0) / (T - 60.0), rel=1e-6)


def test_weights_shard_with_tensor_parallelism_not_replicas():
    """tp4 puts a quarter of the weights on each card. Data-parallel
    replicas each carry a FULL copy on their own GPUs, so replica
    count must not appear in this calculation."""
    assert weights_per_gpu_gb(80.0, 1) == 80.0
    assert weights_per_gpu_gb(80.0, 4) == 20.0
    assert weights_per_gpu_gb(None, 4) is None
    assert weights_per_gpu_gb(0, 4) is None


def test_a_shape_with_no_room_for_kv_is_refused():
    """Better a refusal than a launch that OOMs, or one that silently
    builds a KV pool too small to mean anything."""
    value, why = to_engine_fraction("trtllm", 0.5, total_vram_gb=T,
                                    weights_gb=60.0)
    assert value is None
    assert "does not cover" in why

    value, why = to_engine_fraction("trtllm", 0.95, total_vram_gb=T,
                                    weights_gb=120.0)
    assert value is None
    assert "raise tensor parallelism" in why


def test_unknown_sizes_fall_back_to_the_engine_default():
    """A guessed conversion that looks authoritative is worse than
    admitting the inputs are missing."""
    value, why = to_engine_fraction("trtllm", 0.95, total_vram_gb=None,
                                    weights_gb=W)
    assert value is None
    assert "could not translate" in why

    from simulator.config import EngineConfig
    from simulator.engines.trtllm import llm_api_options
    opts = llm_api_options(EngineConfig(type="trtllm", model_id="m",
                                        gpu_memory_utilization=0.95))
    # The flag is omitted entirely rather than set to a made-up value.
    assert "free_gpu_memory_fraction" not in opts.get("kv_cache_config", {})


def test_trtllm_options_carry_the_translated_value():
    from simulator.config import EngineConfig
    from simulator.engines.trtllm import llm_api_options

    opts = llm_api_options(EngineConfig(
        type="trtllm", model_id="m", gpu_memory_utilization=0.95,
        vram_per_gpu_gb=T, model_weights_gb=W))
    got = opts["kv_cache_config"]["free_gpu_memory_fraction"]
    assert got != 0.95                                  # not passed through
    assert got == pytest.approx((0.95 * T - W) / (T - W), abs=1e-4)


# ── the translation is checked, not trusted ───────────────────────────

def test_every_engine_reports_the_pool_it_actually_built():
    """Arithmetic nobody verified is worth very little. Each engine
    exposes its KV capacity in tokens, so two engines given 'the same'
    share can be confirmed to hold comparable room."""
    from simulator.engines.base import Engine
    from simulator.engines.trtllm import _Acc, accumulate, snapshot

    # vLLM hides it in LABELS on an Info metric whose value is 1.0.
    vllm = Engine._parse_prometheus(
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="278400"} 1.0')
    assert vllm["kv_cache_tokens"] == 278400 * 16

    sgl = Engine._parse_prometheus("sglang:max_total_num_tokens 4456448.0")
    assert sgl["kv_cache_tokens"] == 4456448.0

    acc = _Acc()
    accumulate([{"iter": 1, "kvCacheStats": {
        "usedNumBlocks": 10, "maxNumBlocks": 139200,
        "tokensPerBlock": 32}}], acc)
    assert snapshot(acc)["kv_cache_tokens"] == 139200 * 32


def test_kv_capacity_sums_across_replicas():
    """Capacity is per replica; the box holds their sum."""
    from simulator.engines.docker_replica import aggregate_replica_metrics

    agg = aggregate_replica_metrics([{"kv_cache_tokens": 1000.0}] * 8)
    assert agg["kv_cache_tokens"] == 8000.0


def test_sglang_holds_back_room_for_activations():
    """--mem-fraction-static covers weights and KV only; activations
    and captured CUDA graphs land on top of it. Passing vLLM's number
    through is not a slightly different allocation, it is an OOM:
    measured on this box, 0.95 left 105 MiB free of 94.97 GiB and the
    server died allocating a 448 MiB workspace."""
    from simulator.engines.vram import SGLANG_ACTIVATION_RESERVE, to_engine_fraction

    value, why = to_engine_fraction("sglang_cuda", 0.95,
                                    total_vram_gb=T, weights_gb=W)
    assert value == pytest.approx(0.95 - SGLANG_ACTIVATION_RESERVE)
    assert value < 0.95
    assert "on top" in why.lower()
    # Enough headroom to matter on a 96 GB card.
    assert (0.95 - value) * T > 5.0


def test_sglang_launch_uses_the_translated_fraction():
    from simulator.config import EngineConfig
    from simulator.engines.sglang_cuda import SGLangCudaEngine
    from simulator.engines.vram import SGLANG_ACTIVATION_RESERVE

    eng = SGLangCudaEngine(EngineConfig(
        type="sglang_cuda", model_id="org/M", replica_devices=[[0]],
        gpu_memory_utilization=0.95, vram_per_gpu_gb=T,
        model_weights_gb=W, max_model_len=8192))
    cmd = eng.build_replica_command(0, [0], "sglang-r0-x")
    got = float(cmd[cmd.index("--mem-fraction-static") + 1])
    assert got == pytest.approx(0.95 - SGLANG_ACTIVATION_RESERVE)


def test_trtllm_options_file_is_mounted_by_absolute_path(tmp_path):
    """Docker reads a relative -v source as a NAMED VOLUME and refuses
    it: "includes invalid characters for a local volume name". capsim's
    run directories are relative throughout, so this is the one place
    it must be made absolute."""
    import os

    from simulator.config import EngineConfig
    from simulator.engines.trtllm import TrtLlmEngine

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        eng = TrtLlmEngine(EngineConfig(
            type="trtllm", model_id="org/M", replica_devices=[[0]]))
        eng.launch  # noqa: B018 - not launching; just the path rule
        from pathlib import Path
        eng._opts_path = (Path("runs/run_1") / "opts.yaml").resolve()
        cmd = eng.build_replica_command(0, [0], "trtllm-r0-x")
        mount = next(c for c in cmd if "capsim-trtllm" in c)
        assert mount.startswith("/"), mount
    finally:
        os.chdir(cwd)
