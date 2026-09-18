"""One memory knob, translated per engine.

The operator sets ONE number: the share of each GPU's total VRAM the
engine may occupy — weights, KV pool and workspace together. That is
what vLLM's ``--gpu-memory-utilization`` has always meant, so every
result capsim has ever recorded keeps its meaning, and it is the
quantity a person can actually reason about ("give it 95% of the
card").

Two of the three engines take that number directly. TensorRT-LLM does
not: its ``free_gpu_memory_fraction`` is the share of whatever is
still FREE ONCE WEIGHTS ARE LOADED, and it governs the KV pool alone.
The same digits mean materially different allocations, so passing 0.95
to both is not one experiment run twice — it is two different
experiments reported as a comparison.

Asking the operator to convert is how that mistake gets made. The
conversion needs only the card size and the weight footprint, both of
which capsim already knows, so it belongs here:

    vLLM / SGLang:   used  = f · T                (weights + KV)
    TensorRT-LLM:    KV    = g · (T − W)

    equal KV  ⟹  g = (f · T − W) / (T − W)

where T is per-GPU VRAM and W the weights resident on that GPU (the
model's size divided by its tensor-parallel width).

This is an estimate, not an identity: engines reserve activation and
communication workspace that no catalog records, and TensorRT-LLM
measures "free" after its own allocations rather than after weights
alone. So the translation is checked rather than trusted — every
engine reports the KV pool it actually built (see ``kv_cache_tokens``
in the metric parsers), and comparing those numbers is what confirms
two engines were given the same room. An engine comparison resting on
arithmetic nobody verified is worth very little.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Engines whose memory knob already means "share of total VRAM",
# INCLUDING the activation and CUDA-graph workspace.
DIRECT_FRACTION_ENGINES = ("vllm_cuda", "vllm_cuda_multi")

# SGLang's --mem-fraction-static covers weights and the KV pool only;
# activations and captured CUDA graphs are allocated ON TOP of it. So
# it is not interchangeable with vLLM's fraction even though both are
# shares of total VRAM, and passing vLLM's number through is an OOM,
# not a slightly different allocation.
#
# Measured on this box: --mem-fraction-static 0.95 on Llama-3.3-70B
# NVFP4 left 105 MiB free of 94.97 GiB and died allocating a 448 MiB
# workspace. The reserve below is empirical, not derived -- there is
# no catalog figure for activation working set, and it grows with
# batch size -- so it is deliberately generous and overridable.
SGLANG_ACTIVATION_RESERVE = 0.07

# Leave the engine some room; 1.0 is an OOM in every implementation.
MAX_FRACTION = 0.98


def weights_per_gpu_gb(model_size_gb: float | None,
                       tensor_parallel: int = 1) -> float | None:
    """Weight bytes resident on ONE GPU.

    Tensor parallelism shards the weights, so tp4 puts a quarter of
    them on each card. Data-parallel replicas do not: each replica
    carries a full copy on its own GPUs, which is why replica count
    does not appear here.
    """
    if not model_size_gb or model_size_gb <= 0:
        return None
    tp = max(1, int(tensor_parallel or 1))
    return float(model_size_gb) / tp


def to_engine_fraction(engine: str, fraction: float, *,
                       total_vram_gb: float | None,
                       weights_gb: float | None) -> tuple[float | None, str]:
    """(value for this engine, human explanation).

    Returns ``(None, reason)`` when the requested share cannot hold
    the weights at all — a launch that would OOM, or one whose KV pool
    would be zero, is refused rather than attempted.
    """
    f = max(0.0, min(MAX_FRACTION, float(fraction)))
    if engine in DIRECT_FRACTION_ENGINES:
        return f, (f"{f:.2f} of total VRAM for weights + KV "
                   f"(this engine's own meaning)")
    if engine == "sglang_cuda":
        g = max(0.05, min(MAX_FRACTION, f - SGLANG_ACTIVATION_RESERVE))
        return g, (f"{g:.2f} static, not {f:.2f}: SGLang allocates "
                   f"activations and CUDA graphs ON TOP of this "
                   f"fraction, so {int(SGLANG_ACTIVATION_RESERVE * 100)}% "
                   f"is held back for them")
    if engine != "trtllm":
        return f, f"{f:.2f} of total VRAM"

    if not total_vram_gb or not weights_gb:
        # Without both numbers the conversion is a guess. TensorRT-LLM's
        # own default is the honest fallback, and saying so beats
        # inventing a number that looks authoritative.
        log.warning(
            "trtllm memory fraction not translated (vram=%s, weights=%s) "
            "— using the engine default", total_vram_gb, weights_gb)
        return None, ("could not translate: per-GPU VRAM or weight size "
                      "unknown, so TensorRT-LLM's own default is used")

    T, W = float(total_vram_gb), float(weights_gb)
    if W >= T:
        return None, (f"weights need {W:.0f} GB but each GPU has "
                      f"{T:.0f} GB — raise tensor parallelism")
    budget = f * T
    if budget <= W:
        return None, (f"{f:.2f} of {T:.0f} GB is {budget:.0f} GB, which "
                      f"does not cover {W:.0f} GB of weights — nothing "
                      f"would be left for KV")
    g = (budget - W) / (T - W)
    g = max(0.01, min(MAX_FRACTION, g))
    return g, (f"{f:.2f} of {T:.0f} GB total, minus {W:.0f} GB of "
               f"weights, leaves {budget - W:.0f} GB of KV — which is "
               f"{g:.2f} of the {T - W:.0f} GB TensorRT-LLM sees free")


def explain(engine: str, fraction: float, *, total_vram_gb: float | None,
            weights_gb: float | None) -> str:
    """Just the sentence, for UI and run metadata."""
    return to_engine_fraction(engine, fraction, total_vram_gb=total_vram_gb,
                              weights_gb=weights_gb)[1]
