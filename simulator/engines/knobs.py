"""One vocabulary of launch knobs, two engine dialects.

The optimizer, the benchmark form and the headline search all reason
about the same handful of dimensions — how many streams the engine
may batch, how many tokens per step, what precision the KV cache is
kept in, how much VRAM the pool may claim. Those are properties of
the *experiment*, not of vLLM. Before this module each one was
spelled as a vLLM command-line flag at config-build time, which made
"which engine?" unaskable: the flags simply would not survive the
question.

So: a canonical dict in, an engine config fragment out. Adding an
engine means adding a translation here, not touching the search.

Where the dialects genuinely disagree, the disagreement is recorded
rather than smoothed over — see ``gpu_memory_utilization``, which
names different quantities in the two engines and is flagged in
``ENGINE_CAVEATS`` so the UI can say so out loud.
"""

from __future__ import annotations

# Engines the benchmark and optimizer may choose between, in the order
# the UI offers them.
GPU_ENGINES = ("vllm_cuda_multi", "trtllm")

ENGINE_LABELS = {
    "vllm_cuda_multi": "vLLM",
    "vllm_cuda": "vLLM (single engine)",
    "trtllm": "TensorRT-LLM",
}

ENGINE_CAVEATS = {
    "trtllm": [
        "GPU memory fraction means something different here: vLLM's is "
        "a share of TOTAL VRAM covering weights and KV together, "
        "TensorRT-LLM's is the share of what remains FREE AFTER weights "
        "load, for KV alone. The same number is not the same setting.",
        "Throughput is reconstructed from per-iteration stats rather "
        "than read off a cumulative counter, because TensorRT-LLM "
        "exposes no token counter. capsim detects any gap in that "
        "stream and reports the total as a lower bound if one occurs.",
    ],
}


def canonical(custom: dict) -> dict:
    """The engine-neutral knobs out of a UI/benchmark request."""
    def _int(key):
        v = custom.get(key)
        try:
            return int(v) if v not in (None, "", "default") else None
        except (TypeError, ValueError):
            return None

    kv = custom.get("kv_cache_dtype")
    if kv in ("", None, "auto"):
        kv = None
    gmu = custom.get("gpu_memory_utilization")
    try:
        gmu = min(0.98, max(0.5, float(gmu))) if gmu is not None else 0.92
    except (TypeError, ValueError):
        gmu = 0.92
    return {
        "max_num_seqs": _int("max_num_seqs"),
        "max_num_batched_tokens": _int("max_num_batched_tokens"),
        "max_model_len": _int("max_model_len") or 16384,
        "gpu_memory_utilization": gmu,
        "kv_cache_dtype": kv,
        "expert_parallel": bool(custom.get("expert_parallel")),
        "trust_remote_code": bool(custom.get("trust_remote_code")),
    }


def _vllm_flags(k: dict) -> list[str]:
    flags: list[str] = []
    if k.get("max_num_seqs"):
        flags += ["--max-num-seqs", str(k["max_num_seqs"])]
    if k.get("max_num_batched_tokens"):
        flags += ["--max-num-batched-tokens",
                  str(k["max_num_batched_tokens"])]
    if k.get("kv_cache_dtype"):
        flags += ["--kv-cache-dtype", str(k["kv_cache_dtype"])]
    if k.get("expert_parallel"):
        flags += ["--enable-expert-parallel"]
    if k.get("trust_remote_code"):
        # Executes model-repo Python inside the engine container, so
        # it is opt-in per run and recorded in the run's engine
        # summary — never a silent default.
        flags += ["--trust-remote-code"]
    return flags


def to_engine_config(engine_type: str, knobs: dict) -> dict:
    """Engine config fields implementing ``knobs`` on ``engine_type``.

    The canonical values are ALSO written onto the config, for every
    engine, so a stored run answers "what shape was this?" without
    anyone having to parse a flag list back out.
    """
    k = dict(knobs)
    out: dict = {
        "max_model_len": k.get("max_model_len") or 16384,
        "gpu_memory_utilization": k.get("gpu_memory_utilization", 0.92),
        # Provenance: the searched shape, engine-independent.
        "max_num_seqs": k.get("max_num_seqs"),
        "max_num_batched_tokens": k.get("max_num_batched_tokens"),
        "kv_cache_dtype": k.get("kv_cache_dtype"),
        "expert_parallel": bool(k.get("expert_parallel")),
        "trust_remote_code": bool(k.get("trust_remote_code")),
    }
    if engine_type == "trtllm":
        # trtllm-serve reads these off the config object directly
        # (see engines/trtllm.build_replica_command), so no flag
        # translation is needed beyond what is already above.
        return out
    flags = _vllm_flags(k)
    if flags:
        out["vllm_extra_flags"] = flags
    return out


def caveats(engine_type: str) -> list[str]:
    return list(ENGINE_CAVEATS.get(engine_type, []))
