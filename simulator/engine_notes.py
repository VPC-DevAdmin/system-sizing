"""Measured tuning levers, and what they actually did on this box.

Every entry here was measured, not read off a blog. The point is that
the obvious levers are not obviously good: on a 70B model with 42 GB
of weights on a 96 GB card, three separate TensorRT-LLM knobs that
look like straightforward throughput wins each made things markedly
worse, and two of them by more than half. A search that offers a knob
without saying what it did last time invites the same afternoon to be
spent twice.

So each lever carries three things: what it does, what it MEASURED
here, and whether it is worth searching. ``searchable`` levers become
arena dimensions; the rest are documentation attached to the engine
card. A lever measured harmful stays offered -- it may well be the
right trade on a smaller model, where weights leave room for both --
but it is off by default and says why.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Lever:
    """One engine knob, with its evidence."""
    key: str                  # arena dimension name
    engine: str
    title: str
    values: list[str]
    default: str
    text: str                 # what the knob does
    measured: str = ""        # what it did HERE, with numbers
    searchable: bool = False  # becomes an arena dimension
    verdict: str = ""         # "help" | "harm" | "required" | "untested"


LEVERS: list[Lever] = [
    Lever(
        key="trtllm_chunked_prefill", engine="trtllm",
        title="TensorRT · chunked prefill",
        values=["off", "on"], default="off", searchable=True,
        verdict="harm",
        text="Lets a long prompt be prefilled in slices so decode is "
             "not stalled behind it. Sounds like a pure win for a "
             "workload mixing prefill and decode.",
        measured="Halved throughput. TensorRT-LLM sizes its KV pool in "
                 "two phases — an estimation dry run, then the real "
                 "allocation — and with this on, the real pool never "
                 "grows past the dry run: 32,768 tokens instead of "
                 "304,160, a ninefold cut. The engine then admits ~128 "
                 "requests per replica instead of a thousand. "
                 "18,397 tok/s against 37,494 with it off.",
    ),
    Lever(
        key="trtllm_cuda_graphs", engine="trtllm",
        title="TensorRT · CUDA graph capture",
        values=["default", "wide"], default="default", searchable=True,
        verdict="untested",
        text="Replays decode steps as a captured graph instead of "
             "dispatching kernels each iteration. 'wide' captures up "
             "to the full batch width; 'default' lets the engine "
             "choose its own set of sizes.",
        measured="Not isolated. Graphs are ON by default either way — "
                 "34 batch sizes were captured in both the fast and "
                 "the slow run — so setting this only changes WHICH "
                 "sizes are captured. An earlier collapse was blamed "
                 "on it and turned out to be chunked prefill; the two "
                 "had been changed together.",
    ),
    Lever(
        key="trtllm_postprocess_workers", engine="trtllm",
        title="TensorRT · detokenisation workers",
        values=["0", "4"], default="0", searchable=True,
        verdict="harm",
        text="Moves detokenisation off the serving loop into worker "
             "processes. The loop is a plausible CPU bottleneck at a "
             "thousand streams per replica.",
        measured="Cut throughput roughly in half: 19,393 tok/s against "
                 "37,494. The KV pool was healthy (2.1M tokens, 11% "
                 "used) but the engine held only 1,137 of 4,096 "
                 "offered streams — per-token IPC to the workers slows "
                 "each response enough that a closed loop cannot keep "
                 "the batch full. Streaming many small tokens is the "
                 "worst case for this knob.",
    ),
    Lever(
        key="trtllm_moe_backend", engine="trtllm",
        title="TensorRT · MoE backend",
        values=["auto", "CUTLASS", "TRTLLM", "VANILLA"], default="auto",
        searchable=True, verdict="harm",
        text="Which kernel family serves the mixture-of-experts GEMMs. "
             "'auto' leaves the engine's own selection alone.",
        measured="Does not help, and the workaround was tested rather "
                 "than assumed. TensorRT-LLM 1.2.1 routes FP8 "
                 "block-scale MoE GEMMs through DeepGEMM, which "
                 "refuses this hardware: 'DeepGEMM only supports "
                 "Hopper (SM90) architectures, but current device "
                 "compute capability is 120'. Setting the backend to "
                 "CUTLASS explicitly — verified present in the "
                 "options document and named 232 times in the engine "
                 "log — does NOT change the selection: DeepGEMM still "
                 "fires and the replica still dies. FP8 MoE is "
                 "unserviceable on SM120 with this release, and no "
                 "top-level setting reaches it.",
    ),
    Lever(
        key="sglang_quantization", engine="sglang_cuda",
        title="SGLang · ModelOpt quantization",
        values=["auto", "modelopt_fp4"], default="auto",
        searchable=False, verdict="required",
        text="Names the checkpoint's quantization explicitly rather "
             "than relying on auto-detection.",
        measured="REQUIRED for NVIDIA NVFP4 checkpoints. They carry "
                 "quantization_config: null in config.json and declare "
                 "themselves only in the hf_quant_config sidecar, "
                 "which the auto-detection path does not read. vLLM "
                 "happens to look there; SGLang does not. capsim now "
                 "derives it from the catalog's precision label.",
    ),
    Lever(
        key="sglang_mem_fraction", engine="sglang_cuda",
        title="SGLang · static memory fraction",
        values=["translated", "raw"], default="translated",
        searchable=False, verdict="required",
        text="SGLang's --mem-fraction-static covers weights and KV "
             "only; activations and captured CUDA graphs are allocated "
             "on top of it.",
        measured="Passing vLLM's 0.95 through unchanged is an OOM, not "
                 "a tighter fit: 105 MiB free of 94.97 GiB and the "
                 "server died allocating a 448 MiB workspace. capsim "
                 "holds 7% back so the same request means the same "
                 "total footprint on both engines.",
    ),
    Lever(
        key="sglang_nccl_port", engine="sglang_cuda",
        title="SGLang · rendezvous port",
        values=["per-replica", "auto"], default="per-replica",
        searchable=False, verdict="required",
        text="Fixes each replica's torch.distributed port instead of "
             "letting it pick one at random.",
        measured="Eight replicas starting together on host networking "
                 "race for a random free port; two pick the same "
                 "number before either binds and the loser dies with "
                 "EADDRINUSE. Cost one candidate of an eight-candidate "
                 "search before it was fixed.",
    ),
]

# Engine-level narrative for the arena's engine card.
ENGINE_NOTES: dict[str, str] = {
    "vllm_cuda_multi":
        "The reference. Every capsim result before today was measured "
        "on it, and on this box it is the only engine that scaled "
        "cleanly to 8,192 concurrent streams — the others peak around "
        "4,096 and one collapses past it.",
    "trtllm":
        "NVIDIA's own server, and the engine vendor headline numbers "
        "are usually quoted on. Its stock configuration is already "
        "close to right for a large model: every top-level lever tried "
        "here made it slower, twice by more than half, and on the "
        "dense Llama-70B it landed ~12% behind vLLM. MoE needs the "
        "CUTLASS backend named explicitly, or it routes through a "
        "Hopper-only kernel and will not start at all; and a model "
        "newer than the container's Transformers is simply unknown to "
        "it, which is a staleness problem rather than a hardware one.",
    "sglang_cuda":
        "RadixAttention prefix caching and an aggressive scheduler. "
        "Competitive at moderate concurrency — it beat vLLM at 4,096 "
        "streams — but it collapsed at 8,192 with KV only 63% full, "
        "so something in its scheduling rather than memory binds "
        "first. Run-to-run spread was 29% against ~8% for the others.",
    "ktransformers":
        "Heterogeneous: MoE experts on the CPU, attention on the GPU. "
        "It answers a question the others cannot — whether a model far "
        "larger than VRAM can be served at all — and its own docs "
        "demonstrate a max batch of 4, so ranking it on tokens/sec "
        "against the GPU-resident engines measures the wrong thing.",
}


def levers_for(engine: str) -> list[Lever]:
    return [x for x in LEVERS if x.engine == engine]


def searchable_dimensions(engines: list[str]) -> dict[str, list[str]]:
    """Arena dimensions contributed by the staged engines' levers."""
    out: dict[str, list[str]] = {}
    for lv in LEVERS:
        if lv.searchable and lv.engine in engines:
            out[lv.key] = list(lv.values)
    return out


def as_dicts(engine: str | None = None) -> list[dict]:
    """JSON view for the API and the arena cards."""
    rows = LEVERS if engine is None else levers_for(engine)
    return [{
        "key": lv.key, "engine": lv.engine, "title": lv.title,
        "values": lv.values, "default": lv.default, "text": lv.text,
        "measured": lv.measured, "searchable": lv.searchable,
        "verdict": lv.verdict,
    } for lv in rows]
