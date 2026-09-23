"""KTransformers v0.7 — kt-kernel CPU experts inside SGLang.

The v0.3 server (``ktransformers.server.main``, GGUF weights, its own
``balance_serve`` scheduler) is ARCHIVED upstream. What KTransformers
ships now is ``kt-kernel``, a CPU MoE library with AMX/AVX512 kernels,
served through a fork of SGLang (``sglang-kt``) whose FusedMoE layer
wraps its quant method with a "kt_ep" wrapper: experts the operator
keeps on the GPU load as usual, the rest are skipped by the GPU
loader and served from host RAM by kt-kernel. So the launch is the
ordinary ``python -m sglang.launch_server`` plus ``--kt-*`` flags,
readiness is SGLang's, and /metrics carries SGLang's ``sglang:`` names
-- the parser ``sglang_cuda`` already uses applies unchanged.

Everything below was read off the v0.7.1 tag of kvcache-ai/
ktransformers (kt-kernel/README.md, doc/en/kt-kernel/*.md, docker/
Dockerfile) and the kvcache-ai/sglang fork's server_args.py; the
docs/ktransformers.md write-up carries the citations.

Three things the launcher has to get right that the v0.3 line never
had to think about:

* **The weight format is chosen by ``--kt-method``, not by the
  engine.** Native methods (FP8, FP8_PERCHANNEL, BF16, RAWINT4, MXFP4)
  read the HF checkpoint itself -- ``--kt-weight-path`` is the SAME
  directory as ``--model`` -- so a block-FP8 DeepSeek/Kimi-K2 or an
  INT4 Kimi-K2-Thinking needs no conversion at all. AMXINT4/AMXINT8
  need ``kt-kernel/scripts/convert_cpu_weights.py`` output (hours for
  a 1T model); LLAMAFILE reads a GGUF directory with the portable
  llamafile kernels, not AMX. capsim reads the method off the
  checkpoint's quantization_config and refuses a shape whose weights
  are not staged, in milliseconds, with the fix named.

* **The published image is a DeepSeek-V4 build.** No v0.6/v0.7 tag was
  ever pushed to Docker Hub; ``DSV4-specific`` is the one image of the
  new line, and it is also the one compiled for SM120 (the kt-kernel
  PyPI wheel stops at SM90). Its entrypoint execs any CMD that does
  not start with ``--`` and otherwise assembles a V4-Flash launch, so
  capsim passes the full ``python -m sglang.launch_server`` line and
  sets the CUDA-arch environment the entrypoint would have derived.

* **``--kt-num-gpu-experts`` is per MoE layer and effectively
  required** -- the fork only warns when it is absent and then has no
  expert mask. capsim always passes it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

from .docker_replica import gpus_arg_for

log = logging.getLogger(__name__)

# The only published image of the v0.7 line (Docker Hub
# approachingai/ktransformers, 2026-09-01, 12.0 GB compressed):
# ktransformers / kt-kernel / sglang-kt 0.7.0.post1, torch 2.9.1+cu128,
# CUDA 12.8.1, flashinfer 0.6.15.post1, transformers 4.57.1, kt-kernel
# compiled with CPUINFER_CUDA_ARCHS=80;86;89;90;100;120 and every CPU
# variant (AMX, AVX512, AVX2 chosen at runtime). The tag name is the
# model it was validated on, not a restriction: it is the fork's
# generic launch_server with the V4 parsers added.
DEFAULT_IMAGE = "approachingai/ktransformers:DSV4-specific"

# Where the HF cache lands inside every capsim engine container
# (docker_replica._mount_args).
CONTAINER_HF_CACHE = "/root/.cache/huggingface"

# ``--kt-method`` values, by what they read.
NATIVE_METHODS = ("FP8", "FP8_PERCHANNEL", "BF16", "RAWINT4", "MXFP4")
AMX_METHODS = ("AMXINT4", "AMXINT8")
GGUF_METHOD = "LLAMAFILE"
METHODS = NATIVE_METHODS + AMX_METHODS + (GGUF_METHOD,)

# The image's own entrypoint holds these back for the driver and the
# scheduler; the kt-kernel README says "physical cores, not
# hyperthreads". capsim's default is cores - 2 (ktransformers.py).
LINUX_NUMA_ROOT = "/sys/devices/system/node"


def kt_method_for(config_doc: dict | None) -> str | None:
    """The native ``--kt-method`` a checkpoint's config.json implies,
    or None when kt-kernel has no native path for it (ModelOpt NVFP4,
    AWQ, GPTQ...) -- the caller then refuses with the reason rather
    than guessing a format that would load garbage or not at all.

    Verified against the checkpoints the tutorials launch: DeepSeek-
    V3/V3.1/V3.2 and Kimi-K2-Instruct carry ``quant_method: fp8`` with
    a ``weight_block_size`` (FP8); Kimi-K2-Thinking / K2.5 carry
    compressed-tensors with 4-bit int weights (RAWINT4); DeepSeek-V4-
    Flash carries mxfp4 (MXFP4); an unquantized bf16 Qwen3-MoE / GLM
    is BF16.
    """
    if not isinstance(config_doc, dict):
        return None
    qc = config_doc.get("quantization_config")
    if not qc:
        dtype = str(config_doc.get("torch_dtype")
                    or config_doc.get("dtype") or "").lower()
        return "BF16" if dtype in ("bfloat16", "bf16") else None
    if not isinstance(qc, dict):
        return None
    qm = str(qc.get("quant_method") or "").lower()
    if qm == "fp8":
        return "FP8" if qc.get("weight_block_size") else "FP8_PERCHANNEL"
    if qm == "mxfp4":
        return "MXFP4"
    if qm == "compressed-tensors":
        groups = qc.get("config_groups") or {}
        for g in groups.values() if isinstance(groups, dict) else []:
            w = (g or {}).get("weights") or {}
            bits, typ = w.get("num_bits"), str(w.get("type") or "").lower()
            if bits == 4 and typ == "int":
                return "RAWINT4"
            if bits == 8 and typ == "float":
                return "FP8" if w.get("block_structure") else "FP8_PERCHANNEL"
    return None


def parse_numa_nodes(names: list[str]) -> int:
    """NUMA node count from the entries of /sys/devices/system/node."""
    return sum(1 for n in names if re.fullmatch(r"node\d+", n))


def numa_node_count() -> int | None:
    """NUMA nodes on a Linux host, None elsewhere -- the launch runs
    docker on the host it sizes for, and a Mac's topology says nothing
    about the Xeon that will run the experts."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        n = parse_numa_nodes(os.listdir(LINUX_NUMA_ROOT))
    except OSError:
        return None
    return n or None


def cuda_arch_env(arch: str | None) -> dict[str, str]:
    """The environment the image's entrypoint would have exported for
    this compute capability (docker/Dockerfile, entrypoint-dsv4): torch
    wants ``12.0+PTX``, flashinfer wants the ``a`` feature suffix.
    Passing the CMD ourselves bypasses that logic, so capsim sets the
    same two variables."""
    if not arch:
        return {}
    a = str(arch).strip()
    major = a.split(".")[0]
    torch_arch = a if major in ("8",) else f"{a}+PTX"
    fi_arch = a if major in ("8",) else f"{a}a"
    return {"TORCH_CUDA_ARCH_LIST": torch_arch,
            "FLASHINFER_CUDA_ARCH_LIST": fi_arch}


# model_type / architecture markers of the checkpoint family the image
# was built for; its config-backup hack is right only for these.
V4_MARKERS = ("deepseek_v4", "deepseek_ref", "deepseekv4")


def config_backup_env(config_doc: dict | None) -> dict[str, str]:
    """Turn off the image's DeepSeek-V4 config substitution for every
    other checkpoint.

    The fork's config loader (hf_transformers_utils) sends ANY
    checkpoint whose architectures mention "deepseek" through a
    temporary V4 path that, with SGLANG_APPLY_CONFIG_BACKUP at its
    default ``auto``, replaces the checkpoint's config.json with a
    packaged V4 one picked by layer count. Kimi-K2-Thinking (61 layers,
    DeepseekV3ForCausalLM) came up as V4-large: 128 heads instead of
    64, a 129k vocabulary instead of 164k, sparse attention with fp8
    KV and 64-token pages -- and its KV pool then outgrew the GPU at
    every memory share. ``none`` makes the loader read the checkpoint's
    own config. Unknown config: leave the image's default alone."""
    if not isinstance(config_doc, dict):
        return {}
    marks = [str(config_doc.get("model_type") or "")]
    marks += [str(a) for a in config_doc.get("architectures") or []]
    if any(m in x.lower() for x in marks for m in V4_MARKERS):
        return {}
    return {"SGLANG_APPLY_CONFIG_BACKUP": "none"}


def launch_argv(model: str, *, port: int, tp: int,
                weight_path: str, method: str,
                cpu_infer: int | None = None,
                threadpool_count: int | None = None,
                gpu_experts: int = 0,
                deferred_experts: int | None = None,
                gpu_prefill_threshold: int | None = None,
                dynamic_expert_update: bool = False,
                context_length: int | None = None,
                max_running_requests: int | None = None,
                chunked_prefill_size: int | None = None,
                mem_fraction_static: float | None = None,
                max_total_tokens: int | None = None,
                trust_remote_code: bool = True,
                extra: list[str] | None = None) -> list[str]:
    """The container CMD: ``python -m sglang.launch_server`` with the
    ``--kt-*`` flags the kvcache-ai fork adds.

    Flag names verified against the fork's server_args.py (kt block,
    "Ktransformer server args") and the launch lines in kt-kernel/
    README.md and the DeepSeek-V3.2 / Kimi-K2 tutorials. The fixed
    tail (flashinfer attention, no shared-expert fusion, mixed chunk)
    is what every tutorial launch carries.
    """
    if method not in METHODS:
        raise ValueError(f"unknown --kt-method {method!r}; expected one of "
                         f"{', '.join(METHODS)}")
    argv = [
        "python", "-m", "sglang.launch_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(int(port)),
        "--tensor-parallel-size", str(int(tp)),
        "--kt-method", method,
        "--kt-weight-path", weight_path,
        # Per MoE layer; the fork has no expert mask without it.
        "--kt-num-gpu-experts", str(int(gpu_experts)),
        # Without this there is no /metrics at all.
        "--enable-metrics",
    ]
    if cpu_infer:
        argv += ["--kt-cpuinfer", str(int(cpu_infer))]
    if threadpool_count:
        argv += ["--kt-threadpool-count", str(int(threadpool_count))]
    if deferred_experts is not None:
        argv += ["--kt-max-deferred-experts-per-token", str(int(deferred_experts))]
    if gpu_prefill_threshold is not None:
        argv += ["--kt-gpu-prefill-token-threshold", str(int(gpu_prefill_threshold))]
    if dynamic_expert_update:
        argv += ["--kt-enable-dynamic-expert-update"]
    if context_length:
        argv += ["--context-length", str(int(context_length))]
    if max_running_requests:
        argv += ["--max-running-requests", str(int(max_running_requests))]
    if chunked_prefill_size:
        argv += ["--chunked-prefill-size", str(int(chunked_prefill_size)),
                 "--max-prefill-tokens", str(int(chunked_prefill_size))]
    if mem_fraction_static is not None:
        argv += ["--mem-fraction-static", str(float(mem_fraction_static))]
    if max_total_tokens:
        argv += ["--max-total-tokens", str(int(max_total_tokens))]
    argv += [
        "--attention-backend", "flashinfer",
        "--disable-shared-experts-fusion",
        "--enable-mixed-chunk",
        # A 1T model takes minutes to load and the first long prefill
        # allocates lazily; SGLang's default watchdog (300 s) reads a
        # slow layerwise prefill as a hang.
        "--watchdog-timeout", "3000",
    ]
    if trust_remote_code:
        argv += ["--trust-remote-code"]
    argv += list(extra or [])
    return argv


# ── Staging ───────────────────────────────────────────────────────────

def _has_safetensors(d: Path) -> bool:
    try:
        return any(p.suffix == ".safetensors" for p in d.iterdir())
    except OSError:
        return False


def weights_missing(*, method: str | None, model_dir: str | None,
                    amx_weight_path: str | None,
                    gguf_path: str | None,
                    checkpoint_known: bool = True) -> str | None:
    """Why a v0.7 launch cannot proceed, or None when it can.

    ``model_dir`` is the staged HF checkpoint on THIS host (None when
    nothing is staged); ``checkpoint_known`` is False when the model
    directory is a container path capsim cannot inspect, in which case
    only the method-level checks apply.
    """
    if method in AMX_METHODS:
        if not amx_weight_path:
            return (f"--kt-method {method} reads AMX-converted expert weights "
                    "and no ktransformers_amx_weight_path is configured; run "
                    "kt-kernel/scripts/convert_cpu_weights.py on the checkpoint "
                    "(see docs/ktransformers.md) and point "
                    "ktransformers_amx_weight_path at its output")
        if not Path(str(amx_weight_path)).exists():
            return (f"ktransformers_amx_weight_path {amx_weight_path!r} does "
                    "not exist on this host")
        return None
    if method == GGUF_METHOD:
        if not gguf_path:
            return ("--kt-method LLAMAFILE reads GGUF expert weights and no "
                    "ktransformers_gguf_path is configured")
        if not Path(str(gguf_path)).exists():
            return (f"ktransformers_gguf_path {gguf_path!r} does not exist on "
                    "this host")
        return None
    # Native: the checkpoint IS the CPU weight source, so staging is
    # checked before the format -- an unstaged model has no format.
    if not checkpoint_known:
        return None
    if not model_dir:
        return ("the v0.7 line reads CPU expert weights from the HF checkpoint "
                "itself and nothing is staged; stage the full checkpoint "
                "(config + tokenizer + safetensors) in Prepare, or stage a "
                "GGUF companion and set ktransformers_gguf_path for the v0.3 "
                "line")
    if not _has_safetensors(Path(model_dir)):
        return (f"the v0.7 line reads CPU expert weights from the HF "
                f"checkpoint itself, and {model_dir} holds no safetensors "
                "(a config-only staging serves the v0.3 GGUF line, not this "
                "one); stage the full checkpoint, or set "
                "ktransformers_generation v0.3 with a GGUF companion")
    if method is None:
        return ("kt-kernel has no native CPU-expert path for this checkpoint's "
                "quantization (it serves block-FP8, per-channel FP8, BF16, "
                "compressed-tensors INT4 and MXFP4 natively); stage the "
                "source checkpoint in one of those formats, or convert it "
                "with kt-kernel/scripts/convert_cpu_weights.py and set "
                "ktransformers_kt_method AMXINT8 with "
                "ktransformers_amx_weight_path")
    return None


def staged_snapshot(model_id: str | None) -> Path | None:
    """The host directory of a hub id's newest cached snapshot, or None."""
    if not model_id or "/" not in model_id:
        return None
    from ..models import _latest_snapshot, _model_dir, hf_cache_dir
    return _latest_snapshot(_model_dir(model_id, hf_cache_dir()))


def container_path_for(host_path: Path, volumes: dict | None) -> str:
    """Where ``host_path`` appears inside the container: through an
    explicit docker volume that contains it, else through the HF cache
    mount every engine container gets."""
    host = str(host_path)
    for h, c in (volumes or {}).items():
        h = str(h).rstrip("/")
        if host == h or host.startswith(h + "/"):
            return c.rstrip("/") + host[len(h):]
    from ..models import hf_cache_dir
    cache = str(hf_cache_dir()).rstrip("/")
    if host == cache or host.startswith(cache + "/"):
        return CONTAINER_HF_CACHE + host[len(cache):]
    return host


def resolve_method(cfg, model_dir: Path | None) -> str | None:
    """Explicit ``ktransformers_kt_method`` (upper-cased), else the
    native method the staged config.json implies, else LLAMAFILE when
    only a GGUF companion is configured, else None."""
    explicit = getattr(cfg, "ktransformers_kt_method", None)
    if explicit:
        m = str(explicit).upper()
        if m not in METHODS:
            raise ValueError(f"ktransformers_kt_method {explicit!r} is not one "
                             f"of {', '.join(METHODS)}")
        return m
    if model_dir is not None:
        try:
            doc = json.loads((model_dir / "config.json").read_text())
        except (OSError, ValueError):
            doc = None
        m = kt_method_for(doc)
        if m:
            return m
        if doc is not None and _has_safetensors(model_dir):
            return None
    if getattr(cfg, "ktransformers_gguf_path", None):
        return GGUF_METHOD
    return None


def build_replica_command(engine, index: int, devices: list[int],
                          container_name: str) -> list[str]:
    """The v0.7 ``docker run`` line for replica ``index`` -- called by
    ``KTransformersEngine`` when the generation resolves to v0.7."""
    from .ktransformers import default_cpu_infer
    cfg = engine.cfg
    volumes = getattr(cfg, "docker_volumes", None) or {}

    # Where the checkpoint is, on the host (for inspection) and in the
    # container (for the flags).
    local = getattr(cfg, "model_local_path", None)
    if local:
        host_dir = Path(local) if Path(local).is_dir() else None
        model_in_container = str(local)
        checkpoint_known = host_dir is not None
    else:
        host_dir = staged_snapshot(cfg.model_id)
        model_in_container = (container_path_for(host_dir, volumes)
                              if host_dir else cfg.model_id)
        checkpoint_known = True

    method = resolve_method(cfg, host_dir)
    amx = getattr(cfg, "ktransformers_amx_weight_path", None)
    gguf = getattr(cfg, "ktransformers_gguf_path", None)
    why = weights_missing(method=method,
                          model_dir=str(host_dir) if host_dir else None,
                          amx_weight_path=amx, gguf_path=gguf,
                          checkpoint_known=checkpoint_known)
    if why:
        raise RuntimeError(f"ktransformers (v0.7) cannot launch: {why}")

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", container_name,
        "--gpus", gpus_arg_for(devices),
        # The expert path moves activations through shared memory and
        # the image's own launch uses --ipc host; SYS_NICE lets the
        # thread pools pin to their NUMA node.
        "--ipc=host",
        "--cap-add", "SYS_NICE",
        "--network", "host",
    ]
    cmd += engine._mount_args()
    for k, v in cuda_arch_env(getattr(cfg, "ktransformers_cuda_arch", None)).items():
        cmd += ["-e", f"{k}={v}"]
    doc = None
    if host_dir is not None:
        try:
            doc = json.loads((host_dir / "config.json").read_text())
        except (OSError, ValueError):
            doc = None
    for k, v in config_backup_env(doc).items():
        cmd += ["-e", f"{k}={v}"]
    if method in AMX_METHODS:
        cmd += ["-v", f"{amx}:/kt-weights:ro"]
        weight_path = "/kt-weights"
    elif method == GGUF_METHOD:
        from ..models import container_cache_path
        weight_path = container_cache_path(gguf)
        if weight_path is None:
            cmd += ["-v", f"{gguf}:/gguf:ro"]
            weight_path = "/gguf"
    else:
        weight_path = model_in_container
    cmd += list(cfg.docker_extra_args or [])
    cmd.append(getattr(cfg, "ktransformers_image", None) or DEFAULT_IMAGE)

    cpu_threads = getattr(cfg, "ktransformers_cpu_threads", None) or default_cpu_infer()
    pools = getattr(cfg, "ktransformers_threadpool_count", None) or numa_node_count()

    from .vram import to_engine_fraction
    mem_fraction = to_engine_fraction(
        "sglang_cuda", cfg.gpu_memory_utilization,
        total_vram_gb=getattr(cfg, "vram_per_gpu_gb", None),
        weights_gb=getattr(cfg, "model_weights_gb", None))[0]

    # The fork sizes the KV pool from the memory it sees free after the
    # GPU share of the weights, and on Kimi-K2-Thinking (23 GB on the
    # GPU, 70 GB free) it asked for ~3 GB per layer over 61 layers and
    # died allocating it, at every share. The cell's own concurrency
    # times its context is all the pool can ever hold; cap it there.
    total_tokens = getattr(cfg, "ktransformers_max_total_tokens", None)
    seqs, ctx = getattr(cfg, "max_num_seqs", None), cfg.max_model_len
    if not total_tokens and seqs and ctx:
        total_tokens = int(seqs) * int(ctx)

    return cmd + launch_argv(
        model_in_container,
        port=engine._port(index),
        tp=len(devices),
        weight_path=weight_path,
        method=method,
        cpu_infer=cpu_threads,
        threadpool_count=pools,
        gpu_experts=int(getattr(cfg, "ktransformers_gpu_experts", 0) or 0),
        deferred_experts=getattr(cfg, "ktransformers_deferred_experts", None),
        gpu_prefill_threshold=getattr(cfg, "ktransformers_gpu_prefill_threshold", None),
        dynamic_expert_update=bool(getattr(cfg, "ktransformers_dynamic_expert_update", False)),
        context_length=cfg.max_model_len,
        max_running_requests=getattr(cfg, "max_num_seqs", None),
        chunked_prefill_size=getattr(cfg, "max_num_batched_tokens", None),
        mem_fraction_static=mem_fraction,
        max_total_tokens=total_tokens,
        trust_remote_code=True,
        extra=list(getattr(cfg, "ktransformers_extra_flags", None) or []),
    )


def ready_url(host: str, port: int) -> str:
    """SGLang's liveness endpoint -- the image's own HEALTHCHECK polls
    it. It answers once the scheduler has loaded the model."""
    return f"http://{host}:{port}/health"
