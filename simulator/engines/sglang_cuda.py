"""SGLang on GPUs — the whole box as N independent replicas.

The existing ``sglang`` engine is the CPU path: it pins to NUMA
nodes, passes ``--device cpu`` and runs a locally built Xeon image.
None of that applies here. This is the same lifecycle as
``vllm_cuda_multi`` (see ``docker_replica``) against upstream's CUDA
image, so SGLang can be measured against vLLM and TensorRT-LLM on
identical terms.

Two places where SGLang is genuinely NOT interchangeable with the
others, both encoded rather than glossed:

* **KV cache dtype has a different vocabulary.** vLLM takes ``fp8``
  and picks a representation; SGLang wants the representation named
  outright (``fp8_e5m2`` / ``fp8_e4m3``) and has no nvfp4 path at
  all. Passing vLLM's spelling straight through is a launch failure,
  so ``kv_dtype_for_sglang`` translates and returns None for values
  SGLang cannot express — the search then records the candidate as
  unreachable instead of measuring a silently different thing.

* **``--mem-fraction-static`` is a fraction of TOTAL GPU memory**,
  like vLLM's ``--gpu-memory-utilization`` and unlike TensorRT-LLM's
  free-memory-after-weights fraction. So the vLLM number carries
  over here and the TensorRT one does not.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from .docker_replica import DockerReplicaEngine, gpus_arg_for
from .vram import to_engine_fraction

log = logging.getLogger(__name__)

DEFAULT_IMAGE = "lmsysorg/sglang:latest"

# Each replica needs its OWN torch.distributed rendezvous port, and
# each LAUNCH needs its own range.
#
# Left to itself SGLang picks a port at random and eight replicas
# starting together race for it — two choose the same number before
# either binds and the loser dies with EADDRINUSE (seen at 40593).
# Fixing the ports solves that but creates a second collision a sweep
# runs into constantly: it tears down eight replicas and immediately
# starts eight more, and the previous set's sockets are still closing.
# Observed at port 42128, one cell into a roofline.
#
# So the base is offset per launch as well as per replica. The stride
# leaves room for the handful of consecutive ports a replica opens
# around its base, and the window stays well clear of the 9100-range
# HTTP ports the replicas serve on.
# The whole scheme must land inside 0-65535, which my first version
# did not: 360 windows of 8 replicas at a 64-port stride needs 184k
# ports and SGLang rejected it with "Port out of range". The arithmetic
# below is asserted by a test rather than trusted.
NCCL_PORT_BASE = 20000
NCCL_PORT_STRIDE = 64
NCCL_PORT_WINDOWS = 64           # 64 x 8 x 64 = 32,768 ports
NCCL_PORT_CEILING = 60000


def nccl_port(index: int, run_id: str = "") -> int:
    """Rendezvous port for one replica of one launch."""
    window = 0
    if run_id:
        # Stable, cheap, and spread: consecutive launches get unrelated
        # windows rather than adjacent ones.
        window = int(hashlib.sha1(run_id.encode()).hexdigest()[:8], 16) \
            % NCCL_PORT_WINDOWS
    port = (NCCL_PORT_BASE + window * NCCL_PORT_STRIDE * 8
            + index * NCCL_PORT_STRIDE)
    assert NCCL_PORT_BASE <= port <= NCCL_PORT_CEILING, port
    return port

# vLLM's KV dtype spelling -> SGLang's. SGLang names the float8
# representation explicitly where vLLM takes a bare "fp8", so the
# alias has to be resolved the SAME WAY vLLM resolves it or the two
# engines run different numeric formats while reporting one
# comparison. vLLM's own docs settle it: "CUDA 11.8+ supports fp8
# (=fp8_e4m3)". NVIDIA's ModelOpt checkpoints agree -- they declare
# kv_cache_quant_algo FP8, which is e4m3.
KV_DTYPE_MAP = {
    "auto": None,
    "fp8": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "fp8_e4m3": "fp8_e4m3",
}


def kv_dtype_for_sglang(dtype: str | None) -> tuple[Optional[str], Optional[str]]:
    """(flag value, reason it is unsupported).

    Returns ``(None, None)`` for "say nothing", ``(value, None)`` for a
    translation, and ``(None, reason)`` when SGLang cannot express the
    requested precision — which must fail the candidate rather than
    quietly measure a different one.
    """
    if not dtype or dtype == "auto":
        return None, None
    if dtype in KV_DTYPE_MAP:
        return KV_DTYPE_MAP[dtype], None
    if dtype in ("nvfp4", "fp4"):
        return None, ("SGLang has no nvfp4 KV cache path — the "
                      "candidate cannot be measured as specified")
    return None, f"SGLang does not accept kv_cache_dtype {dtype!r}"


# Precision labels that need ``--quantization`` spelled out, and the
# value SGLang's registry knows them by. Anything absent from this map
# (compressed-tensors, awq, gptq, plain bf16) declares itself inside
# config.json and IS auto-detected -- naming those explicitly would
# only create a second way to be wrong.
MODELOPT_QUANTIZATION = {
    "nvfp4": "modelopt_fp4",
    "fp4": "modelopt_fp4",
}


def sglang_quantization(model_quant: str | None,
                        explicit: str | None = None) -> str | None:
    """The ``--quantization`` value for a model, or None to let SGLang
    decide. An explicit config setting always wins."""
    if explicit:
        return str(explicit)
    if not model_quant:
        return None
    return MODELOPT_QUANTIZATION.get(str(model_quant).lower())


def launch_argv(model: str, *, port: int, tp: int,
                context_length: int | None = None,
                max_running_requests: int | None = None,
                max_prefill_tokens: int | None = None,
                mem_fraction_static: float | None = None,
                kv_cache_dtype: str | None = None,
                quantization: str | None = None,
                nccl_port: int | None = None,
                expert_parallel: bool = False,
                trust_remote_code: bool = False,
                extra: list[str] | None = None) -> list[str]:
    """The container CMD: ``python3 -m sglang.launch_server ...``."""
    argv = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "0.0.0.0",
        "--port", str(int(port)),
        "--tp", str(int(tp)),
        # Without this there is no /metrics at all, and the sweep
        # measures from engine counters.
        "--enable-metrics",
    ]
    if context_length:
        argv += ["--context-length", str(int(context_length))]
    if max_running_requests:
        argv += ["--max-running-requests", str(int(max_running_requests))]
    if max_prefill_tokens:
        argv += ["--max-prefill-tokens", str(int(max_prefill_tokens))]
    if mem_fraction_static is not None:
        argv += ["--mem-fraction-static", str(float(mem_fraction_static))]
    kv, _reason = kv_dtype_for_sglang(kv_cache_dtype)
    if kv:
        argv += ["--kv-cache-dtype", kv]
    if quantization:
        argv += ["--quantization", quantization]
    if nccl_port:
        argv += ["--nccl-port", str(int(nccl_port))]
    if expert_parallel and tp > 1:
        argv += ["--enable-ep-moe", "--ep-size", str(int(tp))]
    if trust_remote_code:
        argv += ["--trust-remote-code"]
    argv += list(extra or [])
    return argv


class SGLangCudaEngine(DockerReplicaEngine):
    """N GPU-pinned SGLang replicas, sticky-routed by the pool."""

    ENGINE_NAME = "sglang_cuda"

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        cfg = self.cfg
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--gpus", gpus_arg_for(devices),
            # SGLang's workers use shared memory for tensor transport;
            # the 64MB docker default kills tp>1 init.
            "--ipc=host",
            "--network", "host",
        ]
        cmd += self._mount_args()
        for k, v in (cfg.sglang_extra_env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(getattr(cfg, "sglang_image", None) or DEFAULT_IMAGE)

        return cmd + launch_argv(
            cfg.model_local_path or cfg.model_id,
            port=self._port(index),
            tp=len(devices),
            context_length=cfg.max_model_len,
            max_running_requests=getattr(cfg, "max_num_seqs", None),
            max_prefill_tokens=getattr(cfg, "max_num_batched_tokens", None),
            # Translated, not passed through: see vram.py for why
            # this engine's fraction has to be smaller than vLLM's to
            # mean the same allocation.
            mem_fraction_static=to_engine_fraction(
                "sglang_cuda", cfg.gpu_memory_utilization,
                total_vram_gb=getattr(cfg, "vram_per_gpu_gb", None),
                weights_gb=getattr(cfg, "model_weights_gb", None))[0],
            kv_cache_dtype=getattr(cfg, "kv_cache_dtype", None),
            quantization=sglang_quantization(
                getattr(cfg, "model_quant", None),
                cfg.quantization_kind or cfg.quantization),
            nccl_port=nccl_port(index, self._run_id),
            expert_parallel=bool(getattr(cfg, "expert_parallel", False)),
            trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
            extra=list(cfg.sglang_extra_flags or []),
        )
