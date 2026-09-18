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

import logging
from typing import Optional

from .docker_replica import DockerReplicaEngine, gpus_arg_for

log = logging.getLogger(__name__)

DEFAULT_IMAGE = "lmsysorg/sglang:latest"

# vLLM's KV dtype spelling -> SGLang's. SGLang names the float8
# representation explicitly; e5m2 is the conservative default choice
# (wider exponent range, the one vLLM's plain "fp8" historically
# meant on non-Hopper paths).
KV_DTYPE_MAP = {
    "auto": None,
    "fp8": "fp8_e5m2",
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


def launch_argv(model: str, *, port: int, tp: int,
                context_length: int | None = None,
                max_running_requests: int | None = None,
                max_prefill_tokens: int | None = None,
                mem_fraction_static: float | None = None,
                kv_cache_dtype: str | None = None,
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
            # A fraction of TOTAL GPU memory, same quantity vLLM's
            # gpu_memory_utilization names.
            mem_fraction_static=cfg.gpu_memory_utilization,
            kv_cache_dtype=getattr(cfg, "kv_cache_dtype", None),
            expert_parallel=bool(getattr(cfg, "expert_parallel", False)),
            trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
            extra=list(cfg.sglang_extra_flags or []),
        )
