"""Multi-replica CUDA vLLM engine — the whole box as N independent
replicas.

The CUDA analog of ``vllm_dual_socket``: the guided search showed
that on a PCIe box (no NVLink), N independent single- or few-GPU
replicas beat tensor-parallel spans for models that fit — dp8 of
Qwen3-30B delivered ~9.2k tok/s where TP pays an all-reduce per
layer. This engine lets the capacity benchmark MEASURE that shape
instead of extrapolating from one replica × 8.

Config: ``engine.replica_devices`` is a list of GPU-id lists, one
per replica (exactly the assignment the optimizer's placement logic
produces — e.g. dp8: ``[[0],[4],[1],[5],[2],[6],[3],[7]]``; a TP2×4
hybrid: ``[[0,1],[2,3],[4,5],[6,7]]``). Each replica serves on
``port + index`` (host networking). Everything else (image, model,
gmu, extra flags) comes from the same EngineConfig fields as
``vllm_cuda``.

Routing is the simulator's, same as dual-socket: ``replica_urls``
exposes every backend and the pool manager sticks each virtual user
to one replica (±1 balanced), so multi-turn conversations keep their
prefix-cache locality — the property the warm-KV telemetry measures.

The container lifecycle, log capture and metric aggregation live in
``docker_replica.DockerReplicaEngine``, shared with the TensorRT-LLM
engine; only the argv below is vLLM-specific.
"""

from __future__ import annotations

import logging

from .docker_replica import (
    DockerReplicaEngine,
    aggregate_replica_metrics,
    gpus_arg_for,
    remove_stale_engine_containers,
)

log = logging.getLogger(__name__)

__all__ = [
    "VllmCudaMultiEngine",
    "aggregate_replica_metrics",
    "gpus_arg_for",
    "remove_stale_engine_containers",
]


class VllmCudaMultiEngine(DockerReplicaEngine):
    """N GPU-pinned vLLM CUDA replicas, sticky-routed by the pool."""

    ENGINE_NAME = "vllm_cuda_multi"

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        cfg = self.cfg
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--gpus", gpus_arg_for(devices),
            # vLLM's CUDA workers use shared memory for tensor
            # transport; the 64MB docker default kills TP>1 init.
            "--ipc=host",
            "--network", "host",
        ]
        cmd += self._mount_args()
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(cfg.gpu_image)

        model_arg = cfg.model_local_path or cfg.model_id
        inner = [
            "--model", model_arg,
            "--host", "0.0.0.0",
            "--port", str(self._port(index)),
            "--max-model-len", str(cfg.max_model_len),
            # Per-replica TP = the replica's device count; DP is the
            # replica count itself.
            "--tensor-parallel-size", str(len(devices)),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        ]
        if cfg.served_model_name:
            inner += ["--served-model-name", cfg.served_model_name]
        quantization = cfg.quantization_kind or cfg.quantization
        if quantization:
            inner += ["--quantization", quantization]
        inner += list(cfg.vllm_extra_flags or [])
        return cmd + inner
