"""KTransformers — MoE experts on the CPU, attention on the GPU.

This engine is here because of what the box is, not because it is
another GPU server. An XE7740 has two Xeon 6787P sockets with AMX and
2 TB of RAM sitting next to the GPUs, and KTransformers is the engine
built to use exactly that: it keeps attention and the dense path on
the GPU and runs MoE expert matmuls on the CPU, so a model whose
weights dwarf 8x96 GB of VRAM can still be served.

That makes it a different KIND of measurement, and the reports should
not pretend otherwise. The GPU-resident engines compete on tokens per
second at a given concurrency; KTransformers competes on serving a
model the others cannot load at all. Its throughput is bounded by CPU
and memory bandwidth, so a straight tokens/sec ranking against vLLM
flatters vLLM and answers a question nobody asked.

Knobs that have no meaning here (KV precision, batched-token budget,
expert parallelism) are refused by ``knobs.unsupported`` rather than
accepted and ignored -- a silently dropped setting is how a search
concludes the wrong thing.

Concurrency needs the ``balance_serve`` backend; the older single-
stream backends serve one request at a time, which would read as a
catastrophic engine rather than the wrong launch flag.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .docker_replica import DockerReplicaEngine, gpus_arg_for

log = logging.getLogger(__name__)

# NOT ``:latest``. The v0.5 line reorganised into kt-kernel (CPU
# kernels) and kt-sft (fine-tuning) and ARCHIVED the inference server:
# the latest image has no torch in its default interpreter and no
# compiled KTransformersOps, so it cannot serve at all. The ISA-tagged
# v0.3.2 images are the serving builds. AVX512 is the right variant
# for a Granite Rapids Xeon; NATIVE/FANCY/AVX2 exist for other hosts.
DEFAULT_IMAGE = "approachingai/ktransformers:v0.3.2-AVX512"

# The image ships a conda environment and its ENTRYPOINT is
# ``tail -f /dev/null`` -- it is built to be run detached and then
# docker exec'd into. Anything passed as CMD becomes an ARGUMENT TO
# TAIL, so this engine MUST override the entrypoint. That is the exact
# opposite of the TensorRT-LLM and SGLang images, where overriding it
# breaks the launch; do not "make them consistent".
PYTHON = "/opt/conda/bin/python"
WORKDIR = "/workspace/ktransformers"

# Batching backend. The default backend answers one request at a
# time; without this the concurrency sweep measures a queue, not an
# engine.
DEFAULT_BACKEND = "balance_serve"


def serve_argv(model: str, *, port: int,
               gguf_path: str | None = None,
               optimize_config_path: str | None = None,
               max_batch_size: int | None = None,
               max_new_tokens: int | None = None,
               cache_lens: int | None = None,
               chunk_size: int | None = None,
               cpu_threads: int | None = None,
               backend: str = DEFAULT_BACKEND,
               extra: list[str] | None = None) -> list[str]:
    """The CMD after the entrypoint override: ``-m
    ktransformers.server.main ...``.

    Flag names verified against the server's own --help in the
    v0.3.2 image. KTransformers uses snake_case throughout, unlike
    every other engine capsim drives.
    """
    argv = [
        "-m", "ktransformers.server.main",
        "--model_path", model,
        "--host", "0.0.0.0",
        "--port", str(int(port)),
        "--backend_type", backend,
    ]
    if gguf_path:
        argv += ["--gguf_path", gguf_path]
    if optimize_config_path:
        argv += ["--optimize_config_path", optimize_config_path]
    if max_batch_size:
        argv += ["--max_batch_size", str(int(max_batch_size))]
    if max_new_tokens:
        argv += ["--max_new_tokens", str(int(max_new_tokens))]
    if cache_lens:
        argv += ["--cache_lens", str(int(cache_lens))]
    if chunk_size:
        # The nearest thing to a batched-token budget: how many prompt
        # tokens are processed per prefill step.
        argv += ["--chunk_size", str(int(chunk_size))]
    if cpu_threads:
        argv += ["--cpu_infer", str(int(cpu_threads))]
    argv += list(extra or [])
    return argv


class KTransformersEngine(DockerReplicaEngine):
    """KTransformers replicas, sticky-routed by the pool.

    Almost always ONE replica: the expert path wants every core and
    the whole memory bandwidth of the box, so two replicas contend
    with each other rather than doubling throughput.
    """

    ENGINE_NAME = "ktransformers"

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        cfg = self.cfg
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--gpus", gpus_arg_for(devices),
            "--ipc=host",
            "--network", "host",
            # The expert path is the whole point: give the container
            # the machine's memory and cores, not docker's defaults.
            "--shm-size", cfg.docker_shm_size,
            "-w", WORKDIR,
            # See PYTHON above -- the image's entrypoint is
            # `tail -f /dev/null` and would swallow the command.
            "--entrypoint", PYTHON,
        ]
        cmd += self._mount_args()
        gguf = getattr(cfg, "ktransformers_gguf_path", None)
        has_gguf = bool(gguf) and Path(gguf).exists()
        if has_gguf:
            cmd += ["-v", f"{gguf}:/gguf:ro"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(getattr(cfg, "ktransformers_image", None) or DEFAULT_IMAGE)

        return cmd + serve_argv(
            cfg.model_local_path or cfg.model_id,
            port=self._port(index),
            gguf_path="/gguf" if has_gguf else None,
            optimize_config_path=getattr(
                cfg, "ktransformers_optimize_config", None),
            max_batch_size=getattr(cfg, "max_num_seqs", None),
            chunk_size=getattr(cfg, "max_num_batched_tokens", None),
            cache_lens=cfg.max_model_len,
            cpu_threads=getattr(cfg, "ktransformers_cpu_threads", None),
            extra=list(getattr(cfg, "ktransformers_extra_flags", None) or []),
        )

    def _ready_url(self, port: int) -> str:
        return f"http://{self.cfg.host}:{port}/v1/models"
