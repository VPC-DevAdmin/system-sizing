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
import os
import sys
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

# The batch width KTransformers' own documentation demonstrates. The
# GPU engines hold thousands of streams; asking this one for 1,024 is
# asking the wrong question, so the roofline clamps its cells here.
DOCUMENTED_MAX_BATCH = 4

# Injection rules the v0.3.2 image ships, by the HF config's
# model_type. KTransformers needs one to know which tensors go to the
# CPU and which stay on the GPU, and it cannot infer the rule from the
# model id -- but capsim can read model_type from the staged
# config.json. File names verified against the v0.3.2 tag of
# kvcache-ai/ktransformers (ktransformers/optimize/optimize_rules/).
# The ``*-serve.yaml`` rules are the balance_serve variants; the
# ``*-serve-amx.yaml`` siblings exist for the AMX kernel path but
# constrain the GGUF's quant types, so they stay an explicit choice
# (ktransformers_optimize_config). deepseek_v32 (DeepSeek V3.2's
# sparse-attention indexer) has no rule in this image at all.
OPTIMIZE_RULES_DIR = f"{WORKDIR}/ktransformers/optimize/optimize_rules"
OPTIMIZE_RULES = {
    "deepseek_v3": "DeepSeek-V3-Chat-serve.yaml",
    "qwen3_moe": "Qwen3Moe-serve.yaml",
}


def optimize_config_for(arch: str | None) -> str | None:
    """Container path of the optimize rule for an architecture, or
    None when this image has no rule for it (the server then falls
    back to its own default, which loads every tensor on the GPU --
    for a kt_only model that is an OOM, not a slower run)."""
    if not arch:
        return None
    name = OPTIMIZE_RULES.get(str(arch).lower())
    return f"{OPTIMIZE_RULES_DIR}/{name}" if name else None


def parse_cpuinfo_cores(text: str) -> int:
    """Distinct (physical id, core id) pairs in a /proc/cpuinfo dump --
    physical cores across sockets. 0 when the fields are absent (ARM
    hosts, containers with a trimmed cpuinfo)."""
    cores: set[tuple[str, str]] = set()
    phys = core = None
    for line in text.splitlines() + [""]:
        key, _, val = line.partition(":")
        key = key.strip()
        if key == "physical id":
            phys = val.strip()
        elif key == "core id":
            core = val.strip()
        elif not key:
            if phys is not None and core is not None:
                cores.add((phys, core))
            phys = core = None
    return len(cores)


def physical_cores() -> int | None:
    """Physical CPU cores on a Linux host: distinct (physical id,
    core id) pairs from /proc/cpuinfo, falling back to half the
    logical count. None elsewhere -- the launch runs docker on the
    host it sizes for, and a Mac's core count says nothing about the
    Xeon that will run the experts."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        text = ""
    n = parse_cpuinfo_cores(text)
    if n:
        return n
    logical = os.cpu_count()
    return max(1, logical // 2) if logical else None


def default_cpu_infer(cores: int | None = None) -> int | None:
    """``--cpu_infer`` when the operator set nothing: physical cores
    minus two, leaving the GPU driver threads and the server's own
    scheduler a core each. Every core beyond that helps -- the expert
    path is memory-bandwidth bound and wants the whole box -- and
    hyperthreads do not, so the count is physical, not logical. None
    when the host's cores are unknown (the server then picks)."""
    cores = physical_cores() if cores is None else cores
    if not cores:
        return None
    return max(1, int(cores) - 2)


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
            # --ipc=host shares the host's /dev/shm outright, which is
            # what the expert path needs; a --shm-size beside it would
            # be a no-op (it sizes the private /dev/shm --ipc=host
            # replaces).
            "--ipc=host",
            "--network", "host",
            "-w", WORKDIR,
            # See PYTHON above -- the image's entrypoint is
            # `tail -f /dev/null` and would swallow the command.
            "--entrypoint", PYTHON,
        ]
        cmd += self._mount_args()
        gguf = getattr(cfg, "ktransformers_gguf_path", None)
        from .custom import ktransformers_gguf_missing
        why = ktransformers_gguf_missing(gguf)
        if why:
            # Fail in milliseconds with the reason, not in 30 minutes
            # with a health timeout (see ktransformers_gguf_missing).
            raise RuntimeError(f"ktransformers cannot launch: {why}")
        cmd += ["-v", f"{gguf}:/gguf:ro"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(getattr(cfg, "ktransformers_image", None) or DEFAULT_IMAGE)

        optimize = getattr(cfg, "ktransformers_optimize_config", None)
        if not optimize:
            optimize = optimize_config_for(self._model_arch())
        cpu_threads = getattr(cfg, "ktransformers_cpu_threads", None)
        if not cpu_threads:
            cpu_threads = default_cpu_infer()

        return cmd + serve_argv(
            cfg.model_local_path or cfg.model_id,
            port=self._port(index),
            gguf_path="/gguf",
            optimize_config_path=optimize,
            max_batch_size=getattr(cfg, "max_num_seqs", None),
            chunk_size=getattr(cfg, "max_num_batched_tokens", None),
            cache_lens=cfg.max_model_len,
            cpu_threads=cpu_threads,
            extra=list(getattr(cfg, "ktransformers_extra_flags", None) or []),
        )

    def _model_arch(self) -> str | None:
        """``model_type`` from the staged config.json -- the HF cache
        snapshot for a hub id, or the directory itself for a local
        path. None when nothing is staged (the launch will fail on
        the missing config anyway, with the server's own message)."""
        cfg = self.cfg
        local = getattr(cfg, "model_local_path", None)
        if local and Path(local).is_dir():
            try:
                import json
                doc = json.loads((Path(local) / "config.json").read_text())
                return str(doc.get("model_type") or "") or None
            except (OSError, ValueError):
                return None
        if cfg.model_id and "/" in cfg.model_id:
            from ..models import model_arch
            try:
                return model_arch(cfg.model_id)
            except OSError:
                return None
        return None

    def _ready_url(self, port: int) -> str:
        return f"http://{self.cfg.host}:{port}/v1/models"
