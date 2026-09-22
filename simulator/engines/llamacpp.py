"""llama.cpp's ``llama-server`` -- the widest road to the largest models.

Every frontier-scale open model ships a GGUF long before any GPU
engine learns its architecture: Kimi-K2-Thinking (646 GB at UD-Q4_K_XL),
DeepSeek-V3.2 (408 GB), GLM-5.3 (467 GB) and MiniMax-M2.7 all have an
unsloth quant on the Hub, and llama-server loads any of them. On this
box that matters twice over. A model whose weights exceed 8 x 96 GB of
VRAM is served by keeping the MoE expert tensors in host RAM
(``--override-tensor``: ``\\.ffn_.*_exps\\.=CPU``) while attention, the
dense layers and the KV cache stay on the GPUs -- the same split
KTransformers makes, but for every architecture llama.cpp knows rather
than the two the KTransformers v0.3.2 image has an injection rule for.

Like KTransformers, this is a different KIND of measurement. With the
experts on the CPU the decode rate is bounded by memory bandwidth, not
by the GPUs, and a tokens/sec ranking against vLLM answers a question
nobody asked. What it answers is whether the largest models can be
served at all, and at what rate.

Image and tag policy
--------------------
``ghcr.io/ggml-org/llama.cpp:server-cuda`` is the CUDA server build
(verified on GHCR: the ``server-cuda`` index carries linux/amd64 and
arm64, ~2.6 GB compressed; the CUDA-less ``server`` tag runs on the CPU
only). It is a FLOATING tag rebuilt on every upstream release, which is
right for a project that adds model architectures weekly: the pinned
siblings (``server-cuda-v0.4.1`` for the 2026-09-14 release,
``server-cuda-b<build>`` for older builds) exist for reproducing a run
and go in ``llamacpp_image``. Built from ``.devops/cuda.Dockerfile`` on
a ``nvidia/cuda:12.8.1-runtime`` base -- CUDA 12.8 is what SM120
(Blackwell) needs -- with ``ENTRYPOINT ["/app/llama-server"]``, so the
docker argv after the image is llama-server's own argument list and
the entrypoint is NOT overridden (the opposite of the KTransformers
image, whose entrypoint is ``tail -f /dev/null``).

Verified against the upstream server README and ``server-task.cpp``
(master, 2026-09): ``GET /health`` answers 503 ``{"error": ... "Loading
model"}`` until the model is loaded and 200 ``{"status": "ok"}`` after;
``/v1/models`` reports the ``-m`` path as the model id unless
``--alias`` names it; ``/metrics`` (behind ``--metrics``) exports
``llamacpp:prompt_tokens_total``, ``llamacpp:tokens_predicted_total``,
``llamacpp:requests_processing`` and ``llamacpp:requests_deferred``.
The ``llamacpp:kv_cache_usage_ratio`` / ``kv_cache_tokens`` gauges of
older builds are no longer emitted; they are parsed when present.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

from .docker_replica import DockerReplicaEngine, gpus_arg_for
from .ktransformers import physical_cores

log = logging.getLogger(__name__)

# See the module docstring for the tag policy.
DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"

# The image's own entrypoint. Recorded so a test can assert the launch
# never overrides it; nothing here passes --entrypoint.
ENTRYPOINT = "/app/llama-server"

# Where the GGUF directory is mounted inside the container.
GGUF_MOUNT = "/gguf"

# Server slots the roofline asks for. llama-server batches its slots
# through one llama_decode() per step and its own docs demonstrate it
# in the tens of slots; 32 keeps every slot's share of the KV pool
# useful (the pool is split evenly, see ``context_tokens``) where the
# GPU engines' thousands of streams would leave each slot a handful of
# tokens. The roofline clamps its cells here.
DOCUMENTED_MAX_BATCH = 32

# The expert-offload pattern: every MoE expert tensor
# (blk.N.ffn_{gate,up,down}_exps) stays in host RAM, attention and the
# dense path go to the GPUs. Written with the dots escaped, as the
# llama.cpp docs spell it; unsloth's guides use the unescaped
# ``.ffn_.*_exps.`` which matches the same tensors (``.`` matches a
# dot) -- the escaped form cannot also match an unrelated name.
OFFLOAD_PATTERN = r"\.ffn_.*_exps\.=CPU"

# Auto rule for ``llamacpp_offload_experts``: the experts go to the CPU
# when the GGUF would not leave room on the replica's GPUs. 0.85 is
# the share the weights may take -- the rest is the KV pool, the
# compute buffers and the graph's activations.
OFFLOAD_VRAM_SHARE = 0.85

# KV cache precision. llama-server quantises K and V separately; q8_0
# is its 8-bit cache, the nearest thing to vLLM's fp8. There is no
# 4-bit float cache (q4_0 exists but is a different quantisation) so
# nvfp4 is refused, not approximated -- knobs.unsupported.
KV_CACHE_TYPES = {
    "fp8": "q8_0", "fp8_e4m3": "q8_0", "fp8_e5m2": "q8_0",
}

_FIRST_SHARD = re.compile(r"-00001-of-\d{5}\.gguf$")


def gguf_entry_file(directory: str | Path) -> str:
    """The file ``-m`` names inside a staged GGUF directory: the one
    ``.gguf`` when the quant is a single file, or the first shard
    (``-00001-of-0000N``) when it is split -- llama.cpp opens the rest
    from the first shard's own split metadata. ValueError when the
    directory holds no usable GGUF (an interrupted download), so the
    launch fails before docker rather than after it."""
    d = Path(directory)
    try:
        files = sorted(f for f in d.iterdir()
                       if f.is_file() and f.suffix == ".gguf"
                       and f.stat().st_size > 0)
    except OSError as e:
        raise ValueError(f"{d} is not a readable directory") from e
    if not files:
        raise ValueError(f"no .gguf file under {d}")
    firsts = [f for f in files if _FIRST_SHARD.search(f.name)]
    if firsts:
        return firsts[0].name
    if len(files) == 1:
        return files[0].name
    raise ValueError(
        f"{d} holds {len(files)} .gguf files and none is a first shard "
        f"(-00001-of-0000N); one quant per directory")


def gguf_size_gb(directory: str | Path) -> float | None:
    """Bytes of every ``.gguf`` under the directory, in GB -- what the
    offload rule compares against VRAM. None when unreadable."""
    try:
        return sum(f.stat().st_size for f in Path(directory).iterdir()
                   if f.is_file() and f.suffix == ".gguf") / 1e9
    except OSError:
        return None


def offload_by_default(gguf_gb: float | None, vram_per_gpu_gb: float | None,
                       n_gpus: int) -> bool:
    """The auto rule: experts to the CPU when the GGUF exceeds
    OFFLOAD_VRAM_SHARE of the replica's VRAM. Unknown VRAM or an
    unreadable GGUF means offload ON -- the choice that always loads;
    a fully GPU-resident launch of a model that does not fit costs a
    30-minute health timeout to discover."""
    if gguf_gb is None or not vram_per_gpu_gb or n_gpus < 1:
        return True
    return float(gguf_gb) > OFFLOAD_VRAM_SHARE * float(vram_per_gpu_gb) * n_gpus


def default_threads(cores: int | None = None) -> int | None:
    """``-t`` when the operator set nothing: physical cores minus two,
    the same rule KTransformers uses and for the same reason -- the
    expert matmuls are bandwidth-bound and want every physical core,
    hyperthreads add nothing, and the GPU driver and the HTTP threads
    need a core each. None when the host's cores are unknown (the
    server then picks)."""
    cores = physical_cores() if cores is None else cores
    if not cores:
        return None
    return max(1, int(cores) - 2)


def context_tokens(max_model_len: int, slots: int) -> int:
    """The ``-c`` value. llama-server's context is ONE pool split
    evenly across its slots (``n_ctx_slot = n_ctx / n_parallel``), so
    a 4k model length at 32 slots is a 128k pool -- passing 4k would
    give every slot 128 tokens and the sweep would measure truncation."""
    return int(max_model_len) * max(1, int(slots))


def serve_argv(gguf_file: str, *, port: int, gguf_mount: str = GGUF_MOUNT,
               max_model_len: int,
               slots: int | None = None,
               threads: int | None = None,
               offload_experts: bool = False,
               jinja: bool = False,
               chat_template: str | None = None,
               kv_cache_type: str | None = None,
               batch_tokens: int | None = None,
               alias: str | None = None,
               extra: list[str] | None = None) -> list[str]:
    """llama-server's argument list (the CMD after the image).

    Flag spellings verified against the server README's argument
    table: ``-m``, ``--host``/``--port``, ``-c``, ``-np``, ``-cb``,
    ``-ngl``, ``-sm``, ``-fa``, ``-t``, ``-ot``, ``-ctk``/``-ctv``,
    ``-b``/``-ub``, ``-a``, ``--metrics``.
    """
    slots = int(slots or DOCUMENTED_MAX_BATCH)
    argv = [
        "-m", f"{gguf_mount}/{gguf_file}",
        "--host", "0.0.0.0",
        "--port", str(int(port)),
        # Every layer on the GPUs, split by layer across all of them.
        # Expert tensors are then pulled back to the CPU by -ot below
        # when offloading; without -ngl the default is 'auto'.
        "-ngl", "999",
        "-sm", "layer",
        "-fa", "on",
        "-c", str(context_tokens(max_model_len, slots)),
        "-np", str(slots),
        "-cb",
        "--metrics",
        # llama-server parses its own chat output through a grammar
        # derived from the model's jinja template (reasoning, tool
        # calls). DeepSeek-V3.2 loaded, generated, and then answered
        # every request HTTP 500: "The model produced output that does
        # not match the expected peg-native format" (XE7740 giants
        # pass). A capacity benchmark counts tokens; it does not need
        # thoughts extracted or tool calls parsed, so the parser is
        # off: thoughts stay in message.content and the legacy chat
        # formatter (which has no output grammar) builds the prompt.
        "--reasoning-format", "none",
    ]
    if not jinja:
        # The legacy formatter refuses a template it does not know
        # ("this custom template is not supported, try using --jinja"
        # -- DeepSeek-V3.2), so name a built-in one. ChatML is the
        # generic choice: tokens are what the benchmark counts.
        argv += ["--no-jinja", "--chat-template", chat_template or "chatml"]
    if alias:
        argv += ["-a", alias]
    if threads:
        argv += ["-t", str(int(threads))]
    if offload_experts:
        argv += ["-ot", OFFLOAD_PATTERN]
    if kv_cache_type:
        argv += ["-ctk", kv_cache_type, "-ctv", kv_cache_type]
    if batch_tokens:
        # Logical and physical batch: the nearest thing to a batched-
        # token budget. -ub cannot exceed -b, so both are set.
        argv += ["-b", str(int(batch_tokens)), "-ub", str(int(batch_tokens))]
    argv += list(extra or [])
    return argv


def parse_llamacpp_metrics(text: str) -> dict[str, float]:
    """``/metrics`` in capsim's canonical keys.

    ``prompt_tokens_total`` EXCLUDES cached tokens upstream and
    ``prompt_tokens_cached_total`` counts the reused ones, so the
    prefix-cache view is hits = cached, queries = cached + processed.
    """
    wanted = {
        "llamacpp:prompt_tokens_total": "prompt_tokens_total",
        "llamacpp:tokens_predicted_total": "generation_tokens_total",
        "llamacpp:requests_processing": "num_running",
        "llamacpp:requests_deferred": "queue_depth",
        "llamacpp:prompt_tokens_cached_total": "prefix_cache_hits",
        # Older builds only; harmless when absent.
        "llamacpp:kv_cache_usage_ratio": "kv_cache_used_pct",
        "llamacpp:kv_cache_tokens": "kv_cache_tokens",
    }
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value_str = line.rpartition(" ")
        name = name_part.split("{", 1)[0].strip()
        if name in wanted:
            try:
                out[wanted[name]] = float(value_str)
            except ValueError:
                pass
    if "prefix_cache_hits" in out:
        out["prefix_cache_queries"] = (out["prefix_cache_hits"]
                                       + out.get("prompt_tokens_total", 0.0))
        if out["prefix_cache_queries"] > 0:
            out["prefix_cache_hit_rate"] = (out["prefix_cache_hits"]
                                            / out["prefix_cache_queries"])
    if "kv_cache_used_pct" in out and out["kv_cache_used_pct"] <= 1.0:
        out["kv_cache_used_pct"] *= 100.0
    return out


class LlamaCppEngine(DockerReplicaEngine):
    """llama-server replicas, sticky-routed by the pool.

    Almost always ONE replica. With the experts offloaded the decode
    path wants every core and all the memory bandwidth of the box, so
    a second replica contends rather than doubles; and even fully on
    the GPUs a replica is spread by layer across its whole device
    group, so the roofline runs it as one whole-box replica.
    """

    ENGINE_NAME = "llamacpp"

    def __init__(self, engine_config):
        super().__init__(engine_config)
        self._api_model_name: str | None = None

    def _gguf_dir(self) -> str:
        cfg = self.cfg
        gguf = getattr(cfg, "llamacpp_gguf_path", None)
        from .custom import llamacpp_gguf_missing
        why = llamacpp_gguf_missing(gguf)
        if why:
            # Milliseconds with the reason, not a health timeout.
            raise RuntimeError(f"llamacpp cannot launch: {why}")
        return str(gguf)

    def offload_experts(self, devices: list[int]) -> bool:
        """The lever, or the auto rule against this replica's VRAM."""
        explicit = getattr(self.cfg, "llamacpp_offload_experts", None)
        if explicit is not None:
            return bool(explicit)
        return offload_by_default(gguf_size_gb(self._gguf_dir()),
                                  getattr(self.cfg, "vram_per_gpu_gb", None),
                                  len(devices))

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        cfg = self.cfg
        gguf_dir = self._gguf_dir()
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--gpus", gpus_arg_for(devices),
            # The offloaded experts are mmap'd from the GGUF and read
            # through the page cache; sharing the host's IPC namespace
            # is what the CPU path wants.
            "--ipc=host",
            "--network", "host",
        ]
        cmd += self._mount_args()
        from ..models import container_cache_path
        gguf_mount = container_cache_path(gguf_dir)
        if gguf_mount is None:
            cmd += ["-v", f"{gguf_dir}:{GGUF_MOUNT}:ro"]
            gguf_mount = GGUF_MOUNT
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(getattr(cfg, "llamacpp_image", None) or DEFAULT_IMAGE)

        threads = getattr(cfg, "llamacpp_cpu_threads", None) or default_threads()
        kv = getattr(cfg, "kv_cache_dtype", None)
        kv_type = KV_CACHE_TYPES.get(str(kv)) if kv and kv != "auto" else None
        if kv and kv != "auto" and not kv_type:
            # knobs.unsupported refuses this at config time; a hand-
            # written config gets the same answer here.
            raise RuntimeError(
                f"llamacpp has no KV cache type for {kv!r}; "
                f"supported: {', '.join(sorted(KV_CACHE_TYPES))}")
        return cmd + serve_argv(
            gguf_entry_file(gguf_dir),
            port=self._port(index),
            gguf_mount=gguf_mount,
            max_model_len=cfg.max_model_len,
            slots=getattr(cfg, "max_num_seqs", None),
            threads=threads,
            offload_experts=self.offload_experts(devices),
            jinja=bool(getattr(cfg, "llamacpp_jinja", False)),
            chat_template=getattr(cfg, "llamacpp_chat_template", None),
            kv_cache_type=kv_type,
            batch_tokens=getattr(cfg, "max_num_batched_tokens", None),
            alias=getattr(cfg, "served_model_name", None) or cfg.model_id,
            extra=list(getattr(cfg, "llamacpp_extra_flags", None) or []),
        )

    def _ready_url(self, port: int) -> str:
        # /v1/models answers 200 with a null ``meta`` while the model
        # is still loading; /health is the endpoint that waits.
        return f"http://{self.cfg.host}:{port}/health"

    @property
    def api_model_name(self) -> str:
        """What ``/v1/models`` reports. The launch passes ``--alias``
        so this is the configured name; asking the server is what
        makes an operator's ``llamacpp_extra_flags`` alias win."""
        if self._api_model_name:
            return self._api_model_name
        for _i, _d, port, _cid, _s in self._replicas:
            try:
                r = httpx.get(f"http://{self.cfg.host}:{port}/v1/models",
                              timeout=2.0)
                data = r.json().get("data") or []
                if r.status_code == 200 and data and data[0].get("id"):
                    self._api_model_name = str(data[0]["id"])
                    return self._api_model_name
            except Exception:  # noqa: BLE001
                continue
            break
        return super().api_model_name

    def parse_metrics(self, text: str) -> dict[str, float]:
        return parse_llamacpp_metrics(text)
