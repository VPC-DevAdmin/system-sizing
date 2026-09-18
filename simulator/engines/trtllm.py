"""TensorRT-LLM engine — ``trtllm-serve`` as a whole-box, N-replica
target, measured on the same footing as vLLM.

Same lifecycle as ``vllm_cuda_multi`` (see ``docker_replica``); the
differences are all in three places, and each one is a trap that
cost real debugging time:

1. **Launch through the image's own entrypoint.** The container's
   ``ENV LD_LIBRARY_PATH`` does NOT include ``/usr/local/tensorrt/lib``
   — that path is added by ``/etc/bash.bashrc`` via ``BASH_ENV``, and
   the image ENTRYPOINT (``nvidia_entrypoint.sh``) runs through bash.
   Overriding the entrypoint to run ``trtllm-serve`` directly dies with
   ``ImportError: libnvonnxparser.so.10``. So we pass the command as
   CMD and never touch ``--entrypoint``.

2. **Metrics are iteration stats, not Prometheus.** ``/metrics``
   returns a JSON *list* of per-iteration snapshots (camelCase keys);
   the Prometheus endpoint lives at ``/prometheus/metrics``, is only
   mounted when ``return_perf_metrics`` is set, and exposes only
   latency histograms — no running/queued/KV/token counters, which is
   precisely what the sweep measures. So we read the JSON.

3. **The JSON endpoint DRAINS a queue.** Each GET consumes the
   iteration stats it returns. capsim has two independent pollers (the
   telemetry loop and the headline sweep's per-second sampler); if both
   hit the endpoint they get disjoint halves of the stream and BOTH
   undercount. So this engine owns the draining: one background thread
   polls each replica and accumulates into monotonic counters, and
   ``get_metrics()`` returns a snapshot without any I/O. That restores
   the vLLM contract — an idempotent read, safe for N callers.

Because the stats are per-iteration rather than cumulative, dropped
iterations would silently understate throughput. Every snapshot
carries a strictly increasing ``iter`` index, so we detect gaps and
report them rather than quietly reporting a low number.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

from .docker_replica import DockerReplicaEngine, gpus_arg_for

log = logging.getLogger(__name__)

DEFAULT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"

# How often the background thread drains each replica's stats queue.
# Fast enough that the engine-side ring buffer cannot overflow between
# reads at realistic iteration rates (tens per second), slow enough
# that draining costs nothing measurable.
POLL_INTERVAL_S = 0.5


@dataclass
class _Acc:
    """Monotonic totals rebuilt from per-iteration snapshots."""
    gen_tokens: float = 0.0
    prompt_tokens: float = 0.0
    last_iter: Optional[int] = None
    dropped_iters: int = 0
    # Latest gauges, carried forward between drains.
    gauges: dict = field(default_factory=dict)


def iteration_gen_tokens(stat: dict) -> float:
    """Decode tokens produced in ONE iteration.

    The pytorch backend reports inflight-batching stats, where each
    generating request emits ``avgNumDecodedTokensPerIter`` tokens
    (1.0 normally, >1 under speculative decoding). The tensorrt
    backend reports static-batching stats, which count gen tokens
    directly.
    """
    sbs = stat.get("staticBatchingStats") or {}
    if sbs.get("numGenTokens") is not None:
        return float(sbs["numGenTokens"])
    ibs = stat.get("inflightBatchingStats") or {}
    if ibs:
        n = float(ibs.get("numGenRequests") or 0)
        per = ibs.get("avgNumDecodedTokensPerIter")
        return n * (float(per) if per else 1.0)
    return 0.0


def iteration_prompt_tokens(stat: dict) -> float:
    """Prefill tokens processed in ONE iteration."""
    for key in ("inflightBatchingStats", "staticBatchingStats"):
        blk = stat.get(key) or {}
        if blk.get("numCtxTokens") is not None:
            return float(blk["numCtxTokens"])
    return 0.0


def accumulate(stats: list, acc: _Acc) -> _Acc:
    """Fold a drained batch of iteration stats into monotonic totals.

    ``stats`` arrives oldest-first. Snapshots at or below the last
    index we already counted are ignored, so an overlapping read can
    never double-count; a jump in the index means the engine produced
    iterations we never saw, which is recorded as a gap.
    """
    for stat in stats:
        if not isinstance(stat, dict):
            continue
        idx = stat.get("iter")
        if idx is not None:
            idx = int(idx)
            if acc.last_iter is not None:
                if idx <= acc.last_iter:
                    continue                      # already counted
                if idx > acc.last_iter + 1:
                    acc.dropped_iters += idx - acc.last_iter - 1
            acc.last_iter = idx
        acc.gen_tokens += iteration_gen_tokens(stat)
        acc.prompt_tokens += iteration_prompt_tokens(stat)

        g: dict[str, float] = {}
        if stat.get("numActiveRequests") is not None:
            g["num_running"] = float(stat["numActiveRequests"])
        if stat.get("numQueuedRequests") is not None:
            g["queue_depth"] = float(stat["numQueuedRequests"])
        kv = stat.get("kvCacheStats") or {}
        used, mx = kv.get("usedNumBlocks"), kv.get("maxNumBlocks")
        if used is not None and mx:
            g["kv_cache_used_pct"] = 100.0 * float(used) / float(mx)
        if kv.get("cacheHitRate") is not None:
            g["prefix_cache_hit_rate"] = float(kv["cacheHitRate"])
        if kv.get("reusedBlocks") is not None:
            g["prefix_cache_hits"] = float(kv["reusedBlocks"])
            if kv.get("missedBlocks") is not None:
                g["prefix_cache_queries"] = (
                    float(kv["reusedBlocks"]) + float(kv["missedBlocks"]))
        if g:
            acc.gauges = g
    return acc


def snapshot(acc: _Acc) -> dict[str, float]:
    """One replica's metrics in capsim's canonical vocabulary."""
    out = dict(acc.gauges)
    out["generation_tokens_total"] = acc.gen_tokens
    out["prompt_tokens_total"] = acc.prompt_tokens
    return out


def llm_api_options(cfg) -> dict:
    """The ``--extra_llm_api_options`` document.

    Several knobs capsim searches have no command-line flag on
    ``trtllm-serve`` and are only reachable through this YAML — KV
    cache dtype above all, which is the single highest-leverage
    dimension on a KV-bound box.
    """
    kv: dict = {}
    dtype = getattr(cfg, "kv_cache_dtype", None)
    if dtype and str(dtype) != "auto":
        kv["dtype"] = str(dtype)
    if getattr(cfg, "gpu_memory_utilization", None) is not None:
        # NOTE: NOT the same quantity as vLLM's --gpu-memory-utilization.
        # vLLM's is a fraction of TOTAL VRAM covering weights + KV;
        # TensorRT-LLM's is the fraction of what remains FREE AFTER
        # weights are loaded, for KV alone. Same operator intent ("how
        # hard to push the KV pool"), different denominator — so the
        # two engines' numbers are not interchangeable and a run
        # records which engine produced it.
        kv["free_gpu_memory_fraction"] = float(cfg.gpu_memory_utilization)
    opts: dict = {}
    if kv:
        opts["kv_cache_config"] = kv
    # Mounts /prometheus/metrics and enables per-request perf records.
    # The JSON iteration stats this engine reads are always on, but the
    # histograms are a useful cross-check on client-side latency.
    opts["return_perf_metrics"] = True
    extra = getattr(cfg, "trtllm_llm_api_options", None) or {}
    for k, v in extra.items():
        if k == "kv_cache_config" and isinstance(v, dict):
            opts.setdefault("kv_cache_config", {}).update(v)
        else:
            opts[k] = v
    return opts


class TrtLlmEngine(DockerReplicaEngine):
    """N GPU-pinned ``trtllm-serve`` replicas, sticky-routed."""

    ENGINE_NAME = "trtllm"

    def __init__(self, engine_config):
        super().__init__(engine_config)
        self._acc: dict[int, _Acc] = {}
        self._acc_lock = threading.Lock()
        self._poller: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._opts_path: Optional[Path] = None
        self._warned_gaps = False

    # ── Launch ────────────────────────────────────────────────────────

    def launch(self, log_dir: str | Path = "runs") -> None:
        # The options YAML must exist on the host before any container
        # starts — it is bind-mounted read-only into every replica.
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        opts = llm_api_options(self.cfg)
        self._opts_path = d / "trtllm_llm_api_options.yaml"
        self._opts_path.write_text(yaml.safe_dump(opts, sort_keys=False))
        log.info("trtllm extra options -> %s: %s", self._opts_path, opts)
        super().launch(log_dir)
        self._start_poller()

    def shutdown(self) -> None:
        self._stop_poller()
        super().shutdown()

    def build_replica_command(self, index: int, devices: list[int],
                              container_name: str) -> list[str]:
        cfg = self.cfg
        image = getattr(cfg, "trtllm_image", None) or DEFAULT_IMAGE
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--gpus", gpus_arg_for(devices),
            # TensorRT-LLM's workers use shared memory for tensor
            # transport; the 64MB docker default kills tp>1 init.
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "--network", "host",
        ]
        cmd += self._mount_args()
        if self._opts_path is not None:
            cmd += ["-v", f"{self._opts_path}:/etc/capsim-trtllm.yaml:ro"]
        cmd += list(cfg.docker_extra_args or [])
        cmd.append(image)

        model_arg = cfg.model_local_path or cfg.model_id
        # NO --entrypoint: the image's own entrypoint puts
        # /usr/local/tensorrt/lib on LD_LIBRARY_PATH. Overriding it
        # breaks the TensorRT import before serve ever runs.
        inner = [
            "trtllm-serve", "serve", model_arg,
            "--host", "0.0.0.0",
            "--port", str(self._port(index)),
            "--backend", getattr(cfg, "trtllm_backend", None) or "pytorch",
            "--tp_size", str(len(devices)),
            "--max_seq_len", str(cfg.max_model_len),
        ]
        mns = getattr(cfg, "max_num_seqs", None)
        if mns:
            inner += ["--max_batch_size", str(int(mns))]
        mbt = getattr(cfg, "max_num_batched_tokens", None)
        if mbt:
            inner += ["--max_num_tokens", str(int(mbt))]
        if getattr(cfg, "expert_parallel", False) and len(devices) > 1:
            inner += ["--ep_size", str(len(devices))]
        if getattr(cfg, "trust_remote_code", False):
            inner += ["--trust_remote_code"]
        if self._opts_path is not None:
            inner += ["--extra_llm_api_options", "/etc/capsim-trtllm.yaml"]
        inner += list(getattr(cfg, "trtllm_extra_flags", None) or [])
        return cmd + inner

    # ── Metrics: drain in one place, serve cached ─────────────────────

    def _metrics_url(self, port: int) -> str:
        # The JSON iteration stats. /prometheus/metrics carries only
        # latency histograms — none of the counters the sweep needs.
        return f"http://{self.cfg.host}:{port}/metrics"

    def _start_poller(self) -> None:
        self._poll_stop.clear()
        self._poller = threading.Thread(
            target=self._poll_loop, name="trtllm-stats", daemon=True)
        self._poller.start()

    def _stop_poller(self) -> None:
        self._poll_stop.set()
        if self._poller is not None:
            self._poller.join(timeout=5)
            self._poller = None

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            for i, _d, port, _cid, _s in list(self._replicas):
                try:
                    r = httpx.get(self._metrics_url(port), timeout=5.0)
                    if r.status_code != 200:
                        continue
                    stats = r.json()
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(stats, list):
                    stats = [stats]
                with self._acc_lock:
                    acc = self._acc.setdefault(i, _Acc())
                    accumulate(stats, acc)
            self._poll_stop.wait(POLL_INTERVAL_S)

    def get_metrics(self) -> dict[str, float]:
        """Cached, idempotent — no I/O, so any number of callers may
        read it without stealing each other's iteration stats."""
        from .docker_replica import aggregate_replica_metrics
        with self._acc_lock:
            per_replica = [snapshot(a) for a in self._acc.values()]
            dropped = sum(a.dropped_iters for a in self._acc.values())
        if dropped and not self._warned_gaps:
            self._warned_gaps = True
            log.warning(
                "trtllm: %d iteration stat(s) were produced but never "
                "read — token totals are a LOWER bound. Lower "
                "POLL_INTERVAL_S or raise iter_stats_max_iterations.",
                dropped)
        return aggregate_replica_metrics(per_replica)

    @property
    def dropped_iterations(self) -> int:
        """Iterations the engine produced that we never sampled. Any
        nonzero value means the token totals understate reality."""
        with self._acc_lock:
            return sum(a.dropped_iters for a in self._acc.values())
