"""TensorRT-LLM engine: launch argv, and the iteration-stat
accounting the headline number depends on."""

from __future__ import annotations

import pytest

from simulator.config import EngineConfig
from simulator.engines.trtllm import (
    DEFAULT_IMAGE,
    TrtLlmEngine,
    _Acc,
    accumulate,
    iteration_gen_tokens,
    iteration_prompt_tokens,
    llm_api_options,
    serve_argv,
    snapshot,
)


def _cfg(**kw) -> EngineConfig:
    base = dict(type="trtllm", model_id="org/M", port=9100,
                replica_devices=[[0], [1], [2], [3]], max_model_len=8192)
    base.update(kw)
    return EngineConfig(**base)


def _stat(i, *, active=0, queued=0, gen_reqs=0, ctx_tokens=0,
          per_iter=1.0, used=None, mx=None):
    s = {
        "iter": i,
        "numActiveRequests": active,
        "numQueuedRequests": queued,
        "inflightBatchingStats": {
            "numGenRequests": gen_reqs,
            "numCtxTokens": ctx_tokens,
            "avgNumDecodedTokensPerIter": per_iter,
        },
    }
    if used is not None:
        s["kvCacheStats"] = {"usedNumBlocks": used, "maxNumBlocks": mx,
                             "cacheHitRate": 0.25,
                             "reusedBlocks": 30, "missedBlocks": 70}
    return s


# ── Launch ────────────────────────────────────────────────────────────

def test_launch_never_overrides_the_entrypoint():
    """The image's ENV LD_LIBRARY_PATH omits /usr/local/tensorrt/lib —
    that path is added by /etc/bash.bashrc, which only the image's own
    entrypoint runs. --entrypoint dies on libnvonnxparser.so.10."""
    eng = TrtLlmEngine(_cfg())
    cmd = eng.build_replica_command(2, [2], "trtllm-r2-x")
    assert "--entrypoint" not in cmd
    # The server is the CMD, placed after the image.
    assert cmd.index(DEFAULT_IMAGE) < cmd.index("trtllm-serve")
    assert cmd[cmd.index("trtllm-serve") + 1] == "serve"


def test_replica_argv_carries_the_shape():
    eng = TrtLlmEngine(_cfg(max_num_seqs=2048, max_num_batched_tokens=8192,
                            trust_remote_code=True))
    cmd = eng.build_replica_command(1, [1], "trtllm-r1-x")
    j = " ".join(cmd)
    assert "--gpus device=1" in j
    assert "--ipc=host" in j
    assert "--port 9101" in j                  # port + index
    assert "--max_batch_size 2048" in j        # NOT --max-num-seqs
    assert "--max_num_tokens 8192" in j
    assert "--max_seq_len 8192" in j
    assert "--trust_remote_code" in j
    assert "/root/.cache/huggingface" in j

    # A tp2 replica: quoted device pair, tp from the group width.
    eng = TrtLlmEngine(_cfg(replica_devices=[[0, 1], [2, 3]]))
    cmd = eng.build_replica_command(0, [0, 1], "trtllm-r0-x")
    assert '"device=0,1"' in cmd
    assert cmd[cmd.index("--tp_size") + 1] == "2"


def test_expert_parallel_only_applies_above_tp1():
    ep = dict(expert_parallel=True)
    single = serve_argv("m", port=1, tp=1, **ep)
    multi = serve_argv("m", port=1, tp=4, **ep)
    assert "--ep_size" not in single
    assert multi[multi.index("--ep_size") + 1] == "4"


def test_kv_dtype_travels_in_the_options_yaml():
    """trtllm-serve has no KV-dtype flag; the YAML is the only route,
    and it is the highest-leverage dimension on a KV-bound box."""
    opts = llm_api_options(_cfg(kv_cache_dtype="fp8",
                                gpu_memory_utilization=0.95))
    assert opts["kv_cache_config"]["dtype"] == "fp8"
    assert opts["kv_cache_config"]["free_gpu_memory_fraction"] == 0.95
    assert opts["return_perf_metrics"] is True
    # "auto" means "say nothing" — not a literal dtype.
    assert "dtype" not in llm_api_options(
        _cfg(kv_cache_dtype="auto")).get("kv_cache_config", {})


def test_caller_options_merge_without_clobbering_kv_config():
    opts = llm_api_options(_cfg(
        kv_cache_dtype="fp8",
        trtllm_llm_api_options={"kv_cache_config": {"enable_block_reuse": False},
                                "print_iter_log": True}))
    assert opts["kv_cache_config"]["dtype"] == "fp8"
    assert opts["kv_cache_config"]["enable_block_reuse"] is False
    assert opts["print_iter_log"] is True


# ── Iteration-stat accounting ─────────────────────────────────────────

def test_gen_tokens_from_inflight_batching():
    # pytorch backend: every generating request emits
    # avgNumDecodedTokensPerIter tokens this iteration.
    assert iteration_gen_tokens(_stat(1, gen_reqs=64)) == 64
    # Speculative decoding emits more than one per request per step.
    assert iteration_gen_tokens(_stat(1, gen_reqs=64, per_iter=2.5)) == 160
    assert iteration_prompt_tokens(_stat(1, ctx_tokens=4096)) == 4096


def test_gen_tokens_prefers_static_batching_when_present():
    """The tensorrt backend counts gen tokens directly."""
    s = _stat(1, gen_reqs=64)
    s["staticBatchingStats"] = {"numGenTokens": 999, "numCtxTokens": 7}
    assert iteration_gen_tokens(s) == 999


def test_accumulate_builds_a_monotonic_counter():
    """The sweep takes deltas of generation_tokens_total; TensorRT-LLM
    has no such counter, so it is rebuilt from per-iteration stats."""
    acc = _Acc()
    accumulate([_stat(i, gen_reqs=10, ctx_tokens=5) for i in range(1, 6)], acc)
    snap = snapshot(acc)
    assert snap["generation_tokens_total"] == 50
    assert snap["prompt_tokens_total"] == 25
    # A later drain continues the same totals rather than restarting.
    accumulate([_stat(i, gen_reqs=10) for i in range(6, 9)], acc)
    assert snapshot(acc)["generation_tokens_total"] == 80
    assert acc.dropped_iters == 0


def test_overlapping_reads_never_double_count():
    """Re-delivering a snapshot we already folded in would inflate the
    headline number — the worst possible failure mode."""
    acc = _Acc()
    accumulate([_stat(i, gen_reqs=10) for i in (1, 2, 3)], acc)
    accumulate([_stat(i, gen_reqs=10) for i in (2, 3, 4)], acc)
    assert snapshot(acc)["generation_tokens_total"] == 40    # not 60


def test_gaps_are_counted_not_hidden():
    """Per-iteration stats cannot be resampled. A gap means the total
    understates reality, and that must be visible rather than passed
    off as a measurement."""
    acc = _Acc()
    accumulate([_stat(1, gen_reqs=10), _stat(7, gen_reqs=10)], acc)
    assert acc.dropped_iters == 5
    assert snapshot(acc)["generation_tokens_total"] == 20     # a floor


def test_gauges_come_from_the_latest_iteration():
    acc = _Acc()
    accumulate([_stat(1, active=10, queued=0, used=10, mx=100),
                _stat(2, active=64, queued=12, used=80, mx=100)], acc)
    snap = snapshot(acc)
    assert snap["num_running"] == 64
    assert snap["queue_depth"] == 12
    assert snap["kv_cache_used_pct"] == 80.0
    assert snap["prefix_cache_hits"] == 30
    assert snap["prefix_cache_queries"] == 100


def test_get_metrics_is_idempotent_and_does_no_io():
    """capsim has two independent pollers (telemetry loop, sweep
    sampler). TensorRT-LLM's /metrics DRAINS its queue, so if either
    poller touched it directly they would split the stream and both
    undercount. get_metrics must be a pure read of cached state."""
    eng = TrtLlmEngine(_cfg(replica_devices=[[0], [1]]))
    for replica in (0, 1):
        acc = eng._acc.setdefault(replica, _Acc())
        accumulate([_stat(i, gen_reqs=10, active=5, used=50, mx=100)
                    for i in range(1, 4)], acc)
    first = eng.get_metrics()
    second = eng.get_metrics()
    assert first == second                       # no draining side effect
    # Counters SUM across replicas (whole-box), gauges average.
    assert first["generation_tokens_total"] == 60
    assert first["num_running"] == 10
    assert first["kv_cache_used_pct"] == 50.0
    assert eng.dropped_iterations == 0


def test_hit_rate_survives_multi_replica_rollup():
    from simulator.engines.docker_replica import aggregate_replica_metrics
    agg = aggregate_replica_metrics([{"prefix_cache_hit_rate": 0.2},
                                     {"prefix_cache_hit_rate": 0.4}])
    assert agg["prefix_cache_hit_rate"] == pytest.approx(0.3)
