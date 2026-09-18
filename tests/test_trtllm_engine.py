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


# ── The arena searches engines too ────────────────────────────────────

def test_candidate_summary_emits_the_right_dialect():
    """The arena's dimensions are engine-neutral; the args it produces
    must not be. A vLLM flag handed to trtllm-serve is a launch
    failure, and vice versa."""
    from simulator.search import (
        Objective, SearchParams, SearchSpace, candidate_summary)

    space = SearchSpace(
        name="t",
        engine="vllm_cuda",
        model_variants={"m": {"model": "org/M"}},
        dimensions={"engine": ["vllm_cuda_multi", "trtllm"],
                    "model_variant": ["m"], "tp": [1], "dp": [1],
                    "max_num_seqs": [2048], "kv_cache_dtype": ["fp8"]},
        device_groups=[[0]],
        objective=Objective(),
        search=SearchParams(),
    )
    vllm = candidate_summary(
        {"engine": "vllm_cuda_multi", "model_variant": "m", "tp": 1,
         "dp": 1, "max_num_seqs": 2048, "kv_cache_dtype": "fp8"}, space)
    trt = candidate_summary(
        {"engine": "trtllm", "model_variant": "m", "tp": 1, "dp": 1,
         "max_num_seqs": 2048, "kv_cache_dtype": "fp8"}, space)

    assert "--max-num-seqs" in vllm["engine_args"]
    assert "--kv-cache-dtype" in vllm["engine_args"]
    # trtllm-serve spells it differently, and has NO kv dtype flag —
    # it rides in the options YAML instead.
    assert "--max_batch_size" in trt["engine_args"]
    assert "--max-num-seqs" not in trt["engine_args"]
    assert "--kv-cache-dtype" not in trt["engine_args"]
    assert trt["kv_cache_dtype"] == "fp8"
    assert trt["engine"] == "trtllm"


def test_arena_offers_only_staged_engines(monkeypatch):
    """An engine whose image is not on the box is not a choice. A host
    with one runtime gets no engine dimension at all, so combinatorics
    are unchanged until a second one is actually pulled."""
    import simulator.engine_runtimes as er
    from simulator.arena import full_arena

    monkeypatch.setattr(er, "local_images",
                        lambda: {"vllm/vllm-openai:latest"})
    assert "engine" not in full_arena(catalog=[])["dimensions"]

    monkeypatch.setattr(er, "local_images", lambda: {
        "vllm/vllm-openai:latest",
        "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"})
    dims = full_arena(catalog=[])["dimensions"]
    assert dims["engine"] == ["vllm_cuda_multi", "trtllm"]


# ── Prepare: staging a runtime is what unlocks it ─────────────────────

def test_engines_endpoint_reports_staging(monkeypatch):
    from fastapi.testclient import TestClient

    import simulator.engine_runtimes as er
    from simulator.service import create_app

    monkeypatch.setattr(er, "local_images",
                        lambda: {"vllm/vllm-openai:latest"})
    c = TestClient(create_app())
    d = c.get("/api/engines").json()
    assert d["available"] == ["vllm_cuda_multi"]
    by = {r["engine"]: r for r in d["runtimes"]}
    assert by["trtllm"]["staged"] is False
    # The size is part of the offer: these images are tens of GB.
    assert by["trtllm"]["approx_gb"] > 50


def test_pull_refuses_when_the_disk_cannot_take_it(monkeypatch):
    """A 59 GB pull onto a volume without room wedges the host. Refuse
    with the actual numbers instead of discovering it at 98%."""
    from fastapi.testclient import TestClient

    import simulator.engine_runtimes as er
    from simulator.service import create_app

    monkeypatch.setattr(er, "local_images", lambda: set())
    monkeypatch.setattr(er, "image_store_root", lambda: {
        "path": "/var/lib/containerd", "free_gb": 5.0,
        "note": "images are stored by containerd, not under "
                "Docker's data-root"})
    c = TestClient(create_app())
    r = c.post("/api/engines/pull", json={"engine": "trtllm"})
    assert r.status_code == 507
    detail = r.json()["detail"]
    assert "59 GB" in detail and "5 GB" in detail
    # Names the directory that actually fills up, not data-root.
    assert "/var/lib/containerd" in detail


def test_benchmark_config_builds_for_either_engine(monkeypatch):
    from simulator.service import _build_custom_config

    import simulator.arena as arena
    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 96.0})

    def _cfg_for(engine, tmp):
        import yaml
        p = _build_custom_config({
            "model_id": "nvidia/Qwen3.6-35B-A3B-NVFP4", "engine": engine,
            "replicas": 8, "tp": 1, "max_num_seqs": 2048,
            "kv_cache_dtype": "fp8", "gpu_memory_utilization": 0.95,
        }, tmp)
        return yaml.safe_load(p.read_text())["engine"]

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        tmp = Path(td)
        v = _cfg_for("vllm_cuda_multi", tmp)
        t = _cfg_for("trtllm", tmp)

    assert v["type"] == "vllm_cuda_multi"
    assert "--max-num-seqs" in v["vllm_extra_flags"]
    assert t["type"] == "trtllm"
    # TensorRT-LLM takes no vLLM flags; the shape rides on the config.
    assert "vllm_extra_flags" not in t
    assert t["max_num_seqs"] == 2048
    assert t["kv_cache_dtype"] == "fp8"
    # Both record the SAME canonical shape, which is what makes an
    # engine comparison a comparison.
    for k in ("max_num_seqs", "kv_cache_dtype", "gpu_memory_utilization"):
        assert v[k] == t[k]


def test_a_tensorrt_winner_promotes_to_a_tensorrt_profile(tmp_path):
    """Promoting a TensorRT winner as a vLLM profile would silently
    re-measure the winning shape on the engine that did not win it."""
    import yaml

    from simulator.promote import _write_profile

    name, path = _write_profile(
        name_hint="Optimized Qwen3",
        model_id="nvidia/Qwen3.6-35B-A3B-NVFP4",
        engine_fields={"max_model_len": 8192, "kv_cache_dtype": "fp8",
                       "tensor_parallel_size": 1},
        extra_flags=["--max_batch_size", "2048"],
        gpu_device_ids=None,
        replica_devices=[[0], [1], [2], [3]],
        engine_type="trtllm",
        provenance=["engine: trtllm"],
        warnings=[],
        out_dir=tmp_path,
    )
    eng = yaml.safe_load(path.read_text())["engine"]
    assert eng["type"] == "trtllm"
    assert eng["replica_devices"] == [[0], [1], [2], [3]]
    assert eng["kv_cache_dtype"] == "fp8"
    # Never a vLLM flag list, and never the vLLM image.
    assert "vllm_extra_flags" not in eng
    assert "gpu_image" not in eng
    assert eng["trtllm_extra_flags"] == ["--max_batch_size", "2048"]


def test_containerd_root_is_read_from_its_own_config(tmp_path):
    """docker info does not expose it, and under the containerd
    snapshotter it -- not data-root -- is the directory that fills up.
    Reporting free space from the wrong one is a confident wrong
    answer, which is how a 59 GB pull wedges a root volume."""
    from simulator.engine_runtimes import _containerd_root

    cfg = tmp_path / "config.toml"
    cfg.write_text('version = 3\nroot = "/data/containerd"\n'
                   'state = "/run/containerd"\n')
    assert _containerd_root(str(cfg)) == "/data/containerd"

    # Commented-out settings are not settings.
    cfg.write_text('# root = "/wrong"\nversion = 3\n')
    assert _containerd_root(str(cfg), default=str(tmp_path)) == str(tmp_path)

    # No config at all -> containerd's built-in default, when present.
    missing = str(tmp_path / "nope.toml")
    assert _containerd_root(missing, default=str(tmp_path)) == str(tmp_path)
    assert _containerd_root(missing, default=str(tmp_path / "absent")) is None
