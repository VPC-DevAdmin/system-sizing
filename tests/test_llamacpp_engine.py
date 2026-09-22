"""llama.cpp: a GGUF-fed engine with an expert-offload switch.

The reason it exists is the largest models -- every one of them has a
GGUF and llama-server loads any architecture llama.cpp knows -- so the
tests pin the launch shape those models need: the first shard of a
split quant, the expert-offload pattern chosen by the GGUF's size
against the replica's VRAM, one pool of context split over its slots,
and refusals in milliseconds where a launch would otherwise burn the
health timeout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simulator.config import EngineConfig
from simulator.engines.knobs import GGUF_ENGINES, GPU_ENGINES, unsupported
from simulator.engines.llamacpp import (
    DEFAULT_IMAGE,
    DOCUMENTED_MAX_BATCH,
    ENTRYPOINT,
    OFFLOAD_PATTERN,
    LlamaCppEngine,
    context_tokens,
    default_threads,
    gguf_entry_file,
    offload_by_default,
    parse_llamacpp_metrics,
    serve_argv,
)

HW = {"count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
      "vram_per_gpu_gb": 96.0}


def _gguf_dir(tmp_path: Path, *names: str, size: int = 4) -> Path:
    d = tmp_path / "gguf"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"G" * size)
    return d


def _cfg(gguf: Path, **kw) -> EngineConfig:
    base = dict(type="llamacpp", model_id="moonshotai/Kimi-K2-Thinking",
                port=9100, replica_devices=[[0, 1, 2, 3, 4, 5, 6, 7]],
                max_model_len=4096, max_num_seqs=32, vram_per_gpu_gb=96.0,
                llamacpp_gguf_path=str(gguf))
    base.update(kw)
    return EngineConfig(**base)


def _argv(cmd: list[str]) -> list[str]:
    return cmd[cmd.index(DEFAULT_IMAGE) + 1:]


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ── Registry, runtime, label ──────────────────────────────────────────

def test_llamacpp_is_a_first_class_engine():
    """Picker membership, a label, a runtime row with the CUDA server
    image, a factory branch -- each is a place a new engine has been
    forgotten before."""
    from simulator.engine_runtimes import RUNTIMES, runtime_status
    from simulator.engines import make_engine
    from simulator.engines.knobs import ENGINE_LABELS, caveats

    assert "llamacpp" in GPU_ENGINES and "llamacpp" in GGUF_ENGINES
    assert ENGINE_LABELS["llamacpp"] == "llama.cpp"
    assert RUNTIMES["llamacpp"]["image"] == DEFAULT_IMAGE
    assert DEFAULT_IMAGE == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    row = next(r for r in runtime_status(set()) if r["engine"] == "llamacpp")
    assert row["label"] == "llama.cpp" and not row["staged"]
    assert row["caveats"] == caveats("llamacpp") and row["caveats"]
    eng = make_engine("llamacpp", EngineConfig(
        type="llamacpp", model_id="org/M", port=9100, replica_devices=[[0]]))
    assert isinstance(eng, LlamaCppEngine)
    assert EngineConfig(type="llamacpp", model_id="org/M", port=9100
                        ).base_url == "http://127.0.0.1:9100/v1"


def test_the_container_prefix_is_swept():
    from simulator.engines.docker_replica import CAPSIM_CONTAINER_PREFIXES
    assert "llamacpp-" in CAPSIM_CONTAINER_PREFIXES


# ── argv construction ─────────────────────────────────────────────────

def test_launch_uses_the_images_own_entrypoint(tmp_path, monkeypatch):
    """The server image's ENTRYPOINT is /app/llama-server, so the argv
    after the image IS llama-server's argument list; nothing overrides
    the entrypoint (the opposite of the KTransformers image)."""
    import simulator.engines.llamacpp as lc
    monkeypatch.setattr(lc, "physical_cores", lambda: 172)
    gguf = _gguf_dir(tmp_path, "Kimi-K2-Thinking-UD-Q4_K_XL.gguf")
    eng = LlamaCppEngine(_cfg(gguf))
    cmd = eng.build_replica_command(
        0, [0, 1, 2, 3, 4, 5, 6, 7], "llamacpp-r0-x")
    assert ENTRYPOINT == "/app/llama-server"
    assert "--entrypoint" not in cmd
    assert cmd[:4] == ["docker", "run", "-d", "--rm"]
    assert cmd[cmd.index("--name") + 1] == "llamacpp-r0-x"
    # Every GPU of the replica, quoted the way docker's CSV parser needs.
    assert cmd[cmd.index("--gpus") + 1] == '"device=0,1,2,3,4,5,6,7"'
    assert "--ipc=host" in cmd and "host" == cmd[cmd.index("--network") + 1]
    assert f"{gguf}:/gguf:ro" in cmd
    argv = _argv(cmd)
    assert _flag(argv, "-m") == "/gguf/Kimi-K2-Thinking-UD-Q4_K_XL.gguf"
    assert _flag(argv, "--host") == "0.0.0.0" and _flag(argv, "--port") == str(eng._port(0))
    assert _flag(argv, "-ngl") == "999" and _flag(argv, "-sm") == "layer"
    assert _flag(argv, "-fa") == "on"
    assert "-cb" in argv and "--metrics" in argv
    assert _flag(argv, "-np") == "32"
    # One pool split across the slots: 4096 x 32.
    assert _flag(argv, "-c") == str(4096 * 32)
    assert _flag(argv, "-a") == "moonshotai/Kimi-K2-Thinking"
    assert _flag(argv, "-t") == "170"


def test_sharded_quant_loads_from_its_first_shard(tmp_path):
    """unsloth splits a 646 GB quant into fourteen files; llama.cpp
    opens the rest from the first shard's split metadata."""
    names = [f"Kimi-K2-Thinking-UD-Q4_K_XL-{i:05d}-of-00014.gguf"
             for i in range(1, 15)]
    gguf = _gguf_dir(tmp_path, *reversed(names))
    assert gguf_entry_file(gguf) == names[0]
    cmd = LlamaCppEngine(_cfg(gguf)).build_replica_command(0, [0], "lc-r0-x")
    assert _flag(_argv(cmd), "-m") == f"/gguf/{names[0]}"
    # One file: that file. Two unsharded files: ambiguous, refused.
    assert gguf_entry_file(_gguf_dir(tmp_path / "one", "m.gguf")) == "m.gguf"
    with pytest.raises(ValueError, match="none is a first shard"):
        gguf_entry_file(_gguf_dir(tmp_path / "two", "a.gguf", "b.gguf"))
    # An interrupted download leaves an empty file: not loadable.
    with pytest.raises(ValueError, match="no .gguf"):
        gguf_entry_file(_gguf_dir(tmp_path / "empty", "m.gguf", size=0))
    with pytest.raises(ValueError):
        gguf_entry_file(tmp_path / "missing")


def test_expert_offload_defaults_to_the_ggufs_size_against_vram(tmp_path):
    """On when the GGUF exceeds 85% of the replica's VRAM, off
    otherwise; the lever overrides either way; unknown VRAM offloads
    (the choice that always loads)."""
    assert offload_by_default(660.0, 96.0, 8) is True
    # Kimi-K2 at UD-Q4_K_XL is 646.2 GB: just under 85% of 768 GB, so
    # by default it loads GPU-resident and the lever forces offload.
    assert offload_by_default(646.2, 96.0, 8) is (646.2 > 0.85 * 96 * 8)
    assert offload_by_default(140.8, 96.0, 8) is False   # MiniMax-M2.7 fits
    assert offload_by_default(140.8, 96.0, 1) is True    # ...not on one card
    assert offload_by_default(407.8, 96.0, 8) is False   # DeepSeek-V3.2 at Q4 fits 8 cards
    assert offload_by_default(None, 96.0, 8) is True
    assert offload_by_default(10.0, None, 8) is True

    # Through the launcher: the directory's bytes decide.
    big = _gguf_dir(tmp_path / "big", "m.gguf", size=700)
    small = _gguf_dir(tmp_path / "small", "m.gguf", size=100)
    # vram_per_gpu_gb is in GB and the fake files are bytes, so pin a
    # VRAM figure the fakes straddle: 0.85 x 200e-9 x 4 = 6.8e-7 GB.
    vram = 200e-9
    on = LlamaCppEngine(_cfg(big, vram_per_gpu_gb=vram)
                        ).build_replica_command(0, [0, 1, 2, 3], "x")
    off = LlamaCppEngine(_cfg(small, vram_per_gpu_gb=vram)
                         ).build_replica_command(0, [0, 1, 2, 3], "x")
    assert _flag(_argv(on), "-ot") == OFFLOAD_PATTERN
    assert OFFLOAD_PATTERN == r"\.ffn_.*_exps\.=CPU"
    assert "-ot" not in _argv(off)
    forced_off = LlamaCppEngine(_cfg(big, vram_per_gpu_gb=vram,
                                     llamacpp_offload_experts=False)
                                ).build_replica_command(0, [0, 1, 2, 3], "x")
    forced_on = LlamaCppEngine(_cfg(small, vram_per_gpu_gb=vram,
                                    llamacpp_offload_experts=True)
                               ).build_replica_command(0, [0, 1, 2, 3], "x")
    assert "-ot" not in _argv(forced_off) and "-ot" in _argv(forced_on)


def test_threads_default_to_physical_cores_minus_two(monkeypatch, tmp_path):
    import simulator.engines.llamacpp as lc
    assert default_threads(172) == 170
    assert default_threads(2) == 1
    assert default_threads(0) is None
    monkeypatch.setattr(lc, "physical_cores", lambda: None)
    gguf = _gguf_dir(tmp_path, "m.gguf")
    assert "-t" not in _argv(LlamaCppEngine(_cfg(gguf)).build_replica_command(
        0, [0], "x"))
    explicit = LlamaCppEngine(_cfg(gguf, llamacpp_cpu_threads=64)
                              ).build_replica_command(0, [0], "x")
    assert _flag(_argv(explicit), "-t") == "64"


def test_context_is_one_pool_split_over_the_slots():
    """n_ctx_slot = n_ctx / n_parallel upstream: a 4k model length at
    32 slots must be a 128k pool, or every slot gets 128 tokens."""
    assert context_tokens(4096, 32) == 131072
    assert context_tokens(8192, 1) == 8192
    argv = serve_argv("m.gguf", port=1, max_model_len=4096)
    assert _flag(argv, "-np") == str(DOCUMENTED_MAX_BATCH)
    assert _flag(argv, "-c") == str(4096 * DOCUMENTED_MAX_BATCH)
    assert DOCUMENTED_MAX_BATCH == 32


def test_knobs_map_to_llama_servers_flags(tmp_path):
    """fp8 KV -> q8_0 for K and V; batched tokens -> -b and -ub; extra
    flags ride at the end; an image override replaces the default."""
    gguf = _gguf_dir(tmp_path, "m.gguf")
    cmd = LlamaCppEngine(_cfg(gguf, kv_cache_dtype="fp8",
                              max_num_batched_tokens=2048,
                              llamacpp_extra_flags=["--no-webui"],
                              llamacpp_image="ghcr.io/ggml-org/llama.cpp:server-cuda-v0.4.1",
                              served_model_name="kimi")
                         ).build_replica_command(0, [0], "x")
    assert "ghcr.io/ggml-org/llama.cpp:server-cuda-v0.4.1" in cmd
    assert DEFAULT_IMAGE not in cmd
    argv = cmd[cmd.index("ghcr.io/ggml-org/llama.cpp:server-cuda-v0.4.1") + 1:]
    assert _flag(argv, "-ctk") == "q8_0" and _flag(argv, "-ctv") == "q8_0"
    assert _flag(argv, "-b") == "2048" and _flag(argv, "-ub") == "2048"
    assert argv[-1] == "--no-webui"
    assert _flag(argv, "-a") == "kimi"
    # auto / unset KV: no cache-type flags at all.
    plain = _argv(LlamaCppEngine(_cfg(gguf, kv_cache_dtype="auto")
                                 ).build_replica_command(0, [0], "x"))
    assert "-ctk" not in plain and "-ctv" not in plain
    with pytest.raises(RuntimeError, match="no KV cache type"):
        LlamaCppEngine(_cfg(gguf, kv_cache_dtype="nvfp4")
                       ).build_replica_command(0, [0], "x")


def test_health_is_the_endpoint_that_waits(tmp_path):
    """/v1/models answers 200 with a null meta while loading; /health
    is 503 until the model is up and 200 after."""
    eng = LlamaCppEngine(_cfg(_gguf_dir(tmp_path, "m.gguf")))
    assert eng._ready_url(9100) == "http://127.0.0.1:9100/health"
    assert eng._metrics_url(9100) == "http://127.0.0.1:9100/metrics"


# ── Refusals ──────────────────────────────────────────────────────────

def test_launch_refuses_without_a_gguf(tmp_path):
    """Milliseconds with the reason, not a 30-minute health timeout."""
    with pytest.raises(RuntimeError, match="no llamacpp_gguf_path"):
        LlamaCppEngine(_cfg(tmp_path, llamacpp_gguf_path=None)
                       ).build_replica_command(0, [0], "x")
    with pytest.raises(RuntimeError, match="not a directory"):
        LlamaCppEngine(_cfg(tmp_path / "nope")).build_replica_command(0, [0], "x")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no .gguf file"):
        LlamaCppEngine(_cfg(empty)).build_replica_command(0, [0], "x")


def test_knob_refusals():
    """nvfp4 has no cache type; expert parallelism is meaningless in a
    single process; fp8 and auto pass; the GPU memory share is ignored
    (documented in the caveats), never refused."""
    assert unsupported("llamacpp", {"kv_cache_dtype": "nvfp4"})
    assert "nvfp4" in unsupported("llamacpp", {"kv_cache_dtype": "nvfp4"})
    assert unsupported("llamacpp", {"expert_parallel": True})
    assert unsupported("llamacpp", {"kv_cache_dtype": "fp8"}) is None
    assert unsupported("llamacpp", {"kv_cache_dtype": None,
                                    "gpu_memory_utilization": 0.95}) is None
    from simulator.engines.knobs import caveats
    assert any("GPU memory share" in c for c in caveats("llamacpp"))


def test_custom_engine_resolves_the_companion_and_refuses_without(tmp_path,
                                                                  monkeypatch):
    """The same path KTransformers takes: the catalog's GGUF companion
    is resolved once staged, the Prepare button is named when it is
    not, and a model without one gets the configure-the-path refusal.
    The companion's allow-list is honoured, and an explicit path is
    outside its judgement."""
    from simulator.engines.custom import ShapeError, custom_engine

    cache = tmp_path / "hf"
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    spec = {"repo": "unsloth/K-GGUF", "file": "UD-Q4_K_XL", "size_gb": 646.2,
            "engines": ["llamacpp"]}
    catalog = [{"id": "moonshotai/K", "family": "k", "quant": "other",
                "kt_only": True, "gguf": spec},
               {"id": "org/Plain", "family": "p", "quant": "bf16", "gguf": None}]
    base = {"model_id": "moonshotai/K", "device": "gpu", "engine": "llamacpp",
            "replicas": 1, "tp": 1, "kv_cache_dtype": "auto"}

    with pytest.raises(ShapeError, match="stage the GGUF companion in Prepare"):
        custom_engine(base, hw=HW, catalog=catalog)
    with pytest.raises(ShapeError, match="llamacpp_gguf_path is configured"):
        custom_engine({**base, "model_id": "org/Plain"}, hw=HW, catalog=catalog)
    # kt_only: the GPU engines are refused, naming both GGUF engines.
    with pytest.raises(ShapeError, match="ktransformers, llamacpp"):
        custom_engine({**base, "engine": "vllm_cuda_multi"}, hw=HW,
                      catalog=catalog)
    # The allow-list: KTransformers may not run this companion.
    with pytest.raises(ShapeError, match="for llamacpp only"):
        custom_engine({**base, "engine": "ktransformers", "tp": 1}, hw=HW,
                      catalog=catalog)

    rev = cache / "hub" / "models--unsloth--K-GGUF" / "snapshots" / "abc"
    shard = rev / "UD-Q4_K_XL"
    shard.mkdir(parents=True)
    for i in (1, 2):
        (shard / f"K-UD-Q4_K_XL-{i:05d}-of-00002.gguf").write_bytes(b"gguf")
    eng = custom_engine(base, hw=HW, catalog=catalog)
    assert eng["type"] == "llamacpp"
    assert eng["llamacpp_gguf_path"] == str(shard)
    # One replica at tp1 is the WHOLE box: llama-server splits by layer
    # across every GPU it sees, crossing PCIe domains without an
    # all-reduce. A tp > 1 confines it to one domain like any engine.
    assert eng["replica_devices"] == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert custom_engine({**base, "tp": 4}, hw=HW, catalog=catalog
                         )["replica_devices"] == [[0, 1, 2, 3]]
    assert custom_engine({**base, "replicas": 2}, hw=HW, catalog=catalog
                         )["replica_devices"] == [[0], [1]]
    cmd = LlamaCppEngine(EngineConfig(**eng)).build_replica_command(
        0, eng["replica_devices"][0], "llamacpp-r0-x")
    assert ":/gguf:ro" not in " ".join(cmd)  # reached through the cache mount
    # Inside the cache the shard is reached through the cache mount, so
    # its relative symlink into blobs/ still resolves in the container.
    assert _flag(_argv(cmd), "-m").endswith(
        "/hub/models--unsloth--K-GGUF/snapshots/abc/UD-Q4_K_XL/K-UD-Q4_K_XL-00001-of-00002.gguf")
    assert _flag(_argv(cmd), "-m").startswith("/root/.cache/huggingface/")

    # An explicit directory wins, and is not bound by the allow-list.
    mine = _gguf_dir(tmp_path / "mine", "m.gguf")
    eng = custom_engine({**base, "llamacpp_gguf_path": str(mine)},
                        hw=HW, catalog=catalog)
    assert eng["llamacpp_gguf_path"] == str(mine)
    # The tri-state lever coerces the arena's strings.
    for raw, want in (("on", True), ("off", False), ("auto", None), (True, True)):
        eng = custom_engine({**base, "llamacpp_gguf_path": str(mine),
                             "llamacpp_offload_experts": raw},
                            hw=HW, catalog=catalog)
        assert eng.get("llamacpp_offload_experts") is want, raw


# ── Metrics ───────────────────────────────────────────────────────────

# As llama-server (server-task.cpp, master 2026-09) renders /metrics:
# every item is emitted as llamacpp:<name> with HELP/TYPE lines.
SAMPLE_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 120000
# HELP llamacpp:prompt_tokens_cached_total Number of prompt tokens reused from the cache
# TYPE llamacpp:prompt_tokens_cached_total counter
llamacpp:prompt_tokens_cached_total 30000
# HELP llamacpp:prompt_seconds_total Total time spent processing prompts
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 41.5
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 98765
# HELP llamacpp:tokens_predicted_seconds_total Total time spent generating tokens
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 300.2
# HELP llamacpp:n_decode_total Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 5000
# HELP llamacpp:n_tokens_max Largest observed sequence length (prompt + generation)
# TYPE llamacpp:n_tokens_max counter
llamacpp:n_tokens_max 2200
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 2891.5
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 329.0
# HELP llamacpp:requests_processing Number of requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 31
# HELP llamacpp:requests_deferred Number of requests deferred
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 7
# HELP llamacpp:n_busy_slots_per_decode Average number of busy slots per llama_decode() call
# TYPE llamacpp:n_busy_slots_per_decode gauge
llamacpp:n_busy_slots_per_decode 28.4
"""


def test_metrics_parse_to_the_canonical_keys(tmp_path):
    m = parse_llamacpp_metrics(SAMPLE_METRICS)
    assert m["prompt_tokens_total"] == 120000
    assert m["generation_tokens_total"] == 98765
    assert m["num_running"] == 31
    assert m["queue_depth"] == 7
    # prompt_tokens_total excludes cached tokens upstream, so the
    # prefix-cache view is hits = cached, queries = cached + processed.
    assert m["prefix_cache_hits"] == 30000
    assert m["prefix_cache_queries"] == 150000
    assert m["prefix_cache_hit_rate"] == pytest.approx(0.2)
    assert "kv_cache_used_pct" not in m
    # The engine's parser is this one, and it aggregates like the rest.
    eng = LlamaCppEngine(_cfg(_gguf_dir(tmp_path, "m.gguf")))
    assert eng.parse_metrics(SAMPLE_METRICS)["generation_tokens_total"] == 98765
    from simulator.engines.docker_replica import aggregate_replica_metrics
    agg = aggregate_replica_metrics([m, m])
    assert agg["generation_tokens_total"] == 2 * 98765
    assert agg["prefix_cache_hit_rate"] == pytest.approx(0.2)
    # Older builds' KV gauges are read when present, as a percentage.
    old = parse_llamacpp_metrics(
        SAMPLE_METRICS + "llamacpp:kv_cache_usage_ratio 0.37\n"
        "llamacpp:kv_cache_tokens 131072\n")
    assert old["kv_cache_used_pct"] == pytest.approx(37.0)
    assert old["kv_cache_tokens"] == 131072


def test_api_model_name_is_what_v1_models_reports(tmp_path, monkeypatch):
    """The launch passes --alias, so the configured name is the answer
    unless the server says otherwise; without a replica up, the
    configured name."""
    import simulator.engines.llamacpp as lc

    eng = LlamaCppEngine(_cfg(_gguf_dir(tmp_path, "m.gguf")))
    assert eng.api_model_name == "moonshotai/Kimi-K2-Thinking"

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"object": "list", "data": [{"id": "served-as", "object": "model",
                                                "owned_by": "llamacpp"}]}
    calls = []
    monkeypatch.setattr(lc.httpx, "get", lambda url, timeout: (calls.append(url), R)[1])
    eng._replicas.append((0, [0], 9100, "cid", None))
    assert eng.api_model_name == "served-as"
    assert calls == ["http://127.0.0.1:9100/v1/models"]
    assert eng.api_model_name == "served-as" and len(calls) == 1   # cached


# ── Roofline ──────────────────────────────────────────────────────────

def test_roofline_gives_a_gguf_only_giant_llamacpp_cells(tmp_path):
    """Kimi-K2-Thinking: beyond VRAM, GGUF companion for llama.cpp
    only. It gets llamacpp cells, no GPU-engine cells, and no
    KTransformers cell (the allow-list). DeepSeek-V3.1, whose companion
    both GGUF engines serve, gets both. A GPU-fitting model with a
    llama.cpp companion gets GPU cells AND a llamacpp cell."""
    from simulator.roofline import (
        KT_MAX_MODEL_LEN,
        cell_overrides,
        cells,
        engine_defaults,
        engines_for,
        score_models,
    )

    catalog = [
        {"id": "moonshotai/Kimi", "family": "kimi", "series": "Kimi K2",
         "quant": "other", "params_b": 1026.0, "moe": True,
         "approx_size_gb": 594, "min_vram_gb": 40, "kt_only": True,
         "gguf": {"repo": "u/Kimi-GGUF", "file": "UD-Q4_K_XL",
                  "size_gb": 646.2, "engines": ["llamacpp"]}},
        {"id": "deepseek-ai/V31", "family": "v31", "series": "DeepSeek",
         "quant": "fp8", "params_b": 685.0, "moe": True,
         "approx_size_gb": 687, "min_vram_gb": 24, "kt_only": True,
         "gguf": {"repo": "u/V31-GGUF", "file": "UD-Q4_K_XL", "size_gb": 386.9,
                  "engines": ["ktransformers", "llamacpp"]}},
        {"id": "MiniMaxAI/M", "family": "m", "series": "MiniMax",
         "quant": "fp8", "params_b": 228.7, "moe": True,
         "approx_size_gb": 230, "min_vram_gb": 255,
         "gguf": {"repo": "u/M-GGUF", "file": "UD-Q4_K_XL", "size_gb": 140.8,
                  "engines": ["llamacpp"]}},
        {"id": "org/NoGguf", "family": "n", "series": "N", "quant": "fp8",
         "params_b": 30.0, "approx_size_gb": 60, "min_vram_gb": 70},
    ]
    by = {c.id: c for c in score_models(catalog, vram_per_gpu_gb=96,
                                        host_ram_gb=2048, cache=tmp_path)}
    kimi = by["moonshotai/Kimi"]
    assert not kimi.fits_gpu and kimi.kt_eligible and kimi.fits
    assert kimi.gguf_engines == ["llamacpp"]
    assert "llama.cpp" in kimi.why and "KTransformers" not in kimi.why
    assert by["deepseek-ai/V31"].gguf_engines == ["ktransformers", "llamacpp"]
    assert "KTransformers / llama.cpp" in by["deepseek-ai/V31"].why
    info = {c.id: c.info() for c in by.values()}
    assert info["moonshotai/Kimi"]["gguf_engines"] == ["llamacpp"]
    assert info["org/NoGguf"]["gguf_engines"] == ["ktransformers", "llamacpp"]

    assert engines_for("llamacpp", info["moonshotai/Kimi"])
    assert not engines_for("ktransformers", info["moonshotai/Kimi"])
    assert not engines_for("vllm_cuda_multi", info["moonshotai/Kimi"])
    assert engines_for("ktransformers", info["deepseek-ai/V31"])
    assert not engines_for("llamacpp", info["org/NoGguf"])
    assert engines_for("llamacpp", None)

    assert engine_defaults("llamacpp") == {"replicas": 1, "kv_cache_dtype": "auto",
                                           "max_model_len": KT_MAX_MODEL_LEN}
    notes: list[str] = []
    plan = cells(list(info), ["vllm_cuda_multi", "ktransformers", "llamacpp"],
                 {"max_num_seqs": [1024, 2048], "output_tokens": [128, 256]},
                 engine_shape={"replicas": 8, "tp": 1, "kv_cache_dtype": "fp8",
                               "gpu_memory_utilization": 0.95},
                 model_info=info, notes=notes)
    grid: dict[str, dict[str, list]] = {}
    for c in plan:
        grid.setdefault(c["model"], {}).setdefault(c["engine"], []).append(c)
    assert set(grid["moonshotai/Kimi"]) == {"llamacpp"}
    assert set(grid["deepseek-ai/V31"]) == {"ktransformers", "llamacpp"}
    assert set(grid["MiniMaxAI/M"]) == {"vllm_cuda_multi", "llamacpp"}
    assert set(grid["org/NoGguf"]) == {"vllm_cuda_multi"}
    # Two batch widths clamp to the 32 documented slots -> one cell per
    # output length; one replica, kv auto, the KT context, no tp.
    lc = grid["moonshotai/Kimi"]["llamacpp"]
    assert len(lc) == 2
    assert all(c["max_num_seqs"] == DOCUMENTED_MAX_BATCH for c in lc)
    assert all(c["replicas"] == 1 and c["kv_cache_dtype"] == "auto"
               and c["max_model_len"] == KT_MAX_MODEL_LEN for c in lc)
    assert all("tp" not in c or c["tp"] == 1 for c in lc)
    assert grid["deepseek-ai/V31"]["ktransformers"][0]["max_num_seqs"] == 4
    # The GPU-fitting model's llamacpp cell is whole-box, not tp4 x 2.
    mm = grid["MiniMaxAI/M"]
    assert mm["vllm_cuda_multi"][0]["tp"] == 4
    assert mm["llamacpp"][0]["replicas"] == 1
    assert mm["llamacpp"][0].get("tp", 1) == 1
    # And the notes say why the others are missing.
    assert any("Kimi: no ktransformers cells" in n and "llamacpp only" in n
               for n in notes)
    assert any("Kimi: no vllm_cuda_multi cells" in n and "llama.cpp only" in n
               for n in notes)
    assert any("NoGguf: no llamacpp cells" in n and "no GGUF companion" in n
               for n in notes)
    ov = cell_overrides(lc[0])
    assert ov["engine"] == "llamacpp" and ov["replicas"] == 1
    # The override passes the builder once the companion is staged.
    from simulator.engines.custom import ShapeError, custom_engine
    with pytest.raises(ShapeError, match="GGUF"):
        custom_engine({**ov, "model_id": "moonshotai/Kimi", "device": "gpu"},
                      hw=HW, catalog=catalog)
    mine = _gguf_dir(tmp_path / "mine", "m.gguf")
    eng = custom_engine({**ov, "model_id": "moonshotai/Kimi", "device": "gpu",
                         "llamacpp_gguf_path": str(mine)},
                        hw=HW, catalog=catalog)
    assert eng["type"] == "llamacpp" and eng["max_num_seqs"] == DOCUMENTED_MAX_BATCH
    assert eng["replica_devices"] == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_packaged_catalog_carries_the_giants(tmp_path):
    """The real catalog: Kimi-K2-Thinking, GLM-5.3 and DeepSeek-V3.2 are
    llama.cpp-only GGUF giants; DeepSeek-V3.1 keeps both engines."""
    from simulator.model_catalog import load_model_catalog
    from simulator.roofline import engines_for, score_models

    catalog = load_model_catalog(user_dir=tmp_path / "none")
    by = {c.id: c for c in score_models(catalog, vram_per_gpu_gb=96,
                                        host_ram_gb=2048, cache=tmp_path)}
    for mid in ("moonshotai/Kimi-K2-Thinking", "zai-org/GLM-5.3",
                "deepseek-ai/DeepSeek-V3.2"):
        c = by[mid]
        assert not c.fits_gpu and c.fits and c.gguf_engines == ["llamacpp"], mid
        assert engines_for("llamacpp", c.info()) and not engines_for(
            "ktransformers", c.info())
    v31 = by["deepseek-ai/DeepSeek-V3.1"]
    assert v31.gguf_engines == ["ktransformers", "llamacpp"]
    assert engines_for("ktransformers", v31.info())
    mm = by["MiniMaxAI/MiniMax-M2.7"]
    assert mm.fits_gpu and engines_for("llamacpp", mm.info())


def test_gguf_inside_the_cache_is_reached_through_the_cache_mount(tmp_path, monkeypatch):
    """A Hub snapshot's shards are relative symlinks into blobs/; a
    private /gguf bind-mount of the snapshot directory left them
    dangling ('No such file' on the XE7740). Inside the cache the
    launcher references the directory through the cache mount instead."""
    from simulator.config import EngineConfig
    from simulator.engines.llamacpp import LlamaCppEngine
    from simulator.models import container_cache_path

    cache = tmp_path / "cache"
    snap = cache / "hub" / "models--u--M-GGUF" / "snapshots" / "abc" / "UD-Q4_K_XL"
    snap.mkdir(parents=True)
    (snap / "M-UD-Q4_K_XL-00001-of-00002.gguf").write_bytes(b"x")
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    assert container_cache_path(snap) == (
        "/root/.cache/huggingface/hub/models--u--M-GGUF/snapshots/abc/UD-Q4_K_XL")
    assert container_cache_path("/elsewhere/gguf") is None
    cfg = EngineConfig(type="llamacpp", model_id="u/M", port=9100,
                       replica_devices=[[0, 1, 2, 3, 4, 5, 6, 7]],
                       llamacpp_gguf_path=str(snap), max_model_len=4096)
    cmd = LlamaCppEngine(cfg).build_replica_command(0, list(range(8)), "llamacpp-r0-x")
    joined = " ".join(cmd)
    assert ":/gguf:ro" not in joined
    assert "-m /root/.cache/huggingface/hub/models--u--M-GGUF/snapshots/abc/UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf" in joined


def test_output_grammar_parser_is_off_for_benchmarks():
    """DeepSeek-V3.2 loaded and generated, then answered every request
    HTTP 500 because llama-server's template-derived output grammar
    rejected its own reply. Tokens are what a capacity benchmark
    counts, so thoughts stay unparsed and the legacy formatter builds
    the prompt; an operator can turn jinja back on."""
    from simulator.engines.llamacpp import serve_argv
    argv = serve_argv("model.gguf", port=9100, max_model_len=4096, slots=32)
    assert argv[argv.index("--reasoning-format") + 1] == "none"
    assert "--no-jinja" in argv
    assert argv[argv.index("--chat-template") + 1] == "chatml"
    named = serve_argv("model.gguf", port=9100, max_model_len=4096, slots=32,
                       chat_template="deepseek3")
    assert named[named.index("--chat-template") + 1] == "deepseek3"
    on = serve_argv("model.gguf", port=9100, max_model_len=4096, slots=32,
             jinja=True)
    assert "--no-jinja" not in on and "--reasoning-format" in on
