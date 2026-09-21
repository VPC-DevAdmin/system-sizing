"""KTransformers: a heterogeneous engine with its own launch rules."""

from __future__ import annotations

import tempfile

import pytest

from simulator.config import EngineConfig
from simulator.engines.knobs import canonical, unsupported
from simulator.engines.ktransformers import (
    DEFAULT_IMAGE,
    PYTHON,
    KTransformersEngine,
    serve_argv,
)

# A staged GGUF directory: the server loads weights from GGUF only, and
# the launcher refuses to start without one (see the test below).
GGUF_DIR = tempfile.mkdtemp(prefix="kt-gguf-")


def test_gguf_path_resolves_from_a_staged_catalog_companion(tmp_path, monkeypatch):
    """An operator stages the catalog's GGUF companion in Prepare and
    never types a path: the launcher resolves --gguf_path to the
    snapshot directory holding it. Unstaged, the refusal names the
    Prepare button rather than a config key."""
    from simulator.engines.custom import ShapeError, custom_engine

    cache = tmp_path / "hf"
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    spec = {"repo": "org/M-GGUF", "file": "M-Q4_K_M.gguf", "size_gb": 1.0}
    catalog = [{"id": "org/M", "family": "m", "quant": "bf16", "gguf": spec},
               {"id": "org/Plain", "family": "p", "quant": "bf16", "gguf": None}]
    hw = {"count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
          "vram_per_gpu_gb": 96.0}
    base = {"model_id": "org/M", "device": "gpu", "engine": "ktransformers",
            "replicas": 1, "tp": 1, "kv_cache_dtype": "auto"}

    with pytest.raises(ShapeError, match="stage the GGUF companion in Prepare"):
        custom_engine(base, hw=hw, catalog=catalog)
    # No companion at all: today's "configure ktransformers_gguf_path".
    with pytest.raises(ShapeError, match="ktransformers_gguf_path is configured"):
        custom_engine({**base, "model_id": "org/Plain"}, hw=hw, catalog=catalog)

    rev = cache / "hub" / "models--org--M-GGUF" / "snapshots" / "deadbeef"
    rev.mkdir(parents=True)
    (rev / "M-Q4_K_M.gguf").write_bytes(b"gguf")
    eng = custom_engine(base, hw=hw, catalog=catalog)
    assert eng["type"] == "ktransformers"
    assert eng["ktransformers_gguf_path"] == str(rev)
    cmd = KTransformersEngine(_cfg(ktransformers_gguf_path=eng["ktransformers_gguf_path"])
                              ).build_replica_command(0, [0], "ktransformers-r0-x")
    assert ":/gguf:ro" not in " ".join(cmd)  # reached through the cache mount

    # An explicit path still wins over the companion.
    explicit = tmp_path / "mine"
    explicit.mkdir()
    eng = custom_engine({**base, "ktransformers_gguf_path": str(explicit)},
                        hw=hw, catalog=catalog)
    assert eng["ktransformers_gguf_path"] == str(explicit)


def _cfg(**kw) -> EngineConfig:
    base = dict(type="ktransformers", model_id="deepseek-ai/DeepSeek-V3",
                port=9100, replica_devices=[[0, 1, 2, 3]],
                max_model_len=32768, ktransformers_gguf_path=GGUF_DIR)
    base.update(kw)
    return EngineConfig(**base)


def test_the_entrypoint_must_be_overridden():
    """This image's ENTRYPOINT is `tail -f /dev/null` — it is built to
    be run detached and exec'd into, so a CMD becomes an ARGUMENT TO
    TAIL. That is the exact opposite of the TensorRT-LLM and SGLang
    images, where overriding the entrypoint breaks the launch."""
    eng = KTransformersEngine(_cfg())
    cmd = eng.build_replica_command(0, [0, 1, 2, 3], "ktransformers-r0-x")
    assert cmd[cmd.index("--entrypoint") + 1] == PYTHON
    # The server module is the CMD, after the image.
    assert cmd.index(DEFAULT_IMAGE) < cmd.index("-m")
    assert cmd[cmd.index("-m") + 1] == "ktransformers.server.main"
    # Its imports resolve only from the source tree.
    assert cmd[cmd.index("-w") + 1] == "/workspace/ktransformers"


def test_default_image_is_a_serving_build():
    """:latest reorganised into kernels + fine-tuning and ARCHIVED the
    inference server — no torch in its default interpreter and no
    compiled ops, so it cannot serve at all."""
    assert DEFAULT_IMAGE != "approachingai/ktransformers:latest"
    assert "v0.3" in DEFAULT_IMAGE


def test_batching_backend_is_explicit():
    """The default backend answers one request at a time; without
    balance_serve the concurrency sweep measures a queue, not an
    engine."""
    argv = serve_argv("m", port=1)
    assert argv[argv.index("--backend_type") + 1] == "balance_serve"


def test_snake_case_flags_match_the_servers_own_help():
    argv = serve_argv("org/M", port=9100, max_batch_size=4,
                      chunk_size=256, cache_lens=32768, cpu_threads=64,
                      gguf_path="/gguf",
                      optimize_config_path="/rules.yaml")
    j = " ".join(argv)
    assert "--model_path org/M" in j            # not --model-path
    assert "--max_batch_size 4" in j
    assert "--chunk_size 256" in j
    assert "--cache_lens 32768" in j
    assert "--cpu_infer 64" in j
    assert "--gguf_path /gguf" in j
    assert "--optimize_config_path /rules.yaml" in j


def test_gpu_only_knobs_are_refused_not_ignored():
    """A silently dropped setting is how a search concludes the wrong
    thing."""
    assert unsupported("ktransformers",
                       canonical({"kv_cache_dtype": "fp8"}))
    assert unsupported("ktransformers",
                       canonical({"expert_parallel": True}))
    # Batched tokens DOES map here — to the prefill chunk size.
    assert unsupported("ktransformers",
                       canonical({"max_num_batched_tokens": 2048})) is None


def test_benchmark_refuses_a_gpu_only_knob(monkeypatch, tmp_path):
    from fastapi import HTTPException

    import simulator.arena as arena
    from simulator.service import _build_custom_config

    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 96.0})
    with pytest.raises(HTTPException) as e:
        _build_custom_config({
            "model_id": "org/M", "engine": "ktransformers",
            "replicas": 1, "tp": 4, "kv_cache_dtype": "fp8",
        }, tmp_path)
    assert e.value.status_code == 422
    assert "KV cache precision" in e.value.detail


def test_every_engine_prefix_is_swept_before_launch():
    """A leftover container of ANY engine holds port 9100 and answers
    health checks for the wrong server."""
    from simulator.engines.docker_replica import CAPSIM_CONTAINER_PREFIXES

    for engine in ("vllm-", "trtllm-", "sglang-", "ktransformers-"):
        assert engine in CAPSIM_CONTAINER_PREFIXES


def test_shm_size_is_not_passed_beside_ipc_host():
    """--ipc=host shares the host's /dev/shm; --shm-size sizes the
    private one it replaces, so the flag was a no-op that read as a
    deliberate setting."""
    eng = KTransformersEngine(_cfg(docker_volumes={}))
    cmd = eng.build_replica_command(0, [0], "ktransformers-r0-x")
    assert "--ipc=host" in cmd
    assert "--shm-size" not in cmd


def test_launch_without_staged_gguf_is_refused_up_front(tmp_path):
    """Observed on the XE7740: with no GGUF configured the server fell
    back to './DeepSeek-V2-Lite-Chat-GGUF', raised FileNotFoundError,
    and its scheduler lingered -- every roofline cell burned the full
    30-minute health timeout. The launcher and the shape gate now
    refuse in milliseconds and say what to stage."""
    from simulator.engines.custom import ShapeError, custom_engine

    eng = KTransformersEngine(_cfg(ktransformers_gguf_path=None))
    with pytest.raises(RuntimeError, match="GGUF"):
        eng.build_replica_command(0, [0], "ktransformers-r0-x")
    eng = KTransformersEngine(_cfg(ktransformers_gguf_path=str(tmp_path / "nope")))
    with pytest.raises(RuntimeError, match="does not exist"):
        eng.build_replica_command(0, [0], "ktransformers-r0-x")

    hw = {"count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
          "vram_per_gpu_gb": 96.0}
    base = {"model_id": "org/M", "device": "gpu", "engine": "ktransformers",
            "replicas": 1, "tp": 1, "kv_cache_dtype": "auto"}
    with pytest.raises(ShapeError, match="ktransformers_gguf_path"):
        custom_engine(base, hw=hw)
    staged = tmp_path / "gguf"
    staged.mkdir()
    eng_doc = custom_engine({**base, "ktransformers_gguf_path": str(staged)}, hw=hw)
    assert eng_doc["ktransformers_gguf_path"] == str(staged)
    cmd = KTransformersEngine(_cfg(ktransformers_gguf_path=str(staged))
                              ).build_replica_command(0, [0], "ktransformers-r0-x")
    assert f"{staged}:/gguf:ro" in cmd and "--gguf_path" in cmd


def test_optimize_rule_is_picked_from_the_staged_config(tmp_path, monkeypatch):
    """KTransformers cannot infer its injection rule from the model
    id, but capsim can read model_type from the staged config.json:
    deepseek_v3 and qwen3_moe map to the serve rules the v0.3.2 image
    ships; an explicit ktransformers_optimize_config wins; an
    architecture without a rule passes nothing."""
    from simulator.engines.ktransformers import OPTIMIZE_RULES_DIR, optimize_config_for

    assert optimize_config_for("deepseek_v3") == \
        f"{OPTIMIZE_RULES_DIR}/DeepSeek-V3-Chat-serve.yaml"
    assert optimize_config_for("qwen3_moe") == \
        f"{OPTIMIZE_RULES_DIR}/Qwen3Moe-serve.yaml"
    assert OPTIMIZE_RULES_DIR.startswith("/workspace/ktransformers/")
    assert optimize_config_for("deepseek_v32") is None
    assert optimize_config_for(None) is None

    def argv_of(cfg):
        cmd = KTransformersEngine(cfg).build_replica_command(0, [0], "kt-r0-x")
        return cmd[cmd.index(DEFAULT_IMAGE):]

    # From a local model directory.
    local = tmp_path / "local"
    local.mkdir()
    (local / "config.json").write_text('{"model_type": "deepseek_v3"}')
    argv = argv_of(_cfg(model_local_path=str(local)))
    assert argv[argv.index("--optimize_config_path") + 1] == \
        f"{OPTIMIZE_RULES_DIR}/DeepSeek-V3-Chat-serve.yaml"
    # From the HF cache snapshot of a hub id.
    cache = tmp_path / "hf"
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    rev = cache / "hub" / "models--Qwen--Big" / "snapshots" / "r"
    rev.mkdir(parents=True)
    (rev / "config.json").write_text('{"model_type": "qwen3_moe"}')
    argv = argv_of(_cfg(model_id="Qwen/Big"))
    assert argv[argv.index("--optimize_config_path") + 1].endswith("Qwen3Moe-serve.yaml")
    # Explicit wins; unknown architecture -> no flag at all.
    argv = argv_of(_cfg(model_id="Qwen/Big", ktransformers_optimize_config="/mine.yaml"))
    assert argv[argv.index("--optimize_config_path") + 1] == "/mine.yaml"
    (rev / "config.json").write_text('{"model_type": "deepseek_v32"}')
    assert "--optimize_config_path" not in argv_of(_cfg(model_id="Qwen/Big"))
    assert "--optimize_config_path" not in argv_of(_cfg(model_id="Nobody/Staged"))


def test_cpu_infer_defaults_to_physical_cores_minus_two(monkeypatch):
    """The expert path wants every physical core except the two the
    driver and scheduler need; hyperthreads add nothing on a
    bandwidth-bound kernel. Explicit ktransformers_cpu_threads wins;
    off Linux the host's cores mean nothing and the flag is omitted."""
    from simulator.engines import ktransformers as kt

    assert kt.default_cpu_infer(64) == 62
    assert kt.default_cpu_infer(2) == 1
    assert kt.default_cpu_infer(0) is None

    cpuinfo = "".join(
        f"processor\t: {i}\nphysical id\t: {i // 8}\ncore id\t: {(i % 8) % 4}\n\n"
        for i in range(16))          # 2 sockets x 4 cores x 2 threads
    assert kt.parse_cpuinfo_cores(cpuinfo) == 8
    assert kt.parse_cpuinfo_cores("processor\t: 0\nflags\t: neon\n") == 0

    monkeypatch.setattr(kt, "physical_cores", lambda: 172)
    cmd = KTransformersEngine(_cfg()).build_replica_command(0, [0], "kt-r0-x")
    assert cmd[cmd.index("--cpu_infer") + 1] == "170"
    cmd = KTransformersEngine(_cfg(ktransformers_cpu_threads=32)
                              ).build_replica_command(0, [0], "kt-r0-x")
    assert cmd[cmd.index("--cpu_infer") + 1] == "32"
    monkeypatch.setattr(kt, "physical_cores", lambda: None)
    assert "--cpu_infer" not in KTransformersEngine(_cfg()).build_replica_command(
        0, [0], "kt-r0-x")
    monkeypatch.undo()
    monkeypatch.setattr(kt.sys, "platform", "darwin")
    assert kt.physical_cores() is None


def test_kt_only_models_are_refused_by_gpu_engines(tmp_path):
    """The catalog says the weights exceed the GPUs; a vLLM launch
    would spend the health timeout discovering that."""
    from simulator.engines.custom import ShapeError, custom_engine

    catalog = [{"id": "org/Huge", "family": "h", "quant": "fp8", "kt_only": True,
                "gguf": {"repo": "org/Huge-GGUF", "file": "Q4", "size_gb": 1.0}}]
    hw = {"count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
          "vram_per_gpu_gb": 96.0}
    base = {"model_id": "org/Huge", "device": "gpu", "replicas": 1, "tp": 4}
    for engine in ("vllm_cuda_multi", "sglang_cuda", "trtllm"):
        with pytest.raises(ShapeError, match="kt_only"):
            custom_engine({**base, "engine": engine}, hw=hw, catalog=catalog)
    staged = tmp_path / "gguf"
    staged.mkdir()
    eng = custom_engine({**base, "engine": "ktransformers", "tp": 1,
                         "kv_cache_dtype": "auto",
                         "ktransformers_gguf_path": str(staged)},
                        hw=hw, catalog=catalog)
    assert eng["type"] == "ktransformers"


def test_gguf_inside_the_cache_is_reached_through_the_cache_mount(tmp_path, monkeypatch):
    """Same fault as llama.cpp: the Qwen3-235B and DeepSeek cells on the
    XE7740 died with 'No such file' because the snapshot's shard
    symlinks pointed outside a private /gguf mount."""
    from simulator.models import container_cache_path

    cache = tmp_path / "cache"
    snap = cache / "hub" / "models--u--M-GGUF" / "snapshots" / "abc" / "Q4_K_M"
    snap.mkdir(parents=True)
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    eng = KTransformersEngine(_cfg(ktransformers_gguf_path=str(snap)))
    cmd = eng.build_replica_command(0, [0], "ktransformers-r0-x")
    joined = " ".join(cmd)
    assert ":/gguf:ro" not in joined
    assert "--gguf_path " + container_cache_path(snap) in joined
    # Outside the cache a private mount is still the only way.
    ext = tmp_path / "ext"
    ext.mkdir()
    cmd = KTransformersEngine(_cfg(ktransformers_gguf_path=str(ext))
                              ).build_replica_command(0, [0], "ktransformers-r0-x")
    assert f"{ext}:/gguf:ro" in " ".join(cmd) and "--gguf_path /gguf" in " ".join(cmd)
