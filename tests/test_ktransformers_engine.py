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
