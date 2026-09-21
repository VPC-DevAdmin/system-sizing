"""KTransformers v0.7: kt-kernel CPU experts served through SGLang.

The facts these tests pin were read off the v0.7.1 tag of
kvcache-ai/ktransformers (kt-kernel/README.md, the DeepSeek-V3.2 and
Kimi-K2 tutorials, docker/Dockerfile) and the kvcache-ai/sglang fork's
server_args.py -- docs/ktransformers.md has the citations. None of
this has run on the box yet; the tests guard the launch line, the
refusals and the generation choice, not throughput.
"""

from __future__ import annotations

import json

import pytest

from simulator.config import EngineConfig
from simulator.engines import ktransformers as kt
from simulator.engines import ktransformers_v2 as v2
from simulator.engines.ktransformers import KTransformersEngine


def _cfg(**kw) -> EngineConfig:
    base = dict(type="ktransformers", model_id="deepseek-ai/DeepSeek-V3.2",
                port=9100, replica_devices=[[0, 1, 2, 3]],
                max_model_len=32768, gpu_memory_utilization=0.9)
    base.update(kw)
    return EngineConfig(**base)


def _stage(tmp_path, monkeypatch, model_id: str, config: dict,
           safetensors: bool = True):
    """A staged HF snapshot in a private cache; returns its directory."""
    cache = tmp_path / "hf"
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    rev = (cache / "hub" / ("models--" + model_id.replace("/", "--"))
           / "snapshots" / "r1")
    rev.mkdir(parents=True, exist_ok=True)
    (rev / "config.json").write_text(json.dumps(config))
    if safetensors:
        (rev / "model-00001-of-00002.safetensors").write_bytes(b"st")
    return rev


FP8_BLOCK = {"model_type": "deepseek_v3", "torch_dtype": "bfloat16",
             "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                     "weight_block_size": [128, 128]}}
RAWINT4 = {"model_type": "deepseek_v3", "torch_dtype": "bfloat16",
           "quantization_config": {"quant_method": "compressed-tensors",
                                   "config_groups": {"group_0": {"weights": {
                                       "num_bits": 4, "type": "int"}}}}}


# ── The launch line ───────────────────────────────────────────────────

def test_v07_is_sglang_launch_server_with_kt_flags():
    """The v0.3 server module is archived; the current line is the
    kvcache-ai SGLang fork's launch_server plus --kt-* flags, exactly
    as kt-kernel/README.md and the DeepSeek-V3.2 tutorial spell them."""
    argv = v2.launch_argv("/m", port=9100, tp=4, weight_path="/m",
                          method="FP8", cpu_infer=170, threadpool_count=2,
                          gpu_experts=8, deferred_experts=1,
                          gpu_prefill_threshold=2048,
                          dynamic_expert_update=True,
                          context_length=32768, max_running_requests=4,
                          chunked_prefill_size=4096,
                          mem_fraction_static=0.83)
    j = " ".join(argv)
    assert argv[:3] == ["python", "-m", "sglang.launch_server"]
    assert "ktransformers.server.main" not in j
    for flag in ("--model /m", "--kt-weight-path /m", "--kt-method FP8",
                 "--kt-cpuinfer 170", "--kt-threadpool-count 2",
                 "--kt-num-gpu-experts 8",
                 "--kt-max-deferred-experts-per-token 1",
                 "--kt-gpu-prefill-token-threshold 2048",
                 "--kt-enable-dynamic-expert-update",
                 "--tensor-parallel-size 4", "--context-length 32768",
                 "--max-running-requests 4", "--chunked-prefill-size 4096",
                 "--max-prefill-tokens 4096", "--mem-fraction-static 0.83",
                 "--attention-backend flashinfer",
                 "--disable-shared-experts-fusion", "--enable-mixed-chunk",
                 "--trust-remote-code", "--enable-metrics"):
        assert flag in j, flag
    # kebab-case throughout, unlike the v0.3 server.
    assert "--model_path" not in j and "--cpu_infer" not in j


def test_gpu_experts_are_always_passed():
    """The fork only WARNS when --kt-num-gpu-experts is absent and then
    has no expert mask; zero is a launch, silence is not."""
    argv = v2.launch_argv("/m", port=1, tp=1, weight_path="/m", method="BF16")
    assert argv[argv.index("--kt-num-gpu-experts") + 1] == "0"
    assert "--kt-max-deferred-experts-per-token" not in argv
    assert "--kt-gpu-prefill-token-threshold" not in argv


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="kt-method"):
        v2.launch_argv("/m", port=1, tp=1, weight_path="/m", method="INT9")


# ── Generation choice ─────────────────────────────────────────────────

def test_generation_resolves_from_setting_then_image_then_staging(
        tmp_path, monkeypatch):
    """Explicit setting wins; then the image tag (ISA-suffixed v0.3.x
    tags are the archived line, anything else the SGLang line); then
    what is staged: safetensors or AMX weights -> v0.7, a lone GGUF
    -> v0.3, nothing -> v0.7 (whose refusal names what to stage)."""
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    assert kt.resolve_generation(_cfg(ktransformers_generation="v0.3")) == "v0.3"
    assert kt.resolve_generation(_cfg(ktransformers_generation="V0.7")) == "v0.7"
    with pytest.raises(ValueError, match="ktransformers_generation"):
        kt.resolve_generation(_cfg(ktransformers_generation="v9"))

    assert kt.generation_from_image("approachingai/ktransformers:v0.3.2-AVX512") == "v0.3"
    assert kt.generation_from_image("approachingai/ktransformers:v0.2.4post1-AVX2") == "v0.3"
    assert kt.generation_from_image("approachingai/ktransformers:DSV4-specific") == "v0.7"
    assert kt.generation_from_image("approachingai/ktransformers:v0.5.3") == "v0.7"
    assert kt.generation_from_image(None) is None
    assert kt.resolve_generation(_cfg(ktransformers_image=kt.DEFAULT_IMAGE,
                                      ktransformers_amx_weight_path="/x")) == "v0.3"

    # Staging decides when nothing is said.
    cache = tmp_path / "hf"
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    assert kt.resolve_generation(_cfg(ktransformers_gguf_path=str(gguf))) == "v0.3"
    assert kt.resolve_generation(_cfg()) == "v0.7"
    assert kt.resolve_generation(_cfg(ktransformers_amx_weight_path="/amx",
                                      ktransformers_gguf_path=str(gguf))) == "v0.7"
    assert kt.resolve_generation(_cfg(ktransformers_kt_method="FP8",
                                      ktransformers_gguf_path=str(gguf))) == "v0.7"
    _stage(tmp_path, monkeypatch, "deepseek-ai/DeepSeek-V3.2", FP8_BLOCK)
    assert kt.resolve_generation(_cfg(ktransformers_gguf_path=str(gguf))) == "v0.7"
    # Config-only staging (what Prepare does for kt_only entries) is
    # not a checkpoint the v0.7 line can read.
    _stage(tmp_path, monkeypatch, "org/ConfigOnly", FP8_BLOCK, safetensors=False)
    assert kt.resolve_generation(_cfg(model_id="org/ConfigOnly",
                                      ktransformers_gguf_path=str(gguf))) == "v0.3"


def test_v03_launch_is_unchanged_behind_the_switch(tmp_path):
    """The archived line keeps its entrypoint override, its module and
    its GGUF mount when selected."""
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    eng = KTransformersEngine(_cfg(ktransformers_generation="v0.3",
                                   ktransformers_gguf_path=str(gguf)))
    cmd = eng.build_replica_command(0, [0], "kt-r0-x")
    assert cmd[cmd.index("--entrypoint") + 1] == kt.PYTHON
    assert cmd[cmd.index("-m") + 1] == "ktransformers.server.main"
    assert kt.DEFAULT_IMAGE in cmd
    assert eng._ready_url(9100).endswith("/v1/models")


# ── The v0.7 docker line ──────────────────────────────────────────────

def test_v07_replica_command_for_a_native_fp8_checkpoint(tmp_path, monkeypatch):
    """A staged block-FP8 DeepSeek needs no conversion: --model and
    --kt-weight-path are the same snapshot directory, seen through the
    HF cache mount; the method is read off quantization_config; the
    image is the SM120 build with the CUDA-arch environment its own
    entrypoint would have set; readiness is SGLang's /health."""
    rev = _stage(tmp_path, monkeypatch, "deepseek-ai/DeepSeek-V3.2", FP8_BLOCK)
    monkeypatch.setattr(kt, "physical_cores", lambda: 172)
    monkeypatch.setattr(v2, "numa_node_count", lambda: 2)
    eng = KTransformersEngine(_cfg(max_num_seqs=4, max_num_batched_tokens=4096))
    assert eng.generation == "v0.7"
    cmd = eng.build_replica_command(0, [0, 1, 2, 3], "ktransformers-r0-x")
    j = " ".join(cmd)

    assert v2.DEFAULT_IMAGE in cmd
    assert "--entrypoint" not in cmd            # the image execs a CMD
    assert "--ipc=host" in cmd and "SYS_NICE" in cmd
    assert "-e TORCH_CUDA_ARCH_LIST=12.0+PTX" in j
    assert "-e FLASHINFER_CUDA_ARCH_LIST=12.0a" in j
    in_container = "/root/.cache/huggingface" + str(rev)[len(str(tmp_path / "hf")):]
    assert cmd[cmd.index("--model") + 1] == in_container
    assert cmd[cmd.index("--kt-weight-path") + 1] == in_container
    assert cmd[cmd.index("--kt-method") + 1] == "FP8"
    assert cmd[cmd.index("--kt-cpuinfer") + 1] == "170"
    assert cmd[cmd.index("--kt-threadpool-count") + 1] == "2"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert cmd[cmd.index("--max-running-requests") + 1] == "4"
    assert cmd[cmd.index("--chunked-prefill-size") + 1] == "4096"
    assert "--enable-metrics" in cmd
    assert eng._ready_url(9100) == "http://127.0.0.1:9100/health"
    # The vLLM-style memory share is translated the way sglang_cuda's
    # is, not passed through.
    assert float(cmd[cmd.index("--mem-fraction-static") + 1]) < 0.9


def test_v07_knobs_reach_the_flags(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, "moonshotai/Kimi-K2-Thinking", RAWINT4)
    monkeypatch.setattr(kt, "physical_cores", lambda: None)
    monkeypatch.setattr(v2, "numa_node_count", lambda: None)
    eng = KTransformersEngine(_cfg(
        model_id="moonshotai/Kimi-K2-Thinking",
        ktransformers_cpu_threads=96, ktransformers_threadpool_count=4,
        ktransformers_gpu_experts=30, ktransformers_deferred_experts=1,
        ktransformers_gpu_prefill_threshold=400,
        ktransformers_dynamic_expert_update=True,
        ktransformers_cuda_arch="9.0",
        ktransformers_extra_flags=["--tool-call-parser", "kimi_k2"]))
    cmd = eng.build_replica_command(0, [0, 1], "kt-r0-x")
    j = " ".join(cmd)
    assert "--kt-method RAWINT4" in j
    assert "--kt-cpuinfer 96 --kt-threadpool-count 4" in j
    assert "--kt-num-gpu-experts 30" in j
    assert "--kt-max-deferred-experts-per-token 1" in j
    assert "--kt-gpu-prefill-token-threshold 400" in j
    assert "--kt-enable-dynamic-expert-update" in j
    assert "TORCH_CUDA_ARCH_LIST=9.0+PTX" in j and "FLASHINFER_CUDA_ARCH_LIST=9.0a" in j
    assert j.endswith("--tool-call-parser kimi_k2")

    # Off Linux the thread flags are omitted rather than guessed.
    eng = KTransformersEngine(_cfg(model_id="moonshotai/Kimi-K2-Thinking"))
    cmd = eng.build_replica_command(0, [0], "kt-r0-x")
    assert "--kt-cpuinfer" not in cmd and "--kt-threadpool-count" not in cmd


def test_v07_amx_method_mounts_the_converted_weights(tmp_path, monkeypatch):
    """AMXINT8/AMXINT4 read convert_cpu_weights.py output from a
    separate directory; the checkpoint still supplies the GPU side."""
    _stage(tmp_path, monkeypatch, "deepseek-ai/DeepSeek-V3.2", FP8_BLOCK)
    amx = tmp_path / "dsv32-int8"
    amx.mkdir()
    eng = KTransformersEngine(_cfg(ktransformers_kt_method="amxint8",
                                   ktransformers_amx_weight_path=str(amx)))
    cmd = eng.build_replica_command(0, [0], "kt-r0-x")
    assert f"{amx}:/kt-weights:ro" in cmd
    assert cmd[cmd.index("--kt-weight-path") + 1] == "/kt-weights"
    assert cmd[cmd.index("--kt-method") + 1] == "AMXINT8"
    assert cmd[cmd.index("--model") + 1].startswith("/root/.cache/huggingface/")


def test_v07_llamafile_method_mounts_the_gguf(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, "deepseek-ai/DeepSeek-V3.2", FP8_BLOCK)
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    eng = KTransformersEngine(_cfg(ktransformers_kt_method="LLAMAFILE",
                                   ktransformers_gguf_path=str(gguf)))
    cmd = eng.build_replica_command(0, [0], "kt-r0-x")
    assert f"{gguf}:/gguf:ro" in cmd
    assert cmd[cmd.index("--kt-weight-path") + 1] == "/gguf"
    assert "sglang.launch_server" in cmd


# ── Refusals ──────────────────────────────────────────────────────────

def test_v07_refuses_in_milliseconds_with_the_fix_named(tmp_path, monkeypatch):
    """Every way the weights can be absent gets its own sentence; none
    of them reaches docker."""
    cache = tmp_path / "hf"
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))

    # Nothing staged at all.
    with pytest.raises(RuntimeError, match="nothing is staged"):
        KTransformersEngine(_cfg()).build_replica_command(0, [0], "kt-r0-x")
    # Config-only staging: the v0.3 line's convention, useless here.
    _stage(tmp_path, monkeypatch, "org/ConfigOnly", FP8_BLOCK, safetensors=False)
    with pytest.raises(RuntimeError, match="holds no safetensors"):
        KTransformersEngine(_cfg(model_id="org/ConfigOnly",
                                 ktransformers_generation="v0.7")
                            ).build_replica_command(0, [0], "kt-r0-x")
    # A format kt-kernel has no native path for (ModelOpt NVFP4).
    _stage(tmp_path, monkeypatch, "nvidia/Kimi-K2-Thinking-NVFP4",
           {"model_type": "deepseek_v3",
            "quantization_config": {"quant_method": "modelopt"}})
    with pytest.raises(RuntimeError, match="no native CPU-expert path"):
        KTransformersEngine(_cfg(model_id="nvidia/Kimi-K2-Thinking-NVFP4")
                            ).build_replica_command(0, [0], "kt-r0-x")
    # AMX methods without the converted directory, or with a missing one.
    _stage(tmp_path, monkeypatch, "deepseek-ai/DeepSeek-V3.2", FP8_BLOCK)
    with pytest.raises(RuntimeError, match="convert_cpu_weights.py"):
        KTransformersEngine(_cfg(ktransformers_kt_method="AMXINT4")
                            ).build_replica_command(0, [0], "kt-r0-x")
    with pytest.raises(RuntimeError, match="does not exist"):
        KTransformersEngine(_cfg(ktransformers_kt_method="AMXINT4",
                                 ktransformers_amx_weight_path=str(tmp_path / "no"))
                            ).build_replica_command(0, [0], "kt-r0-x")
    # LLAMAFILE without a GGUF.
    with pytest.raises(RuntimeError, match="LLAMAFILE"):
        KTransformersEngine(_cfg(ktransformers_kt_method="LLAMAFILE")
                            ).build_replica_command(0, [0], "kt-r0-x")
    # A method name the fork does not know.
    with pytest.raises(ValueError, match="ktransformers_kt_method"):
        KTransformersEngine(_cfg(ktransformers_kt_method="int8")
                            ).build_replica_command(0, [0], "kt-r0-x")


# ── Format detection and host facts ───────────────────────────────────

def test_kt_method_is_read_off_the_checkpoint():
    """The formats the v0.7 line serves natively, as their configs
    declare them; anything else is None so the caller refuses."""
    assert v2.kt_method_for(FP8_BLOCK) == "FP8"
    assert v2.kt_method_for({"quantization_config": {"quant_method": "fp8"}}) == "FP8_PERCHANNEL"
    assert v2.kt_method_for(RAWINT4) == "RAWINT4"
    assert v2.kt_method_for({"quantization_config": {"quant_method": "mxfp4"}}) == "MXFP4"
    assert v2.kt_method_for({"torch_dtype": "bfloat16"}) == "BF16"
    assert v2.kt_method_for({"dtype": "bfloat16", "quantization_config": None}) == "BF16"
    assert v2.kt_method_for({"torch_dtype": "float16"}) is None
    assert v2.kt_method_for({"quantization_config": {"quant_method": "modelopt"}}) is None
    assert v2.kt_method_for({"quantization_config": {"quant_method": "awq"}}) is None
    assert v2.kt_method_for(None) is None


def test_numa_and_cuda_arch_helpers(monkeypatch):
    assert v2.parse_numa_nodes(["node0", "node1", "has_cpu", "online", "node10"]) == 3
    assert v2.parse_numa_nodes(["online"]) == 0
    monkeypatch.setattr(v2.sys, "platform", "darwin")
    assert v2.numa_node_count() is None
    assert v2.cuda_arch_env("12.0") == {"TORCH_CUDA_ARCH_LIST": "12.0+PTX",
                                        "FLASHINFER_CUDA_ARCH_LIST": "12.0a"}
    assert v2.cuda_arch_env("8.9") == {"TORCH_CUDA_ARCH_LIST": "8.9",
                                       "FLASHINFER_CUDA_ARCH_LIST": "8.9"}
    assert v2.cuda_arch_env(None) == {}


def test_container_path_follows_the_mount(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    snap = tmp_path / "hf" / "hub" / "models--o--m" / "snapshots" / "r"
    assert v2.container_path_for(snap, None) == \
        "/root/.cache/huggingface/hub/models--o--m/snapshots/r"
    assert v2.container_path_for(snap, {str(tmp_path / "hf"): "/hf"}) == \
        "/hf/hub/models--o--m/snapshots/r"
    assert v2.container_path_for(tmp_path / "elsewhere", {}) == str(tmp_path / "elsewhere")


def test_v07_metrics_use_sglangs_names():
    """The fork is SGLang; its /metrics carries sglang: counters and
    the parser sglang_cuda relies on applies unchanged."""
    eng = KTransformersEngine(_cfg())
    m = eng.parse_metrics(
        "sglang:prompt_tokens_total 120\n"
        "sglang:generation_tokens_total 340\n"
        "sglang:num_running_reqs 3\n"
        "sglang:num_queue_reqs 1\n"
        "sglang:token_usage 0.42\n")
    assert m["prompt_tokens_total"] == 120
    assert m["generation_tokens_total"] == 340
    assert m["num_running"] == 3 and m["queue_depth"] == 1
    assert m["kv_cache_used_pct"] == pytest.approx(42.0)


def test_engine_card_names_both_generations():
    from simulator.engine_notes import ENGINE_NOTES, levers_for
    note = ENGINE_NOTES["ktransformers"]
    assert "v0.7" in note and "v0.3" in note and "SM120" in note
    keys = {lv.key for lv in levers_for("ktransformers")}
    assert {"ktransformers_generation", "ktransformers_kt_method",
            "ktransformers_gpu_experts"} <= keys
