"""SGLang on GPUs: launch argv, and the knobs it spells differently
or cannot express at all."""

from __future__ import annotations

from simulator.config import EngineConfig
from simulator.engines.knobs import canonical, unsupported
from simulator.engines.sglang_cuda import (
    DEFAULT_IMAGE,
    SGLangCudaEngine,
    kv_dtype_for_sglang,
    launch_argv,
)


def _cfg(**kw) -> EngineConfig:
    base = dict(type="sglang_cuda", model_id="org/M", port=9100,
                replica_devices=[[0], [1], [2], [3]], max_model_len=8192)
    base.update(kw)
    return EngineConfig(**base)


def test_replica_argv_carries_the_shape():
    eng = SGLangCudaEngine(_cfg(max_num_seqs=2048,
                                max_num_batched_tokens=8192,
                                gpu_memory_utilization=0.93,
                                vram_per_gpu_gb=95.6, model_weights_gb=42.0))
    cmd = eng.build_replica_command(2, [2], "sglang-r2-x")
    j = " ".join(cmd)
    assert "--gpus device=2" in j
    assert "--ipc=host" in j
    assert "--port 9102" in j                     # port + index
    assert "sglang.launch_server" in j
    assert "--max-running-requests 2048" in j     # NOT --max-num-seqs
    assert "--max-prefill-tokens 8192" in j
    # Translated down: SGLang puts activations on top of this
    # fraction, so passing vLLM's number through is an OOM.
    assert "--mem-fraction-static 0.86" in j
    assert "--context-length 8192" in j
    # Without this there is no /metrics, and the sweep measures from
    # engine counters.
    assert "--enable-metrics" in j
    assert cmd.index(DEFAULT_IMAGE) < cmd.index("python3")


def test_tp_comes_from_the_device_group():
    eng = SGLangCudaEngine(_cfg(replica_devices=[[0, 1], [2, 3]]))
    cmd = eng.build_replica_command(1, [2, 3], "sglang-r1-x")
    assert '"device=2,3"' in cmd
    assert cmd[cmd.index("--tp") + 1] == "2"


def test_kv_dtype_is_translated_not_passed_through():
    """vLLM takes 'fp8' and picks a representation; SGLang wants it
    spelled out. Passing vLLM's spelling straight through is a launch
    failure."""
    # vLLM's bare "fp8" IS e4m3 ("CUDA 11.8+ supports fp8 (=fp8_e4m3)"),
    # so resolving it any other way would run two numeric formats and
    # call the result one comparison.
    assert kv_dtype_for_sglang("fp8") == ("fp8_e4m3", None)
    assert kv_dtype_for_sglang("fp8_e4m3") == ("fp8_e4m3", None)
    assert kv_dtype_for_sglang("auto") == (None, None)
    assert kv_dtype_for_sglang(None) == (None, None)
    argv = launch_argv("m", port=1, tp=1, kv_cache_dtype="fp8")
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8_e4m3"


def test_a_precision_sglang_cannot_express_is_refused():
    """Measuring 'close enough' would answer a different question than
    the one asked — the candidate is unreachable, not slower."""
    value, reason = kv_dtype_for_sglang("nvfp4")
    assert value is None
    assert "nvfp4" in reason
    assert unsupported("sglang_cuda",
                       canonical({"kv_cache_dtype": "nvfp4"})) == reason
    # fp8 is expressible, so nothing is refused.
    assert unsupported("sglang_cuda",
                       canonical({"kv_cache_dtype": "fp8"})) is None
    # ...and vLLM, which does have the path, is unaffected.
    assert unsupported("vllm_cuda_multi",
                       canonical({"kv_cache_dtype": "nvfp4"})) is None


def test_benchmark_refuses_an_inexpressible_shape(monkeypatch, tmp_path):
    from fastapi import HTTPException
    import pytest

    import simulator.arena as arena
    from simulator.service import _build_custom_config

    monkeypatch.setattr(arena, "hardware", lambda: {
        "count": 8, "device_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "vram_per_gpu_gb": 96.0})
    with pytest.raises(HTTPException) as e:
        _build_custom_config({
            "model_id": "org/M", "engine": "sglang_cuda",
            "replicas": 8, "tp": 1, "kv_cache_dtype": "nvfp4",
        }, tmp_path)
    assert e.value.status_code == 422
    assert "nvfp4" in e.value.detail


def test_expert_parallel_needs_more_than_one_gpu():
    assert "--enable-ep-moe" not in launch_argv(
        "m", port=1, tp=1, expert_parallel=True)
    argv = launch_argv("m", port=1, tp=4, expert_parallel=True)
    assert "--enable-ep-moe" in argv
    assert argv[argv.index("--ep-size") + 1] == "4"


def test_sglang_metric_names_match_the_real_collector():
    """Verified against lmsysorg/sglang's observability collector, not
    guessed. The previous mapping looked for num_waiting_reqs (which
    does not exist) and had no token counters at all — so the headline
    sweep, which measures throughput as a delta of
    generation_tokens_total, would have read a flat zero."""
    from simulator.engines.base import Engine

    m = Engine._parse_prometheus("\n".join([
        "sglang:num_running_reqs 128.0",
        "sglang:num_queue_reqs 12.0",
        "sglang:token_usage 0.83",
        "sglang:prompt_tokens_total 123456.0",
        "sglang:generation_tokens_total 987654.0",
        "sglang:cache_hit_rate 0.42",
        "sglang:num_retracted_reqs 3.0",
    ]))
    assert m["num_running"] == 128.0
    assert m["queue_depth"] == 12.0
    assert m["generation_tokens_total"] == 987654.0
    assert m["prompt_tokens_total"] == 123456.0
    assert m["preemptions_total"] == 3.0
    # token_usage is a fraction; capsim's canonical form is a percent.
    assert m["kv_cache_used_pct"] == 83.0
    assert m["prefix_cache_hit_rate"] == 0.42


def test_modelopt_checkpoints_name_their_quantization():
    """NVIDIA's NVFP4 exports carry `quantization_config: null` in
    config.json and declare themselves only in the hf_quant_config
    sidecar, which the ordinary auto-detection path does not read.
    vLLM happens to look there; betting that SGLang does too means the
    weights load as though unquantized."""
    from simulator.engines.sglang_cuda import sglang_quantization

    assert sglang_quantization("nvfp4") == "modelopt_fp4"
    # Formats that DO declare themselves in config.json are left alone
    # — naming them would only add a second way to be wrong.
    assert sglang_quantization("fp8") is None
    assert sglang_quantization("bf16") is None
    assert sglang_quantization(None) is None
    # An explicit config setting always wins.
    assert sglang_quantization("nvfp4", "modelopt_fp8") == "modelopt_fp8"

    argv = launch_argv("m", port=1, tp=1, quantization="modelopt_fp4")
    assert argv[argv.index("--quantization") + 1] == "modelopt_fp4"
    # Absent when there is nothing to say.
    assert "--quantization" not in launch_argv("m", port=1, tp=1)


def test_each_replica_gets_its_own_rendezvous_port():
    """Left to itself SGLang picks a free torch.distributed port at
    random. Eight replicas starting at the same moment on host
    networking race for it: two choose the same number before either
    binds and the loser dies with EADDRINUSE mid-startup, which is
    what happened on this box at port 40593."""
    from simulator.engines.sglang_cuda import NCCL_PORT_STRIDE

    eng = SGLangCudaEngine(_cfg(
        replica_devices=[[i] for i in range(8)]))
    ports = []
    for i in range(8):
        cmd = eng.build_replica_command(i, [i], f"sglang-r{i}-x")
        ports.append(int(cmd[cmd.index("--nccl-port") + 1]))
    assert len(set(ports)) == 8                    # all distinct
    # Spaced, because a replica may open a few consecutive ports.
    assert min(b - a for a, b in zip(ports, ports[1:])) == NCCL_PORT_STRIDE
    # And distinct from the HTTP ports the replicas serve on.
    http = {eng._port(i) for i in range(8)}
    assert not (set(ports) & http)


def test_consecutive_launches_do_not_reuse_the_same_ports():
    """A sweep tears down eight replicas and immediately starts eight
    more. Fixed ports are safe within a launch and collide across
    them, because the previous set's sockets are still closing —
    observed at port 42128, one cell into a roofline."""
    from simulator.engines.sglang_cuda import nccl_port

    a = {nccl_port(i, "launch-aaa") for i in range(8)}
    b = {nccl_port(i, "launch-bbb") for i in range(8)}
    assert len(a) == len(b) == 8
    assert not (a & b), "consecutive launches reused a port"
    # Deterministic for a given launch, so the replicas of one launch
    # agree with each other.
    assert nccl_port(3, "launch-aaa") == nccl_port(3, "launch-aaa")
