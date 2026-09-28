import json
from dataclasses import asdict

import pytest

from simulator.config import Config
from simulator.infrastructure import attention
from simulator.infrastructure.spec import ExperimentSpec


def test_profiler_only_diagnostic_and_explicit_backend(tmp_path):
    spec = ExperimentSpec(project_id="P10", paired_blocks=1)
    source = asdict(Config())
    baseline = attention.config(source, spec, "baseline")
    treatment = attention.config(source, spec, "treatment")
    diagnostic = attention.config(source, spec, "baseline", tmp_path / "traces")
    assert baseline["engine"]["attention_backend"] == "FLASHINFER"
    assert treatment["engine"]["attention_backend"] == "TRITON_ATTN"
    for cfg in (baseline, treatment):
        assert "--profiler-config" not in cfg["engine"]["vllm_extra_flags"]
        assert "--speculative-config" not in cfg["engine"]["vllm_extra_flags"]
    flags = diagnostic["engine"]["vllm_extra_flags"]
    profiler = json.loads(flags[flags.index("--profiler-config") + 1])
    assert profiler["max_iterations"] == 4 and profiler["torch_profiler_with_memory"] is False
    assert "torch_profiler_profile_memory" not in profiler


def test_runtime_backend_must_match_all_replicas(tmp_path):
    for i in range(2):
        d = tmp_path / f"replica-{i}"
        d.mkdir()
        (d / "engine.log").write_text("Using AttentionBackendEnum.FLASHINFER backend\n")
    attention.verify_backend(tmp_path, "FLASHINFER")
    (tmp_path / "replica-1/engine.log").write_text("Using AttentionBackendEnum.TRITON_ATTN backend")
    with pytest.raises(RuntimeError, match="silently changed"):
        attention.verify_backend(tmp_path, "FLASHINFER")


def test_numerical_gate_preserves_differences(tmp_path):
    spec = ExperimentSpec(project_id="P10", paired_blocks=1)
    rows = [
        {"task_id": str(i), "token": "a", "top_logprobs": {str(k): -k * 0.1 for k in range(20)}}
        for i in range(4)
    ]
    for arm in ("baseline", "treatment"):
        p = tmp_path / f"block-00-{arm}"
        p.mkdir()
        (p / "next-token-logprobs.json").write_text(json.dumps(rows))
    assert attention.parity(tmp_path, spec)["passed"]
    rows[2]["top_logprobs"]["5"] -= 0.2
    (tmp_path / "block-00-treatment/next-token-logprobs.json").write_text(json.dumps(rows))
    result = attention.parity(tmp_path, spec)
    assert not result["passed"] and not result["probes"][2]["passed"]


def test_profile_requires_cuda_and_does_not_add_cpu_duration(tmp_path):
    import gzip

    traces = tmp_path / "traces"
    traces.mkdir()
    trace = {"traceEvents": [{"cat": "cpu_op", "ph": "X", "name": "cpu", "dur": 1000}]}
    file = traces / "trace.json.gz"
    with gzip.open(file, "wt") as out:
        json.dump(trace, out)
    with pytest.raises(RuntimeError, match="No CUDA kernel"):
        attention.summarize_profile(tmp_path)
    trace["traceEvents"].extend(
        [
            {"cat": "kernel", "ph": "X", "name": "attention", "dur": 20},
            {"cat": "kernel", "ph": "X", "name": "attention", "dur": 30},
        ]
    )
    with gzip.open(file, "wt") as out:
        json.dump(trace, out)
    result = attention.summarize_profile(tmp_path)
    assert result["kernel_events"] == 2 and result["top_kernels"] == [("attention", 50)]


def test_diagnostic_report_exposes_kernels_without_claiming_speedup(tmp_path, monkeypatch):
    from simulator.infrastructure import campaign
    from simulator.infrastructure.comparison import compare
    from simulator.infrastructure.report import export

    monkeypatch.setattr(campaign, "recorded_config", lambda *a: asdict(Config()))
    spec = ExperimentSpec(project_id="P10", paired_blocks=1)
    source = asdict(Config())
    source["engine"]["type"] = "vllm_cuda"
    monkeypatch.setattr(campaign, "recorded_config", lambda *a: source)
    path = campaign.create(tmp_path, {"model": "model"}, spec, "d" * 32)
    d = path / "block-00-profile"
    d.mkdir()
    (d / "profile-summary.json").write_text(
        json.dumps(
            {
                "kernel_events": 2,
                "top_kernels": [["attention_kernel", 50]],
                "trace_files": ["trace.json"],
            }
        )
    )
    state = json.loads((path / "state.json").read_text())
    state["results"] = [
        dict(
            block=0,
            arm=a,
            metrics=dict(
                attempted=32,
                quality_fraction=1,
                gates_passed=True,
                qualified_per_hour=100,
                latency_p95_s=2,
            ),
        )
        for a in ("baseline", "treatment")
    ]
    (path / "state.json").write_text(json.dumps(state))
    export(path, compare(state["results"], spec))
    claim = json.loads((path / "outputs/P10-O3/claim.json").read_text())
    assert not claim["positive_claim_supported"] and claim["measurement"] is None
    text = (path / "outputs/P10-O3/claim-brief.md").read_text()
    assert (
        "attention_kernel: 50.000 microseconds" in text and "Profiled timings are excluded" in text
    )


@pytest.mark.parametrize("dtype", ["fp8", "bfloat16"])
def test_attention_precision_is_matched_and_verified(tmp_path, dtype):
    spec = ExperimentSpec(project_id="P10", attention_kv_dtype=dtype)
    for arm in ("baseline", "treatment"):
        cfg = attention.config(asdict(Config()), spec, arm)["engine"]
        flags = cfg["vllm_extra_flags"]
        assert cfg["kv_cache_dtype"] == dtype
        assert flags[flags.index("--kv-cache-dtype") + 1] == dtype
    directory = tmp_path / "replica-0"
    directory.mkdir()
    log = directory / "engine.log"
    log.write_text(
        f"Using AttentionBackendEnum.FLASHINFER backend\ndtype=torch.bfloat16, kv_cache_dtype={dtype},"
    )
    native_dtype = "float8_e4m3fn" if dtype == "fp8" else "bfloat16"
    with log.open("a") as stream:
        stream.write(
            f"\nFlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, kv_cache_dtype=torch.{native_dtype}, arch=sm120"
        )
    attention.verify_backend(tmp_path, "FLASHINFER", dtype)
    log.write_text(
        "Using AttentionBackendEnum.FLASHINFER backend\ndtype=torch.bfloat16, kv_cache_dtype=auto,"
    )
    with pytest.raises(RuntimeError, match="precision unverified"):
        attention.verify_backend(tmp_path, "FLASHINFER", dtype)
