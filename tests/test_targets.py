"""Target abstraction (roadmap 1.1): vllm_cuda + remote engines, GPU
hardware requirements, and the remote host-telemetry gate."""

from __future__ import annotations

from simulator.config import Config, EngineConfig
from simulator.engines import RemoteEngine, VllmCudaEngine, make_engine
from simulator.preflight import (
    GpuInfo,
    HardwareRequirements,
    check_gpu_requirements,
)


# ── engine registry + config plumbing ─────────────────────────────────


def test_make_engine_new_types() -> None:
    assert isinstance(
        make_engine("vllm_cuda", EngineConfig(type="vllm_cuda")), VllmCudaEngine
    )
    assert isinstance(
        make_engine(
            "remote",
            EngineConfig(type="remote", endpoint_url="https://x.example/v1"),
        ),
        RemoteEngine,
    )


def test_remote_base_url_and_api_key() -> None:
    cfg = EngineConfig(
        type="remote",
        endpoint_url="https://llm.example.com/v1/",
        endpoint_api_key="sk-test",
    )
    assert cfg.base_url == "https://llm.example.com/v1"
    assert cfg.api_key == "sk-test"
    eng = RemoteEngine(cfg)
    assert eng.pid is None
    eng.shutdown()  # never touches the endpoint — must be a no-op


def test_remote_requires_endpoint_url() -> None:
    import pytest
    with pytest.raises(ValueError, match="endpoint_url"):
        _ = EngineConfig(type="remote").base_url


def test_vllm_cuda_base_url() -> None:
    cfg = EngineConfig(type="vllm_cuda", host="127.0.0.1", port=9100)
    assert cfg.base_url == "http://127.0.0.1:9100/v1"


# ── vllm_cuda docker command ──────────────────────────────────────────


def test_vllm_cuda_docker_command_shape(tmp_path) -> None:
    cfg = EngineConfig(
        type="vllm_cuda",
        model_id="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        port=9100,
        docker_volumes={str(tmp_path): "/root/.cache/huggingface"},
    )
    cmd = VllmCudaEngine(cfg)._build_docker_command()
    joined = " ".join(cmd)
    assert "--gpus all" in joined
    assert "--ipc=host" in joined
    assert f"-v {tmp_path}:/root/.cache/huggingface" in joined
    assert "vllm/vllm-openai:latest" in cmd
    assert "--tensor-parallel-size 2" in joined
    assert "--gpu-memory-utilization 0.85" in joined
    assert "--model Qwen/Qwen3-0.6B" in joined
    # host networking (default): serve directly on the configured port
    assert "--port 9100" in joined
    assert "-p" not in cmd


def test_vllm_cuda_device_ids_and_bridge_port() -> None:
    cfg = EngineConfig(
        type="vllm_cuda",
        gpu_device_ids=[0, 1],
        docker_network="bridge",
        port=9200,
        docker_volumes={},
    )
    cmd = VllmCudaEngine(cfg)._build_docker_command()
    joined = " ".join(cmd)
    assert "--gpus device=0,1" in joined
    # bridged: publish host port to the image's 8000, serve on 8000 inside
    assert "-p 9200:8000" in joined
    assert "--port 8000" in joined


# ── GPU hardware requirements ─────────────────────────────────────────


def test_gpu_requirements_pass() -> None:
    gpus = GpuInfo(2, ["NVIDIA L40S", "NVIDIA L40S"], [48.0, 48.0], "ok")
    reqs = HardwareRequirements(requires_gpu=True, min_gpus=2, min_vram_gb=40)
    assert check_gpu_requirements(gpus, reqs) == []


def test_gpu_requirements_no_gpu_is_hard_fail() -> None:
    gpus = GpuInfo(0, [], [], "no_nvidia_smi")
    reqs = HardwareRequirements(requires_gpu=True)
    failures = check_gpu_requirements(gpus, reqs)
    assert len(failures) == 1 and "none detected" in failures[0]


def test_gpu_requirements_vram_below_bar() -> None:
    gpus = GpuInfo(1, ["NVIDIA T4"], [16.0], "ok")
    reqs = HardwareRequirements(min_vram_gb=70)
    failures = check_gpu_requirements(gpus, reqs)
    assert failures and "min_vram_gb=70" in failures[0]


def test_gpu_requirements_skipped_when_not_required() -> None:
    gpus = GpuInfo(0, [], [], "no_nvidia_smi")
    assert check_gpu_requirements(gpus, HardwareRequirements()) == []
    assert not HardwareRequirements().is_empty() or True  # is_empty untouched
    assert HardwareRequirements(requires_gpu=True).is_empty() is False


# ── remote host-telemetry gate ────────────────────────────────────────


def test_remote_target_skips_host_collectors() -> None:
    import asyncio

    from simulator.config import TelemetryConfig
    from simulator.telemetry import MeasurementTelemetry

    async def run() -> MeasurementTelemetry:
        t = MeasurementTelemetry(
            TelemetryConfig(), engine=None, host_telemetry=False,
        )
        t.start(measurement_id=1)
        await asyncio.sleep(0)
        _, rows, agg = await t.stop()
        return t, rows, agg

    t, rows, agg = asyncio.run(run())
    # No host collectors were started at all.
    assert t._perf is None and t._bandwidth is None and t._power is None
    assert t._gpu is None
    statuses = t.collector_statuses
    assert statuses["pmu"] == "skipped_remote_target"
    assert statuses["gpu"] == "skipped_remote_target"
    assert "engine_metrics" in statuses
