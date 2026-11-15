"""P02 capacity on a plain vLLM server (session_runtime="vllm")."""
import json

import pytest
from pydantic import ValidationError

from simulator import gpu_vendor
from simulator.config import EngineConfig
from simulator.infrastructure.dynamo import arm_config
from simulator.infrastructure.session_capacity_worker import SessionReplicas
from simulator.infrastructure.session_vllm import OwnedNativeSessions
from simulator.infrastructure.spec import ExperimentSpec


def spec(**kw):
    args = dict(
        project_id="P02",
        session_mode="capacity",
        session_gpu_gb=None,
        session_cpu_backend="simple_cpu",
        session_runtime="vllm",
        session_cpu_gb=96,
        duration_s=300,
        warmup_s=0,
        min_requests=2,
        paired_blocks=1,
        session_population_grid=[4, 8],
        session_rate_grid=[0.02, 0.04],
        document_records=16,
        gpu_ids=[0],
    )
    return ExperimentSpec(**(args | kw))


@pytest.fixture
def amd(monkeypatch):
    monkeypatch.setattr(gpu_vendor, "vendor", lambda: "amd")
    monkeypatch.setattr(gpu_vendor, "_device_gids", lambda: ["993", "44"])


def engine(arm, tmp_path, model="openai/gpt-oss-120b"):
    cfg = EngineConfig(type="vllm_cuda", model_id=model, gpu_device_ids=[0],
                       replica_devices=[[0]], port=19200, max_model_len=131072,
                       max_num_seqs=64, kv_cache_dtype="fp8",
                       gpu_image="sha256:" + "a" * 64, docker_volumes={})
    e = OwnedNativeSessions(cfg, spec(), arm, "owner")
    e.directory = tmp_path / arm
    e.directory.mkdir()
    return e


def strip_name(argv):
    i = argv.index("--name")
    return argv[:i] + argv[i + 2:]


def test_runtime_is_capacity_simple_cpu_only():
    assert spec()
    for kw in (dict(session_cpu_backend="kvbm"),
               dict(session_mode="resume", session_gpu_gb=4)):
        with pytest.raises(ValidationError):
            spec(**kw)
    # The default stays Dynamo, so existing XE7740 campaigns are unchanged.
    assert spec(session_runtime="dynamo").session_runtime == "dynamo"


def test_treatment_is_baseline_plus_connector_only(amd, tmp_path):
    base = strip_name(engine("baseline", tmp_path)._build_docker_command())
    treat = strip_name(engine("treatment", tmp_path)._build_docker_command())
    # Evidence mounts differ by directory only; the engine flags match.
    norm = lambda a: [x.replace("/treatment:", "/baseline:") for x in a]  # noqa: E731
    assert norm(treat[:-2]) == norm(base)
    assert treat[-2] == "--kv-transfer-config"
    cfg = json.loads(treat[-1])
    assert cfg["kv_connector"] == "ObservedCPUOffloadConnector"
    assert cfg["kv_connector_module_path"] == "capsim_native_cpu"
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 96 * 1024**3


def test_plain_vllm_command_on_amd(amd, tmp_path):
    argv = engine("treatment", tmp_path)._build_docker_command()
    image = "sha256:" + "a" * 64
    pos = argv.index(image)
    docker, inner = argv[:pos], argv[pos + 1:]
    # Dynamo supervisor is gone: no python3 entrypoint, no runtime plan.
    assert "--entrypoint" not in docker and "/capsim-runtime.py" not in argv
    # AMD device passthrough, not --gpus.
    assert "--gpus" not in docker and "--device=/dev/kfd" in docker
    assert "HIP_VISIBLE_DEVICES=0" in docker
    assert docker[docker.index("--ulimit") + 1] == "memlock=-1"
    assert any(a.endswith(":/capsim_native_cpu.py:ro") for a in docker)
    assert "PYTHONPATH=/" in docker and "CAPSIM_CPU_EVIDENCE_DIR=/evidence" in docker
    for flag, value in (("--max-num-seqs", "64"), ("--kv-cache-dtype", "fp8"),
                        ("--reasoning-parser", "openai_gptoss"), ("--port", "19200")):
        assert inner[inner.index(flag) + 1] == value
    assert "--enable-prefix-caching" in inner
    assert "--kv-cache-memory-bytes" not in inner   # capacity: no artificial GPU cap


def test_non_gptoss_has_no_reasoning_parser(amd, tmp_path):
    argv = engine("baseline", tmp_path, model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")._build_docker_command()
    assert "--reasoning-parser" not in argv


def test_metrics_and_health_share_the_api_port(tmp_path):
    e = engine("baseline", tmp_path)
    assert e.replica_urls == ["http://127.0.0.1:19200/v1"]
    assert e.metrics_urls == ["http://127.0.0.1:19200/metrics"]


def test_replicas_use_the_plain_launcher():
    cfg = EngineConfig(type="vllm_cuda", model_id="test", gpu_device_ids=[0],
                       replica_devices=[[0]], port=19200)
    e = SessionReplicas(cfg, spec(), "baseline", "owner", {0: dict(cpus=[0, 1], node=0)})
    assert type(e.children[0]) is OwnedNativeSessions
    assert e.children[0].cfg.docker_extra_args[-2:] == ["--cpuset-mems", "0"]


def test_arm_config_picks_the_rocm_image(amd):
    doc = {"engine": {"model_id": "m", "kv_cache_dtype": "fp8"}}
    e = arm_config(doc, spec(), "treatment")["engine"]
    assert e["gpu_image"] == gpu_vendor.ROCM_IMAGES["vllm"]
