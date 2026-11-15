"""vLLM levers (extra flags and env) reach the launched container, and the
extra flags add to the searched shape's own flags instead of replacing them."""

from simulator.engines.custom import custom_engine, lever_owner

HW = {"count": 1, "device_groups": [[0]], "vram_per_gpu_gb": 143.7}
BASE = {"engine": "vllm_cuda_multi", "model_id": "org/Model", "tp": 1, "replicas": 1,
        "max_model_len": 2048, "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8",
        "max_num_seqs": 2048, "trust_remote_code": False}


def test_vllm_levers_append_to_shape_flags():
    eng = custom_engine({**BASE,
                         "vllm_extra_flags": ["--attention-backend", "ROCM_AITER_FA"],
                         "vllm_extra_env": {"VLLM_ROCM_USE_AITER": "1"}}, hw=HW, catalog=[])
    flags = eng["vllm_extra_flags"]
    assert flags[-2:] == ["--attention-backend", "ROCM_AITER_FA"]
    assert "--max-num-seqs" in flags and "2048" in flags
    assert eng["vllm_extra_env"] == {"VLLM_ROCM_USE_AITER": "1"}


def test_no_levers_keeps_shape_flags():
    eng = custom_engine(dict(BASE), hw=HW, catalog=[])
    assert "--max-num-seqs" in eng["vllm_extra_flags"]
    assert "vllm_extra_env" not in eng


def test_vllm_levers_belong_to_vllm():
    assert lever_owner("vllm_extra_flags") == "vllm_cuda_multi"
    assert lever_owner("sglang_extra_flags") == "sglang_cuda"


def test_vllm_env_reaches_docker_command(monkeypatch):
    from simulator.config import EngineConfig
    from simulator.engines.vllm_cuda import VllmCudaEngine
    from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine
    monkeypatch.setenv("CAPSIM_GPU_VENDOR", "amd")
    from simulator import gpu_vendor
    gpu_vendor.vendor.cache_clear()
    try:
        cfg = EngineConfig(type="vllm_cuda", model_id="org/Model",
                           vllm_extra_env={"VLLM_ROCM_USE_AITER": "1"},
                           vllm_extra_flags=["--block-size", "64"])
        cfg.gpu_device_ids = [0]
        argv = VllmCudaEngine(cfg)._build_docker_command()
        assert "VLLM_ROCM_USE_AITER=1" in argv
        assert argv.index("VLLM_ROCM_USE_AITER=1") < argv.index("--model")
        assert argv[-2:] == ["--block-size", "64"]
        cfg2 = EngineConfig(type="vllm_cuda_multi", model_id="org/Model",
                            vllm_extra_env={"VLLM_ROCM_USE_AITER": "1"})
        cfg2.replica_devices = [[0]]
        argv2 = VllmCudaMultiEngine(cfg2).build_replica_command(0, [0], "c")
        assert "VLLM_ROCM_USE_AITER=1" in argv2
    finally:
        gpu_vendor.vendor.cache_clear()


def test_vllm_gpu_image_lever_sets_the_image(monkeypatch):
    """A per-run image (a nightly, a local AITER rebuild) is run as named,
    on AMD too; without it the default is still swapped for the ROCm build."""
    from simulator import gpu_vendor
    from simulator.gpu_vendor import ROCM_IMAGES, engine_image
    # Patch the function, not the environment: vendor() is cached, and a
    # cached "amd" would leak into every later test in the session.
    monkeypatch.setattr(gpu_vendor, "vendor", lambda: "amd")
    image = "capsim/vllm-rocm:nightly-aiter-main"
    eng = custom_engine({**BASE, "vllm_gpu_image": image}, hw=HW, catalog=[])
    assert eng["gpu_image"] == image
    assert engine_image("vllm", eng["gpu_image"]) == image
    eng = custom_engine(dict(BASE), hw=HW, catalog=[])
    assert engine_image("vllm", eng["gpu_image"]) == ROCM_IMAGES["vllm"]
