"""gpu_vendor: docker flags and image mapping per GPU stack."""

import pytest

from simulator import gpu_vendor


@pytest.fixture
def amd(monkeypatch):
    monkeypatch.setattr(gpu_vendor, "vendor", lambda: "amd")


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(gpu_vendor, "vendor", lambda: "nvidia")


def test_nvidia_args_keep_gpus_flag(nvidia):
    assert gpu_vendor.docker_gpu_args([0]) == ["--gpus", "device=0"]
    assert gpu_vendor.docker_gpu_args([0, 1]) == ["--gpus", '"device=0,1"']
    assert gpu_vendor.docker_gpu_args(None) == ["--gpus", "all"]


def test_amd_args_pass_devices_through(amd):
    args = gpu_vendor.docker_gpu_args([2, 3])
    assert "--gpus" not in args
    assert "--device=/dev/kfd" in args and "--device=/dev/dri" in args
    assert "HIP_VISIBLE_DEVICES=2,3" in args
    assert not any(a.startswith("ROCR_VISIBLE_DEVICES") for a in args)
    assert not any("VISIBLE" in a for a in gpu_vendor.docker_gpu_args(None))


def test_amd_swaps_only_stock_images(amd):
    assert "rocm" in gpu_vendor.engine_image("vllm", "vllm/vllm-openai:latest")
    assert "rocm" in gpu_vendor.engine_image("sglang", "lmsysorg/sglang:latest")
    assert gpu_vendor.engine_image("vllm", "my/own:tag") == "my/own:tag"


def test_nvidia_images_untouched(nvidia):
    assert gpu_vendor.engine_image("vllm", "vllm/vllm-openai:latest") == "vllm/vllm-openai:latest"


def test_amd_metric_parse(monkeypatch):
    sample = [{"gpu": 0,
               "usage": {"gfx_activity": {"value": 87, "unit": "%"},
                         "umc_activity": {"value": 40, "unit": "%"}},
               "power": {"socket_power": {"value": 431, "unit": "W"}},
               "temperature": {"edge": "N/A", "hotspot": {"value": 71, "unit": "C"}},
               "mem_usage": {"total_vram": {"value": 147440, "unit": "MB"},
                             "used_vram": {"value": 135000, "unit": "MB"}},
               "clock": {"gfx_0": {"clk": {"value": 2100, "unit": "MHz"}},
                         "gfx_1": {"clk": {"value": 2050, "unit": "MHz"}},
                         "mem_0": {"clk": {"value": 1900, "unit": "MHz"}}}}]
    monkeypatch.setattr(gpu_vendor, "_amd_json", lambda *a, **k: sample)
    (d,) = gpu_vendor.amd_sample_devices()
    assert d["power_w"] == 431 and d["sm_util_pct"] == 87
    assert d["sm_clock_mhz"] == 2100 and d["temperature_c"] == 71
    assert round(d["vram_total_gb"]) == 144


def test_amd_runtimes_gate_on_rocm_images(amd):
    from simulator.engine_runtimes import available_engines, runtime_status
    rows = {r["engine"]: r for r in runtime_status(set())}
    assert "trtllm" not in rows and "ktransformers" not in rows
    assert "rocm" in rows["vllm_cuda_multi"]["image"]
    have = {gpu_vendor.ROCM_IMAGES["vllm"], gpu_vendor.ROCM_IMAGES["sglang"]}
    assert set(available_engines(have)) == {"vllm_cuda_multi", "sglang_cuda"}
