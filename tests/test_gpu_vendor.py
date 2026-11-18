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


def test_amd_groups_are_numeric_host_gids(amd, monkeypatch):
    """docker resolves a group NAME inside the image; the vLLM ROCm
    image has no "render" group, so the host's gids go by number."""
    monkeypatch.setattr(gpu_vendor, "_device_gids", lambda: ["993", "44"])
    args = gpu_vendor.docker_gpu_args([0])
    groups = [args[i + 1] for i, a in enumerate(args) if a == "--group-add"]
    assert groups == ["993", "44"]
    assert "render" not in args


def _run(stdout, rc=0):
    import subprocess
    return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=stdout, stderr="")


def test_amd_compute_pids(monkeypatch):
    busy = '[{"gpu": 0, "process_list": [{"process_info": {"name": "N/A", "pid": 4242}}]}]'
    monkeypatch.setattr(gpu_vendor.subprocess, "run", _run(busy))
    assert gpu_vendor.amd_compute_pids() == [4242]
    monkeypatch.setattr(gpu_vendor.subprocess, "run", _run('[{"gpu": 0, "process_list": []}]'))
    assert gpu_vendor.amd_compute_pids() == []
    # ROCm 10 reports an idle card as a string in place of the process record.
    idle = '[{"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]}]'
    monkeypatch.setattr(gpu_vendor.subprocess, "run", _run(idle))
    assert gpu_vendor.amd_compute_pids() == []
    # "Unknown" must never read as "idle".
    monkeypatch.setattr(gpu_vendor.subprocess, "run", _run("", rc=1))
    with pytest.raises(RuntimeError):
        gpu_vendor.amd_compute_pids()


def test_amd_pci_addresses(monkeypatch):
    monkeypatch.setattr(gpu_vendor, "_amd_json",
                        lambda *a, **k: [{"gpu": 0, "bus": {"bdf": "0000:8B:00.0"}}])
    assert gpu_vendor.amd_pci_addresses() == {0: "0000:8b:00.0"}


def test_assert_gpu_idle_on_amd(amd, monkeypatch):
    from types import SimpleNamespace

    from simulator.benchmarks import worker
    eng = SimpleNamespace(cfg=SimpleNamespace(type="vllm_cuda"))
    monkeypatch.setattr(gpu_vendor, "amd_compute_pids", lambda: [])
    worker.assert_gpu_idle(eng)
    monkeypatch.setattr(gpu_vendor, "amd_compute_pids", lambda: [7])
    with pytest.raises(RuntimeError, match="already active"):
        worker.assert_gpu_idle(eng)
