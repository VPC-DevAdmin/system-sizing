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


def test_empty_extra_flags_keep_shape_flags():
    # An empty list is still a lever; it must not wipe the shape's flags
    # (it launched vLLM on its default batch cap and KV precision).
    eng = custom_engine({**BASE, "vllm_extra_flags": []}, hw=HW, catalog=[])
    assert "--max-num-seqs" in eng["vllm_extra_flags"]
    assert "2048" in eng["vllm_extra_flags"]


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


def test_replicas_per_gpu_shares_the_card():
    # Two engine processes on the one GPU: the same device twice, half the
    # memory share each, the batch cap per process.
    eng = custom_engine({**BASE, "replicas_per_gpu": 2}, hw=HW, catalog=[])
    assert eng["type"] == "vllm_cuda_multi"
    assert eng["replica_devices"] == [[0], [0]]
    assert eng["replicas_per_gpu"] == 2
    assert eng["gpu_memory_utilization"] == 0.475
    assert "2048" in eng["vllm_extra_flags"]


def test_replicas_per_gpu_default_is_unchanged():
    eng = custom_engine(dict(BASE), hw=HW, catalog=[])
    assert "replicas_per_gpu" not in eng
    assert eng["gpu_memory_utilization"] == 0.95


def test_replicas_per_gpu_refused_when_weights_do_not_fit():
    import pytest
    from simulator.engines.custom import ShapeError
    catalog = [{"id": "org/Model", "approx_size_gb": 80}]
    with pytest.raises(ShapeError, match="under the 80 GB"):
        custom_engine({**BASE, "replicas_per_gpu": 2}, hw=HW, catalog=catalog)
    with pytest.raises(ShapeError, match="one process per GPU"):
        custom_engine({**BASE, "engine": "trtllm", "replicas_per_gpu": 2},
                      hw=HW, catalog=[])


def test_launcher_accepts_only_the_declared_sharing():
    import pytest
    from types import SimpleNamespace
    from simulator.engines.docker_replica import check_gpu_sharing
    one = SimpleNamespace(replicas_per_gpu=1, gpu_memory_utilization=0.95)
    two = SimpleNamespace(replicas_per_gpu=2, gpu_memory_utilization=0.475)
    check_gpu_sharing([[0], [1]], one)
    check_gpu_sharing([[0], [0]], two)
    with pytest.raises(RuntimeError, match="assigns a GPU twice"):
        check_gpu_sharing([[0], [0]], one)
    with pytest.raises(RuntimeError, match="does not put every GPU"):
        check_gpu_sharing([[0], [1]], two)
    with pytest.raises(RuntimeError, match="more than the card"):
        check_gpu_sharing([[0], [0]], SimpleNamespace(
            replicas_per_gpu=2, gpu_memory_utilization=0.95))


def test_shared_gpu_replicas_start_one_at_a_time(monkeypatch, tmp_path):
    # Started together, each vLLM process counts the other's allocation as
    # its own while sizing the KV cache; the second must wait for the first.
    from types import SimpleNamespace
    from simulator.engines import docker_replica as dr
    from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine

    def order_for(per_gpu, groups):
        eng = VllmCudaMultiEngine.__new__(VllmCudaMultiEngine)
        dr.DockerReplicaEngine.__init__(eng, SimpleNamespace(
            type="vllm_cuda_multi", replica_devices=groups, port=9100,
            host="127.0.0.1", replicas_per_gpu=per_gpu,
            gpu_memory_utilization=0.95 / per_gpu))
        events = []
        monkeypatch.setattr(dr.shutil, "which", lambda _: "/usr/bin/docker")
        monkeypatch.setattr(dr, "remove_stale_engine_containers", lambda: None)
        def launch(i, devices, run_id):
            events.append(("launch", i))
            eng._replicas.append((i, devices, 9100 + i, f"cid{i}", None))
        monkeypatch.setattr(eng, "_launch_replica", launch)
        monkeypatch.setattr(eng, "_wait_for_replica_ready",
                            lambda i, port, cid: events.append(("ready", i)))
        eng.launch(tmp_path)
        return events

    assert order_for(2, [[0], [0]]) == [
        ("launch", 0), ("ready", 0), ("launch", 1), ("ready", 1)]
    assert order_for(1, [[0], [1]]) == [
        ("launch", 0), ("launch", 1), ("ready", 0), ("ready", 1)]
