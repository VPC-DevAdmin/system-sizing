import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from simulator.infrastructure.dynamo import arm_config, transfer_counters, verify_transfer
from simulator.infrastructure.dynamo_runtime import commands
from simulator.infrastructure.spec import ExperimentSpec


def test_dynamo_budget_validation():
    with pytest.raises(ValidationError, match="prefill and one decode"):
        ExperimentSpec(project_id="P01")
    with pytest.raises(ValidationError):
        ExperimentSpec(project_id="P01", gpu_ids=[0, 1], prefill_workers=2)


def test_same_budget_same_precision_only_role_changes():
    s = ExperimentSpec(project_id="P01", gpu_ids=[2, 3])
    cfg = {"engine": {"kv_cache_dtype": "fp8", "vllm_extra_flags": ["--enforce-eager"]}}
    a, b = [arm_config(cfg, s, x) for x in ("baseline", "treatment")]
    assert a == b
    assert cfg["engine"]["vllm_extra_flags"] == ["--enforce-eager"]
    plan = dict(
        port=19200,
        model="/snapshot",
        served_model_name="model",
        context_limit=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="test",
        gpu_ids=[2, 3],
        prefill_workers=1,
    )
    a, b = [commands(dict(plan, arm=x)) for x in ("baseline", "treatment")]
    assert len(a) == len(b) == 3
    assert [r[2]["CUDA_VISIBLE_DEVICES"] for r in b[1:]] == ["0", "1"]
    assert len({r[2]["PROMETHEUS_MULTIPROC_DIR"] for r in b[1:]}) == 2
    for i in (1, 2):
        assert b[i][1][: len(a[i][1])] == a[i][1]
        flags = b[i][1]
        assert flags[flags.index("--disaggregation-mode") + 1] == (
            "prefill" if i == 1 else "decode"
        )
        assert json.loads(flags[-1])["kv_connector"] == "NixlConnector"
        assert json.loads(flags[-1])["kv_role"] == ("kv_producer" if i == 1 else "kv_consumer")
        assert b[i][2]["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "127.0.0.1"
        assert b[i][2]["UCX_NET_DEVICES"] == "lo"
        assert a[i][2]["VLLM_SSM_CONV_STATE_LAYOUT"] == "DS"
        assert b[i][2]["VLLM_SSM_CONV_STATE_LAYOUT"] == "DS"


def test_transfer_requires_bytes_on_decode_and_positive_delta(tmp_path):
    s = SimpleNamespace(prefill_workers=1, gpu_ids=[0, 1])
    metric = 'vllm:nixl_bytes_transferred_sum{model_name="a b"}'
    assert transfer_counters(metric + " 10\nvllm:nixl_bytes_transferred_count 2") == {metric: 10}
    (tmp_path / "before-worker-1.prom").write_text(metric + " 10")
    (tmp_path / "after-worker-1.prom").write_text(metric + " 10")
    with pytest.raises(RuntimeError, match="unverified"):
        verify_transfer(tmp_path, s)
    (tmp_path / "after-worker-1.prom").write_text(metric + " 20")
    verify_transfer(tmp_path, s)
    assert json.loads((tmp_path / "transfer-verification.json").read_text())["verified"]


def test_readiness_requires_all_worker_health(monkeypatch):
    from simulator.config import EngineConfig
    from simulator.infrastructure.dynamo import OwnedDynamo

    states = [200, 503]

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            if url.endswith("/v1/models"):
                return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "model"}]})
            assert url.endswith("/health")
            return SimpleNamespace(status_code=states[0 if ":19201/" in url else 1])

    monkeypatch.setattr("simulator.infrastructure.dynamo.httpx.Client", Client)
    engine = OwnedDynamo(
        EngineConfig(port=19200),
        ExperimentSpec(project_id="P01", gpu_ids=[0, 1]),
        "baseline",
        "a" * 32,
    )
    assert not engine.health_check()
    states[1] = 200
    assert engine.health_check()


def test_transfer_rejects_native_failures(tmp_path):
    s = SimpleNamespace(prefill_workers=1, gpu_ids=[0, 1])
    (tmp_path / "before-worker-1.prom").write_text(
        "vllm:nixl_bytes_transferred_sum 10\nvllm:nixl_num_failed_transfers_total 0"
    )
    (tmp_path / "after-worker-1.prom").write_text(
        "vllm:nixl_bytes_transferred_sum 20\nvllm:nixl_num_failed_transfers_total 1"
    )
    with pytest.raises(RuntimeError, match="failure counters"):
        verify_transfer(tmp_path, s)


def test_supervisor_stops_siblings_after_child_exit(monkeypatch, tmp_path):
    """A failed frontend must not leave its GPU workers alive in the container."""
    import builtins

    from simulator.infrastructure import dynamo_runtime as runtime

    real_open, real_makedirs = builtins.open, runtime.os.makedirs

    def redirected(path):
        return tmp_path / str(path).lstrip("/")

    def open_file(path, *args, **kwargs):
        return real_open(redirected(path), *args, **kwargs)

    def mkdir(path, *args, **kwargs):
        return real_makedirs(redirected(path), *args, **kwargs)

    children = []

    class Child:
        def __init__(self, argv, **kwargs):
            self.code = 1 if not children else None
            self.terminated = False
            children.append(self)

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated = True
            self.code = -15

        def wait(self, **kwargs):
            return self.code

        def kill(self):
            self.code = -9

    monkeypatch.setattr(builtins, "open", open_file)
    monkeypatch.setattr(runtime.os, "makedirs", mkdir)
    monkeypatch.setattr(runtime.subprocess, "Popen", Child)
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        runtime, "commands", lambda p: [("front", ["front"], {}), ("worker", ["worker"], {})]
    )
    with pytest.raises(RuntimeError, match="service exited"):
        runtime.main({})
    assert len(children) == 2 and children[1].terminated


def test_failed_report_does_not_claim_transfer_verified(tmp_path):
    from simulator.infrastructure.report import report_sections

    s = ExperimentSpec(project_id="P01", gpu_ids=[0, 1], paired_blocks=1)
    manifest = {
        "spec": s.model_dump(),
        "harness": "test",
        "source_result": {"model": "m"},
        "cache_policy": "disabled",
        "created_at": 0,
    }
    claim = {"measurement": None, "status": "not_qualified", "reason": "Failed startup"}
    _, methods, _, _ = report_sections(manifest, {"results": []}, {}, claim, tmp_path)
    assert dict(methods)["Transfer evidence"].startswith("Transfer not established")
    d = tmp_path / "block-00-treatment"
    d.mkdir()
    (d / "transfer-verification.json").write_text('{"verified":true}')
    _, methods, _, _ = report_sections(manifest, {"results": []}, {}, claim, tmp_path)
    assert dict(methods)["Transfer evidence"].startswith("Increasing NIXL")


def test_every_decode_worker_must_receive_transfer(tmp_path):
    s = SimpleNamespace(prefill_workers=1, gpu_ids=[0, 1, 2])
    for i, end in ((1, 20), (2, 10)):
        (tmp_path / f"before-worker-{i}.prom").write_text("vllm:nixl_bytes_transferred_sum 10")
        (tmp_path / f"after-worker-{i}.prom").write_text(f"vllm:nixl_bytes_transferred_sum {end}")
    with pytest.raises(RuntimeError, match="worker 2"):
        verify_transfer(tmp_path, s)


def test_idle_worker_and_reset_reject_budget_evidence(tmp_path):
    from simulator.infrastructure.dynamo import verify_worker_activity

    s = SimpleNamespace(gpu_ids=[0, 1])
    for i in range(2):
        (tmp_path / f"before-worker-{i}.prom").write_text(
            'vllm:generation_tokens_total{worker="x"} 10'
        )
        (tmp_path / f"after-worker-{i}.prom").write_text(
            'vllm:generation_tokens_total{worker="x"} 20'
        )
    verify_worker_activity(tmp_path, s)
    (tmp_path / "after-worker-1.prom").write_text('vllm:generation_tokens_total{worker="x"} 10')
    with pytest.raises(RuntimeError, match="served no"):
        verify_worker_activity(tmp_path, s)
    (tmp_path / "after-worker-1.prom").write_text('vllm:generation_tokens_total{worker="x"} 5')
    with pytest.raises(RuntimeError, match="reset"):
        verify_worker_activity(tmp_path, s)


def test_transfer_controls_are_matched_and_frozen():
    plan = dict(
        port=19200,
        model="m",
        served_model_name="m",
        context_limit=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="test",
        gpu_ids=[0, 1],
        prefill_workers=1,
        transfer_settings=dict(
            p01_block_size=128,
            p01_max_batched_tokens=8192,
            p01_transport="auto",
            p01_memtype_cache="disabled",
            p01_cuda_ipc_get="enabled",
            p01_cross_layer_blocks=True,
        ),
    )
    for arm in ("baseline", "treatment"):
        rows = commands(dict(plan, arm=arm))
        for _, argv, env in rows[1:]:
            assert argv[argv.index("--block-size") + 1] == "128"
            assert argv[argv.index("--max-num-batched-tokens") + 1] == "8192"
            assert env["UCX_TLS"] == "all" and env["UCX_NET_DEVICES"] == "all"
            assert env["UCX_MEMTYPE_CACHE"] == "n"
            assert env["UCX_CUDA_IPC_ENABLE_GET_ZCOPY"] == "y"
            if arm == "treatment":
                cfg = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                assert cfg["kv_connector_extra_config"]["enable_cross_layers_blocks"] is True
                assert cfg["kv_load_failure_policy"] == "fail"
            else:
                assert "--kv-transfer-config" not in argv


def test_native_transfer_diagnostics_and_expiry(tmp_path):
    from simulator.infrastructure.dynamo import transfer_diagnostics

    spec = SimpleNamespace(gpu_ids=[0, 1], prefill_workers=1)
    for i in range(2):
        (tmp_path / f"before-worker-{i}.prom").write_text(
            "vllm:nixl_xfer_time_seconds_sum 1\nvllm:nixl_xfer_time_seconds_count 1\nvllm:nixl_num_kv_expired_reqs_total 0"
        )
        (tmp_path / f"after-worker-{i}.prom").write_text(
            "vllm:nixl_xfer_time_seconds_sum 5\nvllm:nixl_xfer_time_seconds_count 3\nvllm:nixl_num_kv_expired_reqs_total 0"
        )
    transfer_diagnostics(tmp_path, spec)
    data = json.loads((tmp_path / "transfer-diagnostics.json").read_text())
    assert data["workers"][1]["means"]["xfer_time_seconds"]["mean"] == 2
    p = tmp_path / "after-worker-0.prom"
    p.write_text(p.read_text().replace("reqs_total 0", "reqs_total 1"))
    with pytest.raises(RuntimeError, match="leases expired"):
        transfer_diagnostics(tmp_path, spec)


def test_gpu_placement_preserves_requested_order(monkeypatch):
    from simulator.infrastructure.dynamo import resolve_gpu_uuids

    a = "GPU-00000000-0000-0000-0000-000000000001"
    b = "GPU-00000000-0000-0000-0000-000000000002"
    monkeypatch.setattr(
        "simulator.infrastructure.dynamo.subprocess.check_output",
        lambda *a, **k: f"0, {a_uuid}\n4, {b_uuid}\n",
    )
    a_uuid, b_uuid = a, b
    assert resolve_gpu_uuids([4, 0]) == [b, a]
    with pytest.raises(RuntimeError, match="absent"):
        resolve_gpu_uuids([7])
    plan = dict(
        port=19200,
        model="m",
        served_model_name="m",
        context_limit=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="test",
        gpu_ids=[4, 0],
        gpu_uuids=[b, a],
        prefill_workers=1,
        arm="treatment",
    )
    rows = commands(plan)
    assert rows[1][0] == "prefill-0" and rows[1][2]["CUDA_VISIBLE_DEVICES"] == b
    assert rows[2][0] == "decode-1" and rows[2][2]["CUDA_VISIBLE_DEVICES"] == a


def test_diagnostic_execution_settings_apply_to_both_arms():
    plan = dict(
        port=19200,
        model="m",
        served_model_name="m",
        context_limit=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="test",
        gpu_ids=[0, 1],
        prefill_workers=1,
        transfer_settings=dict(p01_execution="eager", p01_cuda_launch_blocking=True),
    )
    for arm in ("baseline", "treatment"):
        for _, argv, env in commands(dict(plan, arm=arm))[1:]:
            assert "--no-async-scheduling" in argv
            assert "--enforce-eager" in argv
            assert env["CUDA_LAUNCH_BLOCKING"] == "1"
