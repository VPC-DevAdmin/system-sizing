import asyncio
import json
from types import SimpleNamespace

import pytest

from simulator.infrastructure.spec import ExperimentSpec
from simulator.infrastructure.tenant_worker import compare_tenants, config


def test_three_placements_hold_kv_budget_and_disable_speculation():
    spec = ExperimentSpec(project_id="P06", gpu_ids=[6, 7])
    for arm, devices in [("dedicated", [6, 7]), ("shared", [7, 7]), ("mig", ["MIG-a", "MIG-b"])]:
        result = config({"engine": {"vllm_extra_flags": []}}, spec, arm, devices)["engine"]
        assert result["replica_devices"] == [[d] for d in devices]
        assert "--speculative-config" not in result["vllm_extra_flags"]
        assert result["vllm_extra_flags"][-1] == str(8 * 1024**3)
        assert result["gpu_memory_utilization"] == (0.45 if arm == "shared" else 0.95)


def test_density_requires_each_tenant_and_retains_software_comparator():
    spec = ExperimentSpec(project_id="P06", gpu_ids=[6, 7], paired_blocks=1)
    rows = [
        dict(block=0, arm=a, phase=p, tenant=t, metrics={"gates_passed": True})
        for a in ("dedicated", "shared", "mig")
        for p in ("steady", "noisy")
        for t in range(2)
    ]
    rows[-4]["metrics"]["gates_passed"] = False
    result = compare_tenants(rows, spec)
    assert result["placements"]["shared"]["services_per_gpu"] == 2
    assert result["placements"]["mig"]["services_per_gpu"] is None
    assert not result["qualified"]


def test_partition_restores_and_refuses_unowned_instances(tmp_path, monkeypatch):
    from simulator.infrastructure import partition

    mode = ["Disabled"]
    instances = []
    calls = []
    monkeypatch.setattr(
        partition,
        "gpu_state",
        lambda device: dict(uuid="GPU-a", current=mode[0], pending=mode[0], display="Disabled"),
    )
    monkeypatch.setattr(partition, "devices", lambda uuid: list(instances))

    def command(args, write=False):
        calls.append((args, write))
        if "-lgip" in args:
            return "| 7 MIG 2g.48gb 5 2/2 |"
        if "-lgi" in args:
            return "| 7 MIG 2g.48gb 5 1 |" if instances else ""
        if "-mig" in args:
            mode[0] = "Enabled" if args[-1] == "1" else "Disabled"
        if "-cgi" in args:
            instances[:] = [
                {"profile": "2g.48gb", "uuid": "MIG-a"},
                {"profile": "2g.48gb", "uuid": "MIG-b"},
            ]
        if "-dgi" in args:
            instances.clear()
        return ""

    monkeypatch.setattr(partition, "command", command)
    transaction = partition.Partition(tmp_path / "partition.json", "owner", 7)
    assert transaction.create() == ["MIG-a", "MIG-b"]
    instances.append({"profile": "2g.48gb", "uuid": "MIG-other"})
    with pytest.raises(RuntimeError, match="Unowned"):
        transaction.restore()
    instances.pop()
    transaction.restore()
    assert mode[0] == "Disabled" and not instances
    assert json.loads((tmp_path / "partition.json").read_text())["phase"] == "restored"
    assert all("GPU-a" in args for args, write in calls if write)


def test_fixed_arrivals_count_admission_failures_and_queue(monkeypatch, tmp_path):
    from simulator.infrastructure import tenants

    async def request(*a):
        await asyncio.sleep(0.08)
        return dict(transport_ok=True, correct=True, elapsed_s=0.08, ttft_s=0.02, usage={})

    monkeypatch.setattr(tenants, "request_one", request)
    spec = SimpleNamespace(
        duration_s=0.15,
        tenant_requests_per_second=80,
        concurrency=1,
        request_timeout_s=1,
        response_slo_s=0.1,
        min_requests=1,
        min_quality=0.95,
        max_error_fraction=0.01,
        min_slo_fraction=0.95,
    )
    engine = SimpleNamespace(replica_urls=["http://one"], api_model_name="m", api_key="EMPTY")
    result = asyncio.run(
        tenants.measure_tenant(
            engine,
            [{"id": "a"}],
            spec,
            tmp_path / "rows.jsonl",
            __import__("time").monotonic() + 0.01,
        )
    )
    rows = [json.loads(x) for x in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert result["attempted"] == result["offered_requests"] == 12
    assert any(r.get("error") == "admission_limit" for r in rows)
    assert any(r.get("queue_s", 0) > 0.01 for r in rows)
    assert not result["gates_passed"]


def test_mig_inventory_accepts_driver_column_padding_and_scopes_gpu(monkeypatch):
    from simulator.infrastructure import partition

    output = """GPU 6: NVIDIA GPU (UUID: GPU-other)
  MIG 2g.48gb     Device  0: (UUID: MIG-other)
GPU 7: NVIDIA GPU (UUID: GPU-owned)
  MIG 2g.48gb     Device  0: (UUID: MIG-first)
  MIG 2g.48gb     Device  1: (UUID: MIG-second)
"""
    monkeypatch.setattr(partition, "command", lambda args: output)
    assert partition.devices("GPU-owned") == [
        {"profile": "2g.48gb", "uuid": "MIG-first"},
        {"profile": "2g.48gb", "uuid": "MIG-second"},
    ]
    assert partition.devices("GPU-absent") == []


def test_mig_instance_inventory_uses_approved_privileged_command(monkeypatch):
    from simulator.infrastructure import partition

    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="inventory", stderr="")
    monkeypatch.setattr(partition.subprocess, "run", run)
    partition.command(["mig", "-i", "GPU-owned", "-lgi"])
    assert calls == [["sudo", "-n", "/usr/bin/nvidia-smi", "mig", "-i", "GPU-owned", "-lgi"]]
