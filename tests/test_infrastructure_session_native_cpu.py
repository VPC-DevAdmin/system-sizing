import json

import pytest

from simulator.infrastructure.dynamo_runtime import commands
from simulator.infrastructure.session_capacity_worker import cache_evidence


def test_simple_cpu_keeps_baseline_hybrid_settings_and_cpu_budget():
    plan = dict(
        port=19200,
        model="openai/gpt-oss-120b",
        served_model_name="m",
        context_limit=32768,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        namespace="test",
        gpu_ids=[0],
        sessions={
            "session_cpu_backend": "simple_cpu",
            "session_tier": "cpu",
            "session_cpu_gb": 128,
            "session_gpu_gb": None,
        },
    )
    a, b = [commands(dict(plan, arm=arm))[1] for arm in ("baseline", "treatment")]
    assert a[1] == b[1][:-2]
    assert "--disable-hybrid-kv-cache-manager" not in b[1]
    assert "--enable-prefix-caching" in a[1]
    connector = json.loads(b[1][-1])
    assert connector["kv_connector"] == "ObservedCPUOffloadConnector"
    assert connector["kv_connector_extra_config"]["cpu_bytes_to_use"] == 128 * 1024**3
    assert not any("KVBM" in k for k in b[2])
    plan["sessions"]["session_tier"] = "cpu_disk"
    with pytest.raises(ValueError, match="CPU tier only"):
        commands(dict(plan, arm="treatment"))


def test_external_hits_are_not_mislabeled_completed_cpu_transfers(tmp_path):
    replica = tmp_path / "replica-0"
    replica.mkdir()
    for label, value in [("before", 0), ("after", 128)]:
        (replica / f"{label}-worker-0.prom").write_text(
            f'vllm:external_prefix_cache_hits_total{{model_name="m"}} {value}\n'
        )
    for label in ("before", "after"):
        (replica / f"{label}-native-cpu.json").write_text(
            json.dumps([dict(pid=1, load_blocks=0, store_blocks=0)])
        )
    row = cache_evidence(tmp_path, "treatment", "simple_cpu")[0]
    assert row["external_matched_tokens"] == 128
    assert row["cpu_reload_observed"] is False
    assert "tier_deltas" not in row


def test_completion_watermark_counts_only_finished_copies(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    name = "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector"
    fake = ModuleType(name)

    class Base:
        def __init__(self):
            self.worker_handler = SimpleNamespace(_load_hwm=-1, _store_hwm=-1)

        def bind_connector_metadata(self, meta):
            pass

        def get_finished(self, ids):
            return None, ids

    fake.SimpleCPUOffloadConnector = Base
    monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setenv("CAPSIM_CPU_EVIDENCE_DIR", str(tmp_path))
    location = Path(__file__).parents[1] / "simulator/infrastructure/native_cpu_connector.py"
    spec = importlib.util.spec_from_file_location("test_native_wrapper", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    connector = module.ObservedCPUOffloadConnector()
    meta = SimpleNamespace(
        load_event=2, store_event=3, load_cpu_blocks=[1, 2], store_gpu_blocks=[4, 5, 6]
    )
    connector.bind_connector_metadata(meta)
    connector.get_finished(set())
    assert connector._evidence_totals == dict(load_blocks=0, store_blocks=0)
    connector.worker_handler._load_hwm = 2
    assert connector.get_finished({"r"}) == (None, {"r"})
    assert connector._evidence_totals == dict(load_blocks=2, store_blocks=0)
    connector.worker_handler._store_hwm = 3
    connector.get_finished(set())
    connector.get_finished(set())
    assert connector._evidence_totals == dict(load_blocks=2, store_blocks=3)
    saved = json.loads(next(tmp_path.glob("native-cpu-*.json")).read_text())
    assert saved["load_blocks"] == 2 and saved["store_blocks"] == 3
