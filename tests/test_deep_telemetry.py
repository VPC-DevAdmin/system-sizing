"""Deep telemetry (schema v4): host detail, per-device GPU, engine
token rates, live session phases, and the migration that carries them."""

from __future__ import annotations

import os

import pytest

from simulator.collectors.gpu import parse_smi_csv
from simulator.engines.base import Engine


def test_smi_parse_extended_and_legacy_columns() -> None:
    # Extended 8-column form: temperature + DRAM-controller util.
    devs = parse_smi_csv("93, 60928, 97887, 402.5, 2617, 0x0, 61, 78\n"
                         "12, 512, 97887, 88.0, 1200, 0x4, 45, 5\n")
    assert devs[0]["index"] == 0 and devs[1]["index"] == 1
    assert devs[0]["temperature_c"] == 61.0
    assert devs[0]["mem_util_pct"] == 78.0
    assert devs[1]["throttled"] is True
    # Legacy 6-column form still parses (no extended fields).
    devs = parse_smi_csv("93, 60928, 97887, 402.5, 2617, 0x0\n")
    assert devs[0]["sm_util_pct"] == 93.0
    assert "temperature_c" not in devs[0]


def test_prometheus_token_counters_and_preemptions() -> None:
    text = """# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
vllm:prompt_tokens_total{model_name="m"} 123456.0
vllm:generation_tokens_total{model_name="m"} 98765.0
vllm:num_preemptions_total{model_name="m"} 7.0
vllm:num_requests_running{model_name="m"} 42.0
vllm:kv_cache_usage_perc{model_name="m"} 0.63
"""
    m = Engine._parse_prometheus(text)
    assert m["prompt_tokens_total"] == 123456.0
    assert m["generation_tokens_total"] == 98765.0
    assert m["preemptions_total"] == 7.0
    assert m["num_running"] == 42.0
    assert m["kv_cache_used_pct"] == pytest.approx(63.0)


def test_shared_state_phase_split() -> None:
    import asyncio

    from simulator.virtual_user import SharedState

    async def _run() -> None:
        s = SharedState()
        await s.submit()
        await s.submit()
        assert s.in_flight == 2 and s.prefill_in_flight == 2
        assert s.decode_in_flight == 0

        s.note_first_token()           # one request starts streaming
        assert s.prefill_in_flight == 1 and s.decode_in_flight == 1

        await s.complete(first_token_seen=True)
        assert s.in_flight == 1 and s.prefill_in_flight == 1
        # The other request dies BEFORE its first token — its prefill
        # slot must drain too or the counter leaks upward forever.
        await s.fail(first_token_seen=False)
        assert s.in_flight == 0 and s.prefill_in_flight == 0

        # Warm set is token-weighted: unequal sessions, unequal KV.
        s.enter_warm_think(4000)
        s.enter_warm_think(150)
        s.leave_warm_think(4000)
        assert s.warm_thinking == 1
        assert s.warm_kv_tokens == 150

    asyncio.run(_run())


def test_migration_v4_lifts_legacy_db(tmp_path) -> None:
    import sqlite3

    from simulator.database import SCHEMA_VERSION, Database

    db_path = tmp_path / "run.db"
    # Fresh DB: v4 columns exist and version is stamped.
    db = Database(db_path)
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(measurement_telemetry)")}
    assert {"prefill_tok_s", "decode_tok_s", "preemptions",
            "host_json", "gpu_devices_json"} <= cols
    snap_cols = {r[1] for r in conn.execute("PRAGMA table_info(simulation_snapshots)")}
    assert {"prefill_in_flight", "decode_in_flight", "sessions_warm",
            "sessions_cold", "warm_kv_tokens"} <= snap_cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()
    db.close()

    # Legacy v3 DB: reopening migrates in place.
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE measurement_telemetry ("
                 "telemetry_id INTEGER PRIMARY KEY, measurement_id INTEGER,"
                 "sampled_at_ms INTEGER)")
    conn.execute("CREATE TABLE simulation_snapshots ("
                 "snapshot_id INTEGER PRIMARY KEY, cohort_run_id TEXT,"
                 "snapshot_at_ms INTEGER, phase TEXT, pool_size INTEGER,"
                 "in_flight INTEGER, requests_completed INTEGER, errors INTEGER)")
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()
    db = Database(legacy)
    conn = sqlite3.connect(legacy)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(measurement_telemetry)")}
    assert "host_json" in cols and "decode_tok_s" in cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()
    db.close()


@pytest.mark.skipif(not os.path.exists("/proc/stat"),
                    reason="host collector reads /proc (Linux only)")
def test_host_collector_on_linux() -> None:
    import time

    from simulator.collectors.host import HostCollector
    hc = HostCollector()
    assert hc.is_available()
    first = hc.sample()
    assert first is not None and "mem" in first
    time.sleep(0.3)
    second = hc.sample()
    assert second is not None
    # Second sample carries the rate-derived fields.
    assert "cores_util_pct" in second and second["cores_util_pct"]
    assert "cpu_breakdown_pct" in second
    assert "ctx_switches_s" in second


def test_ipmi_power_parse(monkeypatch) -> None:
    import subprocess as sp

    from simulator.collectors import host as host_mod

    class FakeDone:
        returncode = 0
        stdout = ("    Instantaneous power reading:            542 Watts\n"
                  "    Minimum during sampling period:         321 Watts\n")

    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeDone())
    monkeypatch.setattr(host_mod, "subprocess", sp)
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: "/usr/bin/ipmitool")
    assert host_mod._read_ipmi_power_w() == 542.0
    assert host_mod._probe_ipmi() is True
