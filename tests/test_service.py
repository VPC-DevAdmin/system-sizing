"""Control-plane service (roadmap 2.1/2.2): API surface, run
lifecycle over HTTP with the mock engine, and the telemetry bus
reaching WebSocket clients."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest
from fastapi.testclient import TestClient

from simulator.bus import BUS, EventBus
from simulator.distributions import Constant
from simulator.personas import PERSONAS
from simulator.service import create_app

MOCK_PORT = 19381


@pytest.fixture()
def fast_persona(monkeypatch):
    base = PERSONAS["quick_lookup"]
    persona = dataclasses.replace(
        base,
        id="fast_svc",
        input_tokens=Constant(24),
        output_tokens=Constant(8),
        turns_per_session=Constant(2),
        sessions_before_leaving=Constant(50),
        inter_session_gap_seconds=Constant(0.1),
        read_time_seconds=Constant(0.05),
        active_think_seconds=Constant(0.05),
    )
    monkeypatch.setitem(PERSONAS, "fast_svc", persona)
    return persona


def _write_mock_config(tmp_path) -> str:
    cfg = tmp_path / "mock.yaml"
    cfg.write_text(f"""
engine:
  type: mock
  model_id: mock-model
  port: {MOCK_PORT}
  mock_ttft_ms: 40
  mock_tpot_ms: 3
  mock_capacity_inflight: 4
  mock_jitter: 0.0
simulation:
  target_samples_per_step: 6
  warmup_min_duration_s: 1
  warmup_max_duration_s: 2
  measurement_timeout_s: 60
  max_total_duration_minutes: 3
  request_timeout_s: 30
  ramp_spawn_interval_s: 0.05
telemetry:
  enable_pmu: false
  enable_memory_bandwidth: false
  enable_power: false
  enable_gpu: false
""")
    return str(cfg)


def test_api_surface(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(  # profiles resolve relative to the repo root
        __import__("pathlib").Path(__file__).parent.parent
    )
    with TestClient(create_app(tmp_path / "runs")) as client:
        status = client.get("/api/status").json()
        assert status["service"] == "capsim"
        assert status["active_run"] is None

        profiles = client.get("/api/profiles").json()
        assert "mock" in profiles and "xeon-gpu-qwen3-30b" in profiles

        personas = client.get("/api/personas").json()
        assert any(p["id"] == "quick_lookup" for p in personas)
        assert all("ttft_failure_s" in p for p in personas)

        cohorts = client.get("/api/cohorts").json()
        assert any(c["id"] == "chat_heavy" for c in cohorts)

        assert client.get("/api/runs").json() == []
        # No export yet.
        assert client.get("/api/export/latest").status_code == 404
        # No active run to stop.
        assert client.post("/api/runs/stop").status_code == 409
        # Bad workload kind.
        r = client.post("/api/runs", json={
            "profile": "mock", "workload": {"kind": "nope"},
        })
        assert r.status_code == 422


def test_run_lifecycle_and_ws_events(tmp_path, fast_persona) -> None:
    """Start a mock run through the API, watch bus events arrive on
    the WebSocket, wait for completion, list it, export it."""
    config_path = _write_mock_config(tmp_path)

    with TestClient(create_app(tmp_path / "runs")) as client:
        with client.websocket_connect("/ws/telemetry") as ws:
            r = client.post("/api/runs", json={
                "config": config_path,
                "workload": {"kind": "persona", "id": "fast_svc"},
                "new_run": True,
                "pool_sizes": [3],
            })
            assert r.status_code == 202, r.text

            # A second start while active → 409.
            r2 = client.post("/api/runs", json={
                "config": config_path,
                "workload": {"kind": "persona", "id": "fast_svc"},
            })
            assert r2.status_code == 409

            # Bus events flow to the socket; collect until the run
            # finishes (bounded by a deadline, not a fixed sleep).
            topics = set()
            deadline = time.time() + 60
            while time.time() < deadline:
                event = ws.receive_json()
                topics.add(event["topic"])
                if (
                    event["topic"] == "run"
                    and event["data"].get("event") == "finished"
                ):
                    assert event["data"]["final_status"] == "ok"
                    break
            else:
                pytest.fail(f"run did not finish; topics seen: {topics}")
            assert {"run", "snapshot", "turn", "step"} <= topics

        status = client.get("/api/status").json()
        assert status["active_run"]["running"] is False
        assert status["active_run"]["error"] is None
        assert status["active_run"]["result"]

        runs = client.get("/api/runs").json()
        assert len(runs) == 1
        assert runs[0]["cohorts"][0]["final_status"] == "ok"
        assert runs[0]["cohorts"][0]["steps"] == 1

        exported = client.post("/api/export", json={"slim": False}).json()
        assert exported["cohort_count"] == 1
        doc = client.get("/api/export/latest").json()
        assert doc["schema_version"] == exported["schema_version"]
        assert doc["cohorts"][0]["curve"][0]["sample_size"] >= 6


def test_bus_drops_oldest_on_overflow() -> None:
    async def run() -> tuple[int, dict]:
        bus = EventBus()
        q = bus.subscribe(maxsize=3)
        for i in range(5):
            bus.publish("t", {"i": i})
        first = await q.get()
        return q.qsize(), first

    remaining, first = asyncio.run(run())
    # 5 published into a queue of 3: oldest two dropped.
    assert remaining == 2
    assert first["data"]["i"] == 2


def test_bus_noop_without_subscribers() -> None:
    # Publishing with no subscribers must be safe from any context —
    # including outside an event loop.
    BUS.publish("t", {"x": 1})


def test_ui_served_and_per_run_export(tmp_path, fast_persona) -> None:
    """The packaged no-build UI is served same-origin, and the
    per-run export endpoint builds on first request (Phase 3)."""
    config_path = _write_mock_config(tmp_path)
    with TestClient(create_app(tmp_path / "runs")) as client:
        # Static UI at the root.
        index = client.get("/")
        assert index.status_code == 200
        assert "<title>capsim</title>" in index.text
        assert client.get("/app.js").status_code == 200
        assert client.get("/app.css").status_code == 200
        # API routes still win over the static mount.
        assert client.get("/api/status").status_code == 200

        # Per-run export 404s before any run exists…
        assert client.get("/api/runs/run_01/export").status_code == 404

        client.post("/api/runs", json={
            "config": config_path,
            "workload": {"kind": "persona", "id": "fast_svc"},
            "new_run": True,
            "pool_sizes": [2],
        })
        deadline = time.time() + 60
        while time.time() < deadline:
            status = client.get("/api/status").json()
            if status["active_run"] and not status["active_run"]["running"]:
                break
            time.sleep(0.3)
        assert status["active_run"]["error"] is None

        # …and builds + returns after the run.
        doc = client.get("/api/runs/run_01/export").json()
        assert doc["cohorts"][0]["id"] == "fast_svc"
        # Second call reads the built file (still valid).
        doc2 = client.get("/api/runs/run_01/export").json()
        assert doc2["schema_version"] == doc["schema_version"]
        # Path traversal refused.
        assert client.get("/api/runs/..%2Fsecrets/export").status_code in (404, 422)


def test_live_backfill_endpoint(tmp_path) -> None:
    """A page opened mid-run (or after) gets the run's recent history
    shaped like the live WS events: snapshots, telemetry, turns, and
    the completed steps."""
    import time as _t

    from fastapi.testclient import TestClient

    from simulator.database import Database
    from simulator.service import create_app

    runs = tmp_path / "runs"
    run_dir = runs / "run_01"
    run_dir.mkdir(parents=True)
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid", started_at="2026-09-16T00:00:00Z",
        engine_type="vllm_cuda_multi", model_id="org/M",
        cohort_id="chat_heavy",
        cohort_definition={"name": "c", "description": "",
                           "persona_weights": {"p": 1.0}},
        config={"engine": {"type": "vllm_cuda_multi"}},
    )
    now_ms = int(_t.time() * 1000)
    db.insert_snapshot({
        "cohort_run_id": "crid", "snapshot_at_ms": now_ms - 5000,
        "phase": "measuring", "pool_size": 2048, "in_flight": 130,
        "prefill_in_flight": 10, "decode_in_flight": 120,
        "sessions_warm": 900, "sessions_cold": 1018,
        "warm_kv_tokens": 412000, "requests_completed": 5000,
        "errors": 0, "step_samples": 250, "step_target_samples": 500,
        "loop_lag_ms": 12.0,
    })
    mid = db.insert_measurement({
        "cohort_run_id": "crid", "step_index": 0,
        "target_pool_size": 1024, "measured_avg_pool_size": 1024.0,
        "measured_avg_in_flight": 60.0,
        "measurement_started_at": "2026-09-16T00:05:00Z",
        "measurement_duration_s": 60, "sample_size": 500,
        "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
        "combined_violation_rate": 0.0,
        "combined_target_miss_rate": 0.0,
        "violation_rate_ci_lower": 0.0, "violation_rate_ci_upper": 0.01,
        "ttft_p95_ms": 620.0, "tpot_p95_ms": 26.0,
        "capacity_status": "pass",
    })
    db.insert_telemetry([{
        "measurement_id": mid, "sampled_at_ms": now_ms - 4000,
        "kv_cache_used_pct": 22.0, "cpu_util_bound_avg": 40.0,
        "gpu_sm_util_pct": 55.0, "prefill_tok_s": 15000.0,
        "decode_tok_s": 3500.0,
    }])
    db.insert_events([{
        "measurement_id": mid, "persona_id": "p", "user_id": "u",
        "session_id": "s", "turn_index": 0,
        "submitted_at_ms": now_ms - 6000, "ttft_ms": 300.0,
        "completed_at_ms": now_ms - 4500, "input_tokens": 100,
        "history_tokens": 0, "output_tokens": 80, "tpot_ms": 18.0,
        "end_to_end_ms": 1500.0, "in_flight_at_submit": 50,
        "sla_ttft_violation": 0, "sla_tpot_violation": 0,
    }])
    db.close()

    with TestClient(create_app(runs)) as client:
        doc = client.get("/api/live/backfill").json()
        assert doc["run"]["cohort_id"] == "chat_heavy"
        assert doc["snapshots"][0]["pool_size"] == 2048
        assert doc["snapshots"][0]["warm_kv_tokens"] == 412000
        assert doc["telemetry"][0]["decode_tok_s"] == 3500.0
        assert doc["turns"][0]["ttft_ms"] == 300.0
        s = doc["steps"][0]
        assert s["pool_size"] == 1024 and s["capacity_status"] == "pass"
        # Closed-loop rows carry null open-loop fields (present, not
        # crashing — the read-only path must tolerate any schema age).
        assert s["arrival_rate_per_min"] is None
        assert s["stability"] is None


def test_live_backfill_tolerates_pre_v7_db(tmp_path) -> None:
    """The backfill path opens run.db read-only (no migrations), so it
    must not name late-added columns in SQL — a v6-era run.db without
    the open-loop columns has to backfill cleanly, not 500."""
    import sqlite3 as _sq
    import time as _t

    from fastapi.testclient import TestClient

    from simulator.service import create_app

    runs = tmp_path / "runs"
    run_dir = runs / "run_01"
    run_dir.mkdir(parents=True)
    conn = _sq.connect(run_dir / "run.db")
    conn.executescript("""
      CREATE TABLE cohort_run (
        cohort_run_id TEXT PRIMARY KEY, started_at TEXT, completed_at TEXT,
        engine_type TEXT, model_id TEXT, cohort_id TEXT,
        cohort_definition_json TEXT, config_json TEXT, final_status TEXT);
      CREATE TABLE cohort_measurements (
        measurement_id INTEGER PRIMARY KEY, cohort_run_id TEXT,
        step_index INTEGER, target_pool_size INTEGER, sample_size INTEGER,
        combined_violation_rate REAL, combined_target_miss_rate REAL,
        ttft_p95_ms REAL, tpot_p95_ms REAL, capacity_status TEXT);
      CREATE TABLE simulation_snapshots (
        snapshot_id INTEGER PRIMARY KEY, cohort_run_id TEXT,
        snapshot_at_ms INTEGER, phase TEXT, pool_size INTEGER,
        in_flight INTEGER, requests_completed INTEGER, errors INTEGER);
      CREATE TABLE measurement_telemetry (
        telemetry_id INTEGER PRIMARY KEY, measurement_id INTEGER,
        sampled_at_ms INTEGER, kv_cache_used_pct REAL);
      CREATE TABLE turn_events (
        event_id INTEGER PRIMARY KEY, measurement_id INTEGER,
        completed_at_ms INTEGER, ttft_ms REAL, tpot_ms REAL, error TEXT);
    """)
    conn.execute(
        "INSERT INTO cohort_run VALUES ('crid','2026-09-16T00:00:00Z',"
        "NULL,'vllm_cuda','org/M','chat_heavy','{}','{}','ok')")
    conn.execute(
        "INSERT INTO cohort_measurements VALUES "
        "(1,'crid',0,512,500,0.0,0.0,400.0,20.0,'pass')")
    conn.execute(
        "INSERT INTO simulation_snapshots VALUES "
        f"(1,'crid',{int(_t.time() * 1000)},'measuring',512,40,900,0)")
    conn.commit()
    conn.close()

    with TestClient(create_app(runs)) as client:
        r = client.get("/api/live/backfill")
        assert r.status_code == 200
        doc = r.json()
        s = doc["steps"][0]
        assert s["pool_size"] == 512
        assert s["arrival_rate_per_min"] is None
        assert doc["snapshots"][0]["pool_size"] == 512

        # Empty runs dir degrades to an empty (not erroring) shape.
        empty = create_app(tmp_path / "none")
        with TestClient(empty) as c2:
            assert c2.get("/api/live/backfill").json()["run"] is None


def test_profiles_metadata_and_custom_run(tmp_path, monkeypatch) -> None:
    """Profiles carry operator-facing metadata (label, hardware fit,
    optimized marker); a custom engine shape builds a valid config
    and refuses infeasible topologies."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from simulator import arena as arena_mod
    from simulator.service import create_app

    monkeypatch.chdir(Path(__file__).parent.parent)  # repo profiles
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)

    with TestClient(create_app(tmp_path / "runs")) as client:
        profiles = client.get("/api/profiles").json()
        gpu = profiles["xeon-gpu-qwen3-30b"]
        assert gpu["engine_type"] == "vllm_cuda"
        assert gpu["fits_hardware"] is True
        assert "Qwen3-30B" in gpu["label"]
        assert gpu["optimized"] is False
        mock = profiles["mock"]
        assert mock["fits_hardware"] is True       # utility: always usable
        assert "Self-test" in mock["label"]

        # Infeasible custom shape: refused with the reason.
        r = client.post("/api/runs", json={
            "custom": {"model_id": "org/M", "replicas": 8, "tp": 4},
            "workload": {"kind": "cohort", "id": "chat_heavy"},
        })
        assert r.status_code == 422
        assert "does not fit" in r.json()["detail"]

        # Feasible custom shape: generated config accepted (run will
        # fail later at docker launch in this env — acceptance is
        # what's under test) and the file round-trips the loader.
        r = client.post("/api/runs", json={
            "custom": {"model_id": "org/M", "replicas": 8, "tp": 1,
                       "max_num_seqs": 256, "kv_cache_dtype": "fp8"},
            "workload": {"kind": "cohort", "id": "chat_heavy"},
        })
        assert r.status_code == 202, r.text
        from simulator.config import load_config
        c = load_config(tmp_path / "runs" / "custom_benchmark.yaml")
        assert c.engine.type == "vllm_cuda_multi"
        assert len(c.engine.replica_devices) == 8
        assert "--kv-cache-dtype" in c.engine.vllm_extra_flags
        client.post("/api/runs/stop")


def test_persona_and_cohort_summaries(tmp_path, monkeypatch) -> None:
    """The catalog endpoints carry human-readable summaries: analytic
    token/turn/think numbers per persona, weight-blended per cohort."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from simulator.service import create_app

    monkeypatch.chdir(Path(__file__).parent.parent)
    with TestClient(create_app(tmp_path / "runs")) as client:
        personas = client.get("/api/personas").json()
        p = next(x for x in personas if x["id"] == "quick_lookup")
        s = p["summary"]
        assert s["input_tokens"]["median"] > 0
        assert s["input_tokens"]["p90"] >= s["input_tokens"]["median"]
        assert s["output_tokens"]["median"] > 0
        assert s["turns_per_session"]["mean"] >= 1
        # Think gap = read + active think, summed per quantile.
        assert s["think_gap_s"]["median"] > 0

        cohorts = client.get("/api/cohorts").json()
        c = next(x for x in cohorts if x["id"] == "chat_heavy")
        b = c["blended"]
        assert b and b["input_tokens"] > 0 and b["think_gap_s"] > 0


def test_distribution_summaries() -> None:
    import math

    from simulator.distributions import (
        Constant,
        Discrete,
        LogNormal,
        summarize,
    )
    s = summarize(LogNormal.from_median(400, 0.5))
    assert s["median"] == pytest.approx(400)
    assert s["mean"] == pytest.approx(400 * math.exp(0.125))
    assert s["p90"] == pytest.approx(400 * math.exp(1.2816 * 0.5))

    s = summarize(Discrete({1: 0.6, 2: 0.3, 4: 0.1}))
    assert s["median"] == 1 and s["p90"] == 2
    assert s["mean"] == pytest.approx(1.6)

    s = summarize(Constant(7))
    assert s == {"median": 7, "mean": 7, "p90": 7}
