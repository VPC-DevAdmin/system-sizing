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
