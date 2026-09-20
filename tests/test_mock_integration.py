"""End-to-end pipeline integration over the mock engine (roadmap 2.3
/ 2.4): real HTTP + SSE + AsyncOpenAI client + virtual users +
measurement loop + DB + export, no hardware, ~10 s wall time.

This is the test the mock engine exists for: if it passes, the whole
run path works — the same path capsim smoke exercises against real
engines on benchmark boxes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from simulator.config import Config
from simulator.distributions import Constant
from simulator.export import export_dir, validate_export
from simulator.personas import PERSONAS, Cohort

MOCK_PORT = 19273


@pytest.fixture()
def fast_persona(monkeypatch):
    """A near-zero-think clone of quick_lookup so the closed loop
    cycles in ~0.3 s per turn instead of tens of seconds."""
    base = PERSONAS["quick_lookup"]
    persona = dataclasses.replace(
        base,
        id="fast_test",
        input_tokens=Constant(24),
        output_tokens=Constant(8),
        turns_per_session=Constant(2),
        sessions_before_leaving=Constant(50),
        inter_session_gap_seconds=Constant(0.1),
        read_time_seconds=Constant(0.05),
        active_think_seconds=Constant(0.05),
    )
    monkeypatch.setitem(PERSONAS, "fast_test", persona)
    return persona


def _mock_config(tmp_path) -> Config:
    cfg = Config()
    cfg.engine.type = "mock"
    cfg.engine.model_id = "mock-model"
    cfg.engine.port = MOCK_PORT
    cfg.engine.mock_ttft_ms = 40
    cfg.engine.mock_tpot_ms = 3
    cfg.engine.mock_capacity_inflight = 4
    cfg.engine.mock_jitter = 0.0
    sim = cfg.simulation
    sim.target_samples_per_step = 6
    sim.warmup_min_duration_s = 1
    sim.warmup_max_duration_s = 2
    sim.measurement_timeout_s = 60
    sim.max_total_duration_minutes = 3
    sim.request_timeout_s = 30
    sim.ramp_spawn_interval_s = 0.05
    tel = cfg.telemetry
    tel.enable_pmu = False
    tel.enable_memory_bandwidth = False
    tel.enable_power = False
    tel.enable_gpu = False
    cfg.output.db_directory = str(tmp_path / "runs")
    return cfg


def test_mock_end_to_end_run_and_export(tmp_path, fast_persona) -> None:
    from simulator.runner import run_cohort

    cfg = _mock_config(tmp_path)
    cohort = Cohort(
        id="mock_smoke", name="Mock smoke", description="integration",
        persona_weights={"fast_test": 1.0},
    )
    db_path = asyncio.run(run_cohort(
        cfg, cohort, fixed_grid_pool_sizes=[3], new_run=True,
    ))

    from simulator.database import Database
    db = Database(db_path)
    run_row = db.fetchone(
        "SELECT final_status, collectors_json FROM cohort_run"
    )
    assert run_row["final_status"] == "ok"
    m = db.fetchone(
        "SELECT sample_size, combined_violation_rate, ttft_p50_ms, "
        "avg_kv_cache_pct FROM cohort_measurements"
    )
    assert m["sample_size"] >= 6
    # Uncontended mock (pool 3 < capacity 4 in-flight): everything
    # passes SLA and TTFT sits near the configured 40 ms.
    assert m["combined_violation_rate"] == 0.0
    assert m["ttft_p50_ms"] is not None and m["ttft_p50_ms"] < 1000
    # Engine-metrics path exercised the mock's /metrics.
    collectors = json.loads(run_row["collectors_json"])
    assert collectors["engine_metrics"] == "ok"
    assert collectors["pmu"] == "disabled"
    turns = db.fetchone("SELECT COUNT(*) AS n FROM turn_events")
    assert turns["n"] >= 6
    db.close()

    # Export the run and hold it to the schema contract.
    doc, _ = export_dir(cfg.output.db_directory)
    errors = validate_export(doc)
    assert not errors, "\n".join(errors)
    cohort_doc = doc["cohorts"][0]
    assert cohort_doc["id"] == "mock_smoke"
    assert cohort_doc["curve"][0]["sample_size"] >= 6
    assert cohort_doc["collectors"]["engine_metrics"] == "ok"


def test_user_stop_is_recorded_as_cancelled(tmp_path, fast_persona, monkeypatch) -> None:
    """Ctrl-C mid-run stamps the cohort_run 'cancelled' — the same
    word the open-loop runner and the service use — never
    'interrupted'."""
    import simulator.runner as runner
    from simulator.database import Database

    async def boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_measurement_step", boom)
    cfg = _mock_config(tmp_path)
    cfg.engine.port = MOCK_PORT + 2
    cohort = Cohort(id="cancel_test", name="t", description="t",
                    category="test", persona_weights={"fast_test": 1.0})
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(runner.run_cohort(cfg, cohort, new_run=True))
    db_files = list((tmp_path / "runs").glob("run_*/run.db"))
    assert len(db_files) == 1
    db = Database(db_files[0])
    row = db.fetchone("SELECT final_status FROM cohort_run")
    db.close()
    assert row["final_status"] == "cancelled"
