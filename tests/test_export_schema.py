"""Export-contract validation (roadmap 0.1).

The export document is a versioned public contract:
``docs/export_schema/buyer_page_data.schema.json`` is the schema,
``EXPORT_SCHEMA_VERSION`` in ``simulator/export.py`` is the stamp. Any
structural change to the export must bump the version and keep these
tests green — they build a representative run.db through the real
``Database`` write path, export it (full and slim), and validate the
resulting documents against the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from simulator.database import Database
from simulator.export import EXPORT_SCHEMA_VERSION, export_dir

SCHEMA_PATH = (
    Path(__file__).parent.parent
    / "docs" / "export_schema" / "buyer_page_data.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _validate(doc: dict, schema: dict) -> None:
    jsonschema.validate(
        doc, schema,
        cls=jsonschema.validators.validator_for(schema),
    )


def _build_run_db(tmp_path: Path) -> Path:
    """Populate a run_01/run.db with one cohort covering the export's
    branches: multiple steps (pass → marginal → fail), telemetry
    samples, turn events, and the collector-status summary."""
    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    db = Database(run_dir / "run.db")
    db.insert_run(
        cohort_run_id="crid",
        started_at="2026-01-01T00:00:00Z",
        engine_type="vllm",
        model_id="Qwen/Test",
        cohort_id="chat_heavy",
        cohort_definition={
            "name": "Chat heavy",
            "description": "test mix",
            "category": "mix",
            "persona_weights": {"chat": 1.0},
        },
        config={"engine": {"type": "vllm", "model_id": "Qwen/Test"}},
    )
    for i, (pool, status, viol) in enumerate(
        [(8, "pass", 0.0), (16, "marginal", 0.1), (32, "fail", 0.5)]
    ):
        mid = db.insert_measurement({
            "cohort_run_id": "crid", "step_index": i,
            "target_pool_size": pool,
            "measured_avg_pool_size": float(pool),
            "measured_avg_in_flight": pool * 0.2,
            "measurement_started_at": "2026-01-01T00:00:00Z",
            "measurement_duration_s": 60, "sample_size": 100,
            "ttft_violation_rate": viol, "tpot_violation_rate": 0.0,
            "combined_violation_rate": viol,
            "ttft_target_miss_rate": viol, "tpot_target_miss_rate": 0.0,
            "combined_target_miss_rate": viol,
            "violation_rate_ci_lower": max(0.0, viol - 0.05),
            "violation_rate_ci_upper": viol + 0.05,
            "ttft_p50_ms": 800.0, "ttft_p95_ms": 2400.0,
            "tpot_p50_ms": 90.0, "tpot_p95_ms": 140.0,
            "avg_kv_cache_pct": 40.0 + 10 * i,
            "capacity_status": status, "target_status": status,
        })
        db.insert_telemetry([{
            "measurement_id": mid, "sampled_at_ms": 1000 * s,
            "kv_cache_used_pct": 40.0, "cpu_util_avg": 70.0,
            "cpu_util_bound_avg": 95.0, "memory_used_gb": 100.0,
            "engine_rss_gb": 60.0, "freq_mhz_mean": 3000.0,
        } for s in range(3)])
        db.insert_events([{
            "measurement_id": mid, "persona_id": "chat",
            "user_id": "u1", "session_id": "s1", "turn_index": 0,
            "submitted_at_ms": 0, "ttft_ms": 800.0,
            "completed_at_ms": 5000, "input_tokens": 200,
            "history_tokens": 0, "output_tokens": 150,
            "tpot_ms": 90.0, "end_to_end_ms": 5000.0,
            "in_flight_at_submit": 2, "sla_ttft_violation": 0,
            "sla_tpot_violation": 0,
        }])
    db.update_cohort_run("crid", {
        "collectors_json": json.dumps({
            "pmu": "ok", "memory_bandwidth": "perf_uncore_unavailable",
            "power": "ok", "engine_metrics": "ok", "frequency": "ok",
            "cpu_util": "ok", "memory": "ok", "engine_rss": "no_data",
        }),
    })
    db.finalise_run("crid", "2026-01-01T00:30:00Z", "ok")
    db.close()
    return run_dir


def test_full_export_validates(tmp_path, schema) -> None:
    _build_run_db(tmp_path)
    doc, out_path = export_dir(tmp_path)
    _validate(doc, schema)
    assert doc["schema_version"] == EXPORT_SCHEMA_VERSION
    # The written file must be byte-identical in structure to the
    # returned doc — downstream consumers read the file.
    _validate(json.loads(out_path.read_text()), schema)
    cohort = doc["cohorts"][0]
    assert cohort["collectors"]["pmu"] == "ok"
    assert cohort["collectors"]["memory_bandwidth"] == "perf_uncore_unavailable"
    # Full export carries the per-step time-series fields.
    assert "telemetry_samples" in cohort["curve"][0]
    assert "turns" in cohort["curve"][0]


def test_slim_export_validates(tmp_path, schema) -> None:
    _build_run_db(tmp_path)
    doc, _ = export_dir(tmp_path, slim=True)
    _validate(doc, schema)
    assert doc["meta"]["slim"] is True
    cohort = doc["cohorts"][0]
    # Slim drops the heavy per-step time-series fields entirely.
    assert "telemetry_samples" not in cohort["curve"][0]
    assert "turns" not in cohort["curve"][0]
    assert "timeline" not in cohort["curve"][0]


def test_legacy_db_export_validates(tmp_path, schema) -> None:
    """A run from before collectors_json existed exports with
    collectors=None and still validates — the contract's nullable
    fields are the read-only legacy-tolerance story."""
    run_dir = _build_run_db(tmp_path)
    import sqlite3
    conn = sqlite3.connect(run_dir / "run.db")
    conn.execute("UPDATE cohort_run SET collectors_json = NULL")
    conn.commit()
    conn.close()
    doc, _ = export_dir(tmp_path)
    _validate(doc, schema)
    assert doc["cohorts"][0]["collectors"] is None


def test_schema_version_is_semver() -> None:
    parts = EXPORT_SCHEMA_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
