"""Open-loop methodology: aggregation units, DB shape, and an
end-to-end mock-engine run (Poisson arrivals through worker
subprocesses → stability verdicts → rate-space export)."""

from __future__ import annotations

import asyncio
import json

import simulator.stability
from simulator.config import Config
from simulator.export import _open_loop_summary, export_dir, validate_export
from simulator.open_loop import _summarize_turns

MOCK_PORT = 19377


def _turn(ok=True, miss=False):
    return {
        "ttft_ms": 80.0 if ok else 12000.0,
        "tpot_ms": 10.0,
        "ttfct_ms": 80.0,
        "reasoning_tokens": 0,
        "ttft_violation": not ok,
        "tpot_violation": False,
        "ttft_target_miss": miss or not ok,
        "tpot_target_miss": False,
    }


def test_summarize_turns_rates_and_status():
    turns = [_turn() for _ in range(95)] + [_turn(ok=False) for _ in range(5)]
    s = _summarize_turns(turns)
    assert s["sample_size"] == 100
    assert abs(s["combined_violation_rate"] - 0.05) < 1e-9
    assert s["ttft_p50_ms"] == 80.0
    assert s["capacity_status"] in ("marginal", "pass")  # Wilson-gated


def test_summarize_turns_empty():
    assert _summarize_turns([]) == {"sample_size": 0}


def _m(rate, stability, status="pass", sessions=None, sess_dur=None, n=100):
    return {
        "arrival_rate_per_min": rate,
        "stability": stability,
        "capacity_status": status,
        "sample_size": n,
        "active_sessions_mean": sessions,
        "mean_session_duration_s": sess_dur,
    }


def test_open_loop_summary_full_curve_with_littles_law():
    ms = [
        _m(60, "stable", sessions=40, sess_dur=45.0),
        _m(120, "stable", sessions=85, sess_dur=44.0),
        _m(240, "divergent", status="fail"),
        _m(170, "stable", status="marginal", sessions=120, sess_dur=46.0),
        _m(205, "divergent", status="fail"),
    ]
    s = _open_loop_summary(ms)
    assert s["coverage"] == "full_curve"
    assert s["rate_max_per_min"] == 170
    assert s["rate_ceiling_per_min"] == 205
    assert s["rate_sla_per_min"] == 120
    # Little's law at the SLA point: 120/min = 2/s × 44 s ≈ 88.
    assert s["derived_concurrent_sessions_at_sla"] == 88
    assert not s["rates_are_lower_bounds"]
    assert "grows without bound" in s["zones"]["degraded"]


def test_open_loop_summary_client_limited_is_lower_bound():
    ms = [
        _m(60, "stable", sessions=40, sess_dur=45.0),
        _m(120, "client_limited"),
    ]
    s = _open_loop_summary(ms)
    assert s["coverage"] == "client_limited"
    assert s["rates_are_lower_bounds"]
    assert "generator saturated" in s["zones"]["degraded"]


def test_open_loop_summary_none_for_closed_loop_rows():
    assert _open_loop_summary([{"target_pool_size": 8}]) is None


def test_fresh_db_has_v7_columns(tmp_path):
    from simulator.database import Database
    db = Database(tmp_path / "t.db")
    cols = {r["name"] for r in db.fetchall(
        "SELECT name FROM pragma_table_info('cohort_measurements')")}
    assert {"arrival_rate_per_min", "stability", "queue_depth_slope_per_min",
            "arrival_tardiness_p99_ms", "load_workers"} <= cols
    snap_cols = {r["name"] for r in db.fetchall(
        "SELECT name FROM pragma_table_info('simulation_snapshots')")}
    assert {"arrival_rate_per_min", "queue_depth", "active_sessions"} <= snap_cols
    run_cols = {r["name"] for r in db.fetchall(
        "SELECT name FROM pragma_table_info('cohort_run')")}
    assert "mode" in run_cols
    db.close()


def _open_loop_config(tmp_path) -> Config:
    cfg = Config()
    cfg.engine.type = "mock"
    cfg.engine.model_id = "mock-model"
    cfg.engine.port = MOCK_PORT
    cfg.engine.mock_ttft_ms = 40
    cfg.engine.mock_tpot_ms = 3
    cfg.engine.mock_capacity_inflight = 64
    cfg.engine.mock_jitter = 0.0
    sim = cfg.simulation
    sim.open_loop_initial_rate_per_s = 4.0
    sim.open_loop_max_rate_per_s = 8.0      # tiny rail → capped fast
    sim.open_loop_window_s = 10
    sim.open_loop_refine_window_s = 10
    sim.open_loop_warmup_s = 2
    sim.open_loop_drain_timeout_s = 10
    sim.max_total_duration_minutes = 3
    sim.request_timeout_s = 30
    tel = cfg.telemetry
    tel.enable_pmu = False
    tel.enable_memory_bandwidth = False
    tel.enable_power = False
    tel.enable_gpu = False
    cfg.output.db_directory = str(tmp_path / "runs")
    return cfg


def test_open_loop_end_to_end_mock(tmp_path, monkeypatch):
    """Whole pipeline: worker subprocess generates Poisson arrivals of
    real quick_lookup sessions against the mock engine; windows get
    stability verdicts; the run caps at the (deliberately tiny) rate
    rail and exports as a lower bound in rate space."""
    monkeypatch.setattr(simulator.stability, "MIN_SAMPLES", 8)
    from simulator.open_loop import run_cohort_open_loop
    from simulator.personas import cohort_from_persona

    cfg = _open_loop_config(tmp_path)
    db_path = asyncio.run(run_cohort_open_loop(
        cfg, cohort_from_persona("quick_lookup"), new_run=True,
    ))

    from simulator.database import Database
    db = Database(db_path)
    run_row = db.fetchone(
        "SELECT final_status, mode FROM cohort_run")
    assert run_row["final_status"] == "ok"
    assert run_row["mode"] == "open_loop"
    ms = db.fetchall(
        "SELECT arrival_rate_per_min, stability, sample_size, "
        "active_sessions_mean, load_workers, stability_detail "
        "FROM cohort_measurements ORDER BY step_index")
    assert len(ms) >= 2
    rates = [m["arrival_rate_per_min"] for m in ms]
    assert 240.0 in rates and 480.0 in rates  # 4/s and 8/s in per-min
    for m in ms:
        assert m["stability"] == "stable"
        assert m["load_workers"] >= 1
        detail = json.loads(m["stability_detail"])
        assert detail["verdict"] in ("stable", "inconclusive")
    # Turns flowed from the worker subprocess into the DB.
    n_turns = db.fetchone("SELECT COUNT(*) AS n FROM turn_events")["n"]
    assert n_turns > 0
    # Live snapshots carry the open-loop pressure fields.
    snap = db.fetchone(
        "SELECT COUNT(*) AS n FROM simulation_snapshots "
        "WHERE arrival_rate_per_min IS NOT NULL")
    assert snap["n"] > 0
    # The engine-launch phase is visible in snapshots — a page opened
    # (or backfilled) during a slow model load must show "launching",
    # not the previous run's charts.
    launch = db.fetchone(
        "SELECT COUNT(*) AS n FROM simulation_snapshots "
        "WHERE phase LIKE 'launching engine%'")
    assert launch["n"] > 0
    db.close()

    doc, _ = export_dir(cfg.output.db_directory)
    errors = validate_export(doc)
    assert not errors, "\n".join(errors)
    cohort_doc = doc["cohorts"][0]
    assert cohort_doc["methodology"] == "open_loop"
    ol = cohort_doc["open_loop"]
    assert ol is not None
    assert ol["coverage"] == "capped"          # rail reached, still stable
    assert ol["rates_are_lower_bounds"]
    assert ol["rate_max_per_min"] == 480.0
    assert cohort_doc["capacity_is_lower_bound"]
    assert "arrival_rate_per_min" in cohort_doc["curve"][0]
