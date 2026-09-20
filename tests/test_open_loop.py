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


def test_engine_broken_fuse_thresholds():
    from simulator.open_loop import _engine_broken
    # All-failure, zero completions: trips fast.
    assert _engine_broken(25, 0)
    assert not _engine_broken(24, 0)
    # Mostly-failure trips even with some completions.
    assert _engine_broken(500, 100)
    # Genuine overload (timeouts amid thousands of completions) never
    # looks like a broken engine.
    assert not _engine_broken(500, 5000)
    assert not _engine_broken(0, 0)


def test_smoke_test_surfaces_engine_error():
    """A dead/unreachable engine fails the smoke preflight with a
    message pointing at the replica, before any load is generated."""
    import pytest

    from simulator.open_loop import EngineBrokenError, smoke_test_engine

    class FakeEngine:
        replica_urls = ["http://127.0.0.1:9"]  # discard port — refuses
        api_model_name = "m"
        api_key = "EMPTY"

    with pytest.raises(EngineBrokenError, match="smoke request"):
        asyncio.run(smoke_test_engine(FakeEngine(), timeout_s=2.0))


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
    sim.open_loop_settle_max_s = 2       # no settling extension
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


def test_fuse_spares_a_slow_engine_that_is_still_generating():
    """A 2048-token turn takes ~290s at load, longer than a 120s
    window — a healthy engine can show zero completions while every
    request streams. Generated tokens, not completions, separate
    'slow' from 'broken'."""
    from simulator.open_loop import _engine_broken

    # The real regression: 91 client timeouts, no completions yet,
    # but the engine generated millions of tokens in the window.
    assert _engine_broken(91, 0, 2_000_000) is False
    assert _engine_broken(500, 10, 5_000_000) is False
    # Genuinely broken: up, failing everything, generating nothing.
    assert _engine_broken(91, 0, 0) is True
    assert _engine_broken(200, 10, 0) is True
    # Counter unavailable: fall back to the original test rather than
    # silently weakening the fuse.
    assert _engine_broken(91, 0, None) is True
    assert _engine_broken(10, 0, None) is False       # under threshold


def test_request_timeout_scales_with_pinned_output_length():
    """Client patience follows the workload's own SLA, so the engine
    (not the stopwatch) decides the outcome."""
    from simulator.config import SimulationConfig
    from simulator.open_loop import (
        REQUEST_TIMEOUT_CEILING_S,
        _request_timeout_for,
    )
    from simulator.personas import cohort_from_persona

    sim = SimulationConfig()
    # Ordinary conversational personas are unaffected.
    assert _request_timeout_for(
        cohort_from_persona("conversational"), sim) == sim.request_timeout_s
    # A pinned long-output headline workload gets real headroom.
    headline = _request_timeout_for(
        cohort_from_persona("headline_generation"), sim)
    assert headline > sim.request_timeout_s
    assert headline <= REQUEST_TIMEOUT_CEILING_S


def test_fuse_keeps_token_baseline_across_failed_scrapes():
    """One failed /metrics scrape must not re-arm the completions-only
    fuse: the token evidence from the last good scrape stands (A6)."""
    from simulator.open_loop import OpenLoopRunner, _ErrorFuse

    class FlakyEngine:
        def __init__(self, values):
            self._values = list(values)

        def get_metrics(self):
            v = self._values.pop(0)
            if v is None:
                raise ConnectionError("scrape failed")
            return {"generation_tokens_total": v}

    runner = OpenLoopRunner.__new__(OpenLoopRunner)
    runner._last_engine_metrics = {}
    runner._queue_gauge_seen = False
    runner._tokens_last_good = None

    # Baseline scrape fails at window start: the fuse still arms on
    # the last good value from before the window.
    runner.engine = FlakyEngine([1000, None, 1500, None, 2000])
    asyncio.run(runner._sample_engine())            # 1000 (good)
    asyncio.run(runner._sample_engine())            # failed scrape
    assert runner._tokens_last_good == 1000
    fuse = _ErrorFuse(0, 0, runner._tokens_last_good)
    assert fuse.tokens0 == 1000

    asyncio.run(runner._sample_engine())            # 1500
    assert not fuse.tripped(30, 0, runner._tokens_last_good)
    asyncio.run(runner._sample_engine())            # failed scrape
    # 30 client timeouts, zero completions, tokens still flowing per
    # the last good scrape: NOT broken. (Old code: tok_now None ->
    # completions-only test -> abort.)
    assert runner._tokens_last_good == 1500
    assert not fuse.tripped(30, 0, runner._tokens_last_good)
    asyncio.run(runner._sample_engine())            # 2000
    assert not fuse.tripped(200, 10, runner._tokens_last_good)

    # A counter that never existed still falls back to the strict test.
    assert _ErrorFuse(0, 0, None).tripped(30, 0, None)
    # A counter that stopped moving does not shield a broken engine.
    assert _ErrorFuse(0, 0, 2000).tripped(30, 0, 2000)


# ── Settling detector (A4) ──────────────────────────────────────────


def test_population_settled_rejects_a_ramp_and_accepts_a_plateau():
    import random

    from simulator.open_loop import _population_settled

    rng = random.Random(4)
    # Linear ramp 0 → 120 sessions over 240 s: at any point the drift
    # across a 60 s trailing window is 30 sessions — far outside 5 %.
    ramp = [0.5 * t + rng.gauss(0, 1.5) for t in range(240)]
    settled, drift = _population_settled(ramp, 60)
    assert not settled and drift > 0.2
    # Poisson-jittered plateau around 100: settled.
    flat = [100 + rng.gauss(0, 4) for _ in range(120)]
    settled, drift = _population_settled(flat, 60)
    assert settled and drift < 0.05
    # Ramp then plateau: not settled while the trailing window still
    # covers the ramp, settled once it is all plateau.
    series = ramp + flat
    assert not _population_settled(series[:250], 60)[0]
    assert _population_settled(series, 60)[0]
    # Too few samples for the trailing window: never settled.
    assert not _population_settled(flat[:30], 60)[0]


def test_warmup_plan_scales_with_session_length():
    """Minimum warmup as before; the settling cap and trailing window
    grow with the measured mean session duration (A4)."""
    from simulator.open_loop import OpenLoopRunner

    class Pool:
        def __init__(self, mean):
            self.mean = mean

        def aggregate(self):
            return {"mean_session_s": self.mean} if self.mean else {}

    def plan(mean, cap=None, window=60):
        r = OpenLoopRunner.__new__(OpenLoopRunner)
        r.cfg = Config()
        r.cfg.simulation.open_loop_settle_max_s = cap
        r.cfg.simulation.open_loop_settle_window_s = window
        r.pool = Pool(mean)
        return r._warmup_plan()

    # No session length known yet (first window): 90 s minimum, cap
    # 300 s, 60 s trailing window.
    assert plan(None) == (90.0, 300.0, 60)
    # Short sessions: same minimum, same cap.
    assert plan(20.0) == (90.0, 300.0, 60)
    # document_qa-length sessions: 300 s minimum, 1.5 W cap, 0.2 W
    # trailing window.
    assert plan(1000.0) == (300.0, 1500.0, 200)
    # Very long sessions: trailing window capped at 300 s.
    assert plan(2000.0) == (300.0, 3000.0, 300)
    # An explicit cap bounds the extension but never undercuts the
    # minimum warmup.
    assert plan(1000.0, cap=600) == (300.0, 600.0, 200)
    assert plan(1000.0, cap=10) == (300.0, 300.0, 200)


# ── A6: smaller measurement biases ──────────────────────────────────


def test_marginal_window_is_an_sla_fail_for_the_search():
    """One rule everywhere: the SLA gate is capacity_status == 'pass'
    (Wilson upper bound < 5 %), so a marginal window does not pass.
    Also pins the summary's own classification of 5/100 as marginal,
    which is what makes the rule bite."""
    turns = [_turn() for _ in range(95)] + [_turn(ok=False) for _ in range(5)]
    s = _summarize_turns(turns)
    assert s["capacity_status"] == "marginal"
    assert (s["capacity_status"] == "pass") is False
    clean = _summarize_turns([_turn() for _ in range(200)])
    assert clean["capacity_status"] == "pass"


def test_served_mean_prefers_engine_running_batch():
    """The practical-significance basis is the engine's running batch
    (num_running), not the client's in-flight count, which also holds
    the queued requests — the very thing that grows in overload."""
    from simulator.open_loop import _served_mean

    running = [8.0] * 60
    inflight = [8.0 + t for t in range(60)]      # queue building up
    mean, basis = _served_mean(running, inflight)
    assert (mean, basis) == (8.0, "engine_running")
    # No running gauge (or too few scrapes): client in-flight fallback.
    assert _served_mean([], inflight)[1] == "client_in_flight"
    assert _served_mean([8.0] * 10, inflight)[1] == "client_in_flight"
    assert _served_mean([], []) == (None, "none")


def test_mean_session_duration_counts_natural_ends_only(monkeypatch):
    """Sessions cut short by a cancel (trim / drain) or a failed turn
    must not feed the Little's-law session length."""
    import simulator.arrivals as arrivals
    from simulator.virtual_user import SharedState

    async def fake_user(*, stats, cancel_event, **_kw):
        # 'natural' sessions take 0.3 s and complete; the others are
        # cancelled after 0.05 s (no sessions_completed increment).
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=0.3)
            return
        except asyncio.TimeoutError:
            stats.sessions_completed += 1

    monkeypatch.setattr(arrivals, "run_virtual_user", fake_user)

    async def main():
        launcher = arrivals.SessionArrivalLauncher(
            persona_weights={"quick_lookup": 1.0},
            clients=[object()], model_id="m", corpus=None,
            state=SharedState(), request_timeout_s=5,
        )
        launcher.start()
        launcher.set_outstanding(6)
        await asyncio.sleep(0.05)
        launcher.set_rate(0.0)                 # stop respawns
        launcher.trim_active(3)                # cancel the 3 newest
        await asyncio.sleep(0.5)
        durations = list(launcher.stats.session_durations_s)
        done = launcher.stats.sessions_done
        await launcher.stop()
        return durations, done

    durations, done = asyncio.run(main())
    assert done == 6
    assert len(durations) == 3                 # only the natural ends
    assert all(d >= 0.29 for d in durations)
