"""Headline sweep — saturation ladder, stop conditions, mode swap."""

from __future__ import annotations

import time

from simulator.headline_sweep import (
    Rung,
    engine_held,
    peak_rung,
    should_stop,
)


def _r(c, out, in_flight=None, **kw):
    return Rung(concurrency=c, in_flight=in_flight if in_flight is not None else c,
                queue_depth=0.0, out_tok_s=out, prompt_tok_s=100.0,
                total_tok_s=(out or 0) + 100.0, **kw)


def test_engine_held_detects_the_batch_ceiling():
    # Engine running what we offered.
    assert engine_held(512, 508) is True
    # Offered 4096 but only 2050 running: the rest are queued, so the
    # engine's own ceiling has been found.
    assert engine_held(4096, 2050) is False
    # Missing metric must not fabricate a ceiling.
    assert engine_held(4096, None) is True


def test_peak_is_the_highest_sustained_output_rate():
    rungs = [_r(64, 9000.0), _r(128, 18000.0), _r(256, 21000.0),
             _r(512, 20500.0)]
    assert peak_rung(rungs).concurrency == 256
    assert peak_rung([]) is None
    # A rung that produced nothing can never be the headline.
    assert peak_rung([_r(64, None)]) is None


def test_sweep_stops_when_the_engine_stops_holding_the_offer():
    # A ceiling is growth STOPPING: 256 offered, but the running
    # count barely moved past the 128 the previous rung already held.
    rungs = [_r(64, 9000.0), _r(128, 18000.0),
             _r(256, 19000.0, in_flight=130.0)]
    reason = should_stop(rungs, 3.0)
    assert reason and "batch ceiling" in reason

    # Whereas a rung that grew 128 -> 150 has not hit any ceiling,
    # even though it fell short of the 256 offered.
    growing = [_r(64, 9000.0), _r(128, 18000.0),
               _r(256, 19000.0, in_flight=150.0)]
    assert should_stop(growing, 3.0) is None


def test_sweep_stops_when_throughput_plateaus():
    # Still climbing: keep going.
    climbing = [_r(64, 9000.0), _r(128, 18000.0), _r(256, 26000.0)]
    assert should_stop(climbing, 3.0) is None
    # Last two rungs added under 3% over the best before them.
    flat = [_r(64, 9000.0), _r(128, 20000.0), _r(256, 20200.0),
            _r(512, 20300.0)]
    reason = should_stop(flat, 3.0)
    assert reason and "plateau" in reason
    # Too early to judge.
    assert should_stop([_r(64, 9000.0)], 3.0) is None


def test_headline_persona_detection_drives_the_mode_swap():
    from simulator.headline_shapes import is_headline_persona
    assert is_headline_persona("headline_generation") is True
    assert is_headline_persona("headline_ingest") is True
    assert is_headline_persona("conversational") is False
    assert is_headline_persona(None) is False


def test_selecting_a_headline_workload_runs_a_sweep(tmp_path, monkeypatch):
    """The UI contract: pick a headline workload and you get the
    saturation instrument, not the arrival-rate capacity search."""
    import asyncio

    from fastapi.testclient import TestClient

    from simulator.service import create_app
    from tests.test_service import _write_mock_config

    called = {}

    async def fake_sweep(cfg, cohort, **kw):
        called["cohort"] = cohort.id
        called["max_concurrency"] = kw.get("max_concurrency")
        called["ladder"] = kw.get("ladder_override")
        await asyncio.sleep(0)
        return tmp_path / "headline_sweep.json"

    async def fake_open_loop(*a, **kw):
        called["open_loop"] = True
        await asyncio.sleep(0)

    import simulator.headline_sweep as hs
    import simulator.open_loop as ol
    monkeypatch.setattr(hs, "run_headline_sweep", fake_sweep)
    monkeypatch.setattr(ol, "run_cohort_open_loop", fake_open_loop)

    config_path = _write_mock_config(tmp_path)
    with TestClient(create_app(tmp_path / "runs")) as client:
        r = client.post("/api/runs", json={
            "config": config_path,
            "workload": {"kind": "persona", "id": "headline_generation"},
            "max_concurrency": 2048,
            "ladder": [8, 1, 2, 4, 4],
        })
        assert r.status_code == 202, r.text
        for _ in range(50):
            if called:
                break
            time.sleep(0.05)
        bad = client.post("/api/runs", json={
            "config": config_path,
            "workload": {"kind": "persona", "id": "headline_generation"},
            "ladder": [0, 4],
        })
        assert bad.status_code == 422
    assert "headline_generation" in called.get("cohort", "")
    assert called["max_concurrency"] == 2048
    # The request's rungs, sorted and de-duplicated, replace the default ladder.
    assert called["ladder"] == [1, 2, 4, 8]
    assert "open_loop" not in called       # the capacity search stayed out


def test_ordinary_workload_still_uses_the_capacity_search(tmp_path,
                                                          monkeypatch):
    """The swap must be surgical — non-headline workloads are
    untouched."""
    import asyncio

    from fastapi.testclient import TestClient

    from simulator.service import create_app
    from tests.test_service import _write_mock_config

    called = {}

    async def fake_sweep(cfg, cohort, **kw):
        called["sweep"] = True
        await asyncio.sleep(0)

    async def fake_open_loop(cfg, cohort, **kw):
        called["open_loop"] = getattr(cohort, "id", "?")
        await asyncio.sleep(0)

    import simulator.headline_sweep as hs
    import simulator.open_loop as ol
    monkeypatch.setattr(hs, "run_headline_sweep", fake_sweep)
    monkeypatch.setattr(ol, "run_cohort_open_loop", fake_open_loop)

    config_path = _write_mock_config(tmp_path)
    with TestClient(create_app(tmp_path / "runs")) as client:
        r = client.post("/api/runs", json={
            "config": config_path,
            "workload": {"kind": "persona", "id": "conversational"},
        })
        assert r.status_code == 202, r.text
        for _ in range(50):
            if called:
                break
            time.sleep(0.05)
    assert "sweep" not in called
    assert "conversational" in called.get("open_loop", "")


def test_ceiling_needs_settling_and_stalled_growth():
    """A sweep once aborted at rung two — "the engine held 77 of 128
    offered streams, its own batch ceiling" — on an engine already
    measured holding 6,430. A ceiling means growth STOPPED, not that
    one unsettled rung came up short."""
    def r(c, inflight, out, steady=True):
        return Rung(concurrency=c, in_flight=inflight, queue_depth=0.0,
                    out_tok_s=out, prompt_tok_s=0.0, total_tok_s=out,
                    steady_state=steady)

    # The exact regression: second rung, never settled, 77 of 128.
    assert should_stop([r(64, 64.0, 8226),
                        r(128, 76.8, 8307, steady=False)], 3.0) is None
    # Settled but running count still climbing — not a ceiling.
    assert should_stop([r(64, 64.0, 8226), r(128, 100.0, 9000)], 3.0) is None
    # Genuine ceiling: settled, no growth past the previous best, and
    # well short of what was offered.
    reason = should_stop([r(4096, 4095.0, 54000),
                          r(8192, 4095.0, 52000)], 3.0)
    assert reason and "batch ceiling" in reason


def test_peak_never_reports_an_unsettled_rung_over_a_settled_one():
    """An unsettled rung is the engine discharging or filling a
    backlog, and it is routinely the BIGGEST number in the sweep --
    which is precisely why it must not become the headline.

    Taken from a real confirmation sweep: the top rung read 57,647
    tok/s with 456 queued and a 15.8-second p95 TTFT, having never
    settled, while the best settled rung was 20,999."""
    from simulator.headline_sweep import Rung, peak_rung

    rungs = [
        Rung(concurrency=2048, in_flight=2045.0, queue_depth=0.0,
             out_tok_s=20999.0, prompt_tok_s=None, total_tok_s=34268.0,
             steady_state=True),
        Rung(concurrency=4096, in_flight=4086.0, queue_depth=0.0,
             out_tok_s=20371.0, prompt_tok_s=None, total_tok_s=33242.0,
             steady_state=False),
        Rung(concurrency=8192, in_flight=7603.0, queue_depth=456.0,
             out_tok_s=57647.0, prompt_tok_s=None, total_tok_s=94072.0,
             steady_state=False),
    ]
    pk = peak_rung(rungs)
    assert pk.out_tok_s == 20999.0
    assert pk.steady_state is True


def test_a_sweep_that_never_settled_still_reports_something():
    """Reporting nothing hides the run; reporting the best transient
    with steady_state=False attached lets the caller judge it."""
    from simulator.headline_sweep import Rung, peak_rung

    rungs = [
        Rung(concurrency=1024, in_flight=1000.0, queue_depth=0.0,
             out_tok_s=9000.0, prompt_tok_s=None, total_tok_s=None,
             steady_state=False),
        Rung(concurrency=2048, in_flight=2000.0, queue_depth=0.0,
             out_tok_s=15000.0, prompt_tok_s=None, total_tok_s=None,
             steady_state=False),
    ]
    pk = peak_rung(rungs)
    assert pk.out_tok_s == 15000.0
    assert pk.steady_state is False        # visible, not hidden


def test_reasoning_only_completions_are_kept_apart_from_failures():
    """gpt-oss-20b's winning rung recorded 30,736 'errors', of which
    4,419 were HarmonyError failures in the engine log and the rest
    were reasoning-only completions the client files as
    no_content_tokens. They are different things."""
    from simulator.headline_sweep import _Acc
    acc = _Acc()
    acc.add([{"ttft_ms": 10.0, "tpot_ms": 2.0},
             {"error": "no_content_tokens"},
             {"error": "HarmonyError"},
             {"error": "hard_timeout"},
             {"error": "no_content_tokens"}])
    assert len(acc.ttft) == 1 and acc.errors == 2 and acc.no_content == 2


def test_a_rung_of_nothing_but_errors_means_the_engine_died():
    from simulator.headline_sweep import Rung, engine_dead
    dead = Rung(concurrency=2048, in_flight=None, queue_depth=None, out_tok_s=None,
                prompt_tok_s=None, total_tok_s=None, samples=0, errors=464855)
    assert engine_dead(dead)
    quiet = Rung(concurrency=512, in_flight=512.0, queue_depth=0.0, out_tok_s=None,
                 prompt_tok_s=None, total_tok_s=None, samples=0, errors=0)
    assert not engine_dead(quiet)                      # still filling, no errors
    fine = Rung(concurrency=512, in_flight=500.0, queue_depth=0.0, out_tok_s=4000.0,
                prompt_tok_s=100.0, total_tok_s=4100.0, samples=900, errors=12)
    assert not engine_dead(fine)


def test_few_completions_read_sglangs_gauge_not_the_wave_counter():
    """Kimi on KTransformers finished 64 requests in one chunk and the
    finish-time counter read 264 tok/s; the scheduler logged 47-65.
    Under two completions per stream the gauge, averaged over the rung
    from its first fresh reading, is the rate."""
    from simulator.headline_sweep import _Acc, wave_bound
    assert wave_bound(64, 64.0) and wave_bound(0, 4.0)
    assert not wave_bound(2048, 1024.0) and not wave_bound(0, None)
    acc = _Acc()
    for v in (99.0, 99.0, 99.0):      # the previous rung's interval
        acc.add_gauge(v)
    assert acc.gauge == []
    for v in (46.9, 46.9, 65.1, 65.1, None):
        acc.add_gauge(v)
    assert acc.gauge == [46.9, 46.9, 65.1, 65.1]
    assert acc.gauge_updates == 2
    acc.add([{"ttft_ms": 1}, {"error": "no_content_tokens"}])
    assert acc.completed == 2


def test_chunk_rate_source_defaults_to_the_counter():
    from simulator.headline_search import Chunk
    assert Chunk(running=4, queue=0, out_rate=33.2, prompt_rate=1).rate_source == "counter"


def test_sglangs_gen_throughput_is_parsed_and_summed_over_replicas():
    from simulator.engines.base import Engine
    from simulator.engines.docker_replica import aggregate_replica_metrics
    m = Engine._parse_prometheus('sglang:gen_throughput{model_name="k"} 46.93\n')
    assert m["gen_throughput"] == 46.93
    assert aggregate_replica_metrics([m, m])["gen_throughput"] == 2 * 46.93


def test_the_peak_skips_a_rung_whose_requests_failed():
    from simulator.headline_sweep import Rung, peak_rung
    clean = Rung(concurrency=128, in_flight=128, queue_depth=0, out_tok_s=59.7,
                 prompt_tok_s=None, total_tok_s=None, samples=64)
    timed_out = Rung(concurrency=512, in_flight=256, queue_depth=350,
                     out_tok_s=96.8, prompt_tok_s=None, total_tok_s=None,
                     errors=256)
    slow = Rung(concurrency=64, in_flight=64, queue_depth=0, out_tok_s=37.9,
                prompt_tok_s=None, total_tok_s=None)
    assert peak_rung([slow, clean, timed_out]) is clean
    # Nothing served anywhere: still report the best, not nothing.
    assert peak_rung([timed_out]) is timed_out


def test_a_shortfall_far_below_capacity_is_not_a_batch_ceiling():
    """gpt-oss-20b's confirmation stopped at 49 of 128 offered on an
    engine with 8 x 2048 slots; the streams were ending early, not the
    engine refusing them."""
    from simulator.headline_sweep import Rung, should_stop

    def r(c, inflight, out):
        return Rung(concurrency=c, in_flight=inflight, queue_depth=0, out_tok_s=out,
                    prompt_tok_s=None, total_tok_s=None, steady_state=True)
    rungs = [r(64, 59.5, 7886.1), r(128, 49.0, 9389.3)]
    assert should_stop(rungs, 3.0) is not None                 # the old misfire
    assert should_stop(rungs, 3.0, capacity=16384) is None     # climbs on
    # A KV-bound ceiling well inside capacity still stops the climb.
    kv = [r(1024, 906.4, 3270.7), r(2048, 992.8, 3109.4), r(4096, 958.9, 2787.8)]
    assert "ceiling" in should_stop(kv, 3.0, capacity=2048)


def test_a_sweep_drains_the_engine_after_an_aborted_workload():
    import asyncio

    from simulator.runner import _drain_engine

    class Eng:
        def __init__(self):
            self.calls = 0

        def get_metrics(self):
            self.calls += 1
            return {"num_running": 5, "queue_depth": 3} if self.calls < 3 else {}
    e = Eng()
    asyncio.run(_drain_engine(e, timeout_s=5, poll_s=0.01))
    assert e.calls == 3


# ── Hardening: whole-period counter rates, client token totals, fixed batch ──

def _reading(at, *rows, total=None):
    return {"replicas_total": total if total is not None else len(rows), "at": at,
            "replicas": [{"replica": i, "port": 9100 + i,
                          "generation_tokens_total": g, "prompt_tokens_total": p}
                         for i, g, p in rows]}


def test_counter_growth_sums_each_replicas_own_growth():
    from simulator.headline_sweep import counter_growth
    g = counter_growth(_reading(1.0, (0, 1000, 400), (1, 5000, 900)),
                       _reading(601.0, (0, 13_001_000, 5_000_400), (1, 14_005_000, 5_500_900)))
    assert g["ok"] and g["reason"] is None
    assert g["generation_tokens"] == 27_000_000
    assert g["prompt_tokens"] == 10_500_000
    # The raw readings are kept, replica by replica.
    assert g["replicas"][1]["generation_tokens_start"] == 5000
    assert g["replicas"][1]["generation_tokens_end"] == 14_005_000


def test_counter_growth_refuses_a_missing_replica_or_a_restart():
    from simulator.headline_sweep import counter_growth
    # One replica silent at the end: its history must not become a rate.
    g = counter_growth(_reading(1.0, (0, 1000, 400), (1, 5000, 900)),
                       _reading(601.0, (0, 9000, 800), total=2))
    assert not g["ok"] and "did not answer" in g["reason"]
    assert g["generation_tokens"] is None
    # A counter that went backwards is an engine restart.
    g = counter_growth(_reading(1.0, (0, 1000, 400), (1, 5000, 900)),
                       _reading(601.0, (0, 9000, 800), (1, 30, 10)))
    assert not g["ok"] and "backwards" in g["reason"]
    # Nothing at all read.
    assert not counter_growth({}, {})["ok"]


def test_a_batched_stream_reports_the_servers_token_count():
    """With --stream-interval 8 a chunk carries eight tokens: the chunk
    count is not a token count, the usage record is."""
    import asyncio
    from types import SimpleNamespace as NS

    from simulator.streaming import consume_with_tiers

    def chunk(text=None, usage=None):
        choices = [NS(delta=NS(content=text, reasoning=None, reasoning_content=None))] \
            if usage is None else []
        return NS(choices=choices, usage=usage)

    class Stream:
        def __init__(self, items):
            self._it = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration from None

    async def create():
        return Stream([chunk("a b c d e f g h ")] * 32
                      + [chunk(usage=NS(completion_tokens=256, prompt_tokens=142))])

    r = asyncio.run(consume_with_tiers(
        create, pre_ttft_timeout_s=5, inter_token_timeout_s=5, hard_timeout_s=5))
    assert r.error is None
    assert r.output_tokens == 32                 # chunks
    assert r.usage_completion_tokens == 256      # tokens
    assert r.usage_prompt_tokens == 142


def test_a_fixed_batch_starts_only_its_budget():
    from simulator.arrivals import ArrivalStats, SessionArrivalLauncher
    lau = object.__new__(SessionArrivalLauncher)
    lau._stopped, lau._sessions, lau._budget = False, {}, None
    lau._rate_per_s, lau._outstanding = 0.0, 0
    lau._rate_changed = __import__("asyncio").Event()
    lau.stats = ArrivalStats()
    started = []

    def spawn(**_kw):
        started.append(len(started))
        lau._sessions[len(started)] = object()
    lau._spawn_session = spawn
    lau.set_outstanding(4, budget=6)
    assert len(started) == 4                     # four at a time
    for _ in range(5):                           # each finish asks for a refill
        lau._sessions.pop(next(iter(lau._sessions)))
        lau._refill()
    assert len(started) == 6                     # and never more than six
    lau.set_outstanding(4)                       # no budget: the closed loop again
    assert lau._budget is None and len(lau._sessions) == 4


def test_the_pool_splits_a_batch_budget_across_workers():
    import asyncio

    from simulator.open_loop import WorkerPool
    pool = object.__new__(WorkerPool)
    pool._workers = [object(), object(), object()]
    sent = []

    async def send(_w, msg):
        sent.append(msg)
    pool._send = send
    asyncio.run(pool.set_outstanding(8, budget=50))
    assert [m["n"] for m in sent] == [3, 3, 2]
    assert [m["budget"] for m in sent] == [17, 17, 16]
    sent.clear()
    asyncio.run(pool.set_outstanding(8))
    assert all("budget" not in m for m in sent)


def _mock_cfg(tmp_path, port, **engine):
    from simulator.config import Config
    cfg = Config()
    cfg.engine.type = "mock"
    cfg.engine.model_id = "mock-model"
    cfg.engine.port = port
    cfg.engine.mock_ttft_ms, cfg.engine.mock_tpot_ms = 20, 1
    cfg.engine.mock_capacity_inflight, cfg.engine.mock_jitter = 64, 0.0
    for k, v in engine.items():
        setattr(cfg.engine, k, v)
    cfg.simulation.headline_clear_s = 1
    cfg.simulation.headline_measure_s = 2
    cfg.simulation.open_loop_max_workers = 2
    cfg.telemetry.enable_pmu = cfg.telemetry.enable_memory_bandwidth = False
    cfg.telemetry.enable_power = cfg.telemetry.enable_gpu = False
    cfg.output.db_directory = str(tmp_path / "runs")
    return cfg


def test_a_held_rung_reports_the_whole_period_from_the_counters(tmp_path):
    import asyncio
    import json

    from simulator.headline_sweep import run_headline_sweep
    from simulator.personas import cohort_from_persona
    cfg = _mock_cfg(tmp_path, 18431, mock_stream_interval=8)
    out = asyncio.run(run_headline_sweep(
        cfg, cohort_from_persona("headline_generation"), new_run=True,
        ladder_override=[4], hold_s=5))
    summary = json.loads(out.read_text())
    rung, n_out = summary["rungs"][0], int(summary["shape"]["output_tokens"])
    assert rung["rate_source"] == "counter_hold" and rung["steady_state"]
    c = rung["counters"]
    assert c["ok"] and len(c["replicas"]) == 1
    assert 4.5 <= rung["hold_s"] <= 8
    # The published rate is the counter growth over the hold, nothing else.
    assert rung["out_tok_s"] == round(c["generation_tokens"] / c["elapsed_s"], 1)
    assert len(rung["window_rates"]) >= 2
    # Eight tokens a chunk: the client's token total comes from usage, and
    # time per token is per token, not per chunk.
    assert rung["client_usage_requests"] > 0
    assert rung["client_completion_tokens"] == n_out * rung["client_usage_requests"]
    assert rung["client_content_chunks"] < rung["client_completion_tokens"] / 4
    assert rung["tpot_p50_ms"] < 4


def test_a_fixed_batch_matches_client_and_engine_totals(tmp_path):
    import asyncio
    import json

    from simulator.headline_sweep import run_headline_sweep
    from simulator.personas import cohort_from_persona
    cfg = _mock_cfg(tmp_path, 18432)
    out = asyncio.run(run_headline_sweep(
        cfg, cohort_from_persona("headline_generation"), new_run=True,
        ladder_override=[4], batch_requests=12))
    summary = json.loads(out.read_text())
    rung, n_out = summary["rungs"][0], int(summary["shape"]["output_tokens"])
    b = rung["batch"]
    assert rung["rate_source"] == "counter_batch"
    assert b["completed"] == 12 and b["errors"] == 0 and not b["stalled"]
    assert b["client_completion_tokens"] == 12 * n_out
    assert b["engine_generation_tokens"] == 12 * n_out
    assert b["client_over_engine"] == 1.0


def test_hold_and_batch_options_reach_the_sweep(tmp_path, monkeypatch):
    import asyncio

    from fastapi.testclient import TestClient

    from simulator.service import create_app
    from tests.test_service import _write_mock_config

    called = {}

    async def fake_sweep(cfg, cohort, **kw):
        called.update(hold_s=kw.get("hold_s"), batch=kw.get("batch_requests"),
                      reuse=cfg.simulation.prefix_reuse)
        await asyncio.sleep(0)
        return tmp_path / "headline_sweep.json"

    import simulator.headline_sweep as hs
    monkeypatch.setattr(hs, "run_headline_sweep", fake_sweep)
    config_path = _write_mock_config(tmp_path)
    req = {"config": config_path,
           "workload": {"kind": "persona", "id": "headline_generation"}}
    with TestClient(create_app(tmp_path / "runs")) as client:
        r = client.post("/api/runs", json={**req, "ladder": [8192], "hold_s": 600,
                                           "prefix_reuse": 0})
        assert r.status_code == 202, r.text
        for _ in range(50):
            if called:
                break
            time.sleep(0.05)
        assert client.post("/api/runs", json={
            **req, "hold_s": 600, "batch_requests": 50_000}).status_code == 422
        assert client.post("/api/runs", json={**req, "hold_s": -1}).status_code == 422
    assert called["hold_s"] == 600 and called["batch"] is None
    assert called["reuse"] == "off"


def test_telemetry_ignores_a_scrape_that_missed_a_replica():
    from simulator.telemetry import _scrape_incomplete
    assert not _scrape_incomplete({"generation_tokens_total": 5})        # one process
    assert not _scrape_incomplete({"replicas_total": 2.0, "replicas_scraped": 2.0})
    assert _scrape_incomplete({"replicas_total": 2.0, "replicas_scraped": 1.0})
    assert _scrape_incomplete({"replicas_total": 2.0})
