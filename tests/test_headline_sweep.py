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
        })
        assert r.status_code == 202, r.text
        for _ in range(50):
            if called:
                break
            time.sleep(0.05)
    assert "headline_generation" in called.get("cohort", "")
    assert called["max_concurrency"] == 2048
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


def test_few_completions_use_littles_law_not_the_wave_counter():
    """Kimi on KTransformers finished 32 requests every ~87 s; the
    finish-time counter read 130 or 265 tok/s depending on how many
    waves the window caught. 32 streams x 128 tokens / 87 s = 47."""
    from simulator.headline_sweep import _Acc, little_rate
    acc = _Acc()
    acc.add([{"ttft_ms": 10000, "tpot_ms": 600, "end_to_end_ms": 86800,
              "output_tokens": 100, "reasoning_tokens": 28,
              "input_tokens": 128} for _ in range(32)])
    gen, prompt = little_rate(32.0, acc)
    assert abs(gen - 32 * 128 / 86.8) < 0.1
    assert abs(prompt - 32 * 128 / 86.8) < 0.1
    assert little_rate(None, acc) is None
    assert little_rate(4.0, _Acc()) is None


def test_chunk_rate_source_defaults_to_the_counter():
    from simulator.headline_search import Chunk
    assert Chunk(running=4, queue=0, out_rate=33.2, prompt_rate=1).rate_source == "counter"
