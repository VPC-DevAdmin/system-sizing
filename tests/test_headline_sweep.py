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
    rungs = [_r(64, 9000.0), _r(128, 18000.0),
             _r(256, 19000.0, in_flight=150.0)]
    reason = should_stop(rungs, 3.0)
    assert reason and "batch ceiling" in reason


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
