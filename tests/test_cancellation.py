"""Open-loop cancellation (improvement plan A1): trimming or draining
sessions must abort their in-flight HTTP streams so the ENGINE drops
the requests — not merely stop the session after its current turn."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest

from simulator.arrivals import SessionArrivalLauncher
from simulator.config import Config
from simulator.distributions import Constant
from simulator.engines.mock import MockEngine
from simulator.personas import PERSONAS
from simulator.tokenizer_corpus import TokenCorpus
from simulator.virtual_user import SharedState

MOCK_PORT = 19381


@pytest.fixture()
def slow_engine():
    """Mock engine whose turns take ~8 s (400 tokens x 20 ms), so a
    cancelled turn is unmistakably 'aborted early' rather than
    'finished anyway'."""
    cfg = Config().engine
    cfg.type = "mock"
    cfg.model_id = "mock-model"
    cfg.port = MOCK_PORT
    cfg.mock_ttft_ms = 20
    cfg.mock_tpot_ms = 20
    cfg.mock_capacity_inflight = 1000
    cfg.mock_jitter = 0.0
    engine = MockEngine(cfg)
    engine.launch()
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture()
def long_turn_persona(monkeypatch):
    base = PERSONAS["quick_lookup"]
    persona = dataclasses.replace(
        base,
        id="long_turn_test",
        input_tokens=Constant(16),
        output_tokens=Constant(400),
        turns_per_session=Constant(3),
        read_time_seconds=Constant(0.05),
        active_think_seconds=Constant(0.05),
    )
    monkeypatch.setitem(PERSONAS, "long_turn_test", persona)
    return persona


async def _wait_until(pred, timeout_s: float, what: str) -> float:
    """Return the seconds it took for ``pred`` to become true."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if pred():
            return time.monotonic() - t0
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _launcher(engine, state, persona_id="long_turn_test", **kw):
    from openai import AsyncOpenAI
    return SessionArrivalLauncher(
        persona_weights={persona_id: 1.0},
        clients=[AsyncOpenAI(base_url=engine.base_url, api_key="EMPTY")],
        model_id=engine.api_model_name,
        corpus=TokenCorpus("mock-model"),
        state=state,
        request_timeout_s=60,
        **kw,
    )


def test_trim_active_aborts_in_flight_requests(slow_engine, long_turn_persona):
    """After ``trim_active`` the client in-flight count AND the
    engine's own in-flight count fall within a couple of seconds —
    long before the ~8 s turns would have finished — and the
    aborted turns are recorded as cancelled, not as errors or
    completed samples."""
    async def main():
        state = SharedState()
        launcher = _launcher(slow_engine, state)
        launcher.start()
        launcher.set_rate(40.0)
        await _wait_until(lambda: state.in_flight >= 4, 5.0, "4 streams in flight")
        launcher.set_rate(0.0)
        await asyncio.sleep(0.2)   # let every spawned session submit
        n_active = launcher.stats.sessions_active
        assert n_active >= 4
        assert slow_engine.state["in_flight"] >= 4

        trimmed = launcher.trim_active(0)
        assert trimmed == n_active
        t_client = await _wait_until(
            lambda: state.in_flight == 0, 3.0, "client in-flight to drain")
        t_engine = await _wait_until(
            lambda: slow_engine.state["in_flight"] == 0, 3.0,
            "engine to drop the aborted requests")
        await _wait_until(
            lambda: launcher.stats.sessions_active == 0, 3.0,
            "sessions to finish")
        await launcher.stop()
        return t_client, t_engine, n_active, state

    t_client, t_engine, n_active, state = asyncio.run(main())
    assert t_client < 2.5 and t_engine < 2.5
    assert state.cancelled == n_active
    assert state.errors == 0
    assert state.completed == 0
    assert state.events.empty()   # no synthetic failure events


def test_tier_abort_releases_the_engine_slot(slow_engine, monkeypatch):
    """A tier abort (hard timeout here) closes the response, so the
    engine stops generating instead of running the request to the end
    of its 8 s budget."""
    base = PERSONAS["quick_lookup"]
    persona = dataclasses.replace(
        base, id="abort_test",
        input_tokens=Constant(16), output_tokens=Constant(400),
        turns_per_session=Constant(1), hard_timeout_s=0.5,
    )
    monkeypatch.setitem(PERSONAS, "abort_test", persona)

    async def main():
        state = SharedState()
        launcher = _launcher(slow_engine, state, persona_id="abort_test")
        launcher.start()
        launcher.set_outstanding(2)
        await _wait_until(lambda: state.errors >= 2, 5.0, "two hard timeouts")
        launcher.set_outstanding(0)
        launcher.cancel_active_sessions()
        t_engine = await _wait_until(
            lambda: slow_engine.state["in_flight"] == 0, 3.0,
            "engine to drop the aborted requests")
        await launcher.stop()
        return t_engine, state

    t_engine, state = asyncio.run(main())
    assert t_engine < 2.5
    assert state.errors >= 2
