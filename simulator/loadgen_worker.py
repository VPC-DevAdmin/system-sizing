"""Load-generator worker process for the open-loop methodology.

One worker owns one asyncio event loop and the full lifecycle of the
sessions it spawns. The coordinator (``simulator.open_loop``) runs k
of these as subprocesses, each generating Poisson arrivals at λ/k —
superposition makes the combined traffic exactly Poisson(λ), so the
generator scales horizontally and the *tool* is never the capacity
limit: when a worker's arrival tardiness climbs, the coordinator adds
another worker instead of letting client exhaustion masquerade as an
engine knee.

Protocol (line-delimited JSON):

  stdin  ← {"cmd": "rate", "per_s": 2.5}     set this worker's λ share
           {"cmd": "drain"}                   rate→0 + abort sessions
           {"cmd": "mark"}                    a window opened: scope the
                                              tardiness p99 from here
           {"cmd": "stop"}                    clean shutdown
  stdout → {"t": "ready"}                     init done (tokenizer loaded)
           {"t": "turn", ...}                 one completed/failed turn,
                                              SLA flags pre-computed
           {"t": "stat", ...}                 1 Hz worker heartbeat

Launched as ``python -m simulator.loadgen_worker <config.json>`` with
config {persona_weights, replica_urls, api_key, api_model_name,
model_id, request_timeout_s, reasoning_effort, seed, worker_index}.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from .arrivals import SessionArrivalLauncher
from .virtual_user import SharedState, TurnEvent


def _turn_to_wire(e: TurnEvent) -> dict:
    """Serialize a TurnEvent with SLA verdicts resolved.

    The persona object lives in this process; the coordinator gets
    flags, not thresholds, so it never needs to re-resolve personas.
    """
    return {
        "t": "turn",
        "persona_id": e.persona_id,
        "user_id": e.user_id,
        "session_id": e.session_id,
        "turn_index": e.turn_index,
        "submitted_at_ms": e.submitted_at_ms,
        "completed_at_ms": e.completed_at_ms,
        "ttft_ms": e.ttft_ms,
        "ttfct_ms": e.ttfct_ms,
        "tpot_ms": e.tpot_ms,
        "end_to_end_ms": e.end_to_end_ms,
        "input_tokens": e.input_tokens,
        "history_tokens": e.history_tokens,
        "output_tokens": e.output_tokens,
        "reasoning_tokens": e.reasoning_tokens,
        "in_flight_at_submit": e.in_flight_at_submit,
        "error": e.error,
        "ttft_violation": e.ttft_violation(),
        "tpot_violation": e.tpot_violation(),
        "ttft_target_miss": e.ttft_target_miss(),
        "tpot_target_miss": e.tpot_target_miss(),
    }


async def _amain(config: dict) -> bool:
    from openai import AsyncOpenAI

    from .tokenizer_corpus import TokenCorpus

    out = sys.stdout

    def emit(obj: dict) -> None:
        out.write(json.dumps(obj, separators=(",", ":")) + "\n")
        out.flush()

    state = SharedState()
    corpus = TokenCorpus(config["model_id"])
    clients = [
        AsyncOpenAI(base_url=url, api_key=config.get("api_key") or "EMPTY")
        for url in config["replica_urls"]
    ]
    launcher = SessionArrivalLauncher(
        persona_weights=config["persona_weights"],
        clients=clients,
        model_id=config["api_model_name"],
        corpus=corpus,
        state=state,
        request_timeout_s=int(config.get("request_timeout_s") or 300),
        rng_seed=int(config.get("seed") or 0xC0FFEE),
        reasoning_effort=config.get("reasoning_effort"),
    )
    launcher.start()
    emit({"t": "ready", "worker": config.get("worker_index", 0)})

    stop_event = asyncio.Event()

    async def _stdin_loop() -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while not stop_event.is_set():
            line = await reader.readline()
            if not line:  # coordinator closed our stdin — shut down
                stop_event.set()
                return
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            cmd = msg.get("cmd")
            if cmd == "rate":
                launcher.set_rate(float(msg.get("per_s") or 0.0))
            elif cmd == "drain":
                launcher.set_rate(0.0)
                launcher.cancel_active_sessions()
            elif cmd == "trim":
                launcher.trim_active(int(msg.get("target") or 0))
            elif cmd == "mark":
                launcher.mark_window()
            elif cmd == "outstanding":
                launcher.set_outstanding(int(msg.get("n") or 0))
            elif cmd == "restart":
                # Saturation-mode shape swap: abort every in-flight
                # session (the engine cancels aborted requests) —
                # each one's finally-refill respawns it immediately,
                # and the respawn picks the persona up fresh, i.e.
                # with the NEW shape. Outstanding count is untouched.
                launcher.cancel_active_sessions()
            elif cmd == "reload_personas":
                # Shape search rewrites the cell persona overlay and
                # changes shape ON THE FLY — the launcher looks the
                # persona up fresh at every spawn, so a registry
                # reload is all a shape change needs.
                from .personas import reload_personas
                try:
                    reload_personas()
                except Exception as e:  # noqa: BLE001
                    print(f"persona reload failed: {e}", file=sys.stderr)
            elif cmd == "stop":
                stop_event.set()
                return

    async def _event_pump() -> None:
        while not stop_event.is_set():
            try:
                event = await asyncio.wait_for(state.events.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            emit(_turn_to_wire(event))

    async def _stat_loop() -> None:
        # Event-loop lag measured the same way the closed-loop
        # snapshot recorder does: how late the 1 Hz heartbeat wakes.
        expected = time.monotonic() + 1.0
        while not stop_event.is_set():
            await asyncio.sleep(max(0.0, expected - time.monotonic()))
            lag_ms = max(0.0, (time.monotonic() - expected) * 1000.0)
            expected = time.monotonic() + 1.0
            s = launcher.stats
            emit({
                "t": "stat",
                "worker": config.get("worker_index", 0),
                "at_ms": int(time.time() * 1000),
                "arrivals_total": s.arrivals_total,
                "tardy_total": s.tardy_total,
                "sessions_active": s.sessions_active,
                "sessions_done": s.sessions_done,
                "in_flight": state.in_flight,
                "prefill_in_flight": state.prefill_in_flight,
                "completed": state.completed,
                "errors": state.errors,
                "cancelled": state.cancelled,
                "tardiness_p99_ms": round(s.tardiness_p99_ms(), 1),
                "loop_lag_ms": round(lag_ms, 1),
                "mean_session_s": (
                    round(s.mean_session_s(), 1)
                    if s.mean_session_s() is not None else None
                ),
            })

    tasks = [
        asyncio.create_task(_stdin_loop(), name="stdin"),
        asyncio.create_task(_event_pump(), name="event_pump"),
        asyncio.create_task(_stat_loop(), name="stat"),
    ]
    # Every loop exits on stop_event; an exception in ANY of them ends
    # the worker immediately with a non-zero exit code. Swallowing it
    # (the old ``gather(return_exceptions=True)`` at shutdown) left a
    # worker generating at a stale rate and deaf to commands — a
    # failure the coordinator could not see.
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    failure: BaseException | None = None
    for t in done:
        if not t.cancelled() and t.exception() is not None:
            failure = t.exception()
            print(f"loadgen worker task {t.get_name()!r} failed: "
                  f"{type(failure).__name__}: {failure}", file=sys.stderr)
            break
    stop_event.set()
    await launcher.stop()
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return failure is None


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m simulator.loadgen_worker <config.json>",
              file=sys.stderr)
        raise SystemExit(2)
    with open(sys.argv[1]) as f:
        config = json.load(f)
    if not asyncio.run(_amain(config)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
