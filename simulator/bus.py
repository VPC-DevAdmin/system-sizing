"""In-process telemetry event bus (roadmap 2.2).

Single-process pub/sub between the run machinery and live subscribers
(the control-plane service's WebSocket clients). Publishers call
``BUS.publish(topic, payload)`` unconditionally — with no subscribers
it's a no-op costing a set check, so the runner needs no plumbing to
know whether a service is attached. The DB remains the source of
truth; the bus is a live mirror, never a store.

Topics and payload shape (all JSON-serializable, all carrying ``ts``
milliseconds since epoch added at publish):

* ``run``        — lifecycle: {event: started|finished|failed,
                   cohort_run_id, cohort_id, engine, model, ...}
* ``snapshot``   — 1 Hz pool state: phase, pool_size, in_flight,
                   requests_completed, errors, step progress.
* ``telemetry``  — 1 Hz collector sample for the active measurement
                   window (kv %, cpu, rss, freq, gpu, ...).
* ``turn``       — one completed virtual-user turn: latencies, token
                   counts, SLA flags.
* ``step``       — a finished measurement step: pool size, violation
                   rates, percentiles, status.

Subscribers get an ``asyncio.Queue`` of ``{topic, ts, data}`` dicts.
A slow subscriber never blocks the run: on overflow the OLDEST event
is dropped to make room (live views prefer fresh state over history).
Queues belong to the loop that subscribed; the whole design assumes
one process, one loop — which is exactly what ``capsim serve`` runs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        """Fan an event out to every subscriber; never blocks, never
        raises. Free (a set check) when nobody is listening."""
        if not self._subscribers:
            return
        event = {"topic": topic, "ts": int(time.time() * 1000), "data": data}
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event: a live view wants fresh state.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception:  # noqa: BLE001 — a broken queue never stops a run
                self._subscribers.discard(q)


# Process-wide bus. The runner, telemetry, and measurement loop publish
# here; ``capsim serve`` subscribes its WebSocket clients.
BUS = EventBus()
