"""Open-loop session arrival generation.

The partly-open traffic model: *sessions* arrive as a Poisson process
at a controllable rate λ (arrivals don't care how busy the engine is —
that independence is what makes queue divergence observable at all).
*Within* a session, behavior stays closed-loop — turn N+1 waits for
turn N plus the persona's think time — exactly what humans do and
exactly what ``run_virtual_user`` already implements (one virtual user
== one session under the per-session-respawn model).

Client honesty is built in as **arrival tardiness**: every arrival has
a wall-clock scheduled time; if the generator can't spawn it on time,
the lateness is recorded. Tardy arrivals mean the *client* is the
bottleneck — the coordinator scales out workers, and only reports
``client_limited`` when adding workers stops helping.

Poisson superposition makes sharding exact: k workers each generating
at λ/k produce precisely a Poisson process at λ, so the generator
scales horizontally without distorting the traffic shape.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from .personas import PERSONAS
from .virtual_user import SharedState, UserStats, _now_ms, run_virtual_user

log = logging.getLogger(__name__)


# An arrival this late (vs its wall-clock schedule) counts as tardy —
# the generator, not the engine, is falling behind. Cumulative tardy
# counts let the coordinator compute an honest PER-WINDOW tardy
# fraction (a trailing-percentile buffer would smear one bad burst
# across several windows).
TARDY_THRESHOLD_MS = 500.0


@dataclass
class ArrivalStats:
    """Live counters the worker reports to the coordinator each second."""
    arrivals_total: int = 0
    tardy_total: int = 0        # cumulative arrivals later than TARDY_THRESHOLD_MS
    sessions_active: int = 0
    sessions_done: int = 0
    # Recent arrival-lateness samples (ms): actual spawn − scheduled.
    tardiness_ms: deque = field(default_factory=lambda: deque(maxlen=2000))
    # Durations (s) of recently completed sessions — feeds the
    # Little's-law concurrency derivation and warmup sizing.
    session_durations_s: deque = field(default_factory=lambda: deque(maxlen=500))

    def tardiness_p99_ms(self) -> float:
        if not self.tardiness_ms:
            return 0.0
        vals = sorted(self.tardiness_ms)
        return float(vals[min(len(vals) - 1, int(0.99 * len(vals)))])

    def mean_session_s(self) -> float | None:
        if not self.session_durations_s:
            return None
        return sum(self.session_durations_s) / len(self.session_durations_s)


class SessionArrivalLauncher:
    """Spawns one ``run_virtual_user`` session per Poisson arrival.

    Cost scales with *in-flight streams*, not simulated population —
    a session sleeping through a think gap is just a timer. That is
    the structural reason open-loop generation doesn't die at the
    client the way a closed pool does.
    """

    def __init__(
        self,
        *,
        persona_weights: dict[str, float],
        clients: list,
        model_id: str,
        corpus,
        state: SharedState,
        request_timeout_s: int,
        rng_seed: int = 0xC0FFEE,
        reasoning_effort: str | None = None,
        capture_token_timestamps: bool = False,
    ):
        if not clients:
            raise ValueError("SessionArrivalLauncher requires at least one client")
        self._weights = dict(persona_weights)
        self._clients = clients
        self._model_id = model_id
        self._corpus = corpus
        self._state = state
        self._request_timeout_s = request_timeout_s
        self._rng = random.Random(rng_seed)
        self._reasoning_effort = reasoning_effort
        self._capture = capture_token_timestamps

        self._rate_per_s: float = 0.0
        self._outstanding: int = 0
        self._rate_changed = asyncio.Event()
        self._stopped = False
        self._sessions: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._next_client = 0
        self._task: asyncio.Task | None = None
        self.stats = ArrivalStats()

    # ── Control ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._arrival_loop())

    def set_rate(self, per_s: float) -> None:
        self._outstanding = 0          # rate mode cancels saturation mode
        self._rate_per_s = max(0.0, float(per_s))
        self._rate_changed.set()

    def set_outstanding(self, n: int) -> None:
        """Saturation mode: keep exactly ``n`` sessions active —
        spawn up to ``n`` now, respawn as each finishes. With
        zero-think single-turn personas this is the classic
        max-throughput benchmark loop (closed-loop self-throttling is
        exactly what we want here: the engine stays fully fed and
        measurement happens on ITS counters)."""
        self._rate_per_s = 0.0
        self._rate_changed.set()       # stop scheduled arrivals
        self._outstanding = max(0, int(n))
        self._refill()

    def _refill(self) -> None:
        target = getattr(self, "_outstanding", 0)
        while not self._stopped and len(self._sessions) < target:
            self.stats.tardiness_ms.append(0.0)
            self._spawn_session()

    def cancel_active_sessions(self) -> None:
        """Abort in-flight sessions (drain aid — aborted HTTP requests
        are cancelled inside the engine too, so the backlog clears)."""
        for ev in self._cancel_events.values():
            ev.set()

    def trim_active(self, target: int) -> int:
        """Cancel the NEWEST sessions beyond ``target`` active.

        Used after overshooting the stability knee: fall back to a
        known-stable operating point without tearing the population
        down to zero. Newest first — they have the least conversation
        history invested, and their queued/in-flight requests are
        exactly the excess the engine is choking on (an aborted HTTP
        request is cancelled inside the engine, freeing its queue
        slot)."""
        excess = len(self._sessions) - max(0, int(target))
        if excess <= 0:
            return 0
        for user_id in list(self._sessions.keys())[-excess:]:
            ev = self._cancel_events.get(user_id)
            if ev is not None:
                ev.set()
        return excess

    async def stop(self) -> None:
        self._stopped = True
        self._rate_changed.set()
        self.cancel_active_sessions()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._sessions:
            await asyncio.gather(
                *self._sessions.values(), return_exceptions=True,
            )
        self._sessions.clear()
        self._cancel_events.clear()

    # ── Internals ────────────────────────────────────────────────────

    def _pick_persona_id(self) -> str:
        items = list(self._weights.items())
        return self._rng.choices(
            [pid for pid, _ in items], weights=[w for _, w in items], k=1,
        )[0]

    async def _arrival_loop(self) -> None:
        next_at: float | None = None  # monotonic time of next arrival
        while not self._stopped:
            rate = self._rate_per_s
            if rate <= 0:
                next_at = None
                self._rate_changed.clear()
                await self._rate_changed.wait()
                continue
            now = time.monotonic()
            if next_at is None:
                next_at = now + self._rng.expovariate(rate)
            delay = next_at - now
            if delay > 0:
                # Sleep interruptibly so a rate change re-plans the
                # NEXT gap immediately (the current scheduled arrival
                # keeps its time — arrivals already committed to the
                # wall clock stay pinned to it).
                self._rate_changed.clear()
                try:
                    await asyncio.wait_for(
                        self._rate_changed.wait(), timeout=delay,
                    )
                    continue  # rate changed — re-evaluate
                except asyncio.TimeoutError:
                    pass
            if self._stopped:
                break
            # Arrival fires. Tardiness = how late we actually are
            # relative to the wall-clock schedule — the honest signal
            # that the CLIENT (not the engine) is falling behind.
            lateness_ms = max(0.0, (time.monotonic() - next_at) * 1000.0)
            self.stats.tardiness_ms.append(lateness_ms)
            if lateness_ms > TARDY_THRESHOLD_MS:
                self.stats.tardy_total += 1
            self._spawn_session()
            next_at += self._rng.expovariate(rate)

    def _spawn_session(self) -> None:
        persona_id = self._pick_persona_id()
        persona = PERSONAS[persona_id]
        user_id = uuid.uuid4().hex
        stats = UserStats(
            user_id=user_id,
            persona_id=persona_id,
            spawned_at_ms=_now_ms(),
            sessions_target=1,
            pool_size_at_spawn=len(self._sessions),
        )
        cancel_event = asyncio.Event()
        sub_rng = random.Random(self._rng.random())
        client = self._clients[self._next_client % len(self._clients)]
        self._next_client += 1
        started = time.monotonic()

        async def _session() -> None:
            try:
                await run_virtual_user(
                    persona=persona,
                    rng=sub_rng,
                    client=client,
                    model_id=self._model_id,
                    corpus=self._corpus,
                    state=self._state,
                    stats=stats,
                    request_timeout_s=self._request_timeout_s,
                    cancel_event=cancel_event,
                    capture_token_timestamps=self._capture,
                    # No initial phase offset: open-loop arrivals are
                    # already exponentially spaced — an extra random
                    # sleep would just blur the arrival process we're
                    # deliberately shaping.
                    initial_phase_offset_enabled=False,
                    reasoning_effort=self._reasoning_effort,
                )
            finally:
                self.stats.sessions_done += 1
                self.stats.session_durations_s.append(
                    time.monotonic() - started,
                )
                self._sessions.pop(user_id, None)
                self._cancel_events.pop(user_id, None)
                self.stats.sessions_active = len(self._sessions)
                # Saturation mode: replace the finished session so the
                # outstanding count holds.
                if getattr(self, "_outstanding", 0) > 0:
                    self._refill()

        task = asyncio.create_task(_session(), name=f"ol:{persona_id}:{user_id[:8]}")
        self._sessions[user_id] = task
        self._cancel_events[user_id] = cancel_event
        self.stats.arrivals_total += 1
        self.stats.sessions_active = len(self._sessions)
