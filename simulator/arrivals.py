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
    # Arrival-lateness samples (ms): actual spawn − scheduled, since
    # the last ``reset_tardiness`` — the coordinator resets it when a
    # measurement window opens, so the reported p99 is THAT window's
    # (a trailing buffer smeared one bad burst across later windows).
    tardiness_ms: deque = field(default_factory=lambda: deque(maxlen=10000))
    # Durations (s) of recently completed sessions — feeds the
    # Little's-law concurrency derivation and warmup sizing. Only
    # sessions that ran their full turn count count: one ended by a
    # trim / drain / cancel or by a failed turn is not a session
    # length, and mixing those in shortened W and inflated the
    # derived concurrency (L = λ·W).
    session_durations_s: deque = field(default_factory=lambda: deque(maxlen=500))

    def tardiness_p99_ms(self) -> float:
        if not self.tardiness_ms:
            return 0.0
        vals = sorted(self.tardiness_ms)
        return float(vals[min(len(vals) - 1, int(0.99 * len(vals)))])

    def reset_tardiness(self) -> None:
        """Start a new tardiness scope (a measurement window opened).
        The cumulative ``tardy_total`` counter is untouched — the
        per-window tardy FRACTION is a delta of that."""
        self.tardiness_ms.clear()

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
        reasoning_overhead_tokens: int | None = None,
        capture_token_timestamps: bool = False,
        history_window_turns: int | None = None,
        context_limit_tokens: int | None = None,
        planner_timeouts: dict | None = None,
        system_prompt_tokens: dict[str, int] | None = None,
        system_prompt_in_input: bool = False,
        stream_usage: bool = False,
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
        self._reasoning_overhead_tokens = reasoning_overhead_tokens
        self._capture = capture_token_timestamps
        self._history_window_turns = history_window_turns
        self._context_limit_tokens = context_limit_tokens
        self._planner_timeouts = planner_timeouts
        self._system_prompt_tokens = dict(system_prompt_tokens or {})
        self._system_prompts: dict[str, tuple[str, int]] = {}
        self._system_prompt_in_input = bool(system_prompt_in_input)
        self._stream_usage = bool(stream_usage)

        self._rate_per_s: float = 0.0
        self._outstanding: int = 0
        # Fixed-batch mode: sessions still to start (None = no limit).
        self._budget: int | None = None
        self._rate_changed = asyncio.Event()
        self._stopped = False
        self._sessions: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._next_client = 0

        self._task: asyncio.Task | None = None
        self.stats = ArrivalStats()

    def _system_prompt(self, persona_id: str) -> tuple[str, int] | None:
        """(text, tokens) shared by every session of the persona; built
        once and identical across workers (seeded by the persona id)."""
        n = int(self._system_prompt_tokens.get(persona_id) or 0)
        if n <= 0:
            return None
        if persona_id not in self._system_prompts:
            from .tokenizer_corpus import shared_system_prompt
            self._system_prompts[persona_id] = shared_system_prompt(
                self._corpus, persona_id, n)
        return self._system_prompts[persona_id]

    # ── Control ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._arrival_loop())

    def set_rate(self, per_s: float) -> None:
        self._outstanding = 0          # rate mode cancels saturation mode
        self._budget = None
        self._rate_per_s = max(0.0, float(per_s))
        self._rate_changed.set()

    def set_outstanding(self, n: int, budget: int | None = None) -> None:
        """Saturation mode: keep exactly ``n`` sessions active —
        spawn up to ``n`` now, respawn as each finishes. With
        zero-think single-turn personas this is the classic
        max-throughput benchmark loop (closed-loop self-throttling is
        exactly what we want here: the engine stays fully fed and
        measurement happens on ITS counters).

        ``budget`` makes it a fixed batch: at most that many sessions are
        started from now on, ``n`` at a time, and none replaces the last
        ones to finish. Every token the engine then generates belongs to
        a request the client saw end, so the two counts can be compared."""
        self._rate_per_s = 0.0
        self._rate_changed.set()       # stop scheduled arrivals
        self._outstanding = max(0, int(n))
        self._budget = None if budget is None else max(0, int(budget))
        self._refill()

    def _refill(self) -> None:
        target = getattr(self, "_outstanding", 0)
        while not self._stopped and len(self._sessions) < target:
            if self._budget is not None:
                if self._budget <= 0:
                    break
                self._budget -= 1
            self.stats.tardiness_ms.append(0.0)
            self._spawn_session()

    def seed_to(self, target: int, weights: dict[str, float] | None = None) -> None:
        """Bring the active population to ``target`` now: start the
        shortfall as sessions already part-way through (random turn,
        earlier history built), or cancel the newest beyond it. The
        Poisson arrivals keep running alongside, so the population
        starts at its Little's-law size instead of ramping to it."""
        short = int(target) - len(self._sessions)
        if short < 0:
            self.trim_active(int(target))
            return
        for _ in range(short):
            pid = None
            if weights:
                items = [(k, v) for k, v in weights.items() if v > 0]
                if items:
                    pid = self._rng.choices([k for k, _ in items],
                                            weights=[v for _, v in items],
                                            k=1)[0]
            self._spawn_session(persona_id=pid, mid_session=True)

    def mark_window(self) -> None:
        """A measurement window opened: scope the tardiness percentile
        to arrivals from here on."""
        self.stats.reset_tardiness()

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
        drawn_rate: float | None = None  # rate the pending gap was drawn at
        while not self._stopped:
            rate = self._rate_per_s
            if rate <= 0:
                next_at = None
                self._rate_changed.clear()
                await self._rate_changed.wait()
                continue
            now = time.monotonic()
            if next_at is None or rate != drawn_rate:
                # Poisson arrivals are memoryless, so the wait still to
                # come at a new rate is a fresh draw at that rate. A gap
                # drawn at the old rate would hold the new one back: a
                # first gap drawn at 0.12/min kept every session out of
                # four rate steps (20 minutes) on a slow CPU agent run.
                next_at = now + self._rng.expovariate(rate)
                drawn_rate = rate
            delay = next_at - now
            if delay > 0:
                # Sleep interruptibly so a rate change re-plans the
                # pending gap at the new rate.
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

    def _spawn_session(self, persona_id: str | None = None,
                       mid_session: bool = False) -> None:
        persona_id = persona_id or self._pick_persona_id()
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
                    # Seeded sessions join mid-cycle, so they get the
                    # offset; Poisson arrivals are already spread.
                    initial_phase_offset_enabled=mid_session,
                    reasoning_effort=self._reasoning_effort,
                    reasoning_overhead_tokens=self._reasoning_overhead_tokens,
                    start_mid_session=mid_session,
                    history_window_turns=self._history_window_turns,
                    context_limit_tokens=self._context_limit_tokens,
                    planner_timeouts=self._planner_timeouts,
                    system_prompt=self._system_prompt(persona.id),
                    system_prompt_in_input=self._system_prompt_in_input,
                    stream_usage=self._stream_usage,
                )
            finally:
                self.stats.sessions_done += 1
                if stats.sessions_completed > 0:
                    # Natural end only — see ArrivalStats.
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
