"""Pool manager — maintains the active virtual user set at the target size."""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI

from .personas import Cohort, PERSONAS
from .tokenizer_corpus import TokenCorpus
from .virtual_user import (
    SharedState,
    UserStats,
    run_virtual_user,
    _now_ms,
)

log = logging.getLogger(__name__)


@dataclass
class _ActiveUser:
    stats: UserStats
    task: asyncio.Task
    cancel_event: asyncio.Event


class PoolManager:
    """Spawns and replaces virtual users to maintain a target pool size."""

    def __init__(
        self,
        cohort: Cohort,
        client: AsyncOpenAI,
        model_id: str,
        corpus: TokenCorpus,
        state: SharedState,
        request_timeout_s: int,
        rng_seed: int = 0xC0FFEE,
        on_user_terminated=None,
        capture_token_timestamps: bool = False,
    ):
        self.cohort = cohort
        self.client = client
        self.model_id = model_id
        self.corpus = corpus
        self.state = state
        self.request_timeout_s = request_timeout_s
        self._rng = random.Random(rng_seed)
        self._users: dict[str, _ActiveUser] = {}
        self._target_size = 0
        self._reaper_task: asyncio.Task | None = None
        self._stopped = False
        self._on_user_terminated = on_user_terminated
        self._capture_token_timestamps = capture_token_timestamps

    @property
    def target_size(self) -> int:
        return self._target_size

    @property
    def active_count(self) -> int:
        return len(self._users)

    def all_user_stats(self) -> list[UserStats]:
        return [u.stats for u in self._users.values()]

    def start(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def stop(self) -> None:
        self._stopped = True
        for u in list(self._users.values()):
            u.cancel_event.set()
        # Give them a moment to exit cleanly
        if self._users:
            await asyncio.gather(
                *(u.task for u in self._users.values()), return_exceptions=True
            )
        self._users.clear()
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except (asyncio.CancelledError, Exception):
                pass

    async def set_target_size(self, n: int) -> None:
        self._target_size = n
        # Spawn up to target
        while not self._stopped and len(self._users) < n:
            self._spawn_one(replaced_user_id=None)
        # If shrinking, cancel surplus (rare in adaptive ramping)
        if len(self._users) > n:
            surplus = len(self._users) - n
            for u in list(self._users.values())[:surplus]:
                u.cancel_event.set()

    # -- Internals -----------------------------------------------------------

    def _pick_persona_id(self) -> str:
        items = list(self.cohort.persona_weights.items())
        weights = [w for _, w in items]
        ids = [pid for pid, _ in items]
        return self._rng.choices(ids, weights=weights, k=1)[0]

    def _spawn_one(self, replaced_user_id: str | None) -> None:
        persona_id = self._pick_persona_id()
        persona = PERSONAS[persona_id]
        user_id = uuid.uuid4().hex
        sessions_target = persona.sessions_before_leaving.sample_int(self._rng)
        stats = UserStats(
            user_id=user_id,
            persona_id=persona_id,
            spawned_at_ms=_now_ms(),
            sessions_target=sessions_target,
            pool_size_at_spawn=self._target_size,
            replaced_user_id=replaced_user_id,
        )
        cancel_event = asyncio.Event()
        # Each user gets its own RNG stream, derived from the pool's RNG so runs are reproducible
        sub_rng = random.Random(self._rng.random())
        task = asyncio.create_task(
            run_virtual_user(
                persona=persona,
                rng=sub_rng,
                client=self.client,
                model_id=self.model_id,
                corpus=self.corpus,
                state=self.state,
                stats=stats,
                request_timeout_s=self.request_timeout_s,
                cancel_event=cancel_event,
                capture_token_timestamps=self._capture_token_timestamps,
            ),
            name=f"vu:{persona_id}:{user_id[:8]}",
        )
        self._users[user_id] = _ActiveUser(stats=stats, task=task, cancel_event=cancel_event)

    async def _reaper_loop(self) -> None:
        """Periodically detect terminated users and replace them."""
        try:
            while not self._stopped:
                await asyncio.sleep(0.5)
                done = [uid for uid, u in self._users.items() if u.task.done()]
                for uid in done:
                    u = self._users.pop(uid)
                    if self._on_user_terminated:
                        try:
                            self._on_user_terminated(u.stats)
                        except Exception:
                            log.exception("on_user_terminated callback failed")
                    if u.task.exception() is not None:
                        log.warning("VU task error: %s", u.task.exception())
                # Refill to target (replaces leavers)
                while not self._stopped and len(self._users) < self._target_size:
                    self._spawn_one(replaced_user_id=None)
        except asyncio.CancelledError:
            pass
