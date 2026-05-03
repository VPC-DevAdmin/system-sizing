"""Virtual user runtime.

One async task per simulated user, executing a persona's session/turn lifecycle
against an OpenAI-compatible endpoint with streaming responses.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from .personas import Persona
from .tokenizer_corpus import TokenCorpus

log = logging.getLogger(__name__)


@dataclass
class TurnEvent:
    """One completed request — captured for analysis."""
    user_id: str
    persona_id: str
    session_id: str
    turn_index: int
    submitted_at_ms: int
    completed_at_ms: int
    ttft_ms: float
    tpot_ms: float
    end_to_end_ms: float
    input_tokens: int
    history_tokens: int
    output_tokens: int
    in_flight_at_submit: int
    in_flight_avg_during: float = 0.0
    in_flight_peak_during: int = 0
    error: Optional[str] = None
    persona: Persona = field(default=None, repr=False)  # for SLA verdict at capture time

    def ttft_violation(self) -> bool:
        return self.ttft_ms / 1000.0 > self.persona.ttft_floor_seconds

    def tpot_violation(self) -> bool:
        return self.tpot_ms > self.persona.tpot_floor_ms


@dataclass
class UserStats:
    user_id: str
    persona_id: str
    spawned_at_ms: int
    sessions_target: int
    sessions_completed: int = 0
    turns_total: int = 0
    pool_size_at_spawn: int = 0
    replaced_user_id: Optional[str] = None
    terminated_at_ms: Optional[int] = None


class SharedState:
    """In-flight tracking and event publishing."""

    def __init__(self):
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._completed = 0
        self._errors = 0
        self.events: asyncio.Queue[TurnEvent] = asyncio.Queue()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def errors(self) -> int:
        return self._errors

    async def submit(self) -> int:
        async with self._lock:
            self._in_flight += 1
            return self._in_flight

    async def complete(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._completed += 1

    async def fail(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._errors += 1


def _now_ms() -> int:
    return int(time.time() * 1000)


async def run_virtual_user(
    persona: Persona,
    rng: random.Random,
    client: AsyncOpenAI,
    model_id: str,
    corpus: TokenCorpus,
    state: SharedState,
    stats: UserStats,
    request_timeout_s: int,
    cancel_event: asyncio.Event,
) -> None:
    """Run one virtual user to completion (or until cancelled)."""
    sessions_target = stats.sessions_target
    history_messages: list[dict] = []
    history_token_count = 0

    try:
        for session_index in range(sessions_target):
            if cancel_event.is_set():
                break
            history_messages = []
            history_token_count = 0
            session_id = uuid.uuid4().hex
            n_turns = persona.turns_per_session.sample_int(rng)

            for turn in range(n_turns):
                if cancel_event.is_set():
                    return

                input_tokens_target = persona.input_tokens.sample_int(rng)
                output_tokens_target = persona.output_tokens.sample_int(rng)
                query_text = corpus.make_text(input_tokens_target, rng)

                messages = list(history_messages) + [
                    {"role": "user", "content": query_text}
                ]

                in_flight_at_submit = await state.submit()
                submitted_at = time.monotonic()
                submitted_at_ms = _now_ms()
                ttft_obs: float | None = None
                output_text_parts: list[str] = []
                output_tokens = 0
                error: str | None = None

                try:
                    stream = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model_id,
                            messages=messages,
                            max_tokens=output_tokens_target,
                            stream=True,
                            temperature=0.7,
                        ),
                        timeout=request_timeout_s,
                    )
                    async for chunk in stream:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta and delta.content:
                            if ttft_obs is None:
                                ttft_obs = time.monotonic() - submitted_at
                            output_text_parts.append(delta.content)
                            output_tokens += 1
                except asyncio.TimeoutError:
                    error = "timeout"
                except Exception as e:  # noqa: BLE001
                    error = type(e).__name__

                completed_at = time.monotonic()
                e2e_ms = (completed_at - submitted_at) * 1000.0

                if error or ttft_obs is None or output_tokens < 1:
                    await state.fail()
                    if error is None:
                        error = "no_tokens"
                    log.debug("user %s turn failed: %s", stats.user_id, error)
                    # Treat as session-aborting for safety
                    break

                ttft_ms = ttft_obs * 1000.0
                tpot_ms = (e2e_ms - ttft_ms) / max(1, output_tokens - 1)

                response_text = "".join(output_text_parts)
                event = TurnEvent(
                    user_id=stats.user_id,
                    persona_id=persona.id,
                    session_id=session_id,
                    turn_index=turn,
                    submitted_at_ms=submitted_at_ms,
                    completed_at_ms=_now_ms(),
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    end_to_end_ms=e2e_ms,
                    input_tokens=corpus.count(query_text),
                    history_tokens=history_token_count,
                    output_tokens=output_tokens,
                    in_flight_at_submit=in_flight_at_submit,
                    persona=persona,
                )

                await state.complete()
                await state.events.put(event)

                # Append turn to history (byte-identical reuse for prefix cache)
                history_messages.append({"role": "user", "content": query_text})
                history_messages.append({"role": "assistant", "content": response_text})
                history_token_count += event.input_tokens + output_tokens
                stats.turns_total += 1

                if turn < n_turns - 1:
                    think = persona.think_time_seconds.sample(rng)
                    try:
                        await asyncio.wait_for(cancel_event.wait(), timeout=think)
                        return  # cancel fired
                    except asyncio.TimeoutError:
                        pass

            stats.sessions_completed += 1

            if session_index < sessions_target - 1:
                gap = persona.inter_session_gap_seconds.sample(rng)
                try:
                    await asyncio.wait_for(cancel_event.wait(), timeout=gap)
                    return
                except asyncio.TimeoutError:
                    pass
    finally:
        stats.terminated_at_ms = _now_ms()
