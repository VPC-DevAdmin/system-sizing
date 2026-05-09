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
from .streaming import consume_with_tiers
from .tokenizer_corpus import TokenCorpus

log = logging.getLogger(__name__)


# Reasoning-model token overhead. When the engine is a reasoning model
# (engine.reasoning=True in YAML), the chain-of-thought is streamed
# BEFORE the answer and consumes ``max_tokens`` budget. Personas
# describe answer length (the user-facing metric); the wire-level
# budget needs headroom on top so reasoning doesn't crowd out content.
#
# Without this bump, short-output personas (quick_lookup median ~30
# tokens) hit the May 2026 GPT-OSS symptom: every request truncates
# during reasoning with finish_reason=length, content never starts,
# the simulator records ``no_content_tokens`` errors on every turn,
# and the convergence detector deadlocks at warmup waiting for
# completions that never arrive.
#
# Values calibrated from GPT-OSS-20B observed reasoning-phase length
# at each effort level. Conservative — better to over-budget and let
# the engine emit content than under-budget and lose the entire turn.
#
# 2026-05-08 calibration update: AMD analyst_team / document_qa
# turns at reasoning_effort=medium emitted 532 reasoning tokens on
# the no-content-tokens cases (i.e. reasoning was BIGGER than the
# 250-token budget we'd given). Bumped medium 250 → 600 to cover
# observed reasoning-phase lengths with margin. Other levels scaled
# proportionally to stay monotonic and roughly 2× of the original
# calibration. Real reasoning models on real prompts use a lot more
# than the early back-of-envelope estimates suggested.
REASONING_OVERHEAD_TOKENS = {
    "minimal": 100,
    "low":     250,
    "medium":  600,
    "high":    1200,
}


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
    # Tier-3 opt-in: per-emitted-chunk (elapsed_ms_from_submit, cumulative_tokens).
    # Empty list when capture is off. Persisted as JSON on the turn_events row.
    token_timestamps: list = field(default_factory=list, repr=False)
    # Reasoning-model support. ``ttft_ms`` always tracks the first
    # user-visible token (reasoning OR content); ``ttfct_ms`` tracks
    # specifically when the content/answer phase started — equals
    # ttft_ms for non-reasoning models. ``reasoning_tokens`` counts
    # chain-of-thought chunks separately from content output_tokens.
    ttfct_ms: float = 0.0
    reasoning_tokens: int = 0

    def ttft_violation(self) -> bool:
        """Hard SLA violation — past the FAILURE threshold. This is
        what gates ``capacity_status`` (the buyer-page capacity
        number). A failed request (timeout, HTTP error, empty stream)
        is always a violation regardless of timing — the user waited
        and didn't get a useful response."""
        if self.error is not None:
            return True
        return self.ttft_ms / 1000.0 > self.persona.ttft_failure_seconds

    def tpot_violation(self) -> bool:
        if self.error is not None:
            return True
        return self.tpot_ms > self.persona.tpot_failure_ms

    def ttft_target_miss(self) -> bool:
        """Soft target miss — past the TARGET threshold (looser than
        failure). Informational quality signal, not a capacity gate.
        Errors count as target misses too."""
        if self.error is not None:
            return True
        return self.ttft_ms / 1000.0 > self.persona.ttft_target_seconds

    def tpot_target_miss(self) -> bool:
        if self.error is not None:
            return True
        return self.tpot_ms > self.persona.tpot_target_ms


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
        # Live progress for the in-flight measurement step. Both
        # written from the single ``run_measurement_step`` task — no
        # lock needed; the snapshot recorder reads them directly.
        # ``step_target_samples`` is 0 outside of a measurement window
        # (warmup, ramp, idle) so the dashboard can render "—".
        self.step_samples: int = 0
        self.step_target_samples: int = 0

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
    capture_token_timestamps: bool = False,
    initial_phase_offset_enabled: bool = True,
    reasoning_effort: str | None = None,
) -> None:
    """Run one virtual user to completion (or until cancelled)."""
    sessions_target = stats.sessions_target
    history_messages: list[dict] = []
    history_token_count = 0

    try:
        # Random initial phase offset: sleep for a uniform-random
        # fraction of one think-time sample before issuing the first
        # request. Without this, all users spawned in the same ramp
        # burst hit /chat on the same event-loop tick and stay
        # synchronised for many cycles. With it, users land
        # distributed across the request/think cycle from the moment
        # they start.
        if initial_phase_offset_enabled:
            sample = (
                persona.read_time_seconds.sample(rng)
                + persona.active_think_seconds.sample(rng)
            )
            offset = sample * rng.random()
            if offset > 0:
                try:
                    await asyncio.wait_for(cancel_event.wait(), timeout=offset)
                    return  # cancelled mid-offset
                except asyncio.TimeoutError:
                    pass

        # Per-session respawn model: each virtual user runs ONE
        # session (multi-turn within the session, history accumulates),
        # then terminates. The pool manager spawns a replacement.
        #
        # This replaces the prior model where a user ran N sessions
        # with inter-session gaps. The old model represented "long-
        # lived authenticated user" lifecycle, which doesn't reflect
        # what the engine actually sees — from the engine's
        # perspective, a user returning after a long gap is
        # indistinguishable from a fresh user (KV is evicted, no
        # shared content between sessions). Modeling each session as
        # a separate virtual user lets pool_size mean "active
        # concurrent sessions," which is the buyer-page-intuitive
        # answer.
        #
        # ``sessions_before_leaving`` and ``inter_session_gap_seconds``
        # on the Persona are deprecated — kept on the dataclass for
        # back-compat with older configs but no longer drive runtime.
        if not cancel_event.is_set():
            history_messages = []
            history_token_count = 0
            session_id = uuid.uuid4().hex
            n_turns = persona.turns_per_session.sample_int(rng)

            for turn in range(n_turns):
                if cancel_event.is_set():
                    return

                input_tokens_target = persona.input_tokens.sample_int(rng)
                output_tokens_target = persona.output_tokens.sample_int(rng)
                # Reasoning models consume budget on chain-of-thought
                # before content streams. Bump the wire-level
                # max_tokens by the per-effort overhead so the answer
                # fits even for short-output personas (quick_lookup
                # median ~30 tokens). Persona's output_tokens stays
                # the user-facing answer-length target.
                if reasoning_effort:
                    output_tokens_target += REASONING_OVERHEAD_TOKENS.get(
                        reasoning_effort, 250,
                    )
                query_text = corpus.make_text(input_tokens_target, rng)

                messages = list(history_messages) + [
                    {"role": "user", "content": query_text}
                ]

                in_flight_at_submit = await state.submit()
                submitted_at = time.monotonic()
                submitted_at_ms = _now_ms()

                # Tiered streaming consume. The persona owns the
                # tier budgets — they're scaled from its SLA floors
                # so a stricter persona gets stricter abort policy.
                # ``request_timeout_s`` from config is no longer the
                # primary stream cap; we keep it as an outer
                # safety only via the persona's hard_timeout_s
                # default (override per-persona to tighten).
                # Reasoning-model passthrough: when configured, the
                # OpenAI ``extra_body`` lets us pass a non-OpenAI-
                # canonical ``reasoning_effort`` field through the
                # SDK to vLLM. vLLM ignores it for non-reasoning
                # models, so it's safe to plumb unconditionally — but
                # we only set it when configured to keep request
                # bodies minimal.
                extra_body = (
                    {"reasoning_effort": reasoning_effort}
                    if reasoning_effort else None
                )
                stream_result = await consume_with_tiers(
                    create_stream=lambda: client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=output_tokens_target,
                        stream=True,
                        temperature=0.7,
                        extra_body=extra_body,
                    ),
                    pre_ttft_timeout_s=persona.pre_ttft_timeout_s,
                    inter_token_timeout_s=persona.inter_token_timeout_s,
                    hard_timeout_s=min(persona.hard_timeout_s, request_timeout_s),
                    capture_token_timestamps=capture_token_timestamps,
                )
                ttft_obs: float | None = (
                    stream_result.ttft_ms / 1000.0
                    if stream_result.ttft_ms is not None else None
                )
                output_tokens = stream_result.output_tokens
                output_text_parts: list[str] = (
                    [stream_result.output_text] if stream_result.output_text else []
                )
                token_timestamps: list[list[float]] = stream_result.token_timestamps
                error: str | None = stream_result.error

                completed_at = time.monotonic()
                e2e_ms = (completed_at - submitted_at) * 1000.0

                if error is not None:
                    await state.fail()
                    log.debug("user %s turn failed: %s", stats.user_id, error)
                    # Synthetic TurnEvent so the failure registers
                    # in the SLA framework. Carries whatever partial
                    # progress we captured before aborting — ttft if
                    # we reached the first token (tier 2/3 cases),
                    # output_tokens received before the abort, etc.
                    # Lets post-hoc analysis classify failure modes
                    # ("did the request even start?" vs "did it
                    # stall mid-stream?") via the ``error`` category.
                    partial_ttft_ms = stream_result.ttft_ms
                    partial_tpot_ms = 0.0
                    # Honest TPOT from the partial stream — useful for
                    # distinguishing "tier 2 stalled hard" vs "tier 3
                    # tripped on slow-but-progressing." Counts reasoning
                    # tokens too: when the engine is a reasoning model,
                    # ``no_content_tokens`` errors still streamed real
                    # decode work (just no content), and we want the
                    # rate captured. For non-reasoning models
                    # reasoning_tokens=0 so this collapses to the
                    # original output_tokens-only formula.
                    partial_total = (
                        output_tokens + stream_result.reasoning_tokens
                    )
                    if partial_ttft_ms is not None and partial_total > 1:
                        decode_ms = e2e_ms - partial_ttft_ms
                        partial_tpot_ms = decode_ms / max(1, partial_total - 1)
                    failed_ttft = (
                        partial_ttft_ms
                        if partial_ttft_ms is not None else e2e_ms
                    )
                    failed_event = TurnEvent(
                        user_id=stats.user_id,
                        persona_id=persona.id,
                        session_id=session_id,
                        turn_index=turn,
                        submitted_at_ms=submitted_at_ms,
                        completed_at_ms=_now_ms(),
                        ttft_ms=failed_ttft,
                        tpot_ms=partial_tpot_ms,
                        end_to_end_ms=e2e_ms,
                        input_tokens=corpus.count(query_text),
                        history_tokens=history_token_count,
                        output_tokens=output_tokens,
                        in_flight_at_submit=in_flight_at_submit,
                        persona=persona,
                        error=error,
                        token_timestamps=token_timestamps,
                        # ttfct: reasoning may have started but content
                        # never did → fall back to ttft (so non-reasoning
                        # paths see ttfct == ttft trivially).
                        ttfct_ms=(
                            stream_result.ttfct_ms
                            if stream_result.ttfct_ms is not None
                            else failed_ttft
                        ),
                        reasoning_tokens=stream_result.reasoning_tokens,
                    )
                    await state.events.put(failed_event)
                    # Treat as session-aborting for safety
                    break

                ttft_ms = ttft_obs * 1000.0
                # TPOT measured against the user-VISIBLE token stream:
                # reasoning + content tokens. For non-reasoning models
                # reasoning_tokens=0 so this collapses to the previous
                # formula (output_tokens - 1).
                total_visible_tokens = (
                    output_tokens + stream_result.reasoning_tokens
                )
                tpot_ms = (e2e_ms - ttft_ms) / max(1, total_visible_tokens - 1)
                # ttfct_ms: time to first content token. For non-reasoning
                # models this equals ttft_ms (no reasoning phase).
                ttfct_ms = (
                    stream_result.ttfct_ms
                    if stream_result.ttfct_ms is not None
                    else ttft_ms
                )

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
                    token_timestamps=token_timestamps,
                    ttfct_ms=ttfct_ms,
                    reasoning_tokens=stream_result.reasoning_tokens,
                )

                await state.complete()
                await state.events.put(event)

                # Append turn to history (byte-identical reuse for prefix cache)
                history_messages.append({"role": "user", "content": query_text})
                history_messages.append({"role": "assistant", "content": response_text})
                history_token_count += event.input_tokens + output_tokens
                stats.turns_total += 1

                if turn < n_turns - 1:
                    # Post-response delay = residual reading time
                    # (what the user hasn't caught up to when the
                    # stream ended) + active deliberation. Sampled
                    # independently and summed; both are LogNormal
                    # so the sum is a heavy-tail bimodal-ish
                    # distribution with the right central tendency.
                    delay = (
                        persona.read_time_seconds.sample(rng)
                        + persona.active_think_seconds.sample(rng)
                    )
                    try:
                        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
                        return  # cancel fired
                    except asyncio.TimeoutError:
                        pass

            stats.sessions_completed += 1
        # Session done — return so the pool manager spawns a
        # replacement user. inter_session_gap behavior intentionally
        # removed; see comment above the session block.
    finally:
        stats.terminated_at_ms = _now_ms()
