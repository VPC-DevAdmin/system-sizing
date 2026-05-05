"""Tiered-timeout stream consumer.

Consumes an OpenAI-compatible streaming chat completion with three-tier
termination so a stuck or pathologically-slow request gets aborted
without tying up a slot indefinitely:

    Tier 1 (pre-TTFT):    no first token by ``pre_ttft_timeout_s`` →
                          ``error="ttft_stalled"``
                          (scheduler / admission deadlock)

    Tier 2 (inter-token): no new token for ``inter_token_timeout_s`` →
                          ``error="decode_stalled"``
                          (mid-stream worker hang)

    Tier 3 (hard):        total wall time exceeds ``hard_timeout_s`` →
                          ``error="hard_timeout"``
                          (slow-but-progressing past the budget;
                          engine is functional but overloaded)

Partial-progress is captured in the result regardless of tier — ttft_ms
(if reached), output_tokens (count received before abort), total_ms
(wall time at abort). Lets post-hoc analysis classify failure modes
and inspect tail shape without paying for 15 minutes of "still stuck"
data.

Used by both ``simulator.virtual_user`` and the standalone optimizer
script — single source of truth for the termination policy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


# Connection-establishment cap (separate from the tier policy). The
# initial ``client.chat.completions.create()`` call returns when the
# stream object is ready — usually milliseconds. A long delay here
# means the engine isn't accepting connections at all, which is a
# different failure than "stream emits nothing." Kept short so this
# doesn't soak up the per-cell budget.
CONNECT_TIMEOUT_S = 30.0


@dataclass
class StreamResult:
    """Outcome of a tiered streaming consume.

    On success: ``error is None``, ``ttft_ms`` and ``output_tokens``
    are set, ``total_ms`` is the wall time of the full completion.

    On any tier abort: ``error`` is one of ``"ttft_stalled"``,
    ``"decode_stalled"``, ``"hard_timeout"``. ``ttft_ms`` is set if
    we reached the first token before aborting (tier 2 or 3 cases),
    else None. ``output_tokens`` is the count of tokens received
    before abort. ``total_ms`` is the wall time at abort.

    On other failures (HTTP 5xx, connection refused, etc.):
    ``error`` is the exception class name, partial fields filled in
    where possible.
    """
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    output_tokens: int = 0
    output_text: str = ""
    token_timestamps: list = None  # list[[elapsed_ms, cum_tokens]]
    error: Optional[str] = None

    def __post_init__(self):
        if self.token_timestamps is None:
            self.token_timestamps = []


async def consume_with_tiers(
    create_stream: Callable[[], Any],
    *,
    pre_ttft_timeout_s: float,
    inter_token_timeout_s: float,
    hard_timeout_s: float,
    capture_token_timestamps: bool = False,
) -> StreamResult:
    """Open a streaming chat completion and consume it under the
    three-tier policy.

    ``create_stream`` is a zero-argument callable returning an
    awaitable that produces the stream object — typically
    ``lambda: client.chat.completions.create(..., stream=True)``.
    Decoupling stream creation from the consumer keeps this helper
    agnostic about which OpenAI-SDK call shape the caller wants.
    """
    submitted_at = time.monotonic()
    ttft_obs: Optional[float] = None
    output_text_parts: list[str] = []
    output_tokens = 0
    token_timestamps: list = []
    t_last_chunk = submitted_at

    # Stream creation: bounded by CONNECT_TIMEOUT_S, separate from the
    # tier policy. A failure here is a connection-level problem, not
    # a stuck-stream problem.
    try:
        stream = await asyncio.wait_for(
            create_stream(), timeout=CONNECT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return StreamResult(
            total_ms=(time.monotonic() - submitted_at) * 1000.0,
            error="connection_timeout",
        )
    except Exception as e:  # noqa: BLE001
        return StreamResult(
            total_ms=(time.monotonic() - submitted_at) * 1000.0,
            error=type(e).__name__,
        )

    aiter = stream.__aiter__()
    while True:
        now = time.monotonic()

        # Compute the deadline for the next chunk: minimum of
        #   * remaining tier-1 budget (only if no first token yet)
        #     OR remaining tier-2 budget (after first token)
        #   * remaining tier-3 budget (always applicable)
        if ttft_obs is None:
            tier_label = "ttft_stalled"
            tier_remaining = pre_ttft_timeout_s - (now - submitted_at)
        else:
            tier_label = "decode_stalled"
            tier_remaining = inter_token_timeout_s - (now - t_last_chunk)
        hard_remaining = hard_timeout_s - (now - submitted_at)
        if hard_remaining < tier_remaining:
            tier_remaining = hard_remaining
            tier_label = "hard_timeout"

        if tier_remaining <= 0:
            # We're already past the deadline — abort immediately.
            return _abort_result(
                tier_label, submitted_at, now, ttft_obs,
                output_tokens, output_text_parts, token_timestamps,
            )

        try:
            chunk = await asyncio.wait_for(
                aiter.__anext__(), timeout=tier_remaining,
            )
        except asyncio.TimeoutError:
            # Re-attribute by checking which budget is now exceeded.
            now2 = time.monotonic()
            attributed = _attribute_timeout(
                ttft_obs, submitted_at, t_last_chunk, now2,
                pre_ttft_timeout_s, inter_token_timeout_s, hard_timeout_s,
            )
            return _abort_result(
                attributed, submitted_at, now2, ttft_obs,
                output_tokens, output_text_parts, token_timestamps,
            )
        except StopAsyncIteration:
            break
        except Exception as e:  # noqa: BLE001
            now2 = time.monotonic()
            return StreamResult(
                ttft_ms=ttft_obs * 1000.0 if ttft_obs is not None else None,
                total_ms=(now2 - submitted_at) * 1000.0,
                output_tokens=output_tokens,
                output_text="".join(output_text_parts),
                token_timestamps=token_timestamps,
                error=type(e).__name__,
            )

        # Process the chunk.
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            now_chunk = time.monotonic()
            if ttft_obs is None:
                ttft_obs = now_chunk - submitted_at
            t_last_chunk = now_chunk
            output_text_parts.append(delta.content)
            output_tokens += 1
            if capture_token_timestamps:
                token_timestamps.append([
                    round((now_chunk - submitted_at) * 1000.0, 3),
                    output_tokens,
                ])

    # Stream finished cleanly.
    if ttft_obs is None:
        # No tokens emitted but no exception either — pathological
        # but technically not a tier abort.
        return StreamResult(
            total_ms=(time.monotonic() - submitted_at) * 1000.0,
            output_tokens=0,
            output_text="",
            token_timestamps=token_timestamps,
            error="no_tokens",
        )
    t_end = time.monotonic()
    return StreamResult(
        ttft_ms=ttft_obs * 1000.0,
        total_ms=(t_end - submitted_at) * 1000.0,
        output_tokens=output_tokens,
        output_text="".join(output_text_parts),
        token_timestamps=token_timestamps,
    )


def _attribute_timeout(
    ttft_obs: Optional[float],
    submitted_at: float,
    t_last_chunk: float,
    now: float,
    pre_ttft_timeout_s: float,
    inter_token_timeout_s: float,
    hard_timeout_s: float,
) -> str:
    """Decide which tier triggered an asyncio.TimeoutError. The
    wait_for fired with a per-chunk deadline equal to ``min(active
    tier remaining, tier-3 remaining)``; when it fires we re-evaluate
    the tier budgets to attribute correctly.

    Slack of 0.1 s on each comparison absorbs scheduling jitter — we
    don't want to mis-attribute a genuine tier-2 timeout to tier-3
    just because the wait_for fired 50 ms after the inter-token
    deadline."""
    elapsed_total = now - submitted_at
    if ttft_obs is None and elapsed_total >= pre_ttft_timeout_s - 0.1:
        return "ttft_stalled"
    if elapsed_total >= hard_timeout_s - 0.1:
        return "hard_timeout"
    return "decode_stalled"


def _abort_result(
    tier: str,
    submitted_at: float,
    now: float,
    ttft_obs: Optional[float],
    output_tokens: int,
    output_text_parts: list[str],
    token_timestamps: list,
) -> StreamResult:
    return StreamResult(
        ttft_ms=ttft_obs * 1000.0 if ttft_obs is not None else None,
        total_ms=(now - submitted_at) * 1000.0,
        output_tokens=output_tokens,
        output_text="".join(output_text_parts),
        token_timestamps=token_timestamps,
        error=tier,
    )
