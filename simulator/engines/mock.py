"""Mock engine (roadmap 2.3): a real in-process OpenAI-compatible SSE
server with a synthetic latency model.

Exists so the entire pipeline — AsyncOpenAI client, tiered stream
consumer, virtual users, measurement loop, DB, export — runs without
hardware: UI development, CI integration tests, and deterministic
measurement fixtures. Nothing downstream knows it's fake; the engine
serves real HTTP on ``cfg.port``.

Latency model (the knobs live on EngineConfig as ``mock_*``):

    load       = max(1, in_flight / mock_capacity_inflight)
    ttft       = mock_ttft_ms x load        (+- mock_jitter, uniform)
    per-token  = mock_tpot_ms x load        (+- mock_jitter)

Below capacity the engine is flat; past it, latency scales linearly
with oversubscription — which produces a clean, tunable capacity knee
for the persona SLA machinery to find. ``/metrics`` reports vLLM-style
gauges (running count, synthetic KV usage) so telemetry and the
prefix-cache scrape exercise their real paths.

The server runs uvicorn in a daemon thread with its own event loop, so
stream pacing is independent of the simulator's loop.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Optional

from .base import Engine

log = logging.getLogger(__name__)


def _estimate_tokens(messages: list[dict]) -> int:
    words = sum(
        len(str(m.get("content", "")).split()) for m in (messages or [])
    )
    return max(1, int(words / 0.75))


def _build_app(cfg, state: dict):
    """Starlette app closed over the shared mutable ``state``
    ({"in_flight": int, "completed": int}). Imported lazily so plain
    engine imports never pay for the web stack."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from starlette.routing import Route

    capacity = max(1, cfg.mock_capacity_inflight)
    served_name = cfg.served_model_name or cfg.model_id

    async def health(_request):
        return PlainTextResponse("ok")

    async def models(_request):
        return JSONResponse({
            "object": "list",
            "data": [{"id": served_name, "object": "model",
                      "created": 0, "owned_by": "mock"}],
        })

    async def metrics(_request):
        inflight = state["in_flight"]
        kv = min(95.0, 100.0 * 0.7 * inflight / capacity)
        queries = state["completed"] * 2 + inflight
        hits = int(queries * 0.6)
        return PlainTextResponse(
            f"vllm:num_requests_running {inflight}\n"
            f"vllm:num_requests_waiting {max(0, inflight - capacity)}\n"
            f"vllm:kv_cache_usage_perc {kv / 100.0}\n"
            f"vllm:prefix_cache_queries_total {queries}\n"
            f"vllm:prefix_cache_hits_total {hits}\n"
        )

    async def chat_completions(request):
        import asyncio

        body = await request.json()
        max_tokens = int(body.get("max_tokens") or 64)
        input_tokens = _estimate_tokens(body.get("messages") or [])
        rng = random.Random()

        def _jitter() -> float:
            j = cfg.mock_jitter
            return 1.0 + rng.uniform(-j, j) if j > 0 else 1.0

        created = int(time.time())
        chunk_id = f"chatcmpl-mock-{created}-{rng.randrange(1 << 30)}"

        def _chunk(delta: dict, finish: Optional[str] = None) -> str:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_name,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        async def stream():
            state["in_flight"] += 1
            try:
                # Load factor sampled at admission — one request keeps
                # its pacing for its lifetime (matches how a batch slot
                # behaves) and keeps the math reproducible.
                load = max(1.0, state["in_flight"] / capacity)
                # TTFT scales with prompt length a little: prefill work.
                ttft_s = (
                    (cfg.mock_ttft_ms + 0.05 * input_tokens)
                    / 1000.0 * load * _jitter()
                )
                await asyncio.sleep(ttft_s)
                yield _chunk({"role": "assistant", "content": ""})
                for _ in range(max_tokens):
                    await asyncio.sleep(
                        cfg.mock_tpot_ms / 1000.0 * load * _jitter()
                    )
                    yield _chunk({"content": "tok "})
                yield _chunk({}, finish="stop")
                yield "data: [DONE]\n\n"
                state["completed"] += 1
            finally:
                state["in_flight"] -= 1

        return StreamingResponse(stream(), media_type="text/event-stream")

    return Starlette(routes=[
        Route("/health", health),
        Route("/v1/models", models),
        Route("/metrics", metrics),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ])


class MockEngine(Engine):
    """In-process OpenAI-compatible SSE server with a synthetic knee."""

    def __init__(self, engine_config):
        super().__init__(engine_config)
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self.state = {"in_flight": 0, "completed": 0}

    def launch(self, log_dir: str | Path = "runs") -> None:
        if self._thread is not None:
            raise RuntimeError("Mock engine already launched")
        import uvicorn

        app = _build_app(self.cfg, self.state)
        config = uvicorn.Config(
            app, host=self.cfg.host, port=self.cfg.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="mock-engine", daemon=True,
        )
        self._thread.start()

        deadline = time.time() + 30
        while time.time() < deadline:
            if self._server.started:
                log.info(
                    "Mock engine serving on %s (capacity_inflight=%d, "
                    "ttft=%.0fms, tpot=%.0fms)",
                    self.base_url, self.cfg.mock_capacity_inflight,
                    self.cfg.mock_ttft_ms, self.cfg.mock_tpot_ms,
                )
                return
            if not self._thread.is_alive():
                raise RuntimeError("Mock engine server thread died on startup")
            time.sleep(0.05)
        raise TimeoutError("Mock engine did not start within 30s")

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    @property
    def pid(self) -> Optional[int]:
        return None   # in-process; engine-RSS rollup degrades gracefully

    # Base abstract API — unused (launch is overridden).
    def _build_command(self) -> list[str]:
        raise NotImplementedError("MockEngine runs in-process")

    def _build_env(self) -> dict[str, str]:
        raise NotImplementedError("MockEngine runs in-process")
