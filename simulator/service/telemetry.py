"""Live telemetry: bus events {topic, ts, data} over WebSocket."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..bus import BUS

router = APIRouter()


# ── live telemetry ────────────────────────────────────────────

@router.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    q = BUS.subscribe()

    # Two halves: the sender pushes bus events; the receiver only
    # exists to notice the client going away. Without it a closed
    # tab left the handler parked on q.get() forever, and its
    # subscriber queue kept filling (the bus drops oldest, so it
    # was a leak of one queue per stale tab, not a crash).
    async def _send() -> None:
        while True:
            event = await q.get()
            await ws.send_json(event)

    async def _recv() -> None:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return

    tasks = [asyncio.create_task(_send()), asyncio.create_task(_recv())]
    try:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError,
                                     asyncio.CancelledError):
                t.result()
    finally:
        for t in tasks:
            t.cancel()
        BUS.unsubscribe(q)
