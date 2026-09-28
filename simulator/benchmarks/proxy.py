"""Loopback streaming router with isolated workers and per-replica evidence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.requests import ClientDisconnect

from .process import module_argv
from .storage import read_json, write_json


def worker_count(concurrency: int) -> int:
    return min(32, max(1, (os.cpu_count() or 4) // 4), max(1, math.ceil(concurrency / 256)))


def _stats(urls):
    return {
        "routed": dict.fromkeys(urls, 0),
        "active": 0,
        "errors": 0,
        "client_disconnects": 0,
        "transport_errors": {},
        "upstream_keepalive_expiry_s": 2,
        "max_loop_lag_ms": 0.0,
        "loop_lag_samples": 0,
        "loop_lag_over_100ms": 0,
        "lag_sampling_scope": "active_requests",
        "policy": "round_robin_single_turn",
        "proxy_included": True,
    }


class _ForwardedStream(StreamingResponse):
    """Own the upstream response through headers, streaming and cancellation."""

    def __init__(self, upstream, stats):
        super().__init__(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )
        self.upstream = upstream
        self.stats = stats

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await self.upstream.aclose()
            finally:
                self.stats["active"] -= 1


@asynccontextmanager
async def _serve(urls, model, api_key, timeout_s, sock, stats, offset=0):
    app = FastAPI()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        limits=httpx.Limits(
            max_connections=None, max_keepalive_connections=None, keepalive_expiry=2
        ),
    )
    cursor = offset

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def forward(request: Request):
        nonlocal cursor
        url = urls[cursor % len(urls)]
        cursor += 1
        stats["active"] += 1
        try:
            data = await request.body()
            payload = json.loads(data)
            stats["request_parameters"] = {
                k: v for k, v in payload.items() if k not in ("messages", "prompt")
            }
            req = client.build_request(
                "POST",
                url + "/v1/chat/completions",
                content=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            stats["routed"][url] += 1
            upstream = await client.send(req, stream=True)
        except ClientDisconnect:
            stats["active"] -= 1
            stats["client_disconnects"] += 1
            return Response(status_code=499)
        except asyncio.CancelledError:
            stats["active"] -= 1
            raise
        except (httpx.HTTPError, ValueError) as error:
            stats["active"] -= 1
            stats["errors"] += 1
            key = type(error).__name__
            stats["transport_errors"][key] = stats["transport_errors"].get(key, 0) + 1
            return Response("Upstream request failed", status_code=502)
        if upstream.status_code >= 400:
            stats["errors"] += 1

        return _ForwardedStream(upstream, stats)

    server = uvicorn.Server(
        uvicorn.Config(
            app, log_level="error", access_log=False, lifespan="off", timeout_graceful_shutdown=5
        )
    )
    server.install_signal_handlers = lambda: None
    if hasattr(server, "capture_signals"):
        server.capture_signals = contextlib.nullcontext
    task = asyncio.create_task(server.serve(sockets=[sock]))

    async def lag():
        while True:
            start = time.monotonic()
            await asyncio.sleep(0.25)
            delay = (time.monotonic() - start - 0.25) * 1000
            stats["max_loop_lag_ms"] = max(stats["max_loop_lag_ms"], delay)
            if stats["active"]:
                stats["loop_lag_samples"] += 1
                stats["loop_lag_over_100ms"] += int(delay > 100)

    lag_task = asyncio.create_task(lag())
    try:
        for _ in range(400):
            if task.done():
                await task
                raise RuntimeError("Routing worker exited before readiness")
            if server.started:
                break
            await asyncio.sleep(0.025)
        if not server.started:
            raise RuntimeError("Routing worker did not become ready")
        yield server
    finally:
        server.should_exit = True
        lag_task.cancel()
        await asyncio.gather(lag_task, return_exceptions=True)
        try:
            await asyncio.wait_for(task, 10)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.aclose()


class ProcessStats(dict):
    def __init__(self, urls, directory, processes):
        super().__init__(_stats(urls))
        self.directory = directory
        self.processes = processes
        self.generation = 0

    async def refresh(self, reset=False):
        self.generation += 1
        write_json(self.directory / "control.json", {"generation": self.generation, "reset": reset})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if any(p.returncode is not None for p in self.processes):
                raise RuntimeError("A routing worker exited; inspect router logs")
            records = [
                read_json(self.directory / f"stats-{i}.json")
                if (self.directory / f"stats-{i}.json").exists()
                else {}
                for i in range(len(self.processes))
            ]
            if all(r.get("generation") == self.generation for r in records):
                self.update(
                    routed={u: sum(r["routed"][u] for r in records) for u in self["routed"]},
                    active=sum(r["active"] for r in records),
                    errors=sum(r["errors"] for r in records),
                    client_disconnects=sum(r["client_disconnects"] for r in records),
                    max_loop_lag_ms=max(r["max_loop_lag_ms"] for r in records),
                    loop_lag_samples=sum(r["loop_lag_samples"] for r in records),
                    loop_lag_over_100ms=sum(r["loop_lag_over_100ms"] for r in records),
                    max_worker_lag_over_100ms_fraction=max(
                        r["loop_lag_over_100ms"] / max(1, r["loop_lag_samples"]) for r in records
                    ),
                    workers=len(records),
                )
                kinds = {kind for record in records for kind in record["transport_errors"]}
                self["transport_errors"] = {
                    kind: sum(record["transport_errors"].get(kind, 0) for record in records)
                    for kind in kinds
                }
                parameters = [r["request_parameters"] for r in records if "request_parameters" in r]
                if parameters:
                    self["request_parameters"] = parameters[-1]
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("Routing workers did not acknowledge evidence snapshot")


async def refresh_routing(stats, *, reset=False):
    if isinstance(stats, ProcessStats):
        await stats.refresh(reset)
    elif reset:
        stats.update(
            routed=dict.fromkeys(stats["routed"], 0),
            errors=0,
            client_disconnects=0,
            max_loop_lag_ms=0.0,
            loop_lag_samples=0,
            loop_lag_over_100ms=0,
            transport_errors={},
        )


@asynccontextmanager
async def routing_endpoint(engine, timeout_s: float, *, workers=1, directory=None):
    urls = [u.rstrip("/").removesuffix("/v1") for u in engine.replica_urls]
    if not urls or len(set(urls)) != len(urls):
        raise ValueError("No distinct replica endpoints")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8192)
    port = sock.getsockname()[1]
    processes, streams = [], []
    try:
        if workers == 1:
            stats = _stats(urls)
            async with _serve(urls, engine.api_model_name, engine.api_key, timeout_s, sock, stats):
                yield f"http://127.0.0.1:{port}", stats
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(
                directory / "config.json",
                {
                    "urls": urls,
                    "model": engine.api_model_name,
                    "timeout_s": timeout_s,
                    "workers": workers,
                },
            )
            write_json(directory / "control.json", {"generation": 0, "reset": False})
            for i in range(workers):
                stream = (directory / f"worker-{i}.log").open("wb")
                streams.append(stream)
                proc = await asyncio.create_subprocess_exec(
                    *module_argv(
                        str(Path(__file__).resolve().parents[2]),
                        "simulator.benchmarks.proxy",
                        str(directory),
                        str(i),
                        str(sock.fileno()),
                        str(os.getpid()),
                    ),
                    pass_fds=(sock.fileno(),),
                    env={**os.environ, "CAPSIM_ROUTER_API_KEY": engine.api_key},
                    stdout=stream,
                    stderr=asyncio.subprocess.STDOUT,
                )
                processes.append(proc)
            stats = ProcessStats(urls, directory, processes)
            await stats.refresh()
            yield f"http://127.0.0.1:{port}", stats
    finally:
        for proc in processes:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
        for proc in processes:
            try:
                await asyncio.wait_for(proc.wait(), 10)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        for stream in streams:
            stream.close()
        sock.close()


async def child(directory, index, fd, parent):
    cfg = read_json(directory / "config.json")
    stats = _stats(cfg["urls"])
    sock = socket.socket(fileno=fd)
    generation = -1
    try:
        async with _serve(
            cfg["urls"],
            cfg["model"],
            os.environ.pop("CAPSIM_ROUTER_API_KEY", "EMPTY"),
            cfg["timeout_s"],
            sock,
            stats,
            index,
        ) as server:
            while os.getppid() == parent and not server.should_exit:
                control = read_json(directory / "control.json")
                if control["generation"] != generation:
                    generation = control["generation"]
                    if control["reset"]:
                        if stats["active"]:
                            raise RuntimeError("Cannot reset router with active requests")
                        stats.update(
                            routed=dict.fromkeys(cfg["urls"], 0),
                            errors=0,
                            client_disconnects=0,
                            max_loop_lag_ms=0.0,
                            loop_lag_samples=0,
                            loop_lag_over_100ms=0,
                            transport_errors={},
                        )
                write_json(directory / f"stats-{index}.json", {**stats, "generation": generation})
                await asyncio.sleep(0.25)
    finally:
        sock.close()


if __name__ == "__main__":
    asyncio.run(child(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])))
