"""Fixed offered-load tenant replay, including queueing and rejected admissions."""

import asyncio
import json
import math
import time

import httpx

from .driver import percentile, request_one, summarize


async def measure_tenant(engine, corpus, spec, path, start_at, offset=0, rate=None):
    rate = spec.tenant_requests_per_second if rate is None else rate
    count = math.ceil(spec.duration_s * rate)
    rows = []
    route_pending = [0] * len(engine.replica_urls)
    pending = set()
    all_tasks = []
    failures = []
    slots = asyncio.Semaphore(spec.concurrency)
    with path.open("w") as out:

        def save(row):
            rows.append(row)
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
            if out.tell() > 1024**3:
                raise RuntimeError("Tenant evidence exceeded one GiB")

        async with httpx.AsyncClient(
            timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
        ) as client:

            async def send(index, arrival):
                async with slots:
                    queue = max(0, time.monotonic() - arrival)
                    item = corpus[(offset + index) % len(corpus)]
                    route = index % len(engine.replica_urls)
                    if getattr(engine, "pool_routing", False):
                        route = min(
                            range(len(route_pending)),
                            key=lambda i: (route_pending[i], (i - index) % len(route_pending)),
                        )
                    route_pending[route] += 1
                    try:
                        row = await request_one(
                            client,
                            engine.replica_urls[route],
                            engine.api_model_name,
                            item,
                            spec,
                            str(index),
                            engine.api_key,
                        )
                    finally:
                        route_pending[route] -= 1
                    row["serving_pool"] = route
                    row["queue_s"] = queue
                    row["scheduled_offset_s"] = index / rate
                    row["elapsed_s"] += queue
                    if row["ttft_s"] is not None:
                        row["ttft_s"] += queue
                    save(row)

            try:
                for i in range(count):
                    arrival = start_at + i / rate
                    await asyncio.sleep(max(0, arrival - time.monotonic()))
                    if failures:
                        raise failures[0]
                    if len(pending) >= spec.concurrency * 4:
                        save(
                            {
                                "request_id": str(i),
                                "transport_ok": False,
                                "correct": False,
                                "elapsed_s": 0,
                                "ttft_s": None,
                                "usage": None,
                                "error": "admission_limit",
                                "scheduled_offset_s": i / rate,
                            }
                        )
                        continue
                    task = asyncio.create_task(send(i, arrival))
                    pending.add(task)
                    all_tasks.append(task)

                    def completed(task):
                        pending.discard(task)
                        if not task.cancelled() and task.exception():
                            failures.append(task.exception())

                    task.add_done_callback(completed)
                if all_tasks:
                    await asyncio.gather(*all_tasks)
                await asyncio.sleep(max(0, start_at + spec.duration_s - time.monotonic()))
            finally:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*list(pending), return_exceptions=True)
    result = summarize(rows, time.monotonic() - start_at, spec)
    result["queue_p95_s"] = percentile(
        [r.get("queue_s", 0) for r in rows if r["transport_ok"]], 0.95
    )
    if getattr(spec, "p01_request_rates", None):
        early = [
            r.get("queue_s", 0)
            for r in rows
            if r.get("scheduled_offset_s", 0) < spec.duration_s / 4 and r["transport_ok"]
        ]
        late = [
            r.get("queue_s", 0)
            for r in rows
            if r.get("scheduled_offset_s", 0) >= spec.duration_s * 0.75 and r["transport_ok"]
        ]
        growth = (
            max(0, percentile(late, 0.95) - percentile(early, 0.95)) if early and late else None
        )
        result["queue_growth_s"] = growth
        result["gates_passed"] &= growth is not None and growth <= spec.p01_queue_growth_s
    result["offered_requests"] = count
    result["offered_requests_per_second"] = rate
    result["gates_passed"] &= len(rows) == count
    return result
