"""Bounded closed-loop streaming task driver with per-request correctness.

Throughput denominator includes drain time; failures and SLO misses remain in
attempted totals. This is a capsim driver, not an AIPerf/GuideLLM result.
"""

import asyncio
import json
import math
import time

import httpx

from .workload import grade


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def summarize(rows, elapsed_s, spec):
    complete = [r for r in rows if r["transport_ok"]]
    correct = [r for r in complete if r["correct"]]
    qualified = [r for r in correct if r["elapsed_s"] <= spec.response_slo_s]
    if getattr(spec, "project_id", None) == "P02":
        qualified = [
            r
            for r in qualified
            if r.get("ttft_s") is not None and r["ttft_s"] <= spec.session_ttft_slo_s
        ]
    if getattr(spec, "p01_request_rates", None):
        qualified = [
            r
            for r in qualified
            if r.get("ttft_s") is not None and r["ttft_s"] <= spec.p01_ttft_slo_s
        ]
    n = len(rows)
    quality = len(correct) / n if n else 0
    slo_fraction = len(qualified) / n if n else 0
    errors = 1 - len(complete) / n if n else 1
    prompt_tokens = [(r.get("usage") or {}).get("prompt_tokens") for r in complete]
    prompt_tokens = [v for v in prompt_tokens if isinstance(v, int) and v > 0]
    output_tokens = [(r.get("usage") or {}).get("completion_tokens") for r in complete]
    output_tokens = [v for v in output_tokens if isinstance(v, int) and v > 0]
    return {
        "output_tokens_min": min(output_tokens) if output_tokens else None,
        "output_tokens_max": max(output_tokens) if output_tokens else None,
        "latency_p50_s": percentile([r["elapsed_s"] for r in complete], 0.5),
        "ttft_p50_s": percentile([r["ttft_s"] for r in complete if r["ttft_s"] is not None], 0.5),
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else None,
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "attempted": n,
        "completed": len(complete),
        "correct": len(correct),
        "qualified": len(qualified),
        "slo_fraction": slo_fraction,
        "quality_fraction": quality,
        "error_fraction": errors,
        "elapsed_s_including_drain": elapsed_s,
        "qualified_per_hour": len(qualified) * 3600 / elapsed_s if elapsed_s else 0,
        "latency_p95_s": percentile([r["elapsed_s"] for r in complete], 0.95),
        "correct_latency_p95_s": percentile([r["elapsed_s"] for r in correct], 0.95),
        "ttft_p95_s": percentile([r["ttft_s"] for r in complete if r["ttft_s"] is not None], 0.95),
        "gates_passed": n >= spec.min_requests
        and quality >= spec.min_quality
        and errors <= spec.max_error_fraction
        and slo_fraction >= spec.min_slo_fraction,
    }


async def request_one(client, url, model, item, spec, request_id, api_key="EMPTY"):
    start = time.monotonic()
    row = {
        "request_id": request_id,
        "task_id": item["id"],
        "started_at": time.time(),
        "transport_ok": False,
        "correct": False,
        "ttft_s": None,
        "output": "",
        "finish_reason": None,
        "usage": None,
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "seed": spec.seed,
        "max_tokens": spec.max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if spec.structured_output:
        # Schema constrains format only. It contains no correct record IDs or answers.
        fields = {
            k: {"type": "integer" if k == "quantity" else "string"}
            for k in ("record_id", "location", "part", "quantity", "status", "instructions")
        }
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_records",
                "strict": True,
                "schema": {
                    "type": "array",
                    "minItems": len(item["expected"]),
                    "maxItems": len(item["expected"]),
                    "items": {
                        "type": "object",
                        "properties": fields,
                        "required": list(fields),
                        "additionalProperties": False,
                    },
                },
            },
        }

    async def stream_response():
        async with client.stream(
            "POST",
            url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            response.raise_for_status()
            done = False
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ValueError("Server returned a streaming error")
                if chunk.get("usage"):
                    row["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    content = choice.get("delta", {}).get("content") or ""
                    if content and row["ttft_s"] is None:
                        row["ttft_s"] = time.monotonic() - start
                    row["output"] += content
                    if len(row["output"]) > 1024 * 1024:
                        raise ValueError("Response exceeds one MiB evidence limit")
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
            row["transport_ok"] = done and row["finish_reason"] is not None
            row["correct"] = (
                row["transport_ok"]
                and row["finish_reason"] == "stop"
                and grade(row["output"], item["expected"])
            )

    try:
        await asyncio.wait_for(stream_response(), spec.request_timeout_s)
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        row["error"] = type(exc).__name__
    row["elapsed_s"] = time.monotonic() - start
    return row


async def measure(engine, corpus, spec, path, duration_s=None):
    duration_s = spec.duration_s if duration_s is None else duration_s
    start = time.monotonic()
    deadline = start + duration_s
    rows, counter = [], 0
    urls = list(engine.replica_urls)
    if not urls:
        raise ValueError("No serving endpoints")
    with path.open("w") as out:
        async with httpx.AsyncClient(
            timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
        ) as client:

            async def worker():
                nonlocal counter
                while time.monotonic() < deadline:
                    number = counter
                    counter += 1
                    item = corpus[number % len(corpus)]
                    row = await request_one(
                        client,
                        urls[number % len(urls)],
                        engine.api_model_name,
                        item,
                        spec,
                        str(number),
                        engine.api_key,
                    )
                    rows.append(
                        {
                            k: row[k]
                            for k in ("transport_ok", "correct", "elapsed_s", "ttft_s", "usage")
                        }
                    )
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    if out.tell() > 1024**3 or len(rows) >= 1_000_000:
                        raise RuntimeError("Request evidence exceeded the per-window safety budget")
                    # Bound failures so a down endpoint cannot fill disk at CPU speed.
                    if not row["transport_ok"]:
                        await asyncio.sleep(0.1)

            tasks = [asyncio.create_task(worker()) for _ in range(spec.concurrency)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    return summarize(rows, time.monotonic() - start, spec)


async def capture_metrics(engine, directory, label):
    """Retain native acceptance/cache counters without relabeling their semantics."""
    if hasattr(engine, "capture_metrics"):
        return await engine.capture_metrics(directory, label)
    async with httpx.AsyncClient(timeout=15) as client:
        for index, url in enumerate(engine.replica_urls):
            root = url.rstrip("/").removesuffix("/v1")
            response = await client.get(
                root + "/metrics", headers={"Authorization": f"Bearer {engine.api_key}"}
            )
            response.raise_for_status()
            if len(response.content) > 8 * 1024**2:
                raise ValueError("Native metrics exceed the evidence size limit")
            (directory / f"{label}-replica-{index}.prom").write_text(response.text)


async def measure_finite(engine, items, spec, path):
    """Replay exactly the frozen cohort once, including failures and final drain."""
    if not engine.replica_urls:
        raise ValueError("No serving endpoints")
    rows, queue = [], iter(enumerate(items))
    start = time.monotonic()
    with path.open("w") as out:
        async with httpx.AsyncClient(
            timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
        ) as client:

            async def worker():
                for index, item in queue:
                    row = await request_one(
                        client,
                        engine.replica_urls[index % len(engine.replica_urls)],
                        engine.api_model_name,
                        item,
                        spec,
                        str(index),
                        engine.api_key,
                    )
                    rows.append(row)
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()

            tasks = [
                asyncio.create_task(worker()) for _ in range(min(spec.concurrency, len(items)))
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    return summarize(rows, time.monotonic() - start, spec)
