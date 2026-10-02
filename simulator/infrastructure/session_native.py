"""Native queue-time and cache counters; never infer document residency from blocks."""

import asyncio
import json
import math
import re
import time
from pathlib import Path

import httpx

METRIC = re.compile(r"^(\w[\w:]*)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?$")


def select_metrics(text):
    out = {}
    for line in text.splitlines():
        match = METRIC.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if not any(
            part in name
            for part in (
                "request_queue_time",
                "request_prefill_time",
                "request_decode_time",
                "request_inference_time",
                "num_requests_running",
                "num_requests_waiting",
                "cache_usage",
                "cache_hits",
                "cache_queries",
                "preemptions",
                "onboard_blocks",
                "offload_blocks",
                "matched_tokens",
            )
        ):
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid native queue/cache metric")
        out[name + (labels or "")] = value
    return out


async def monitor_native(engine, directory):
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=15) as client:
        with (directory / "native-queue-cache.jsonl").open("w") as output:
            while True:
                samples = []
                for i, child in enumerate(engine.children):
                    for url in child.metrics_urls + (
                        [f"http://{child.cfg.host}:{child.cfg.port + 20}/metrics"]
                        if child.arm == "treatment" and child.spec.session_cpu_backend == "kvbm"
                        else []
                    ):
                        response = await client.get(url)
                        response.raise_for_status()
                        if len(response.content) > 8 * 1024**2:
                            raise ValueError("Native metrics exceed evidence budget")
                        samples.append(
                            dict(replica=i, url=url, values=select_metrics(response.text))
                        )
                output.write(
                    json.dumps(
                        dict(
                            at_s=time.monotonic() - start,
                            timestamp=time.time(),
                            samples=samples,
                            host=host_snapshot(),
                            cpu_offload=[
                                json.loads(p.read_text())
                                for child in engine.children
                                for p in child.directory.glob("native-cpu-*.json")
                            ],
                        )
                    )
                    + "\n"
                )
                output.flush()
                await asyncio.sleep(2)


def queue_delta(directory, prefix=""):
    total, count = 0.0, 0.0
    buckets = {}
    coverage = 0
    for replica in directory.glob("replica-*"):
        for before in replica.glob(prefix + "before-worker-*.prom"):
            after = before.with_name(before.name.replace(prefix + "before-", prefix + "after-", 1))
            a, b = select_metrics(before.read_text()), select_metrics(after.read_text())
            keys = [k for k in b if "request_queue_time" in k]
            if not keys:
                continue
            if any(k not in a or b[k] < a[k] for k in keys):
                raise ValueError("Queue histogram reset or changed")
            coverage += 1
            for k in keys:
                delta = b[k] - a[k]
                name = k.split("{")[0]
                if name.endswith("_sum"):
                    total += delta
                elif name.endswith("_count"):
                    count += delta
                elif name.endswith("_bucket"):
                    limit = re.search(r'\ble="([^"]+)"', k)
                    if limit:
                        bound = float(limit.group(1))
                        buckets[bound] = buckets.get(bound, 0) + delta
    upper = None
    for bound, n in sorted(buckets.items()):
        if count and n >= count * 0.95:
            upper = bound if math.isfinite(bound) else None
            break
    return dict(
        native_queue_mean_s=total / count if count else None,
        native_queue_p95_bucket_upper_s=upper,
        native_queue_samples=count,
        native_queue_endpoints=coverage,
        native_queue_scope="Whole measurement plus drain; p95 is histogram bucket upper bound, not exact request percentile",
    )


def host_snapshot():
    """Raw host CPU/memory/NUMA counters; not a DRAM bandwidth measurement."""
    paths = [Path("/proc/stat"), Path("/proc/meminfo"), Path("/proc/vmstat")]
    paths += list(Path("/sys/devices/system/node").glob("node*/numastat"))
    out = {}
    for path in paths:
        try:
            out[str(path)] = path.read_text()
        except OSError as exc:
            out[str(path)] = {"unavailable": str(exc)}
    return out
