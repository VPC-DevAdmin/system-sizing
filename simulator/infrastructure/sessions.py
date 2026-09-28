"""Finite document-session replay with evidence of the actual offload tier."""

import asyncio
import json
import math
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx

from ..benchmarks.storage import write_json
from .driver import request_one, summarize
from .dynamo import OwnedDynamo


class OwnedSessions(OwnedDynamo):
    def launch(self, log_dir):
        self.storage_root = None
        if self.arm == "treatment":
            for offset in (20, 30, 31):
                with socket.socket() as sock:
                    if sock.connect_ex((self.cfg.host, self.cfg.port + offset)) == 0:
                        raise RuntimeError(
                            f"KVBM auxiliary port {self.cfg.port + offset} is occupied"
                        )
            available = next(
                int(x.split()[1]) * 1024
                for x in open("/proc/meminfo")
                if x.startswith("MemAvailable:")
            )
            if available < (self.spec.session_cpu_gb + 16) * 1024**3:
                raise RuntimeError("Insufficient host memory for session cache and safety margin")
            if self.spec.session_tier == "cpu_disk":
                root = subprocess.check_output(
                    ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True, timeout=15
                ).strip()
                self.storage_root = root
                if shutil.disk_usage(root).free < (self.spec.session_disk_gb + 20) * 1024**3:
                    raise RuntimeError("Insufficient Docker storage for the ephemeral disk cache")
        super().launch(log_dir)
        mounts = json.loads(
            subprocess.check_output(
                ["docker", "inspect", "--format", "{{json .Mounts}}", self._container_id],
                text=True,
                timeout=15,
            )
        )
        write_json(self.directory / "container-mounts.json", mounts)

    def _build_docker_command(self):
        argv = super()._build_docker_command()
        if self.arm == "treatment" and self.spec.session_tier == "cpu_disk":
            if not Path("/run/udev").is_dir():
                raise RuntimeError(
                    "Disk tier requires read-only /run/udev for cuFile volume discovery"
                )
            pos = argv.index(self.cfg.gpu_image)
            # Anonymous /tmp volume inherits the image's writable tmp permissions.
            # It is block-backed rather than overlayfs, and removed with its owner.
            argv[pos:pos] = ["-v", "/tmp", "-v", "/run/udev:/run/udev:ro"]
        return argv

    def get_metrics(self):
        metrics = super().get_metrics()
        if self.storage_root:
            free = shutil.disk_usage(self.storage_root).free
            if free < 10 * 1024**3:
                raise RuntimeError("Session cache storage fell below 10 GiB free")
            metrics["session_cache_storage_free_bytes"] = free
        return metrics

    async def capture_metrics(self, directory, label):
        await super().capture_metrics(directory, label)
        if self.arm == "treatment":
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"http://{self.cfg.host}:{self.cfg.port + 20}/metrics")
                response.raise_for_status()
                if len(response.content) > 8 * 1024**2:
                    raise ValueError("KVBM metrics exceed evidence budget")
                (directory / f"{label}-kvbm.prom").write_text(response.text)


def tier_counters(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        if any(part in name for part in ("onboard_blocks_", "offload_blocks_", "matched_tokens")):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid tier counter")
            result[key] = value
    return result


def verify_tier(directory, spec):
    before = tier_counters((directory / "resume-before-kvbm.prom").read_text())
    after = tier_counters((directory / "resume-after-kvbm.prom").read_text())
    if set(before) != set(after) or any(after[k] < before[k] for k in before):
        raise RuntimeError("Tier counters reset or changed identity")
    delta = {k: after[k] - before[k] for k in before}
    required = "onboard_blocks_h2d" if spec.session_tier == "cpu" else "onboard_blocks_d2d"
    verified = any(required in k and v > 0 for k, v in delta.items())
    write_json(
        directory / "tier-verification.json",
        {"verified": verified, "requested_tier": spec.session_tier, "counter_deltas": delta},
    )
    if not verified:
        raise RuntimeError(
            "Requested cache tier did not reload measured sessions; increase or adjust cache pressure"
        )


async def finite_replay(engine, items, spec, path):
    rows = []
    queue = iter(enumerate(items))
    start = time.monotonic()
    with path.open("w") as out:
        async with httpx.AsyncClient(
            timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
        ) as client:

            async def worker():
                for index, item in queue:
                    row = await request_one(
                        client,
                        engine.replica_urls[0],
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


async def measure_sessions(engine, corpus, spec, directory):
    cohort = corpus[: spec.session_count]
    pressure = corpus[spec.session_count : spec.session_count + spec.session_churn]
    write_json(
        directory / "session-trace.json",
        {
            "prime": [x["id"] for x in cohort],
            "hot_diagnostic": [cohort[-1]["id"]],
            "pressure": [x["id"] for x in pressure],
            "resume": [x["id"] for x in cohort],
            "scope": "Repeated document sessions; no claim of multi-turn conversation quality",
            "resume_suffix": "\nCheck all requested fields carefully before returning the JSON.",
        },
    )
    for label, items in [("prime", cohort), ("pressure", pressure)]:
        await engine.capture_metrics(directory, label + "-before")
        result = await finite_replay(engine, items, spec, directory / (label + ".jsonl"))
        write_json(directory / (label + "-metrics.json"), result)
        await engine.capture_metrics(directory, label + "-after")
        if (
            result["error_fraction"] > spec.max_error_fraction
            or result["quality_fraction"] < spec.min_quality
        ):
            raise RuntimeError(f"Session {label} phase failed quality or transport gate")
        if label == "prime":
            await engine.capture_metrics(directory, "hot-before")
            hot_spec = spec.model_copy(update={"concurrency": 1, "min_requests": 1})
            hot = await finite_replay(engine, [cohort[-1]], hot_spec, directory / "hot.jsonl")
            write_json(directory / "hot-metrics.json", hot)
            await engine.capture_metrics(directory, "hot-after")
            if not hot["gates_passed"]:
                raise RuntimeError("Hot-session diagnostic failed its quality/SLO gate")
            verify_hot(directory)
    # Resume once per session. Avoid an unbounded loop that would turn misses into hot hits.
    resumed = [
        dict(
            x,
            prompt=x["prompt"]
            + "\nCheck all requested fields carefully before returning the JSON.",
        )
        for x in cohort
    ]
    await engine.capture_metrics(directory, "resume-before")
    result = await finite_replay(engine, resumed, spec, directory / "requests.jsonl")
    await engine.capture_metrics(directory, "resume-after")
    if engine.arm == "treatment":
        verify_tier(directory, spec)
    result["resume_ttft_gate_passed"] = (
        result["ttft_p95_s"] is not None and result["ttft_p95_s"] <= spec.session_ttft_slo_s
    )
    result["gates_passed"] = result["gates_passed"] and result["resume_ttft_gate_passed"]
    return result


def verify_hot(directory):
    snapshots = []
    for label in ("before", "after"):
        counters = {}
        for line in (directory / f"hot-{label}-worker-0.prom").read_text().splitlines():
            if line.startswith(("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")):
                key, value = line.rsplit(" ", 1)
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError("Invalid GPU-cache counter")
                counters[key] = value
        snapshots.append(counters)
    before, after = snapshots
    if not before or set(before) != set(after) or any(after[k] < before[k] for k in before):
        raise RuntimeError("Missing or reset GPU-cache counters")
    deltas = {k: after[k] - before[k] for k in before}
    hits = sum(v for k, v in deltas.items() if k.startswith("vllm:prefix_cache_hits_total"))
    queries = sum(v for k, v in deltas.items() if k.startswith("vllm:prefix_cache_queries_total"))
    verified = queries > 0 and hits / queries >= 0.9
    write_json(
        directory / "hot-verification.json",
        {
            "verified": verified,
            "gpu_cached_tokens": hits,
            "queried_tokens": queries,
            "minimum_fraction": 0.9,
            "scope": "One immediate-return diagnostic; not hot-session capacity",
        },
    )
    if not verified:
        raise RuntimeError(
            "Immediate return was not at least 90% GPU-cache hits; review GPU cache budget"
        )
