"""Equal-budget Dynamo aggregate/disaggregate deployments with transfer evidence."""

import copy
import math
import re
import socket
import subprocess
import threading
from pathlib import Path

import httpx

from ..benchmarks.storage import write_json
from .deployment import OwnedVllm

DYNAMO_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:d95e97babe5a4893a66fb5e4a810de78335ca6873e85e31d337e915d5ae92b5d"
DYNAMO_SOURCE = "2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a"


def arm_config(configuration, spec, arm):
    if arm not in ("baseline", "treatment"):
        raise ValueError("Invalid Dynamo arm")
    cfg = copy.deepcopy(configuration)
    e = cfg["engine"]
    e.update(
        type="vllm_cuda",
        gpu_image=DYNAMO_IMAGE,
        gpu_device_ids=spec.gpu_ids,
        tensor_parallel_size=1,
        replica_devices=[spec.gpu_ids],
        port=19200,
        max_model_len=spec.context_limit,
        max_num_seqs=max(32, spec.concurrency),
        docker_network="host",
        vllm_extra_flags=[],
        kv_cache_dtype=e.get("kv_cache_dtype") or "fp8",
    )
    # This is a matched Dynamo comparison, not a replay of arbitrary roofline flags.
    return cfg


def resolve_gpu_uuids(gpu_ids):
    """Preserve requested physical placement despite container CUDA enumeration."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    )
    mapping = {}
    for line in raw.splitlines():
        index, uuid = [part.strip() for part in line.split(",")]
        if not re.fullmatch(r"GPU-[a-fA-F0-9-]{36}", uuid) or int(index) in mapping:
            raise RuntimeError("Invalid physical GPU inventory")
        mapping[int(index)] = uuid
    if any(index not in mapping for index in gpu_ids):
        raise RuntimeError("Selected physical GPU is absent")
    selected = [mapping[index] for index in gpu_ids]
    if len(set(selected)) != len(selected):
        raise RuntimeError("Duplicate physical GPU UUID")
    return selected


class OwnedDynamo(OwnedVllm):
    def __init__(self, cfg, spec, arm, owner):
        super().__init__(cfg, threading.Event(), owner)
        self.spec, self.arm = spec, arm
        self.directory = None
        self.worker_roles = None
        self.pool_routing = spec.project_id == "P01" and spec.p01_placement != "pooled"
        self.frontend_count = 1

    @property
    def replica_urls(self):
        return [
            f"http://{self.cfg.host}:{self.cfg.port + 40 * i}/v1"
            for i in range(self.frontend_count)
        ]

    @property
    def metrics_urls(self):
        return [
            f"http://{self.cfg.host}:{self.cfg.port + 1 + i}/metrics"
            for i in range(len(self.spec.gpu_ids))
        ]

    def launch(self, log_dir):
        self.directory = Path(log_dir) / "replica-0"
        self.directory.mkdir(parents=True)
        topology = None
        if self.pool_routing:
            from .topology import discover, pool_plan

            topology = pool_plan(
                self.spec.gpu_ids,
                self.spec.prefill_workers,
                self.arm,
                self.spec.p01_placement,
                discover(self.spec.gpu_ids),
            )
            self.frontend_count = len(topology["frontend_cpus"])
        for port in (
            [self.cfg.port + 9]
            + ([self.cfg.port + 40, self.cfg.port + 49] if self.frontend_count == 2 else [])
            + [self.cfg.port + 1 + i for i in range(len(self.spec.gpu_ids))]
            + [self.cfg.port + 10 + i for i in range(len(self.spec.gpu_ids))]
        ):
            with socket.socket() as sock:
                sock.settimeout(1)
                if sock.connect_ex((self.cfg.host, port)) == 0:
                    raise RuntimeError(f"Dynamo auxiliary port {port} is occupied")
        plan = {
            "arm": self.arm,
            "gpu_ids": self.spec.gpu_ids,
            "gpu_uuids": resolve_gpu_uuids(self.spec.gpu_ids),
            "prefill_workers": self.spec.prefill_workers,
            "port": self.cfg.port,
            "model": self.cfg.model_local_path or self.cfg.model_id,
            "served_model_name": self.api_model_name,
            "context_limit": self.spec.context_limit,
            "max_num_seqs": self.cfg.max_num_seqs,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "kv_cache_dtype": self.cfg.kv_cache_dtype or "fp8",
            "quantization": self.cfg.quantization_kind or self.cfg.quantization,
            "namespace": "capsim-" + self.owner,
            "upstream_source": DYNAMO_SOURCE,
        }
        if topology is not None:
            plan.update(topology)
            self.worker_roles = topology["worker_roles"]
            write_json(self.directory / "topology-plan.json", topology)
        if self.spec.project_id == "P01":
            plan["transfer_settings"] = {
                k: v for k, v in self.spec.model_dump().items() if k.startswith("p01_")
            }
        if self.spec.project_id == "P02":
            plan["sessions"] = {
                k: v for k, v in self.spec.model_dump().items() if k.startswith("session_")
            }
        write_json(self.directory / "dynamo-plan.json", plan)
        super().launch(self.directory)

    def _build_docker_command(self):
        # Reuse cache mounts, pinned snapshot, selected GPU list and environment.
        cmd = super()._build_docker_command()
        pos = cmd.index(self.cfg.gpu_image)
        return cmd[:pos] + [
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--entrypoint",
            "python3",
            "-v",
            f"{Path(__file__).with_name('dynamo_runtime.py')}:/capsim-runtime.py:ro",
            "-v",
            f"{self.directory.resolve()}:/evidence",
            self.cfg.gpu_image,
            "/capsim-runtime.py",
            "/evidence/dynamo-plan.json",
        ]

    def health_check(self):
        try:
            with httpx.Client(timeout=2) as client:
                for url in self.replica_urls:
                    response = client.get(url + "/models")
                    if response.status_code != 200 or not response.json().get("data"):
                        return False
                return all(
                    client.get(url.removesuffix("/metrics") + "/health").status_code == 200
                    for url in self.metrics_urls
                )
        except (httpx.HTTPError, ValueError):
            return False

    async def capture_metrics(self, directory, label):
        async with httpx.AsyncClient(timeout=15) as client:
            for i, url in enumerate(self.metrics_urls):
                result = await client.get(url)
                result.raise_for_status()
                if len(result.content) > 8 * 1024**2:
                    raise ValueError("Dynamo metrics exceed the evidence budget")
                (directory / f"{label}-worker-{i}.prom").write_text(result.text)


def transfer_counters(text, kind="bytes"):
    """Keep only explicit NIXL byte counters; histogram counts are not bytes."""
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(\S+(?:\{.*\})?)\s+([-+0-9.eE]+)(?:\s+\d+)?", line)
        if not match:
            continue
        key, value = match.groups()
        name = key.split("{", 1)[0].lower()
        selected = (
            ("bytes" in name and name.endswith("_sum"))
            if kind == "bytes"
            else ("num_failed" in name and name.endswith("_total"))
        )
        if "nixl" in name and selected:
            value = float(value)
            if math.isfinite(value) and value >= 0:
                result[key] = value
    return result


def verify_transfer(directory, spec, worker_roles=None):
    transfer_diagnostics(directory, spec, worker_roles)
    evidence = []
    # Check decode receivers only, avoiding double counting both transfer ends.
    decoders = (
        [i for i, role in enumerate(worker_roles) if role == "decode"]
        if worker_roles
        else range(spec.prefill_workers, len(spec.gpu_ids))
    )
    for i in decoders:
        initial_count = len(evidence)
        before_text = (directory / f"before-worker-{i}.prom").read_text()
        after_text = (directory / f"after-worker-{i}.prom").read_text()
        before = transfer_counters(before_text)
        after = transfer_counters(after_text)
        failed_before = transfer_counters(before_text, "failures")
        failed_after = transfer_counters(after_text, "failures")
        for key, value in failed_after.items():
            if key not in failed_before or value != failed_before[key]:
                raise RuntimeError(
                    "NIXL transfer failure counters changed; inspect native telemetry"
                )
        if any(value < before.get(key, value) for key, value in after.items()):
            raise RuntimeError("NIXL counters reset during measurement")
        for key, value in after.items():
            if key in before and value > before[key]:
                evidence.append(
                    {
                        "worker": i,
                        "metric": key,
                        "before": before[key],
                        "after": value,
                        "delta_bytes": value - before[key],
                    }
                )
        if len(evidence) == initial_count:
            raise RuntimeError(f"Decode worker {i} transfer is unverified during measurement")
    write_json(
        directory / "transfer-verification.json", {"verified": bool(evidence), "counters": evidence}
    )
    if not evidence:
        raise RuntimeError(
            "No increasing NIXL byte counter on decode workers; disaggregation is unverified"
        )


def verify_worker_activity(directory, spec):
    """An idle replica is not evidence for an equal-budget serving comparison."""
    evidence = []
    for i in range(len(spec.gpu_ids)):
        snapshots = []
        for label in ("before", "after"):
            counters = {}
            for line in (directory / f"{label}-worker-{i}.prom").read_text().splitlines():
                if not line.startswith("vllm:generation_tokens_total"):
                    continue
                key, value = line.rsplit(" ", 1)
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError("Invalid worker generation counter")
                counters[key] = value
            snapshots.append(counters)
        before, after = snapshots
        if set(before) != set(after) or any(after[k] < before[k] for k in before):
            raise RuntimeError("Worker counters changed identity or reset during measurement")
        delta = sum(after[k] - before[k] for k in before)
        if delta <= 0:
            raise RuntimeError(f"Worker {i} served no generated tokens during measurement")
        evidence.append({"worker": i, "generated_tokens": delta})
    write_json(directory / "worker-activity.json", {"verified": True, "workers": evidence})


def transfer_diagnostics(directory, spec, worker_roles=None):
    """Native window deltas; concurrent transfer durations are not link bandwidth."""

    def counters(path):
        result = {}
        for line in path.read_text().splitlines():
            if not line.startswith("vllm:nixl_"):
                continue
            key, value = line.rsplit(" ", 1)
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise RuntimeError("Invalid NIXL diagnostic counter")
            if key.split("{", 1)[0].endswith(("_sum", "_count", "_total")):
                result[key] = number
        return result

    workers = []
    for i in range(len(spec.gpu_ids)):
        before_path = directory / f"before-worker-{i}.prom"
        after_path = directory / f"after-worker-{i}.prom"
        if not before_path.exists() or not after_path.exists():
            continue
        before, after = counters(before_path), counters(after_path)
        if set(before) != set(after) or any(after[k] < before[k] for k in before):
            raise RuntimeError("NIXL diagnostic counters reset or changed identity")
        deltas = {k: after[k] - before[k] for k in before}
        if any(v > 0 for k, v in deltas.items() if "num_kv_expired_reqs" in k):
            raise RuntimeError("NIXL prefill KV leases expired during measurement")
        means = {}
        for stem in (
            "xfer_time_seconds",
            "post_time_seconds",
            "bytes_transferred",
            "num_descriptors",
        ):
            sums = sum(
                v for k, v in deltas.items() if k.split("{", 1)[0] == f"vllm:nixl_{stem}_sum"
            )
            count = sum(
                v for k, v in deltas.items() if k.split("{", 1)[0] == f"vllm:nixl_{stem}_count"
            )
            means[stem] = {"sum": sums, "count": count, "mean": sums / count if count else None}
        workers.append(
            {
                "worker": i,
                "role": worker_roles[i]
                if worker_roles
                else "prefill"
                if i < spec.prefill_workers
                else "decode",
                "deltas": deltas,
                "means": means,
            }
        )
    write_json(
        directory / "transfer-diagnostics.json",
        {
            "workers": workers,
            "scope": "Native before/after measurement window; durations can overlap and do not measure physical link bandwidth",
        },
    )
