#!/usr/bin/env python3
"""Mini engine optimiser for CPU LLM inference hosts.

Iterates a registry of vLLM-CPU launch configurations, measures TTFT
and TPOT across a few representative (input-tokens, output-tokens,
concurrency) cells, and reports the configuration that minimises
latency / maximises throughput on this specific host. Designed for the
dual-socket EPYC R7735 platform but works on any host with Docker +
the ``vllm/vllm-openai-cpu:latest-x86_64`` image (or whatever the
``IMAGE`` environment variable points at).

Per config, the flow is:

    cleanup_containers
    docker run … (one container per replica)
    wait for /v1/models → 200 on each replica
    warmup (3 small requests)
    for cell in TEST_CELLS:
        measure (TTFT/TPOT/throughput)
        save JSON  ← incremental
    cleanup_containers
    save JSON

If a launch fails (container exits during init, /v1/models never
responds), the failure is recorded with the last 50 lines of docker
logs and the optimiser moves to the next config.

Disconnect-resilience and resume:

* JSON output at a stable path (``runs/engine_optimizer/run.json`` by
  default) is updated after every cell — interrupting mid-config
  loses at most one cell, not the whole config's data.
* On launch, an existing JSON is loaded; configs with a terminal
  status (``ok`` or ``launch_failed``) are skipped, so re-running
  after a crash / disconnect picks up where we left off. Pass
  ``--new-run`` to wipe and start fresh.
* Best paired with the ``make optimize-engine`` target which nohups
  the script and auto-tails the log so an SSH disconnect can't kill
  a multi-hour optimisation run.

Output: ``runs/engine_optimizer/run.json`` (stable) + a ranked summary
printed to stdout. When stdout is a TTY a rich dashboard is shown;
otherwise plain timestamped progress lines are printed (so the
backgrounded log file is greppable)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise SystemExit(
        "openai package required: pip install openai (or activate the project venv)"
    ) from e
try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as e:
    raise SystemExit(
        "rich package required: pip install rich (or activate the project venv)"
    ) from e


# ── Tunables ────────────────────────────────────────────────────────────

IMAGE = os.environ.get("OPTIMIZER_IMAGE", "vllm/vllm-openai-cpu:latest-x86_64")
MODEL_PATH = os.environ.get(
    "OPTIMIZER_MODEL_PATH", "/models/Qwen3-30B-A3B-Instruct-2507"
)
SERVED_NAME = os.environ.get("OPTIMIZER_SERVED_NAME", "qwen3_30b_a3b")
HOST = "127.0.0.1"
LAUNCH_TIMEOUT_S = int(os.environ.get("OPTIMIZER_LAUNCH_TIMEOUT_S", "1800"))
WARMUP_REQUESTS = 3
# The long-context pain cells (4k input × 16 concurrent) can push a
# single request close to 5-8 minutes on a struggling config. Cap at
# 15 min so an actually-broken config gets a timeout error instead of
# stalling the whole optimizer indefinitely.
REQUEST_TIMEOUT_S = int(os.environ.get("OPTIMIZER_REQUEST_TIMEOUT_S", "900"))


# ── Replica + config dataclasses ────────────────────────────────────────


@dataclass
class ReplicaSpec:
    name: str
    port: int
    cpuset_cpus: str
    cpuset_mems: Optional[str] = None
    env: dict = field(default_factory=dict)


@dataclass
class EngineConfig:
    """A complete launch shape — one or more replicas + their args.

    ``replica_args`` are appended to every replica's command line;
    ``replica_env`` is merged into every replica's env (per-replica
    env in ReplicaSpec wins on collisions). ``shm_size`` is per-
    container.
    """
    name: str
    description: str
    replicas: list[ReplicaSpec]
    replica_args: list[str] = field(default_factory=list)
    replica_env: dict = field(default_factory=dict)
    shm_size: str = "4g"
    expected_outcome: str = ""


@dataclass
class TestCell:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    # Per-request hard cap. When a request exceeds this, abort it and
    # count as a ``timeouts`` increment (separate from ``errors``,
    # separate from ``samples``). Excluded from p50/p95 so the
    # latency stats only describe requests that actually completed.
    # The aggregate throughput field DOES include timed-out requests'
    # zero contribution, so it correctly drops on configs with a tail.
    #
    # Sized as a small multiple of the expected wall-clock time per
    # cell — generous enough that a healthy config never hits it,
    # tight enough that a stuck-tail pathology (e.g. baseline pain16's
    # ~900 s tail) gets killed in seconds rather than minutes.
    timeout_s: float = 60.0


@dataclass
class CellResult:
    cell_name: str
    samples: int
    errors: int
    ttft_p50_ms: Optional[float]
    ttft_p95_ms: Optional[float]
    tpot_p50_ms: Optional[float]
    tpot_p95_ms: Optional[float]
    throughput_out_tok_s: Optional[float]
    error_summary: str = ""
    # Number of requests killed by the per-cell ``timeout_s`` cap.
    # Default 0 so old JSON without this field deserialises cleanly.
    timeouts: int = 0


@dataclass
class ConfigResult:
    name: str
    description: str
    status: str  # "ok" | "launch_failed" | "skipped"
    launch_seconds: Optional[float] = None
    cells: list[CellResult] = field(default_factory=list)
    failure_reason: str = ""


# ── Engine config registry ──────────────────────────────────────────────
#
# 5 starter configs from the user request, plus 4 complementary axes:
# kv-pool size, prefix-cache toggle, block size, and a single-replica
# all-cores variant (the AMD analog of Intel's ctx_kv_xl winner). Each
# config tests an isolated change so the result diffs are meaningful.

def _dual_replica_pair() -> list[ReplicaSpec]:
    return [
        ReplicaSpec(
            name="vllm-r0", port=8000,
            cpuset_cpus="0-31", cpuset_mems="0",
            env={
                "VLLM_CPU_KVCACHE_SPACE": "80",
                "VLLM_CPU_OMP_THREADS_BIND": "0-31",
                "OMP_NUM_THREADS": "32",
            },
        ),
        ReplicaSpec(
            name="vllm-r1", port=8001,
            cpuset_cpus="32-63", cpuset_mems="1",
            env={
                "VLLM_CPU_KVCACHE_SPACE": "80",
                "VLLM_CPU_OMP_THREADS_BIND": "32-63",
                "OMP_NUM_THREADS": "32",
            },
        ),
    ]


CONFIGS: list[EngineConfig] = [
    EngineConfig(
        name="baseline",
        description="Dual-replica BF16, 80GB KV per replica. Reference point.",
        replicas=_dual_replica_pair(),
        expected_outcome="Reproduces run_02 numbers; sanity check.",
    ),
    EngineConfig(
        name="chunked_prefill",
        description="Dual-replica + chunked prefill (4096 tokens/step, 64 seqs).",
        replicas=_dual_replica_pair(),
        replica_args=[
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "4096",
            "--max-num-seqs", "64",
        ],
        expected_outcome=(
            "TPOT p95 drops on cells mixing long+short prompts; "
            "long-prompt TTFT may rise slightly."
        ),
    ),
    EngineConfig(
        name="batch_budget",
        description="Dual-replica + larger batch budget (8192 tokens/step, 96 seqs).",
        replicas=_dual_replica_pair(),
        replica_args=[
            "--max-num-batched-tokens", "8192",
            "--max-num-seqs", "96",
        ],
        expected_outcome=(
            "Higher prefill throughput when many requests arrive together; "
            "individual TTFT may rise."
        ),
    ),
    EngineConfig(
        name="kv_xl",
        description="Dual-replica with 160GB KV per replica (2× baseline).",
        replicas=[
            ReplicaSpec(
                name="vllm-r0", port=8000, cpuset_cpus="0-31", cpuset_mems="0",
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "160",
                    "VLLM_CPU_OMP_THREADS_BIND": "0-31",
                    "OMP_NUM_THREADS": "32",
                },
            ),
            ReplicaSpec(
                name="vllm-r1", port=8001, cpuset_cpus="32-63", cpuset_mems="1",
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "160",
                    "VLLM_CPU_OMP_THREADS_BIND": "32-63",
                    "OMP_NUM_THREADS": "32",
                },
            ),
        ],
        expected_outcome=(
            "Tests whether 80GB KV is actually a ceiling at the knee. "
            "If aggregate metrics don't improve, KV space wasn't the bottleneck."
        ),
    ),
    EngineConfig(
        name="no_prefix_cache",
        description="Dual-replica baseline with prefix caching disabled.",
        replicas=_dual_replica_pair(),
        replica_args=["--no-enable-prefix-caching"],
        expected_outcome=(
            "Tests whether prefix caching is helping or just adding overhead "
            "on AMD. Hypothesis: helps slightly on multi-turn cells, near-zero "
            "on single-turn."
        ),
    ),
    EngineConfig(
        name="block_32",
        description="Dual-replica with KV block size 32 (2× default).",
        replicas=_dual_replica_pair(),
        replica_args=["--block-size", "32"],
        expected_outcome=(
            "Coarser KV blocks reduce per-block overhead; could help on "
            "long-prompt cells with low concurrency."
        ),
    ),
    EngineConfig(
        name="chunked_prefill_block_32",
        description="Dual-replica + chunked prefill + KV block size 32 (stacked).",
        replicas=_dual_replica_pair(),
        replica_args=[
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "4096",
            "--max-num-seqs", "64",
            "--block-size", "32",
        ],
        expected_outcome=(
            "Hypothesis from the first AMD optimizer pass: block_32 wins "
            "broadly (c=1 ttft -21%, short_throughput -72%, long_output "
            "-36%) and chunked_prefill eliminates the c=16 stuck-tail. "
            "Stacking the two should give both effects. If confirmed, "
            "this is the production AMD config."
        ),
    ),
    EngineConfig(
        name="kv_xl_120",
        description="Dual-replica with 120GB KV per replica (1.5× baseline).",
        replicas=[
            ReplicaSpec(
                name="vllm-r0", port=8000, cpuset_cpus="0-31", cpuset_mems="0",
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "120",
                    "VLLM_CPU_OMP_THREADS_BIND": "0-31",
                    "OMP_NUM_THREADS": "32",
                },
            ),
            ReplicaSpec(
                name="vllm-r1", port=8001, cpuset_cpus="32-63", cpuset_mems="1",
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "120",
                    "VLLM_CPU_OMP_THREADS_BIND": "32-63",
                    "OMP_NUM_THREADS": "32",
                },
            ),
        ],
        expected_outcome=(
            "kv_xl (160GB×2 = 320GB) failed health check on a 386GB host — "
            "weight residency + scratch leaves too little margin. 120GB×2 "
            "= 240GB is comfortable. Tests whether KV pool size still "
            "matters once chunked_prefill / block_32 have eliminated the "
            "c=16 scheduler tail. Hypothesis: it doesn't — 80GB was "
            "already enough; the prior pain was scheduling, not KV size."
        ),
    ),
    EngineConfig(
        name="single_replica_64core",
        description="One container, all 64 cores, both NUMA. AMD analog of ctx_kv_xl.",
        replicas=[
            ReplicaSpec(
                name="vllm-single", port=8000,
                cpuset_cpus="0-63", cpuset_mems=None,
                env={
                    # Original 160GB request failed: vLLM CPU's KV
                    # allocator checks against ONE NUMA node's free
                    # memory (not the cross-NUMA total), even when
                    # cpuset_mems is unset. Observed on R7735:
                    # node 0 had 126.38/188.52 GiB free at the
                    # moment of the check. 100GB stays comfortably
                    # under that with margin for scratch + weights.
                    "VLLM_CPU_KVCACHE_SPACE": "100",
                    "VLLM_CPU_OMP_THREADS_BIND": "0-63",
                    "OMP_NUM_THREADS": "64",
                },
            ),
        ],
        shm_size="16g",
        expected_outcome=(
            "Single user's prefill gets 2× cores at the cost of cross-NUMA "
            "memory traffic. Probably wins per-stream TTFT, loses aggregate "
            "throughput vs dual-replica. Tests the latency-vs-throughput axis."
        ),
    ),
    EngineConfig(
        name="tp2_cross_socket",
        description="Single instance, TP=2 across both sockets (64 cores).",
        replicas=[
            ReplicaSpec(
                name="vllm-tp2", port=8000,
                cpuset_cpus="0-63", cpuset_mems=None,
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "120",
                    "VLLM_CPU_OMP_THREADS_BIND": "0-63",
                    "OMP_NUM_THREADS": "64",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                },
            ),
        ],
        replica_args=[
            "--tensor-parallel-size", "2",
            "--distributed-executor-backend", "mp",
        ],
        shm_size="32g",
        expected_outcome=(
            "Higher-risk launch (CPU TP across NUMA isn't battle-tested). "
            "Prefill matmuls split across 64 cores → ~2× per-stream prefill, "
            "but every layer pays Gloo all-reduce. Net win uncertain."
        ),
    ),
    EngineConfig(
        name="tp2_chunked_prefill",
        description="TP=2 + chunked prefill — best operating point for TP if it works.",
        replicas=[
            ReplicaSpec(
                name="vllm-tp2-cp", port=8000,
                cpuset_cpus="0-63", cpuset_mems=None,
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "120",
                    "VLLM_CPU_OMP_THREADS_BIND": "0-63",
                    "OMP_NUM_THREADS": "64",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                },
            ),
        ],
        replica_args=[
            "--tensor-parallel-size", "2",
            "--distributed-executor-backend", "mp",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "4096",
            "--max-num-seqs", "96",
        ],
        shm_size="32g",
        expected_outcome=(
            "Spreads all-reduce traffic over time instead of one big burst. "
            "Skip if tp2_cross_socket is bad."
        ),
    ),
]


# ── Test cells ──────────────────────────────────────────────────────────
#
# Six cells, covering: best-case single-stream latency, typical chat
# under load, short-prompt throughput, the LONG-CONTEXT PAIN POINT
# (4k input × 512 out at c=8 and c=16 — where the AMD baseline today
# falls over), and a low-in / long-out flip that's decode-bound.
#
# Runtime warning: long_pain_16 is intentionally brutal — 16 streams
# each prefill 4096 tokens. Per-request TTFT at AMD baseline rates is
# 30-60 s with queueing, so a single config spends 5-10 minutes on
# this cell alone. Total runtime across 9 configs ≈ 60-90 min. Use
# ``ONLY=...`` to subset if you want a faster pass.

TEST_CELLS: list[TestCell] = [
    # ``timeout_s`` is the per-request cap; tuned ~5× the worst-case
    # healthy completion time so noise doesn't trip it but the
    # 900-second-tail pathologies do. See _AMD_REFERENCE_TIMINGS in
    # the docstring above for the math.
    TestCell("single_stream_short", input_tokens=128,  output_tokens=128,  concurrency=1,  timeout_s=30),
    TestCell("chat_concurrent",     input_tokens=512,  output_tokens=256,  concurrency=4,  timeout_s=60),
    TestCell("short_throughput",    input_tokens=128,  output_tokens=128,  concurrency=8,  timeout_s=60),
    # Pain point: long input + moderate output, two concurrency levels.
    TestCell("long_pain_8",         input_tokens=4096, output_tokens=512,  concurrency=8,  timeout_s=180),
    TestCell("long_pain_16",        input_tokens=4096, output_tokens=512,  concurrency=16, timeout_s=300),
    # Flipped: short input, long output. Decode-dominated; tests
    # whether AMD's memory-bandwidth advantage actually translates.
    TestCell("long_output",         input_tokens=512,  output_tokens=4096, concurrency=4,  timeout_s=360),
]


# ── Docker / orchestration helpers ──────────────────────────────────────


def run(cmd: list[str], check: bool = False, capture: bool = False, timeout: int = 60):
    return subprocess.run(
        cmd, check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
    )


def cleanup_containers(state: "OptimizerState") -> None:
    """Stop AND remove any container whose name starts with ``vllm-``.

    The remove step matters because docker_launch no longer uses
    ``--rm`` — we keep failed containers around between exit and the
    next config so ``docker logs`` can still scrape the failure
    reason. cleanup is the place we actually delete them.
    """
    state.set_phase("cleanup")
    state.append_log("== cleanup ==")
    # Running containers — stop them.
    res = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=vllm-"],
        capture_output=True, text=True,
    )
    cids = res.stdout.split()
    if cids:
        state.append_log(f"stopping {len(cids)} container(s): {' '.join(cids)}")
        subprocess.run(["docker", "stop", "-t", "10", *cids], capture_output=True)
    # Now also remove any vllm-* container including exited ones (no
    # --filter status= so we catch both running and stopped).
    res = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=vllm-"],
        capture_output=True, text=True,
    )
    cids = res.stdout.split()
    if cids:
        state.append_log(f"removing {len(cids)} container(s)")
        subprocess.run(["docker", "rm", "-f", *cids], capture_output=True)
    # Ports take a moment to release.
    time.sleep(2)


def docker_launch(cfg: EngineConfig, replica: ReplicaSpec) -> str:
    """Build and run the docker command for one replica. Returns the
    container ID."""
    # NOT using --rm — we want failed containers to stick around so
    # ``docker logs`` can scrape the failure reason post-mortem.
    # cleanup_containers() does the docker rm -f at the end.
    args = [
        "docker", "run", "-d",
        "--network", "host",
        "--shm-size", cfg.shm_size,
        "--cpuset-cpus", replica.cpuset_cpus,
        "--security-opt", "seccomp=unconfined",
        "--cap-add", "SYS_NICE",
        "-v", "/data/ml/models:/models",
        "--name", replica.name,
    ]
    if replica.cpuset_mems is not None:
        args.extend(["--cpuset-mems", replica.cpuset_mems])
    env = {**cfg.replica_env, **replica.env}
    for k, v in env.items():
        args.extend(["-e", f"{k}={v}"])
    args.append(IMAGE)
    args.extend([
        "--model", MODEL_PATH,
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", str(replica.port),
        "--served-model-name", SERVED_NAME,
    ])
    args.extend(cfg.replica_args)
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"docker run for {replica.name} failed:\n"
            f"  cmd: {' '.join(args)}\n"
            f"  stderr: {res.stderr.strip()}"
        )
    return res.stdout.strip()


def docker_logs_tail(name: str, n: int = 30) -> str:
    res = subprocess.run(
        ["docker", "logs", "--tail", str(n), name],
        capture_output=True, text=True, timeout=5,
    )
    # docker logs writes to stderr by default; combine.
    return (res.stderr or "") + (res.stdout or "")


def container_running(name: str) -> bool:
    res = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"name={name}"],
        capture_output=True, text=True, timeout=5,
    )
    return bool(res.stdout.strip())


async def wait_for_health(
    cfg: EngineConfig, state: "OptimizerState", timeout_s: int = LAUNCH_TIMEOUT_S,
) -> bool:
    """Poll /v1/models on every replica until 200, or fail with the
    last 50 log lines for the offending container."""
    import httpx
    state.set_phase("waiting for health")
    deadline = time.monotonic() + timeout_s
    poll_interval = 5.0
    healthy: set[str] = set()
    while time.monotonic() < deadline:
        for r in cfg.replicas:
            if r.name in healthy:
                continue
            if not container_running(r.name):
                tail = docker_logs_tail(r.name, 50)
                state.append_log(f"!! {r.name} EXITED. last 50:\n{tail}")
                return False
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"http://{HOST}:{r.port}/v1/models")
                if resp.status_code == 200:
                    healthy.add(r.name)
                    state.append_log(f"✓ {r.name} healthy after "
                                     f"{int(time.monotonic() - state.config_started_at)}s")
            except Exception:
                pass
        # Stream a few log lines from whichever container isn't healthy yet.
        for r in cfg.replicas:
            if r.name not in healthy:
                tail = docker_logs_tail(r.name, 5)
                if tail:
                    for line in tail.strip().splitlines()[-3:]:
                        state.append_log(f"[{r.name}] {line}")
                break
        if len(healthy) == len(cfg.replicas):
            return True
        await asyncio.sleep(poll_interval)
    return False


# ── Bench client ────────────────────────────────────────────────────────


def build_prompt(target_tokens: int) -> str:
    """Generate a prompt that tokenises to roughly ``target_tokens``.

    Uses a fixed lexicon repeated to length. Rough rule-of-thumb is
    ~1.3 tokens per English word; we pad to (target_tokens / 1.3)
    words. Exact token count isn't critical — we just need each
    config to see the SAME prompt, which they do since prompts are
    pre-generated."""
    words = (
        "system architecture latency throughput memory bandwidth concurrency "
        "prefix cache token generation prefill decode quantization tensor "
        "parallel attention layer kernel matmul scheduler batch request "
        "user persona simulation benchmark capacity bottleneck saturation "
    ).split()
    needed = int(target_tokens / 1.3) + 1
    out = []
    while len(out) < needed:
        out.extend(words)
    return " ".join(out[:needed])


# Pre-generate prompts once so every config tests the SAME inputs.
def make_prompts(cells: list[TestCell]) -> dict[str, list[str]]:
    return {
        cell.name: [
            build_prompt(cell.input_tokens) + f" [variant {i}]"
            for i in range(cell.concurrency)
        ]
        for cell in cells
    }


@dataclass(frozen=True)
class _ReqResult:
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    timed_out: bool = False


async def _do_stream(
    client: AsyncOpenAI, prompt: str, output_tokens: int,
) -> _ReqResult:
    """Issue one streaming request, return latency / token-count
    measurements. Caller is responsible for the asyncio.wait_for
    wrapper that enforces the per-cell timeout."""
    t_start = time.monotonic()
    t_first: Optional[float] = None
    n_tokens = 0
    stream = await client.chat.completions.create(
        model=SERVED_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_tokens,
        stream=True,
        extra_body={"ignore_eos": True},
        timeout=REQUEST_TIMEOUT_S,
    )
    async for chunk in stream:
        if t_first is None:
            t_first = time.monotonic()
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            # Roughly 1 chunk per token for vLLM's stream; count chunks.
            n_tokens += 1
    if t_first is None:
        return _ReqResult(error="no tokens emitted")
    t_end = time.monotonic()
    return _ReqResult(
        ttft_ms=(t_first - t_start) * 1000.0,
        total_ms=(t_end - t_start) * 1000.0,
        output_tokens=n_tokens,
    )


async def measure_one_request(
    client: AsyncOpenAI, prompt: str, output_tokens: int, timeout_s: float,
) -> _ReqResult:
    """Wrap _do_stream in asyncio.wait_for so a stuck request gets
    killed at ``timeout_s`` and classified as a timeout. The openai
    SDK's own ``timeout`` parameter only governs per-chunk reads, so
    a stream that emits tokens slowly enough to take 15 minutes
    won't trip it — we need our own outer wall-clock cap.
    """
    try:
        return await asyncio.wait_for(
            _do_stream(client, prompt, output_tokens),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return _ReqResult(timed_out=True, error="timeout")
    except Exception as e:  # noqa: BLE001
        return _ReqResult(error=str(e)[:200])


async def measure_cell(
    cfg: EngineConfig, cell: TestCell, prompts: list[str],
    state: "OptimizerState",
) -> CellResult:
    state.set_phase(
        f"cell {cell.name} (in={cell.input_tokens} out={cell.output_tokens} "
        f"c={cell.concurrency} timeout={cell.timeout_s:.0f}s)"
    )
    # Round-robin requests across replicas.
    clients = [
        AsyncOpenAI(base_url=f"http://{HOST}:{r.port}/v1", api_key="EMPTY")
        for r in cfg.replicas
    ]
    tasks = [
        measure_one_request(
            clients[i % len(clients)], prompts[i], cell.output_tokens,
            cell.timeout_s,
        )
        for i in range(cell.concurrency)
    ]
    t0 = time.monotonic()
    results = await asyncio.gather(*tasks)
    dur = time.monotonic() - t0
    ttfts, tpots = [], []
    out_tok_total = 0
    errors = 0
    timeouts = 0
    error_msgs: list[str] = []
    for r in results:
        if r.timed_out:
            timeouts += 1
            continue
        if r.error is not None:
            errors += 1
            error_msgs.append(r.error)
            continue
        if r.ttft_ms is None or r.total_ms is None or r.output_tokens is None:
            errors += 1
            continue
        ttfts.append(r.ttft_ms)
        if r.output_tokens > 0 and r.total_ms > r.ttft_ms:
            tpots.append((r.total_ms - r.ttft_ms) / max(1, r.output_tokens))
        out_tok_total += r.output_tokens
    samples = len(ttfts)
    return CellResult(
        cell_name=cell.name,
        samples=samples,
        errors=errors,
        timeouts=timeouts,
        ttft_p50_ms=_pct(ttfts, 0.5),
        ttft_p95_ms=_pct(ttfts, 0.95),
        tpot_p50_ms=_pct(tpots, 0.5),
        tpot_p95_ms=_pct(tpots, 0.95),
        throughput_out_tok_s=(out_tok_total / dur) if dur > 0 else None,
        error_summary=";".join(sorted(set(error_msgs))[:2]),
    )


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    # Linear interpolation
    i = q * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


# ── Dashboard state + rendering ─────────────────────────────────────────


class OptimizerState:
    def __init__(self, total_configs: int, print_to_stdout: bool = False):
        self.total_configs = total_configs
        self.config_index = 0
        self.config_name = ""
        self.phase = "init"
        self.config_started_at = time.monotonic()
        self.script_started_at = time.monotonic()
        self.log_lines: deque[str] = deque(maxlen=20)
        self.results: list[ConfigResult] = []
        self.current_cells: list[CellResult] = []
        # When True, every ``append_log`` line is also printed to
        # stdout with a timestamp prefix — used for the
        # backgrounded / nohup'd run where there's no live dashboard
        # and the log file is the user's only window into progress.
        self.print_to_stdout = print_to_stdout

    def begin_config(self, idx: int, name: str) -> None:
        self.config_index = idx
        self.config_name = name
        self.phase = "starting"
        self.config_started_at = time.monotonic()
        self.current_cells = []
        self.log_lines.clear()
        if self.print_to_stdout:
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                f"=== Config {idx + 1}/{self.total_configs}: {name} ===",
                flush=True,
            )

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        if self.print_to_stdout:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] phase: {phase}",
                flush=True,
            )

    def append_log(self, msg: str) -> None:
        for line in msg.splitlines():
            if not line.strip():
                continue
            line = line.rstrip()
            self.log_lines.append(line)
            if self.print_to_stdout:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {line}",
                    flush=True,
                )

    def push_cell_result(self, r: CellResult) -> None:
        self.current_cells.append(r)

    def upsert_result(self, r: ConfigResult) -> None:
        """Replace any prior entry for this config name (in place,
        preserving order) or append. Resume marks a config
        ``in_progress`` at start and overwrites with the final
        ``ok`` / ``launch_failed`` later."""
        for i, x in enumerate(self.results):
            if x.name == r.name:
                self.results[i] = r
                return
        self.results.append(r)


def _fmt_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def render(state: OptimizerState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer"),
    )
    layout["body"].split_row(
        Layout(name="status", ratio=1),
        Layout(name="logs", ratio=2),
    )
    layout["footer"].size = max(8, min(20, len(state.results) + 6))

    header = Text()
    header.append("Engine Optimizer  ", style="bold cyan")
    header.append(f"({state.config_index + 1}/{state.total_configs})  ", style="dim")
    header.append(f"elapsed {_fmt_seconds(time.monotonic() - state.script_started_at)}", style="dim")
    layout["header"].update(Panel(header, border_style="cyan"))

    status_tbl = Table(show_header=False, expand=True, box=None)
    status_tbl.add_column("k", style="bold")
    status_tbl.add_column("v")
    status_tbl.add_row("Config", state.config_name or "—")
    status_tbl.add_row("Phase", state.phase)
    status_tbl.add_row(
        "Config elapsed",
        _fmt_seconds(time.monotonic() - state.config_started_at),
    )
    status_tbl.add_row(
        "Cells done", f"{len(state.current_cells)}/{len(TEST_CELLS)}",
    )
    cur_cells_tbl = Table(title="This config — cells", expand=True)
    cur_cells_tbl.add_column("cell")
    cur_cells_tbl.add_column("n", justify="right")
    cur_cells_tbl.add_column("err", justify="right")
    cur_cells_tbl.add_column("t/o", justify="right")
    cur_cells_tbl.add_column("ttft p95", justify="right")
    cur_cells_tbl.add_column("tpot p95", justify="right")
    cur_cells_tbl.add_column("tok/s", justify="right")
    for r in state.current_cells:
        timeouts = getattr(r, "timeouts", 0) or 0
        cur_cells_tbl.add_row(
            r.cell_name, str(r.samples), str(r.errors),
            Text(str(timeouts), style="yellow") if timeouts else "0",
            f"{r.ttft_p95_ms:.0f}" if r.ttft_p95_ms else "—",
            f"{r.tpot_p95_ms:.0f}" if r.tpot_p95_ms else "—",
            f"{r.throughput_out_tok_s:.1f}" if r.throughput_out_tok_s else "—",
        )
    layout["status"].update(Panel(Group(status_tbl, cur_cells_tbl), title="Status"))

    log_text = Text()
    for line in list(state.log_lines)[-15:]:
        log_text.append(line + "\n")
    layout["logs"].update(Panel(log_text, title="Launch / health log"))

    # Cross-config comparison columns chosen for the AMD pain point:
    # best-case latency (sanity), pain TTFT at two concurrencies (does
    # it work? does it scale?), pain throughput, and decode-regime
    # throughput. Full per-cell data is always in the JSON.
    #
    # TTFT cells annotate with timeout count when nonzero —
    # "12087 *2" means 14 of 16 requests completed at 12 s p95, but
    # 2 were killed at the per-cell timeout. Distinguishes a real win
    # from "fast on the few that didn't get stuck."
    summary_tbl = Table(title="Configs (so far)  (* = killed by per-cell timeout)", expand=True)
    summary_tbl.add_column("config")
    summary_tbl.add_column("status")
    summary_tbl.add_column("launch s", justify="right")
    summary_tbl.add_column("c=1 ttft", justify="right")
    summary_tbl.add_column("pain8 ttft", justify="right")
    summary_tbl.add_column("pain16 ttft", justify="right")
    summary_tbl.add_column("pain8 tok/s", justify="right")
    summary_tbl.add_column("longout tok/s", justify="right")
    for cr in state.results:
        c1 = next((c for c in cr.cells if c.cell_name == "single_stream_short"), None)
        p8 = next((c for c in cr.cells if c.cell_name == "long_pain_8"), None)
        p16 = next((c for c in cr.cells if c.cell_name == "long_pain_16"), None)
        lo = next((c for c in cr.cells if c.cell_name == "long_output"), None)
        style = {"ok": "green", "launch_failed": "red", "skipped": "yellow"}.get(cr.status, "")
        summary_tbl.add_row(
            cr.name,
            Text(cr.status, style=style),
            f"{cr.launch_seconds:.0f}" if cr.launch_seconds else "—",
            _fmt_ttft_with_timeouts(c1),
            _fmt_ttft_with_timeouts(p8),
            _fmt_ttft_with_timeouts(p16),
            f"{p8.throughput_out_tok_s:.1f}" if p8 and p8.throughput_out_tok_s else "—",
            f"{lo.throughput_out_tok_s:.1f}" if lo and lo.throughput_out_tok_s else "—",
        )
    layout["footer"].update(summary_tbl)
    return layout


def _fmt_ttft_with_timeouts(c: CellResult | None) -> str:
    """Render p95 TTFT, suffixed with ``*N`` when N requests were
    killed by the per-cell timeout. ``*`` reads as "warning" — the
    p95 is from the surviving requests only, so a "fast" number with
    a high ``*`` count means the config has a tail problem masked by
    the dashboard summary."""
    if c is None or c.ttft_p95_ms is None:
        return "—"
    timeouts = getattr(c, "timeouts", 0) or 0
    base = f"{c.ttft_p95_ms:.0f}"
    if timeouts:
        return f"{base} *{timeouts}"
    return base


# ── Main loop ───────────────────────────────────────────────────────────


async def run_config(
    cfg: EngineConfig, state: OptimizerState, prompts: dict[str, list[str]],
    save: Callable[[], None],
) -> ConfigResult:
    """Run one config end-to-end. ``save`` is invoked after every
    cell so a mid-config crash leaves up-to-date partial data on
    disk and the resume path can pick up at the next config."""
    state.append_log(f"== {cfg.name} ==")
    state.append_log(cfg.description)

    # Persist an in-progress marker BEFORE we start touching docker.
    # If the host crashes during launch, the resume logic will see
    # ``in_progress`` (non-terminal) and re-attempt this config from
    # scratch on next invocation.
    state.upsert_result(ConfigResult(
        name=cfg.name, description=cfg.description, status="in_progress",
    ))
    save()

    cleanup_containers(state)
    state.set_phase("launching")
    launch_t0 = time.monotonic()
    try:
        for r in cfg.replicas:
            cid = docker_launch(cfg, r)
            state.append_log(f"launched {r.name} ({cid[:12]})")
    except Exception as e:  # noqa: BLE001
        state.append_log(f"!! launch failed: {e}")
        cleanup_containers(state)
        result = ConfigResult(
            name=cfg.name, description=cfg.description,
            status="launch_failed", failure_reason=str(e)[:500],
        )
        state.upsert_result(result)
        save()
        return result
    healthy = await wait_for_health(cfg, state)
    launch_seconds = time.monotonic() - launch_t0
    if not healthy:
        # Capture more log context before tearing down.
        for r in cfg.replicas:
            state.append_log(f"-- {r.name} tail --")
            state.append_log(docker_logs_tail(r.name, 50))
        cleanup_containers(state)
        result = ConfigResult(
            name=cfg.name, description=cfg.description,
            status="launch_failed",
            launch_seconds=launch_seconds,
            failure_reason="health check timeout",
        )
        state.upsert_result(result)
        save()
        return result
    state.set_phase("warmup")
    state.append_log(f"warmup ({WARMUP_REQUESTS} reqs)")
    warmup_clients = [
        AsyncOpenAI(base_url=f"http://{HOST}:{r.port}/v1", api_key="EMPTY")
        for r in cfg.replicas
    ]
    warmup_prompt = build_prompt(64)
    for i in range(WARMUP_REQUESTS):
        await measure_one_request(
            warmup_clients[i % len(warmup_clients)], warmup_prompt,
            output_tokens=32, timeout_s=60.0,
        )
    state.append_log("warmup done")
    cell_results: list[CellResult] = []
    for cell in TEST_CELLS:
        cr = await measure_cell(cfg, cell, prompts[cell.name], state)
        cell_results.append(cr)
        state.push_cell_result(cr)
        state.append_log(
            f"{cell.name}: n={cr.samples} err={cr.errors} "
            f"timeouts={cr.timeouts} "
            f"ttft_p95={cr.ttft_p95_ms or 0:.0f}ms "
            f"tpot_p95={cr.tpot_p95_ms or 0:.0f}ms "
            f"tok/s={cr.throughput_out_tok_s or 0:.1f}"
        )
        # Persist after every cell — losing one cell on crash is
        # tolerable; losing N is not.
        state.upsert_result(ConfigResult(
            name=cfg.name, description=cfg.description,
            status="in_progress", launch_seconds=launch_seconds,
            cells=list(cell_results),
        ))
        save()
    cleanup_containers(state)
    final = ConfigResult(
        name=cfg.name, description=cfg.description, status="ok",
        launch_seconds=launch_seconds, cells=cell_results,
    )
    state.upsert_result(final)
    save()
    return final


def _load_existing(path: Path) -> dict | None:
    """Read prior results from the output JSON, if any. Returns None
    on missing file or unparseable content (treat as no prior data)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _completed_names(doc: dict | None) -> set[str]:
    """Configs whose status is terminal — skip on resume.

    ``in_progress`` is intentionally NOT terminal: an interrupted
    config is re-run from scratch. ``launch_failed`` IS terminal:
    that config has been determined not to work on this host, no
    point retrying."""
    if not doc:
        return set()
    return {
        c["name"] for c in doc.get("configs", [])
        if c.get("status") in ("ok", "launch_failed")
    }


def _prior_results(doc: dict | None) -> list[ConfigResult]:
    """Reconstruct ConfigResult dataclasses from JSON for state
    pre-population (so the dashboard / final summary include
    previously-completed configs even on a resume).

    Tolerant of schema drift — both directions:
    * Old JSON missing newly-added fields (e.g. ``timeouts``):
      dataclass defaults fill in.
    * Old JSON with fields that no longer exist on the dataclass:
      filtered out before construction so we don't TypeError.
    """
    if not doc:
        return []
    cell_fields = {f.name for f in dataclasses.fields(CellResult)}
    cfg_fields = {f.name for f in dataclasses.fields(ConfigResult)}
    out: list[ConfigResult] = []
    for c in doc.get("configs", []):
        cell_dicts = c.get("cells", []) or []
        cells = [
            CellResult(**{k: v for k, v in cd.items() if k in cell_fields})
            for cd in cell_dicts
        ]
        merged = {k: v for k, v in c.items() if k in cfg_fields}
        merged["cells"] = cells
        out.append(ConfigResult(**merged))
    return out


async def _iterate_configs(
    state: OptimizerState,
    selected: list[EngineConfig],
    completed: set[str],
    prompts: dict[str, list[str]],
    save: Callable[[], None],
) -> None:
    """Shared driver used by both the dashboard and the non-tty
    paths. Skips configs whose name is in ``completed``."""
    for i, cfg in enumerate(selected):
        if cfg.name in completed:
            continue
        state.begin_config(i, cfg.name)
        await run_config(cfg, state, prompts, save)


async def main_async(
    out_path: Path,
    only: list[str] | None,
    new_run: bool,
) -> None:
    selected = [c for c in CONFIGS if (not only or c.name in only)]
    if only:
        unknown = set(only) - {c.name for c in CONFIGS}
        if unknown:
            raise SystemExit(f"unknown configs: {sorted(unknown)}")

    if new_run and out_path.exists():
        out_path.unlink()
        print(f"--new-run: removed prior {out_path}")

    existing = _load_existing(out_path)
    completed = _completed_names(existing)
    use_dashboard = sys.stdout.isatty()

    state = OptimizerState(
        total_configs=len(selected), print_to_stdout=not use_dashboard,
    )
    # Pre-populate state with prior results so the dashboard /
    # summary reflect the full run, not just what we re-attempted.
    # Restrict to the selected set so a stale entry from an earlier
    # --only invocation doesn't pollute this run.
    selected_names = {c.name for c in selected}
    state.results = [
        r for r in _prior_results(existing) if r.name in selected_names
    ]

    # Always print a startup banner so the user sees clearly whether
    # this is a fresh run or a resume — and exactly which configs the
    # resume will skip / re-run / attempt for the first time.
    print(f"Output: {out_path}")
    if existing is None:
        print(
            f"No prior run at this path — starting fresh "
            f"({len(selected)} configs)."
        )
    else:
        skipped = sorted(completed & selected_names)
        in_prog = sorted(
            c["name"] for c in existing.get("configs", [])
            if c.get("status") == "in_progress" and c.get("name") in selected_names
        )
        todo = [c.name for c in selected if c.name not in completed]
        print(
            f"Loaded prior JSON ({len(existing.get('configs', []))} entries)."
        )
        print(f"  done   ({len(skipped)}): {', '.join(skipped) or '—'}")
        if in_prog:
            print(
                f"  in progress ({len(in_prog)}): {', '.join(in_prog)} "
                f"(will re-run from scratch)"
            )
        print(f"  todo   ({len(todo)}): {', '.join(todo) or '—'}")
        if not todo:
            _print_summary(state)
            _save_json(out_path, state)
            return

    prompts = make_prompts(TEST_CELLS)
    save = lambda: _save_json(out_path, state)

    if use_dashboard:
        console = Console()
        with Live(
            render(state), console=console, refresh_per_second=2, screen=True,
        ) as live:
            stop_render = asyncio.Event()

            async def render_loop() -> None:
                while not stop_render.is_set():
                    live.update(render(state))
                    await asyncio.sleep(0.5)

            rt = asyncio.create_task(render_loop())
            try:
                await _iterate_configs(state, selected, completed, prompts, save)
            finally:
                stop_render.set()
                await rt
    else:
        # Backgrounded / nohup'd run: no dashboard, just timestamped
        # progress lines via state.print_to_stdout=True.
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Engine optimizer starting: {len(selected)} configs, "
            f"output -> {out_path}"
        )
        await _iterate_configs(state, selected, completed, prompts, save)

    _print_summary(state)
    _save_json(out_path, state)
    print(f"\nSaved results -> {out_path}")


def _save_json(path: Path, state: OptimizerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "image": IMAGE,
        "model": MODEL_PATH,
        "served_model_name": SERVED_NAME,
        "host": HOST,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_cells": [dataclasses.asdict(c) for c in TEST_CELLS],
        "configs": [dataclasses.asdict(r) for r in state.results],
    }
    path.write_text(json.dumps(doc, indent=2))


def _print_summary(state: OptimizerState) -> None:
    print("\n=== Engine Optimizer Summary ===\n")
    for cr in state.results:
        if cr.status != "ok":
            print(f"[{cr.status:13}] {cr.name}: {cr.failure_reason}")
            continue
        print(f"[ok           ] {cr.name}  (launched in {cr.launch_seconds:.0f}s)")
        for c in cr.cells:
            timeouts = getattr(c, "timeouts", 0) or 0
            t_flag = f"  t/o={timeouts}" if timeouts else ""
            print(
                f"    {c.cell_name:24}  n={c.samples:2}  err={c.errors}{t_flag}  "
                f"ttft p50={c.ttft_p50_ms or 0:6.0f}ms  p95={c.ttft_p95_ms or 0:6.0f}ms  "
                f"tpot p95={c.tpot_p95_ms or 0:5.0f}ms  "
                f"out_tok/s={c.throughput_out_tok_s or 0:6.1f}"
            )


DEFAULT_OUT = Path("runs/engine_optimizer/run.json")


# ── Read-only watch dashboard (separate process) ────────────────────────
#
# When the optimizer runs in the background (via ``make optimize-engine``),
# stdout is a log file — not a TTY — so the live rich dashboard isn't
# rendered. ``--watch`` (exposed via ``make optimize-dashboard``) opens a
# read-only dashboard from any other terminal, polling the JSON + log
# file the optimizer is writing. Same layout, same data, no interference
# with the running optimizer (read-only).


def _latest_log(log_dir: Path) -> Optional[Path]:
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("optimizer_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _tail_log(path: Path | None, n: int = 20) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", errors="replace") as f:
            return [
                line.rstrip()
                for line in f.readlines()[-n:]
                if line.strip()
            ]
    except OSError:
        return []


def _state_from_disk(
    json_path: Path, log_path: Path | None, total_configs: int,
) -> OptimizerState:
    """Build an OptimizerState by reading the JSON + log file. The
    state's "current config" is whichever entry has status='in_progress'
    (set by the running optimizer at the start of each config and
    overwritten at the end)."""
    state = OptimizerState(total_configs=total_configs)
    doc = _load_existing(json_path)
    if doc is not None:
        state.results = _prior_results(doc)
    for line in _tail_log(log_path):
        state.log_lines.append(line)

    current = next(
        (r for r in state.results if r.status == "in_progress"), None,
    )
    if current is not None:
        for i, c in enumerate(CONFIGS):
            if c.name == current.name:
                state.config_index = i
                break
        state.config_name = current.name
        state.current_cells = list(current.cells)
        state.phase = (
            f"running ({len(current.cells)}/{len(TEST_CELLS)} cells done)"
        )
    elif state.results:
        # No active config — maybe between configs, or the run finished.
        # Show the most-recently-completed entry as "current".
        last = state.results[-1]
        for i, c in enumerate(CONFIGS):
            if c.name == last.name:
                state.config_index = i
                break
        state.config_name = last.name
        state.current_cells = list(last.cells)
        state.phase = f"latest: {last.status}"
    return state


def watch_dashboard(json_path: Path, refresh_s: float = 2.0) -> None:
    """Render a live dashboard against an on-disk run.json. Read-only —
    the running optimizer (in another shell) writes; we just observe."""
    log_dir = json_path.parent
    console = Console()
    state = _state_from_disk(json_path, _latest_log(log_dir), len(CONFIGS))
    if not json_path.exists():
        console.print(
            f"[yellow]Waiting for {json_path} to appear...[/yellow]"
        )
    with Live(
        render(state), console=console, refresh_per_second=1, screen=True,
    ) as live:
        try:
            while True:
                state = _state_from_disk(
                    json_path, _latest_log(log_dir), len(CONFIGS),
                )
                live.update(render(state))
                time.sleep(refresh_s)
        except KeyboardInterrupt:
            pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Iterate engine configs, measure TTFT/TPOT, pick the best."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            "Where to write the JSON artifact. Default is a stable path "
            f"({DEFAULT_OUT}) so re-running picks up from where the previous "
            "invocation stopped. Override with --out path/to/other.json for "
            "ad-hoc one-offs."
        ),
    )
    p.add_argument(
        "--only", nargs="+",
        help=(
            "Run only these config names (default: all). Combines with "
            "resume: configs not in the list are ignored; configs in the "
            "list with terminal status in the existing JSON are still "
            "skipped (use --new-run to force re-run)."
        ),
    )
    p.add_argument(
        "--new-run", action="store_true",
        help=(
            "Wipe the existing JSON at --out and start fresh. Default is "
            "to resume — configs already marked 'ok' or 'launch_failed' "
            "are skipped, in_progress / missing configs are re-run."
        ),
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the registered configs and exit.",
    )
    p.add_argument(
        "--watch", action="store_true",
        help=(
            "Read-only dashboard mode: poll the --out JSON + the latest "
            ".log file in the same directory and render the live view "
            "from another terminal. Use this against a backgrounded "
            "``make optimize-engine`` run."
        ),
    )
    args = p.parse_args()

    if args.list:
        for c in CONFIGS:
            print(f"  {c.name:25} {c.description}")
            if c.expected_outcome:
                print(f"    expect: {c.expected_outcome}")
        return

    if args.watch:
        watch_dashboard(args.out)
        return

    asyncio.run(main_async(args.out, args.only, args.new_run))


if __name__ == "__main__":
    main()
