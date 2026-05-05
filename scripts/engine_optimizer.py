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
    cleanup_containers

If a launch fails (container exits during init, /v1/models never
responds), the failure is recorded with the last 50 lines of docker
logs and the optimiser moves to the next config.

Output: a JSON artifact at ``runs/engine_optimizer_<ts>.json`` plus a
ranked summary printed to stdout. Live progress is shown via a rich-
based dashboard: current config, phase, log tail, and the
results-so-far table.

Tweak the CONFIGS list at the top to add or remove launch shapes —
the dataclass is intentionally flat so a new entry is a few lines."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import statistics
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
REQUEST_TIMEOUT_S = 600


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
        name="single_replica_64core",
        description="One container, all 64 cores, both NUMA. AMD analog of ctx_kv_xl.",
        replicas=[
            ReplicaSpec(
                name="vllm-single", port=8000,
                cpuset_cpus="0-63", cpuset_mems=None,
                env={
                    "VLLM_CPU_KVCACHE_SPACE": "160",
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
# 4 cells span: short single-stream, typical chat at moderate concurrency,
# long-context moderate, and short-prompt at higher concurrency for
# throughput. Light enough to keep one config under ~3 min total.

TEST_CELLS: list[TestCell] = [
    TestCell("single_stream_short",  input_tokens=128, output_tokens=128, concurrency=1),
    TestCell("chat_concurrent",      input_tokens=512, output_tokens=256, concurrency=4),
    TestCell("long_ctx_moderate",    input_tokens=2048, output_tokens=256, concurrency=2),
    TestCell("short_throughput",     input_tokens=128, output_tokens=128, concurrency=8),
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
    """Stop any container whose name starts with ``vllm-`` so we don't
    leave a pinned replica from an earlier config eating cores."""
    state.set_phase("cleanup")
    state.append_log("== cleanup ==")
    res = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=vllm-"],
        capture_output=True, text=True,
    )
    cids = res.stdout.split()
    if cids:
        state.append_log(f"stopping {len(cids)} container(s): {' '.join(cids)}")
        subprocess.run(["docker", "stop", "-t", "10", *cids], capture_output=True)
    # Ports take a moment to release.
    time.sleep(2)


def docker_launch(cfg: EngineConfig, replica: ReplicaSpec) -> str:
    """Build and run the docker command for one replica. Returns the
    container ID."""
    args = [
        "docker", "run", "-d", "--rm",
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


async def measure_one_request(
    client: AsyncOpenAI, prompt: str, output_tokens: int,
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[str]]:
    """Returns (ttft_ms, total_ms, observed_output_tokens, error)."""
    t_start = time.monotonic()
    t_first: Optional[float] = None
    n_tokens = 0
    try:
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
            return None, None, 0, "no tokens emitted"
        t_end = time.monotonic()
        return (t_first - t_start) * 1000.0, (t_end - t_start) * 1000.0, n_tokens, None
    except Exception as e:  # noqa: BLE001
        return None, None, None, str(e)[:200]


async def measure_cell(
    cfg: EngineConfig, cell: TestCell, prompts: list[str],
    state: "OptimizerState",
) -> CellResult:
    state.set_phase(f"cell {cell.name} (in={cell.input_tokens} out={cell.output_tokens} c={cell.concurrency})")
    # Round-robin requests across replicas.
    clients = [
        AsyncOpenAI(base_url=f"http://{HOST}:{r.port}/v1", api_key="EMPTY")
        for r in cfg.replicas
    ]
    tasks = [
        measure_one_request(clients[i % len(clients)], prompts[i], cell.output_tokens)
        for i in range(cell.concurrency)
    ]
    t0 = time.monotonic()
    results = await asyncio.gather(*tasks)
    dur = time.monotonic() - t0
    ttfts, tpots = [], []
    out_tok_total = 0
    errors = 0
    error_msgs: list[str] = []
    for ttft, total, n_out, err in results:
        if err is not None:
            errors += 1
            error_msgs.append(err)
            continue
        if ttft is None or total is None or n_out is None:
            errors += 1
            continue
        ttfts.append(ttft)
        if n_out > 0 and total > ttft:
            tpots.append((total - ttft) / max(1, n_out))
        out_tok_total += n_out
    samples = len(ttfts)
    return CellResult(
        cell_name=cell.name,
        samples=samples,
        errors=errors,
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
    def __init__(self, total_configs: int):
        self.total_configs = total_configs
        self.config_index = 0
        self.config_name = ""
        self.phase = "init"
        self.config_started_at = time.monotonic()
        self.script_started_at = time.monotonic()
        self.log_lines: deque[str] = deque(maxlen=20)
        self.results: list[ConfigResult] = []
        self.current_cells: list[CellResult] = []

    def begin_config(self, idx: int, name: str) -> None:
        self.config_index = idx
        self.config_name = name
        self.phase = "starting"
        self.config_started_at = time.monotonic()
        self.current_cells = []
        self.log_lines.clear()

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def append_log(self, msg: str) -> None:
        for line in msg.splitlines():
            if line.strip():
                self.log_lines.append(line.rstrip())

    def push_cell_result(self, r: CellResult) -> None:
        self.current_cells.append(r)


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
    cur_cells_tbl.add_column("ttft p95", justify="right")
    cur_cells_tbl.add_column("tpot p95", justify="right")
    cur_cells_tbl.add_column("tok/s", justify="right")
    for r in state.current_cells:
        cur_cells_tbl.add_row(
            r.cell_name, str(r.samples), str(r.errors),
            f"{r.ttft_p95_ms:.0f}" if r.ttft_p95_ms else "—",
            f"{r.tpot_p95_ms:.0f}" if r.tpot_p95_ms else "—",
            f"{r.throughput_out_tok_s:.1f}" if r.throughput_out_tok_s else "—",
        )
    layout["status"].update(Panel(Group(status_tbl, cur_cells_tbl), title="Status"))

    log_text = Text()
    for line in list(state.log_lines)[-15:]:
        log_text.append(line + "\n")
    layout["logs"].update(Panel(log_text, title="Launch / health log"))

    summary_tbl = Table(title="Configs (so far)", expand=True)
    summary_tbl.add_column("config")
    summary_tbl.add_column("status")
    summary_tbl.add_column("launch s", justify="right")
    summary_tbl.add_column("c=1 ttft", justify="right")
    summary_tbl.add_column("c=4 ttft", justify="right")
    summary_tbl.add_column("c=1 tok/s", justify="right")
    summary_tbl.add_column("c=8 tok/s", justify="right")
    for cr in state.results:
        c1 = next((c for c in cr.cells if c.cell_name == "single_stream_short"), None)
        c4 = next((c for c in cr.cells if c.cell_name == "chat_concurrent"), None)
        c8 = next((c for c in cr.cells if c.cell_name == "short_throughput"), None)
        style = {"ok": "green", "launch_failed": "red", "skipped": "yellow"}.get(cr.status, "")
        summary_tbl.add_row(
            cr.name,
            Text(cr.status, style=style),
            f"{cr.launch_seconds:.0f}" if cr.launch_seconds else "—",
            f"{c1.ttft_p95_ms:.0f}" if c1 and c1.ttft_p95_ms else "—",
            f"{c4.ttft_p95_ms:.0f}" if c4 and c4.ttft_p95_ms else "—",
            f"{c1.throughput_out_tok_s:.1f}" if c1 and c1.throughput_out_tok_s else "—",
            f"{c8.throughput_out_tok_s:.1f}" if c8 and c8.throughput_out_tok_s else "—",
        )
    layout["footer"].update(summary_tbl)
    return layout


# ── Main loop ───────────────────────────────────────────────────────────


async def run_config(
    cfg: EngineConfig, state: OptimizerState, prompts: dict[str, list[str]],
) -> ConfigResult:
    state.append_log(f"== {cfg.name} ==")
    state.append_log(cfg.description)
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
        return ConfigResult(
            name=cfg.name, description=cfg.description,
            status="launch_failed", failure_reason=str(e)[:500],
        )
    healthy = await wait_for_health(cfg, state)
    launch_seconds = time.monotonic() - launch_t0
    if not healthy:
        # Capture more log context before tearing down.
        for r in cfg.replicas:
            state.append_log(f"-- {r.name} tail --")
            state.append_log(docker_logs_tail(r.name, 50))
        cleanup_containers(state)
        return ConfigResult(
            name=cfg.name, description=cfg.description,
            status="launch_failed",
            launch_seconds=launch_seconds,
            failure_reason="health check timeout",
        )
    state.set_phase("warmup")
    state.append_log(f"warmup ({WARMUP_REQUESTS} reqs)")
    warmup_clients = [
        AsyncOpenAI(base_url=f"http://{HOST}:{r.port}/v1", api_key="EMPTY")
        for r in cfg.replicas
    ]
    warmup_prompt = build_prompt(64)
    for i in range(WARMUP_REQUESTS):
        await measure_one_request(
            warmup_clients[i % len(warmup_clients)], warmup_prompt, output_tokens=32,
        )
    state.append_log("warmup done")
    cell_results: list[CellResult] = []
    for cell in TEST_CELLS:
        cr = await measure_cell(cfg, cell, prompts[cell.name], state)
        cell_results.append(cr)
        state.push_cell_result(cr)
        state.append_log(
            f"{cell.name}: n={cr.samples} err={cr.errors} "
            f"ttft_p95={cr.ttft_p95_ms or 0:.0f}ms "
            f"tpot_p95={cr.tpot_p95_ms or 0:.0f}ms"
        )
    cleanup_containers(state)
    return ConfigResult(
        name=cfg.name, description=cfg.description, status="ok",
        launch_seconds=launch_seconds, cells=cell_results,
    )


async def main_async(out_path: Path, only: list[str] | None) -> None:
    selected = [c for c in CONFIGS if (not only or c.name in only)]
    if only:
        unknown = set(only) - {c.name for c in CONFIGS}
        if unknown:
            raise SystemExit(f"unknown configs: {sorted(unknown)}")
    state = OptimizerState(total_configs=len(selected))
    prompts = make_prompts(TEST_CELLS)

    console = Console()
    with Live(render(state), console=console, refresh_per_second=2, screen=True) as live:
        # Spawn a background renderer task so the dashboard stays fresh
        # while async docker/health calls block.
        stop_render = asyncio.Event()

        async def render_loop() -> None:
            while not stop_render.is_set():
                live.update(render(state))
                await asyncio.sleep(0.5)

        rt = asyncio.create_task(render_loop())
        try:
            for i, cfg in enumerate(selected):
                state.begin_config(i, cfg.name)
                result = await run_config(cfg, state, prompts)
                state.results.append(result)
                # Persist incrementally so a crash mid-run doesn't lose data.
                _save_json(out_path, state)
        finally:
            stop_render.set()
            await rt

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
            print(
                f"    {c.cell_name:24}  n={c.samples:2}  err={c.errors}  "
                f"ttft p50={c.ttft_p50_ms or 0:6.0f}ms  p95={c.ttft_p95_ms or 0:6.0f}ms  "
                f"tpot p95={c.tpot_p95_ms or 0:5.0f}ms  "
                f"out_tok/s={c.throughput_out_tok_s or 0:6.1f}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Iterate engine configs, measure TTFT/TPOT, pick the best."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(f"runs/engine_optimizer_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"),
        help="Where to write the JSON artifact (default: runs/engine_optimizer_<ts>.json)",
    )
    p.add_argument(
        "--only", nargs="+",
        help="Run only these config names (default: all). Useful for incremental work.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the registered configs and exit.",
    )
    args = p.parse_args()

    if args.list:
        for c in CONFIGS:
            print(f"  {c.name:25} {c.description}")
            if c.expected_outcome:
                print(f"    expect: {c.expected_outcome}")
        return

    asyncio.run(main_async(args.out, args.only))


if __name__ == "__main__":
    main()
