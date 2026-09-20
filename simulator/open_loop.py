"""Open-loop cohort orchestration — arrival-rate capacity search.

The closed-loop pool methodology cannot see the capacity limit: a
fixed pool of users throttles its own offered load when the engine
slows (each user's cycle stretches, arrivals/sec drop, the queue
drains — bounded at N by construction). This module replaces the
control variable: *sessions arrive as a Poisson process at rate λ*
regardless of engine state, and capacity is the λ at which the
engine's waiting queue transitions from stationary to divergent.

Per λ window:  set rate → settle → measure (SLA on completed turns +
per-second queue-depth series) → verdict (simulator.stability) →
next λ from the two-knee rate search (simulator.rate_search). After a
divergent window the backlog is drained before the next rate.

Load generation is sharded across worker subprocesses
(simulator.loadgen_worker) — Poisson superposition makes k workers at
λ/k exactly λ. The generator's own honesty signal is *arrival
tardiness*: when arrivals fall behind their wall-clock schedule the
coordinator adds workers, and only when scaling stops helping does a
window get recorded as ``client_limited``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bus import BUS
from .config import Config
from .cpu_binding import expand_thread_binding
from .database import AGGREGATE_COLUMN_NAMES, Database
from .engines import Engine, make_engine
from .measurement import _classify_status, _percentile, _wilson_ci
from .personas import Cohort, get_cohort
from .preflight import preflight_check
from .rate_search import (
    CLIENT_LIMITED,
    DIVERGENT,
    STABLE,
    RateStep,
    RateStepper,
)
from .runs import resolve_run_dir
from .stability import INCONCLUSIVE, assess_queue_stability, theil_sen_slope
from .telemetry import MeasurementTelemetry

log = logging.getLogger(__name__)

# Fraction of a window's arrivals allowed to be tardy (later than
# arrivals.TARDY_THRESHOLD_MS vs their wall-clock schedule) before the
# generator is judged saturated. Computed from cumulative counters so
# the verdict is strictly PER WINDOW — one bad burst can't smear into
# later windows the way a trailing-percentile buffer does.
TARDY_FRACTION_LIMIT = 0.02
TARDY_MIN_COUNT = 5
# Worker event-loop lag limit (same meaning as the closed-loop
# CLIENT_SATURATION_LAG_MS: past this, latencies measure the client).
WORKER_LAG_LIMIT_MS = 1000.0
# Settling detector: the trailing population drift (Theil-Sen slope ×
# trailing window) must be within this fraction of the trailing mean
# before a window opens. See SimulationConfig.open_loop_settle_*.
SETTLE_TOLERANCE = 0.05
SETTLE_WINDOW_CAP_S = 300


def _population_settled(
    series: list[float], window_n: int, tol: float = SETTLE_TOLERANCE,
) -> tuple[bool, float]:
    """Is the active-session population flat over its trailing
    ``window_n`` samples?

    Returns ``(settled, drift_fraction)`` where drift is the Theil-Sen
    slope projected across the trailing window, as a fraction of the
    trailing mean. Settled when that drift is within ``tol`` of the
    mean plus one session (the absolute slack lets a population of a
    dozen integer-valued sessions settle at all). Theil-Sen — the
    median pairwise slope — is what makes this robust to the Poisson
    jitter of arrivals and departures: endpoint noise does not read
    as a ramp, while a genuine ramp toward a new equilibrium does.
    """
    if window_n < 2 or len(series) < window_n:
        return False, float("inf")
    tail = series[-window_n:]
    mean = statistics.fmean(tail)
    drift = abs(theil_sen_slope(tail, 1.0)) * window_n
    limit = tol * mean + 1.0
    return drift <= limit, (drift / mean if mean > 0 else 0.0)


class EngineBrokenError(RuntimeError):
    """The engine is up but cannot serve — surfaced to the user with
    the engine's own error instead of an endless error counter."""


def _engine_broken(errors: int, completions: int,
                   generated_tokens: int | None = None) -> bool:
    """Runaway-error fuse. A broken engine fails every request almost
    instantly, so nothing would ever stop the run on its own — the
    queue stays empty (stable!) while errors pile up forever.

    Completions alone cannot tell breakage from slowness. A window is
    120s; a turn with 2048 pinned output tokens takes ~290s at load,
    so a perfectly healthy engine can show ZERO completions for a
    whole window while every request is still streaming — and any
    client-timeout failures alongside them used to trip this fuse and
    abort a run that had already served 50k turns.

    Generated tokens settle it: an engine that is up but cannot serve
    produces none. If tokens are flowing, the engine is working, and
    overload is the stability statistics' job to judge, not ours.
    When the counter is unavailable we fall back to the old test
    rather than weakening the fuse.
    """
    if generated_tokens is not None and generated_tokens > 0:
        return False
    return (errors >= 25 and completions == 0) or \
           (errors >= 100 and errors > 4 * completions)


class _ErrorFuse:
    """Runaway-error fuse armed at window start.

    Deltas of the workers' cumulative error / completion counters and
    the engine's ``generation_tokens_total`` since arming feed
    ``_engine_broken``. The token side is read from the LAST GOOD
    scrape, not the latest attempt: a single failed /metrics scrape
    used to hand ``None`` to the fuse, which then fell back to the
    completions-only test and could abort a healthy-but-slow run —
    the exact false abort the token counter was added to prevent.
    """

    def __init__(self, errors: int, completions: int,
                 tokens_last_good: int | None):
        self.errors0 = int(errors)
        self.completions0 = int(completions)
        self.tokens0 = tokens_last_good

    def tripped(self, errors: int, completions: int,
                tokens_last_good: int | None) -> bool:
        err_d = int(errors) - self.errors0
        comp_d = int(completions) - self.completions0
        tok_d = (
            int(tokens_last_good) - int(self.tokens0)
            if tokens_last_good is not None and self.tokens0 is not None
            else None
        )
        return _engine_broken(err_d, comp_d, tok_d)


REQUEST_TIMEOUT_CEILING_S = 1800


def _request_timeout_for(cohort, sim) -> int:
    """Client patience sized to the WORKLOAD, not a flat 300s.

    A turn's honest worst case is its own SLA: time-to-first-token
    failure, then one output token per TPOT-failure interval. A
    workload with 2048 pinned output tokens needs ~290s at load, so a
    flat 300s timeout kills requests the engine was still serving —
    and those client-side kills then read as engine failures. We take
    the longest SLA bound across the cohort's personas (capped, and
    never below the configured value) so the ENGINE's behaviour
    decides the outcome; the stability statistics, not the stopwatch,
    are what call a collapse.
    """
    from .distributions import summarize
    from .personas import PERSONAS
    configured = int(sim.request_timeout_s)
    worst = 0.0
    for pid in cohort.persona_weights:
        p = PERSONAS.get(pid)
        if p is None:
            continue
        out_tokens = summarize(p.output_tokens).get("p90") or 0
        worst = max(worst, float(p.ttft_failure_seconds)
                    + float(out_tokens) * float(p.tpot_failure_ms) / 1000.0)
    scaled = min(REQUEST_TIMEOUT_CEILING_S, int(worst))
    if scaled > configured:
        log.info("request timeout raised %ds → %ds for this workload "
                 "(long pinned outputs need more than the default)",
                 configured, scaled)
        return scaled
    return configured


async def smoke_test_engine(engine, timeout_s: float = 180.0) -> None:
    """One real (tiny) completion against EVERY replica before load
    starts. /health only proves the server process is up — an engine
    launched with an unsupported flag combination (e.g. a
    kv-cache-dtype the model's attention backend rejects) passes
    health and then 500s every request. This is the preflight that
    catches it, with the engine's own error text."""
    import httpx
    for url in engine.replica_urls:
        def _probe(u: str = url):
            r = httpx.post(
                f"{u}/chat/completions",
                json={
                    "model": engine.api_model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
                headers={"Authorization": f"Bearer {engine.api_key}"},
                timeout=timeout_s,
            )
            if r.status_code != 200:
                raise EngineBrokenError(
                    f"engine at {u} is up but cannot serve requests "
                    f"(HTTP {r.status_code}): {r.text[:500]}"
                )
        try:
            await asyncio.to_thread(_probe)
        except EngineBrokenError:
            raise
        except Exception as e:  # noqa: BLE001
            raise EngineBrokenError(
                f"smoke request to {url} failed before any load was "
                f"offered: {e}"
            ) from e
    log.info("engine smoke test passed on %d replica(s)",
             len(engine.replica_urls))


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── Worker pool ──────────────────────────────────────────────────────


@dataclass
class _Worker:
    index: int
    proc: asyncio.subprocess.Process
    reader_task: asyncio.Task
    last_stat: dict = field(default_factory=dict)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    watch_task: asyncio.Task | None = None
    dead: bool = False


class WorkerPool:
    """k load-generator subprocesses sharing one total arrival rate.

    A worker that dies is noticed the moment its process exits (each
    worker has a ``proc.wait()`` watcher): it leaves the live set, its
    frozen counters stop being aggregated, the rate is redistributed
    over the survivors so the offered load stays λ, and a death record
    is queued for the window in progress — which is then discarded as
    a measurement of the generator, never of the engine.
    """

    def __init__(self, *, base_config: dict, log_dir: Path, max_workers: int):
        self._base = base_config
        self._log_dir = Path(log_dir)
        self.max_workers = max(1, max_workers)
        self.turn_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._workers: list[_Worker] = []
        self._rate_total = 0.0
        self._stderr_files: list = []
        self._next_index = 0
        self._stopping = False
        self._deaths: list[dict] = []
        self.deaths_total = 0

    @property
    def size(self) -> int:
        """Live workers (a dead one is removed as soon as it exits)."""
        return len(self._workers)

    def take_deaths(self) -> list[dict]:
        """Death records queued since the last call ({index,
        returncode, at_ms}). The window loop polls this each tick."""
        out, self._deaths = self._deaths, []
        return out

    async def scale_to(self, k: int) -> None:
        k = max(1, min(k, self.max_workers))
        while len(self._workers) < k:
            await self._spawn(self._next_index)
        # Redistribute the current rate over the new worker count.
        if self._rate_total > 0:
            await self.set_rate(self._rate_total)

    async def _spawn(self, index: int) -> None:
        self._next_index = index + 1
        cfg = dict(self._base)
        cfg["worker_index"] = index
        cfg["seed"] = (self._base.get("seed") or 0xC0FFEE) + index * 7919
        cfg_path = self._log_dir / f"loadgen_worker_{index}.json"
        cfg_path.write_text(json.dumps(cfg))
        stderr_path = self._log_dir / f"loadgen_worker_{index}.log"
        stderr_file = open(stderr_path, "w")
        self._stderr_files.append(stderr_file)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "simulator.loadgen_worker", str(cfg_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_file,
        )
        worker = _Worker(index=index, proc=proc, reader_task=None)  # type: ignore[arg-type]
        worker.reader_task = asyncio.create_task(self._read_loop(worker))
        worker.watch_task = asyncio.create_task(self._watch(worker))
        self._workers.append(worker)
        # Tokenizer load can take a while on first spawn; don't start
        # the window clock until the worker can actually generate.
        # Raced against the process exiting: a worker that crashes on
        # startup (bad config, import error) fails fast with its log
        # path instead of a 180 s timeout.
        ready = asyncio.ensure_future(worker.ready.wait())
        done, _ = await asyncio.wait(
            {ready, worker.watch_task}, timeout=180.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ready not in done:
            ready.cancel()
            if worker.watch_task in done:
                raise RuntimeError(
                    f"loadgen worker {index} exited with code "
                    f"{proc.returncode} before becoming ready "
                    f"(see {stderr_path})"
                )
            raise RuntimeError(
                f"loadgen worker {index} did not become ready in 180s "
                f"(see {stderr_path})"
            )
        log.info("loadgen worker %d ready (pid=%s)", index, proc.pid)

    async def _watch(self, worker: _Worker) -> None:
        """Notice a worker's death the moment it happens. Without this
        the coordinator kept aggregating the dead worker's frozen
        counters and offering λ·(k−1)/k while recording λ."""
        try:
            rc = await worker.proc.wait()
        except asyncio.CancelledError:
            return
        if self._stopping or worker.dead:
            return
        worker.dead = True
        self.deaths_total += 1
        record = {"index": worker.index, "returncode": rc, "at_ms": _now_ms()}
        self._deaths.append(record)
        if worker in self._workers:
            self._workers.remove(worker)
        log.error(
            "LOAD GENERATOR WORKER %d DIED (exit code %s, pid=%s) — the "
            "window in progress is discarded; see %s",
            worker.index, rc, worker.proc.pid,
            self._log_dir / f"loadgen_worker_{worker.index}.log",
        )
        # Survivors pick up the dead worker's share so the offered
        # load stays λ until the coordinator re-plans the pool.
        if self._rate_total > 0 and self._workers:
            await self.set_rate(self._rate_total)

    async def _read_loop(self, worker: _Worker) -> None:
        try:
            while True:
                line = await worker.proc.stdout.readline()
                if not line:
                    return
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                t = msg.get("t")
                if t == "turn":
                    await self.turn_queue.put(msg)
                elif t == "stat":
                    worker.last_stat = msg
                elif t == "ready":
                    worker.ready.set()
        except asyncio.CancelledError:
            pass

    async def _send(self, worker: _Worker, obj: dict) -> None:
        try:
            worker.proc.stdin.write(
                json.dumps(obj, separators=(",", ":")).encode() + b"\n",
            )
            await worker.proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError):
            log.warning("loadgen worker %d pipe closed", worker.index)

    async def set_rate(self, total_per_s: float) -> None:
        self._rate_total = max(0.0, total_per_s)
        if not self._workers:
            return
        share = self._rate_total / len(self._workers)
        for w in self._workers:
            await self._send(w, {"cmd": "rate", "per_s": share})

    async def drain(self) -> None:
        self._rate_total = 0.0
        for w in self._workers:
            await self._send(w, {"cmd": "drain"})

    async def trim(self, target_total: int) -> None:
        """Cancel newest sessions across workers down to
        ``target_total`` active (evenly split)."""
        if not self._workers:
            return
        per = max(0, int(target_total)) // len(self._workers)
        for w in self._workers:
            await self._send(w, {"cmd": "trim", "target": per})

    async def set_outstanding(self, total: int) -> None:
        """Saturation mode: hold ``total`` sessions active across the
        workers (each respawns as its sessions finish)."""
        self._rate_total = 0.0
        if not self._workers:
            return
        k = len(self._workers)
        base, rem = divmod(max(0, int(total)), k)
        for i, w in enumerate(self._workers):
            await self._send(w, {"cmd": "outstanding",
                                 "n": base + (1 if i < rem else 0)})

    async def reload_personas(self) -> None:
        for w in self._workers:
            await self._send(w, {"cmd": "reload_personas"})

    async def restart_sessions(self) -> None:
        """Abort every in-flight session; in saturation mode each
        respawns immediately with the current (fresh-loaded) persona.
        The instant shape swap: no waiting for old-shape requests."""
        for w in self._workers:
            await self._send(w, {"cmd": "restart"})

    def drain_turn_queue(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self.turn_queue.get_nowait())
            except asyncio.QueueEmpty:
                return out

    def aggregate(self) -> dict:
        """Summed live counters + honesty maxima across workers."""
        stats = [w.last_stat for w in self._workers if w.last_stat]
        if not stats:
            return {}
        sess_durs = [
            s["mean_session_s"] for s in stats
            if s.get("mean_session_s") is not None
        ]
        return {
            "workers": len(self._workers),
            "arrivals_total": sum(s.get("arrivals_total", 0) for s in stats),
            "tardy_total": sum(s.get("tardy_total", 0) for s in stats),
            "sessions_active": sum(s.get("sessions_active", 0) for s in stats),
            "sessions_done": sum(s.get("sessions_done", 0) for s in stats),
            "in_flight": sum(s.get("in_flight", 0) for s in stats),
            "prefill_in_flight": sum(
                s.get("prefill_in_flight", 0) for s in stats),
            "completed": sum(s.get("completed", 0) for s in stats),
            "errors": sum(s.get("errors", 0) for s in stats),
            "cancelled": sum(s.get("cancelled", 0) for s in stats),
            "tardiness_p99_ms": max(
                (s.get("tardiness_p99_ms", 0.0) for s in stats), default=0.0),
            "loop_lag_ms": max(
                (s.get("loop_lag_ms", 0.0) for s in stats), default=0.0),
            "mean_session_s": (
                sum(sess_durs) / len(sess_durs) if sess_durs else None
            ),
            "dead_workers": self.deaths_total,
        }

    async def stop(self) -> None:
        self._stopping = True
        for w in self._workers:
            await self._send(w, {"cmd": "stop"})
        for w in self._workers:
            try:
                await asyncio.wait_for(w.proc.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                w.proc.kill()
            w.reader_task.cancel()
            if w.watch_task is not None:
                w.watch_task.cancel()
        await asyncio.gather(
            *(w.reader_task for w in self._workers),
            *(w.watch_task for w in self._workers if w.watch_task),
            return_exceptions=True,
        )
        for f in self._stderr_files:
            try:
                f.close()
            except Exception:
                pass
        self._workers.clear()


# ── Window measurement ───────────────────────────────────────────────


@dataclass
class _WindowResult:
    stability: str            # stable | divergent | client_limited
    verdict_detail: dict
    sla_pass: bool | None
    violation_rate: float
    target_miss_rate: float
    sample_size: int
    client_saturated: bool
    measurement_id: int | None
    active_sessions_mean: float = 0.0


def _summarize_turns(turns: list[dict]) -> dict:
    """SLA aggregation over wire-format turn dicts — the open-loop
    counterpart of the closed-loop per-window aggregation in
    measurement.run_measurement_step (flags were already resolved
    against personas inside the worker)."""
    n = len(turns)
    if n == 0:
        return {"sample_size": 0}
    ttft = [t["ttft_ms"] for t in turns]
    tpot = [t["tpot_ms"] for t in turns]
    ttfct = [t.get("ttfct_ms") or t["ttft_ms"] for t in turns]
    ttft_v = sum(1 for t in turns if t["ttft_violation"])
    tpot_v = sum(1 for t in turns if t["tpot_violation"])
    comb_v = sum(
        1 for t in turns if t["ttft_violation"] or t["tpot_violation"]
    )
    ttft_m = sum(1 for t in turns if t["ttft_target_miss"])
    tpot_m = sum(1 for t in turns if t["tpot_target_miss"])
    comb_m = sum(
        1 for t in turns if t["ttft_target_miss"] or t["tpot_target_miss"]
    )
    ci_lo, ci_hi = _wilson_ci(comb_v, n)
    return {
        "sample_size": n,
        "ttft_violation_rate": ttft_v / n,
        "tpot_violation_rate": tpot_v / n,
        "combined_violation_rate": comb_v / n,
        "ttft_target_miss_rate": ttft_m / n,
        "tpot_target_miss_rate": tpot_m / n,
        "combined_target_miss_rate": comb_m / n,
        "violation_rate_ci_lower": ci_lo,
        "violation_rate_ci_upper": ci_hi,
        "ttft_p50_ms": _percentile(ttft, 0.5),
        "ttft_p75_ms": _percentile(ttft, 0.75),
        "ttft_p95_ms": _percentile(ttft, 0.95),
        "tpot_p50_ms": _percentile(tpot, 0.5),
        "tpot_p75_ms": _percentile(tpot, 0.75),
        "tpot_p95_ms": _percentile(tpot, 0.95),
        "ttfct_p50_ms": _percentile(ttfct, 0.5),
        "ttfct_p75_ms": _percentile(ttfct, 0.75),
        "ttfct_p95_ms": _percentile(ttfct, 0.95),
        "avg_reasoning_tokens": (
            statistics.fmean([t.get("reasoning_tokens") or 0 for t in turns])
        ),
        "capacity_status": _classify_status(comb_v, n),
        "target_status": _classify_status(comb_m, n),
        "combined_violations": comb_v,
    }


def _turn_row(t: dict, measurement_id: int) -> dict:
    return {
        "measurement_id": measurement_id,
        "persona_id": t["persona_id"],
        "user_id": t["user_id"],
        "session_id": t["session_id"],
        "turn_index": t["turn_index"],
        "submitted_at_ms": t["submitted_at_ms"],
        "ttft_ms": t["ttft_ms"],
        "completed_at_ms": t["completed_at_ms"],
        "input_tokens": t["input_tokens"],
        "history_tokens": t["history_tokens"],
        "output_tokens": t["output_tokens"],
        "tpot_ms": t["tpot_ms"],
        "end_to_end_ms": t["end_to_end_ms"],
        "in_flight_at_submit": t["in_flight_at_submit"],
        "in_flight_avg_during": None,
        "in_flight_peak_during": None,
        "sla_ttft_violation": int(t["ttft_violation"]),
        "sla_tpot_violation": int(t["tpot_violation"]),
        "ttft_target_miss": int(t["ttft_target_miss"]),
        "tpot_target_miss": int(t["tpot_target_miss"]),
        "token_timestamps_json": None,
        "error": t.get("error"),
        "ttfct_ms": t.get("ttfct_ms"),
        "reasoning_tokens": t.get("reasoning_tokens") or 0,
    }


# ── Orchestrator ─────────────────────────────────────────────────────


class OpenLoopRunner:
    """Drives one open-loop cohort run end-to-end."""

    def __init__(
        self,
        cfg: Config,
        cohort: Cohort,
        engine: Engine,
        db: Database,
        cohort_run_id: str,
        run_dir: Path,
    ):
        self.cfg = cfg
        self.cohort = cohort
        self.engine = engine
        self.db = db
        self.cohort_run_id = cohort_run_id
        self.run_dir = run_dir
        sim = cfg.simulation
        self.stepper = RateStepper(
            initial_rate_per_s=sim.open_loop_initial_rate_per_s,
            max_rate_per_s=sim.open_loop_max_rate_per_s,
            resolution_ratio=1.0 + max(
                0.5, getattr(sim, "open_loop_resolution_pct", 5.0)) / 100.0,
        )
        self.pool = WorkerPool(
            base_config={
                "persona_weights": cohort.persona_weights,
                "replica_urls": engine.replica_urls,
                "api_key": engine.api_key,
                "api_model_name": engine.api_model_name,
                "model_id": cfg.engine.model_id,
                "request_timeout_s": _request_timeout_for(cohort, sim),
                "reasoning_effort": (
                    cfg.engine.reasoning_effort if cfg.engine.reasoning
                    else None
                ),
                "seed": 0xC0FFEE,
            },
            log_dir=run_dir,
            max_workers=sim.open_loop_max_workers,
        )
        bind_str = (
            cfg.engine.cpu_bind
            or cfg.engine.vllm_extra_env.get("VLLM_CPU_OMP_THREADS_BIND")
            or ""
        )
        self.telemetry = MeasurementTelemetry(
            cfg.telemetry,
            engine,
            bound_cpus=expand_thread_binding(bind_str) or None,
            engine_pid=getattr(engine, "pid", None),
            artifacts_dir=run_dir,
            host_telemetry=(cfg.engine.type != "remote"),
        )
        self.phase = "idle"
        self.current_rate = 0.0
        self.step_index = 0
        self.client_max_lag_ms = 0.0
        self._last_engine_metrics: dict = {}
        # generation_tokens_total from the last SUCCESSFUL scrape —
        # survives failed scrapes so the error fuse keeps its token
        # evidence (see _ErrorFuse).
        self._tokens_last_good: int | None = None
        self._snapshot_task: asyncio.Task | None = None
        self._last_inflight_mean: float | None = None
        self._last_stable: dict | None = None  # {rate, sessions}
        self._queue_gauge_seen = False

    # ── Engine pressure sampling ────────────────────────────────────

    async def _sample_engine(self) -> dict:
        try:
            m = await asyncio.to_thread(self.engine.get_metrics)
        except Exception:
            m = {}
        self._last_engine_metrics = m
        if m.get("queue_depth") is not None:
            self._queue_gauge_seen = True
        if m.get("generation_tokens_total") is not None:
            self._tokens_last_good = int(m["generation_tokens_total"])
        return m

    # ── Live snapshots (1 Hz, whole run) ────────────────────────────

    async def _snapshot_loop(self) -> None:
        try:
            while True:
                agg = self.pool.aggregate()
                qd = self._last_engine_metrics.get("queue_depth")
                row = {
                    "cohort_run_id": self.cohort_run_id,
                    "snapshot_at_ms": _now_ms(),
                    "phase": self.phase,
                    "pool_size": int(agg.get("sessions_active", 0) or 0),
                    "in_flight": int(agg.get("in_flight", 0) or 0),
                    "prefill_in_flight": agg.get("prefill_in_flight"),
                    "decode_in_flight": (
                        max(0, (agg.get("in_flight", 0) or 0)
                            - (agg.get("prefill_in_flight", 0) or 0))
                        if agg else None
                    ),
                    "requests_completed": int(agg.get("completed", 0) or 0),
                    "errors": int(agg.get("errors", 0) or 0),
                    "loop_lag_ms": agg.get("loop_lag_ms"),
                    "arrival_rate_per_min": round(self.current_rate * 60, 2),
                    "queue_depth": (
                        int(qd) if qd is not None else None
                    ),
                    "active_sessions": int(agg.get("sessions_active", 0) or 0),
                }
                try:
                    self.db.insert_snapshot(row)
                except Exception:
                    log.debug("snapshot insert failed", exc_info=True)
                BUS.publish("snapshot", row)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    # ── Window machinery ────────────────────────────────────────────

    def _warmup_plan(self) -> tuple[float, float, int]:
        """``(minimum_s, cap_s, settle_window_n)`` for the next window.

        The minimum is the configured warmup stretched toward the
        measured mean session duration (capped at 300 s). Past it the
        settling detector decides, up to ``cap_s`` — by default
        max(300 s, 1.5 × mean session duration), the time a rate step
        needs to propagate through essentially every session — on a
        trailing window of max(open_loop_settle_window_s, 0.2 × mean
        session duration) samples, capped at 300, so a slow ramp on
        long sessions is still visible.
        """
        sim = self.cfg.simulation
        base = float(sim.open_loop_warmup_s)
        agg = self.pool.aggregate()
        mean_sess = float(agg.get("mean_session_s") or 0.0) if agg else 0.0
        minimum = max(base, min(300.0, mean_sess)) if mean_sess else base
        cap = sim.open_loop_settle_max_s
        cap_s = (
            max(300.0, 1.5 * mean_sess) if cap is None else float(cap)
        )
        cap_s = max(cap_s, minimum)
        window_n = int(min(
            SETTLE_WINDOW_CAP_S,
            max(sim.open_loop_settle_window_s, 0.2 * mean_sess),
        ))
        return minimum, cap_s, max(2, window_n)

    async def _measure_window(
        self, rate_per_s: float, window_s: int,
    ) -> _WindowResult:
        self.current_rate = rate_per_s
        await self.pool.set_rate(rate_per_s)

        self.phase = "warmup"
        warmup_min, settle_cap, settle_n = self._warmup_plan()
        log.info(
            "rate %.3g/s (%.1f/min): warmup ≥%.0fs (settling cap %.0fs, "
            "trailing window %ds), window %ds, workers=%d",
            rate_per_s, rate_per_s * 60, warmup_min, settle_cap, settle_n,
            window_s, self.pool.size,
        )

        # Runaway-error fuse state: cumulative counters at phase
        # start; checked every tick in warmup AND measurement.
        fuse_agg = self.pool.aggregate()
        fuse = _ErrorFuse(
            fuse_agg.get("errors", 0) if fuse_agg else 0,
            fuse_agg.get("completed", 0) if fuse_agg else 0,
            self._tokens_last_good,
        )
        last_error: list[str] = []

        def _check_fuse(fresh_turns: list[dict]) -> None:
            for t in fresh_turns:
                if t.get("error"):
                    last_error.append(str(t["error"]))
                    del last_error[:-3]
            agg_now = self.pool.aggregate()
            if not agg_now:
                return
            errors = agg_now.get("errors", 0)
            completions = agg_now.get("completed", 0)
            if fuse.tripped(errors, completions, self._tokens_last_good):
                raise EngineBrokenError(
                    f"aborting run: {errors - fuse.errors0} failed requests "
                    f"against {completions - fuse.completions0} completions "
                    f"at {rate_per_s * 60:.0f}/min — the engine is rejecting "
                    f"the load, not serving it (recent errors: "
                    f"{', '.join(last_error) or 'unknown'}). "
                    f"Check the engine log in the run directory."
                )

        # Worker deaths taint the window whenever they land (warmup
        # included): the offered load was not λ. Records from before
        # this window (e.g. during a revert) are cleared first.
        worker_deaths: list[dict] = []
        self.pool.take_deaths()

        def _note_deaths() -> bool:
            worker_deaths.extend(self.pool.take_deaths())
            return bool(worker_deaths)

        # Warmup = the minimum, then extend until the active-session
        # population is flat (or the settling cap is hit).
        active_warm: list[float] = []
        settled = False
        drift = float("inf")
        warm_start = time.monotonic()
        while True:
            await self._sample_engine()
            _check_fuse(self.pool.drain_turn_queue())  # discard settling turns
            if _note_deaths():
                break
            agg_w = self.pool.aggregate()
            if agg_w:
                active_warm.append(float(agg_w.get("sessions_active", 0)))
            elapsed = time.monotonic() - warm_start
            if elapsed >= warmup_min:
                settled, drift = _population_settled(active_warm, settle_n)
                if settled:
                    break
                if elapsed >= settle_cap:
                    log.warning(
                        "rate %.3g/s: population still drifting %.1f%% per "
                        "%ds at the %.0fs settling cap — measuring anyway "
                        "(raise open_loop_settle_max_s for long sessions)",
                        rate_per_s, drift * 100, settle_n, settle_cap,
                    )
                    break
            await asyncio.sleep(1.0)
        warmup_s = time.monotonic() - warm_start
        if settled:
            log.info("rate %.3g/s: population settled after %.0fs "
                     "(drift %.1f%% per %ds)",
                     rate_per_s, warmup_s, drift * 100, settle_n)

        self.phase = "measuring"
        measurement_started_at = datetime.now(timezone.utc).isoformat()
        start_mono = time.monotonic()
        pre_row = {
            "cohort_run_id": self.cohort_run_id,
            "step_index": self.step_index,
            "target_pool_size": 0,
            "measured_avg_pool_size": 0.0,
            "measured_avg_in_flight": 0.0,
            "measurement_started_at": measurement_started_at,
            "measurement_duration_s": 0,
            "sample_size": 0,
            "ttft_violation_rate": 0.0,
            "tpot_violation_rate": 0.0,
            "combined_violation_rate": 0.0,
            "violation_rate_ci_lower": 0.0,
            "violation_rate_ci_upper": 0.0,
            "capacity_status": "pending",
            "target_status": "pending",
            "arrival_rate_per_min": round(rate_per_s * 60, 2),
        }
        measurement_id = self.db.insert_measurement(pre_row)
        self.telemetry.start(measurement_id)

        turns: list[dict] = []
        queue_series: list[float] = []
        inflight_series: list[float] = []
        active_series: list[float] = []
        max_tardiness = 0.0
        max_lag = 0.0
        extended = False
        verdict = None
        # Window-scoped tardy accounting: deltas of the workers'
        # cumulative counters between window start and end.
        agg0 = self.pool.aggregate()
        arrivals0 = agg0.get("arrivals_total", 0) if agg0 else 0
        tardy0 = agg0.get("tardy_total", 0) if agg0 else 0

        try:
            remaining = 0 if worker_deaths else window_s
            while remaining > 0:
                tick_end = time.monotonic() + 1.0
                m = await self._sample_engine()
                agg = self.pool.aggregate()
                qd = m.get("queue_depth")
                if qd is not None:
                    queue_series.append(float(qd))
                if agg:
                    inflight_series.append(float(agg.get("in_flight", 0)))
                    active_series.append(float(agg.get("sessions_active", 0)))
                    max_tardiness = max(
                        max_tardiness, agg.get("tardiness_p99_ms", 0.0))
                    max_lag = max(max_lag, agg.get("loop_lag_ms", 0.0))
                fresh = self.pool.drain_turn_queue()
                _check_fuse(fresh)
                turns.extend(fresh)
                for t in fresh:
                    BUS.publish("turn", {
                        "step_index": self.step_index,
                        "arrival_rate_per_min": round(rate_per_s * 60, 2),
                        "persona_id": t["persona_id"],
                        "user_id": t["user_id"],
                        "session_id": t["session_id"],
                        "turn_index": t["turn_index"],
                        "ttft_ms": t["ttft_ms"],
                        "ttfct_ms": t.get("ttfct_ms"),
                        "tpot_ms": t["tpot_ms"],
                        "end_to_end_ms": t["end_to_end_ms"],
                        "input_tokens": t["input_tokens"],
                        "output_tokens": t["output_tokens"],
                        "reasoning_tokens": t.get("reasoning_tokens") or 0,
                        "in_flight_at_submit": t["in_flight_at_submit"],
                        "ttft_violation": t["ttft_violation"],
                        "tpot_violation": t["tpot_violation"],
                        "error": t.get("error"),
                    })
                await asyncio.sleep(max(0.0, tick_end - time.monotonic()))
                remaining -= 1
                if _note_deaths():
                    remaining = 0  # discard: measure no further

                if remaining <= 0:
                    served_mean = (
                        statistics.fmean(inflight_series)
                        if inflight_series else None
                    )
                    basis = "engine_queue"
                    series = queue_series
                    # Engines without a queue gauge (or scrape
                    # failures): the client's total in-flight count is
                    # the fallback pressure signal — in overload it
                    # grows exactly like the queue (arrivals outpace
                    # completions).
                    if len(series) < len(inflight_series) // 2:
                        series = inflight_series
                        basis = "client_in_flight"
                    verdict = assess_queue_stability(
                        series, served_mean=served_mean,
                    )
                    verdict_dict = verdict.to_dict()
                    verdict_dict["basis"] = basis
                    if worker_deaths:
                        break  # no extension: the window is void
                    if verdict.verdict == INCONCLUSIVE and not extended:
                        extended = True
                        remaining = window_s  # double once, keep sampling
                        log.info(
                            "rate %.3g/s: %s — extending window by %ds",
                            rate_per_s, verdict.reason, window_s,
                        )
                        continue
                    # A stable window with zero completed turns can't
                    # say anything about SLA — extend once for samples.
                    if (
                        verdict.verdict == "stable" and not turns
                        and not extended
                    ):
                        extended = True
                        remaining = window_s
                        continue
        finally:
            duration = int(time.monotonic() - start_mono)
            _, tele_rows, tele_agg = await self.telemetry.stop()
            self.db.insert_telemetry(tele_rows)
            if tele_agg:
                agg_row = {
                    k: v for k, v in tele_agg.items()
                    if k in AGGREGATE_COLUMN_NAMES and v is not None
                }
                if agg_row:
                    self.db.update_measurement(measurement_id, agg_row)
            self.phase = "idle"

        if verdict is None:
            # A worker died before a single measuring tick: nothing
            # to assess, and the window is void anyway.
            verdict = assess_queue_stability(queue_series or inflight_series)
        verdict_dict = verdict.to_dict()
        verdict_dict["basis"] = (
            "engine_queue"
            if len(queue_series) >= len(inflight_series) // 2
            else "client_in_flight"
        )
        verdict_dict["warmup_s"] = round(warmup_s, 1)
        verdict_dict["settled"] = settled

        agg1 = self.pool.aggregate()
        d_arrivals = max(0, (agg1.get("arrivals_total", 0) if agg1 else 0)
                         - arrivals0)
        d_tardy = max(0, (agg1.get("tardy_total", 0) if agg1 else 0) - tardy0)
        tardy_fraction = d_tardy / max(1, d_arrivals)
        client_saturated = (
            (d_tardy >= TARDY_MIN_COUNT
             and tardy_fraction > TARDY_FRACTION_LIMIT)
            or max_lag > WORKER_LAG_LIMIT_MS
        )
        verdict_dict["tardy_fraction"] = round(tardy_fraction, 4)
        verdict_dict["tardy_arrivals"] = d_tardy
        self.client_max_lag_ms = max(self.client_max_lag_ms, max_lag)
        if worker_deaths:
            # The generator, not the engine, failed this window: the
            # offered load was below λ from the moment of death.
            client_saturated = True
            verdict_dict["worker_deaths"] = worker_deaths
            log.error(
                "rate %.3g/s: window discarded — %d load worker(s) died "
                "(%s); the offered load was not the recorded rate",
                rate_per_s, len(worker_deaths),
                ", ".join(f"#{d['index']} rc={d['returncode']}"
                          for d in worker_deaths),
            )

        # Final verdict mapping. An inconclusive verdict that survived
        # the extension gets settled by effect size alone: meaningful
        # growth is divergence, marginal drift is stability — either
        # way the detail JSON records the ambiguity.
        if verdict.verdict == INCONCLUSIVE:
            floor = max(
                10.0,
                0.5 * (statistics.fmean(inflight_series)
                       if inflight_series else 0.0),
            )
            stability = (
                DIVERGENT
                if verdict.growth_over_window >= 0.5 * floor else STABLE
            )
            verdict_dict["settled_by"] = "effect_size_after_extension"
        else:
            stability = verdict.verdict
        if client_saturated:
            stability = CLIENT_LIMITED

        summary = _summarize_turns(turns)
        sla_pass: bool | None = None
        if stability == STABLE and summary["sample_size"] > 0:
            sla_pass = summary["capacity_status"] == "pass"
        # A divergent window means unbounded latency — SLA fails by
        # construction whatever the in-window samples happened to say.
        if stability == DIVERGENT:
            sla_pass = False

        active_mean = (
            statistics.fmean(active_series) if active_series else 0.0
        )
        inflight_mean = (
            statistics.fmean(inflight_series) if inflight_series else 0.0
        )
        self._last_inflight_mean = inflight_mean
        pool_agg = self.pool.aggregate()

        # KV-pool utilization for THIS window, from the per-second
        # engine scrape — the closed-loop path always recorded this;
        # its absence here made the report narrate "KV 0%" while the
        # bottleneck attributor (reading the raw telemetry) said 98%.
        kv_vals = [
            r.get("kv_cache_used_pct") for r in tele_rows
            if r.get("kv_cache_used_pct") is not None
        ]
        final_row: dict = {
            "target_pool_size": int(round(active_mean)),
            "measured_avg_pool_size": round(active_mean, 2),
            "measured_avg_in_flight": round(inflight_mean, 2),
            "avg_kv_cache_pct": (
                round(sum(kv_vals) / len(kv_vals), 2) if kv_vals else None
            ),
            "measurement_duration_s": duration,
            "arrival_rate_per_min": round(rate_per_s * 60, 2),
            "stability": stability,
            "stability_detail": json.dumps(verdict_dict),
            "queue_depth_mean": round(verdict.mean_depth, 2),
            "queue_depth_slope_per_min": round(verdict.slope_per_min, 3),
            "arrival_tardiness_p99_ms": round(max_tardiness, 1),
            "load_workers": self.pool.size,
            "active_sessions_mean": round(active_mean, 2),
            "mean_session_duration_s": pool_agg.get("mean_session_s"),
        }
        if summary["sample_size"] > 0:
            final_row.update({
                k: v for k, v in summary.items()
                if k not in ("combined_violations",)
            })
        else:
            final_row.update({
                "sample_size": 0,
                "capacity_status": "pending",
                "target_status": "pending",
            })
        # A divergent window's capacity_status is 'fail' regardless of
        # the in-window SLA snapshot — a growing queue is a latency
        # cliff in progress, and the samples captured during a short
        # window flatter it.
        if stability == DIVERGENT:
            final_row["capacity_status"] = "fail"
            final_row["target_status"] = "fail"
        # A client-limited window's samples are tainted — the lagging
        # GENERATOR inflated them. "pending", never "pass": pairing
        # "client_limited · pass" would claim an SLA verdict this
        # window cannot honestly give.
        if stability == CLIENT_LIMITED:
            final_row["capacity_status"] = "pending"
            final_row["target_status"] = "pending"
        self.db.update_measurement(measurement_id, final_row)
        if turns:
            self.db.insert_events(
                [_turn_row(t, measurement_id) for t in turns],
            )
        BUS.publish("step", {
            "step_index": self.step_index,
            "pool_size": final_row["target_pool_size"],
            **{k: v for k, v in final_row.items()
               if k not in ("stability_detail",)},
        })
        log.info(
            "rate %.3g/s window done: %s (%s) — samples=%d viol=%.1f%% "
            "queue mean=%.1f slope=%.2f/min tardiness=%.0fms",
            rate_per_s, stability, verdict_dict.get("reason", ""),
            summary.get("sample_size", 0),
            (summary.get("combined_violation_rate", 0.0) or 0.0) * 100,
            verdict.mean_depth, verdict.slope_per_min, max_tardiness,
        )
        self.step_index += 1
        return _WindowResult(
            stability=stability,
            verdict_detail=verdict_dict,
            sla_pass=sla_pass,
            violation_rate=summary.get("combined_violation_rate", 0.0) or 0.0,
            target_miss_rate=(
                summary.get("combined_target_miss_rate", 0.0) or 0.0),
            sample_size=summary.get("sample_size", 0),
            client_saturated=client_saturated,
            measurement_id=measurement_id,
            active_sessions_mean=active_mean,
        )

    def _mark_superseded(self, measurement_id: int | None) -> None:
        """Re-label a client-saturated attempt that is being re-run
        with more workers: it is not a ceiling, it's a discarded
        measurement of the generator."""
        if measurement_id is None:
            return
        try:
            row = self.db.fetchone(
                "SELECT step_index, target_pool_size, stability_detail "
                "FROM cohort_measurements WHERE measurement_id = ?",
                (measurement_id,),
            )
            self.db.update_measurement(measurement_id, {
                "stability": "superseded",
                "capacity_status": "pending",
                "target_status": "pending",
            })
            if row is not None:
                BUS.publish("step", {
                    "step_index": row["step_index"],
                    "pool_size": row["target_pool_size"],
                    "stability": "superseded",
                    "capacity_status": "pending",
                })
        except Exception:
            log.debug("supersede mark failed", exc_info=True)

    async def _revert_to_stable(self) -> None:
        """Fall back after overshooting the knee — WITHOUT tearing the
        population to zero. Arrivals continue at the last known-stable
        rate; only the EXCESS sessions are cancelled (newest first) so
        the queue drains back to the stable density; the bisection then
        probes smaller increments from a warm system. Rebuilding from
        an empty pool would waste a full session-length ramp per
        divergent probe. A divergent FIRST window has no stable point
        to fall back to — full drain then."""
        st = self._last_stable
        if st is None:
            await self._drain()
            return
        self.phase = "reverting to last stable rate"
        self.current_rate = st["rate"]
        await self.pool.set_rate(st["rate"])
        await self.pool.trim(int(st["sessions"]))
        deadline = (time.monotonic()
                    + self.cfg.simulation.open_loop_drain_timeout_s)
        threshold = max(5.0, 0.05 * st["sessions"])
        while time.monotonic() < deadline:
            m = await self._sample_engine()
            qd = m.get("queue_depth")
            if qd is None or qd <= threshold:
                break
            await asyncio.sleep(2.0)
        self.phase = "idle"

    async def _drain(self) -> None:
        """Full teardown fallback — only when there is no stable
        operating point to revert to."""
        self.phase = "draining"
        self.current_rate = 0.0
        await self.pool.drain()
        deadline = time.monotonic() + self.cfg.simulation.open_loop_drain_timeout_s
        while time.monotonic() < deadline:
            m = await self._sample_engine()
            agg = self.pool.aggregate()
            qd = m.get("queue_depth")
            inflight = agg.get("in_flight", 0) if agg else 0
            if (qd is None or qd < 2) and inflight < 2:
                break
            await asyncio.sleep(2.0)
        self.phase = "idle"

    def _plan_workers(self, next_rate: float, last_rate: float) -> int:
        """Predict the in-flight load at the next rate and size the
        worker set for it (Little's law scaling from the last window)."""
        sim = self.cfg.simulation
        if self._last_inflight_mean is None or last_rate <= 0:
            return self.pool.size or 1
        predicted = self._last_inflight_mean * (next_rate / last_rate) * 1.3
        import math
        needed = max(1, math.ceil(predicted / sim.open_loop_inflight_per_worker))
        return max(self.pool.size, min(needed, sim.open_loop_max_workers))

    # ── Main loop ───────────────────────────────────────────────────

    async def run(self) -> str:
        sim = self.cfg.simulation
        # Snapshots first, workers second: worker spawn loads a
        # tokenizer and can take tens of seconds — the live view
        # should say so rather than sit on the previous run's charts.
        self.phase = "verifying engine — smoke request"
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        final_status = "ok"
        run_started = time.monotonic()
        max_total_s = sim.max_total_duration_minutes * 60
        last_rate = 0.0
        try:
            # Real preflight: a broken-but-healthy engine fails HERE,
            # with its own error text, instead of drowning the run in
            # errors (inside try so cleanup still runs on failure).
            await smoke_test_engine(self.engine)
            self.phase = "starting load workers"
            await self.pool.scale_to(1)
            while (rate := self.stepper.next_rate()) is not None:
                if time.monotonic() - run_started > max_total_s:
                    log.warning("max run duration reached; stopping search")
                    final_status = "time_limit"
                    break
                window_s = (
                    sim.open_loop_refine_window_s
                    if self.stepper.in_refinement
                    else sim.open_loop_window_s
                )
                planned = self._plan_workers(rate, last_rate)
                if planned > self.pool.size:
                    self.phase = "starting load workers"
                await self.pool.scale_to(planned)
                result = await self._measure_window(rate, window_s)
                # Generator fell behind: keep adding workers and
                # re-running this rate until the generator keeps up or
                # is genuinely maxed — ONLY a maxed generator records
                # client_limited (the honest "the tool gave out" mark).
                # Each superseded attempt is re-labeled so it never
                # counts as a ceiling in the export or reads as a
                # verdict in the UI.
                # A worker death also lands here (client_saturated):
                # the dead worker is replaced rather than the pool
                # grown, and the void window is superseded by the
                # re-measurement.
                while (
                    result.client_saturated
                    and self.pool.size < sim.open_loop_max_workers
                ):
                    died = result.verdict_detail.get("worker_deaths")
                    target = (
                        self.pool.size + len(died) if died
                        else self.pool.size + 1
                    )
                    log.info(
                        "%s at %.3g/s with %d workers — %s and re-measuring",
                        "worker death" if died else "client saturation",
                        rate, self.pool.size,
                        "replacing" if died else "scaling out",
                    )
                    self._mark_superseded(result.measurement_id)
                    self.phase = "starting load workers"
                    await self.pool.scale_to(target)
                    result = await self._measure_window(rate, window_s)
                self.stepper.record(RateStep(
                    rate_per_s=rate,
                    stability=result.stability,
                    sla_pass=result.sla_pass,
                    violation_rate=result.violation_rate,
                    target_miss_rate=result.target_miss_rate,
                    sample_size=result.sample_size,
                ))
                last_rate = rate
                if result.stability == STABLE:
                    self._last_stable = {
                        "rate": rate,
                        "sessions": result.active_sessions_mean,
                    }
                if result.stability in (DIVERGENT, CLIENT_LIMITED):
                    await self._revert_to_stable()
        except (KeyboardInterrupt, asyncio.CancelledError):
            final_status = "cancelled"
            raise
        finally:
            self.phase = "idle"
            self.current_rate = 0.0
            if self._snapshot_task is not None:
                self._snapshot_task.cancel()
                try:
                    await self._snapshot_task
                except (asyncio.CancelledError, Exception):
                    pass
            await self.pool.stop()
            search = self.stepper.summary()
            try:
                self.db.update_cohort_run(self.cohort_run_id, {
                    "mode": "open_loop",
                    "client_max_lag_ms": round(self.client_max_lag_ms, 1),
                })
            except Exception:
                log.debug("cohort_run patch failed", exc_info=True)
            log.info("open-loop search summary: %s", search)
        return final_status


# ── Entry point (mirrors runner.run_cohort's contract) ───────────────


def _config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


async def run_cohort_open_loop(
    cfg: Config,
    cohort: Cohort | str,
    *,
    engine: Engine | None = None,
    db_path: Path | None = None,
    run_dir: Path | None = None,
    new_run: bool = False,
) -> Path:
    """Open-loop counterpart of ``runner.run_cohort`` — same artifact
    layout (run_NN/run.db, cohort_run + cohort_measurements +
    turn_events + telemetry + snapshots), different methodology."""
    if isinstance(cohort, str):
        cohort = get_cohort(cohort)
    if run_dir is None:
        run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    own_engine = engine is None
    if db_path is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_dir / "run.db"

    # The run row, the "started" event, and a launch heartbeat all go
    # out BEFORE the engine launches. An 8-replica engine spends 5-10
    # minutes loading weights; without these, that whole phase is
    # silent — the live page keeps replaying the PREVIOUS run and the
    # user can't tell "loading" from "hung" from "resumed".
    db = Database(db_path)
    cohort_run_id = uuid.uuid4().hex
    db.insert_run(
        cohort_run_id=cohort_run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        engine_type=cfg.engine.type,
        model_id=cfg.engine.model_id,
        cohort_id=cohort.id,
        cohort_definition={
            "id": cohort.id,
            "name": cohort.name,
            "description": cohort.description,
            "category": cohort.category,
            "persona_weights": cohort.persona_weights,
        },
        config=_config_to_dict(cfg),
    )
    db.update_cohort_run(cohort_run_id, {"mode": "open_loop"})
    BUS.publish("run", {
        "event": "started",
        "cohort_run_id": cohort_run_id,
        "cohort_id": cohort.id,
        "engine": cfg.engine.type,
        "model": cfg.engine.model_id,
        "run_dir": str(run_dir),
        "mode": "open_loop",
    })

    runner = None
    final_status = "error"
    try:
        if own_engine:
            preflight_check(cfg.engine.hardware_requirements)
            engine = make_engine(cfg.engine.type, cfg.engine)
            launch_phase = (
                f"launching engine — loading {cfg.engine.model_id} weights"
            )
            hb_stop = asyncio.Event()

            async def _launch_heartbeat() -> None:
                # 2 s snapshots so both the live stream AND a page
                # opened mid-launch (backfill) show the real phase.
                while not hb_stop.is_set():
                    row = {
                        "cohort_run_id": cohort_run_id,
                        "snapshot_at_ms": _now_ms(),
                        "phase": launch_phase,
                        "pool_size": 0, "in_flight": 0,
                        "requests_completed": 0, "errors": 0,
                    }
                    try:
                        db.insert_snapshot(row)
                    except Exception:
                        pass
                    BUS.publish("snapshot", row)
                    try:
                        await asyncio.wait_for(hb_stop.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        pass

            hb = asyncio.create_task(_launch_heartbeat())
            try:
                await asyncio.to_thread(engine.launch, log_dir=run_dir)
            finally:
                hb_stop.set()
                await hb
        runner = OpenLoopRunner(cfg, cohort, engine, db, cohort_run_id, run_dir)
        final_status = await runner.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        final_status = "cancelled"
        raise
    except Exception as e:  # noqa: BLE001
        final_status = f"error: {type(e).__name__}"
        raise
    finally:
        # End-of-run engine prefix-cache scrape (same as closed loop).
        try:
            m = engine.get_metrics() if engine is not None else {}
            if m.get("prefix_cache_hit_rate") is not None:
                db.update_cohort_run(cohort_run_id, {
                    "prefix_cache_engine_hits": (
                        int(m["prefix_cache_hits"])
                        if m.get("prefix_cache_hits") is not None else None
                    ),
                    "prefix_cache_engine_queries": (
                        int(m["prefix_cache_queries"])
                        if m.get("prefix_cache_queries") is not None else None
                    ),
                    "prefix_cache_engine_hit_rate": float(
                        m["prefix_cache_hit_rate"]),
                })
        except Exception:
            log.debug("end-of-run metrics scrape failed", exc_info=True)
        if runner is not None and runner.telemetry.collector_statuses:
            try:
                db.update_cohort_run(cohort_run_id, {
                    "collectors_json": json.dumps(
                        runner.telemetry.collector_statuses),
                })
            except Exception:
                pass
        db.finalise_run(
            cohort_run_id=cohort_run_id,
            completed_at=datetime.now(timezone.utc).isoformat(),
            status=final_status,
        )
        db.close()
        if own_engine and engine is not None:
            await asyncio.to_thread(engine.shutdown)
        BUS.publish("run", {
            "event": "finished",
            "cohort_run_id": cohort_run_id,
            "cohort_id": cohort.id,
            "final_status": final_status,
            "mode": "open_loop",
        })
    return db_path
