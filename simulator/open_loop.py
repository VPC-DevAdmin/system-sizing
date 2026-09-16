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
from .stability import INCONCLUSIVE, assess_queue_stability
from .telemetry import MeasurementTelemetry

log = logging.getLogger(__name__)

# Arrival lateness past which the generator is judged to be falling
# behind its own schedule — scale out, or report client_limited.
TARDINESS_LIMIT_MS = 500.0
# Worker event-loop lag limit (same meaning as the closed-loop
# CLIENT_SATURATION_LAG_MS: past this, latencies measure the client).
WORKER_LAG_LIMIT_MS = 1000.0


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


class WorkerPool:
    """k load-generator subprocesses sharing one total arrival rate."""

    def __init__(self, *, base_config: dict, log_dir: Path, max_workers: int):
        self._base = base_config
        self._log_dir = Path(log_dir)
        self.max_workers = max(1, max_workers)
        self.turn_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._workers: list[_Worker] = []
        self._rate_total = 0.0
        self._stderr_files: list = []

    @property
    def size(self) -> int:
        return len(self._workers)

    async def scale_to(self, k: int) -> None:
        k = max(1, min(k, self.max_workers))
        while len(self._workers) < k:
            await self._spawn(len(self._workers))
        # Redistribute the current rate over the new worker count.
        if self._rate_total > 0:
            await self.set_rate(self._rate_total)

    async def _spawn(self, index: int) -> None:
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
        self._workers.append(worker)
        # Tokenizer load can take a while on first spawn; don't start
        # the window clock until the worker can actually generate.
        try:
            await asyncio.wait_for(worker.ready.wait(), timeout=180.0)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"loadgen worker {index} did not become ready in 180s "
                f"(see {stderr_path})"
            ) from e
        log.info("loadgen worker %d ready (pid=%s)", index, proc.pid)

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
            "sessions_active": sum(s.get("sessions_active", 0) for s in stats),
            "sessions_done": sum(s.get("sessions_done", 0) for s in stats),
            "in_flight": sum(s.get("in_flight", 0) for s in stats),
            "prefill_in_flight": sum(
                s.get("prefill_in_flight", 0) for s in stats),
            "completed": sum(s.get("completed", 0) for s in stats),
            "errors": sum(s.get("errors", 0) for s in stats),
            "tardiness_p99_ms": max(
                (s.get("tardiness_p99_ms", 0.0) for s in stats), default=0.0),
            "loop_lag_ms": max(
                (s.get("loop_lag_ms", 0.0) for s in stats), default=0.0),
            "mean_session_s": (
                sum(sess_durs) / len(sess_durs) if sess_durs else None
            ),
        }

    async def stop(self) -> None:
        for w in self._workers:
            await self._send(w, {"cmd": "stop"})
        for w in self._workers:
            try:
                await asyncio.wait_for(w.proc.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                w.proc.kill()
            w.reader_task.cancel()
        await asyncio.gather(
            *(w.reader_task for w in self._workers), return_exceptions=True,
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
        )
        self.pool = WorkerPool(
            base_config={
                "persona_weights": cohort.persona_weights,
                "replica_urls": engine.replica_urls,
                "api_key": engine.api_key,
                "api_model_name": engine.api_model_name,
                "model_id": cfg.engine.model_id,
                "request_timeout_s": sim.request_timeout_s,
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
        self._snapshot_task: asyncio.Task | None = None
        self._last_inflight_mean: float | None = None
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

    def _warmup_s(self) -> float:
        base = float(self.cfg.simulation.open_loop_warmup_s)
        agg = self.pool.aggregate()
        mean_sess = agg.get("mean_session_s") if agg else None
        if mean_sess:
            return max(base, min(300.0, float(mean_sess)))
        return base

    async def _measure_window(
        self, rate_per_s: float, window_s: int,
    ) -> _WindowResult:
        self.current_rate = rate_per_s
        await self.pool.set_rate(rate_per_s)

        self.phase = "warmup"
        warmup = self._warmup_s()
        log.info(
            "rate %.3g/s (%.1f/min): warmup %.0fs, window %ds, workers=%d",
            rate_per_s, rate_per_s * 60, warmup, window_s, self.pool.size,
        )
        warm_end = time.monotonic() + warmup
        while time.monotonic() < warm_end:
            await self._sample_engine()
            self.pool.drain_turn_queue()  # discard settling-phase turns
            await asyncio.sleep(1.0)

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

        try:
            remaining = window_s
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

        assert verdict is not None
        verdict_dict = verdict.to_dict()
        verdict_dict["basis"] = (
            "engine_queue"
            if len(queue_series) >= len(inflight_series) // 2
            else "client_in_flight"
        )

        client_saturated = (
            max_tardiness > TARDINESS_LIMIT_MS or max_lag > WORKER_LAG_LIMIT_MS
        )
        self.client_max_lag_ms = max(self.client_max_lag_ms, max_lag)

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

        final_row: dict = {
            "target_pool_size": int(round(active_mean)),
            "measured_avg_pool_size": round(active_mean, 2),
            "measured_avg_in_flight": round(inflight_mean, 2),
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
        )

    async def _drain(self) -> None:
        """Clear the backlog after a divergent window — the next rate
        is only meaningful from an empty queue."""
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
        self.phase = "starting load workers"
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        await self.pool.scale_to(1)
        final_status = "ok"
        run_started = time.monotonic()
        max_total_s = sim.max_total_duration_minutes * 60
        last_rate = 0.0
        try:
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
                # Generator fell behind: add a worker and re-run this
                # rate once — only an already-maxed generator records
                # client_limited (the honest "the tool gave out" mark).
                if (
                    result.client_saturated
                    and self.pool.size < sim.open_loop_max_workers
                ):
                    log.info(
                        "client saturation at %.3g/s with %d workers — "
                        "scaling out and re-measuring",
                        rate, self.pool.size,
                    )
                    await self.pool.scale_to(self.pool.size + 1)
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
                if result.stability in (DIVERGENT, CLIENT_LIMITED):
                    await self._drain()
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
