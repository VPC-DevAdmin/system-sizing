"""Headline sweep — the marketing number, measured the way the
question is actually posed.

This is NOT the capacity methodology and does not pretend to be. The
open-loop runner answers "at what arrival rate does the queue stop
being stationary, and where does the experience break" — the right
question for real users. A headline asks something else entirely:
"what is the largest output-token rate and stream count this box
sustains?" Vendor benchmarks answer that by holding N concurrent
streams, sweeping N, and reporting the peak. There is no arrival
rate, no SLA gate, and no stationarity test, so forcing the question
through the capacity instrument meant disabling that instrument piece
by piece (absurd SLA thresholds to defeat the knee detector,
ignore_eos to defeat sampling, zero think time to defeat the session
model) and still getting an answer that was pessimistic by
construction: the stationarity knee sits strictly below peak
saturated throughput.

So: a closed loop at saturation. Hold N streams in flight, let the
engine self-throttle (it is the bottleneck by construction, which is
exactly what we want here), measure from the ENGINE's own counters
until consecutive chunks agree, then step N up. Stop when throughput
stops improving or the engine can no longer hold what we offer.

Latency is still recorded per rung — not as a pass/fail gate but as
the price of the headline, so the report can show what the number
costs. Telemetry (GPU, power, KV) attaches per rung exactly as it
does for capacity runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bus import BUS
from .config import Config
from .cpu_binding import expand_thread_binding
from .database import AGGREGATE_COLUMN_NAMES, Database
from .engines import make_engine
from .headline_search import Chunk, chunks_converged
from .measurement import _percentile
from .open_loop import EngineBrokenError, WorkerPool, smoke_test_engine
from .personas import Cohort
from .preflight import preflight_check
from .runs import resolve_run_dir
from .telemetry import MeasurementTelemetry

log = logging.getLogger(__name__)

DEFAULT_LADDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass
class Rung:
    """One concurrency rung, measured at steady state."""
    concurrency: int                 # streams offered
    in_flight: float | None          # engine num_running, mean
    queue_depth: float | None
    out_tok_s: float | None          # generation tokens/sec
    prompt_tok_s: float | None
    total_tok_s: float | None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    tpot_p50_ms: float | None = None
    tpot_p95_ms: float | None = None
    samples: int = 0
    errors: int = 0
    kv_cache_pct: float | None = None
    gpu_power_w: float | None = None
    steady_state: bool = True
    measure_s: int = 0
    held: bool = True                # did the engine hold what we offered?


def engine_held(concurrency: int, in_flight: float | None,
                tolerance: float = 0.7) -> bool:
    """Did the engine actually run what we offered it?

    Once the running batch stops tracking the offered stream count,
    the extra streams are queued rather than served — the engine's own
    ceiling (max_num_seqs x replicas, or KV capacity) has been found,
    and further rungs only add queueing delay.
    """
    if in_flight is None:
        return True
    return in_flight >= tolerance * concurrency


def peak_rung(rungs: list[Rung]) -> Rung | None:
    """The headline: highest sustained output token rate."""
    scored = [r for r in rungs if r.out_tok_s]
    if not scored:
        return None
    return max(scored, key=lambda r: r.out_tok_s or 0.0)


def should_stop(rungs: list[Rung], min_gain_pct: float) -> str | None:
    """Stop when the curve has plateaued or the engine is saturated.

    Returns a human reason, or None to keep climbing.
    """
    if not rungs:
        return None
    last = rungs[-1]
    # Derived here rather than trusting the stored flag, so the stop
    # decision cannot drift from the measurement it is based on.
    if not engine_held(last.concurrency, last.in_flight):
        return (f"the engine held {last.in_flight:.0f} of "
                f"{last.concurrency} offered streams — its own batch "
                f"ceiling, so wider offers only add queueing")
    if len(rungs) >= 3:
        best_before = max((r.out_tok_s or 0.0) for r in rungs[:-2])
        recent = max((r.out_tok_s or 0.0) for r in rungs[-2:])
        if best_before > 0 and \
                (recent - best_before) / best_before < min_gain_pct / 100.0:
            return (f"output rate gained under {min_gain_pct:.0f}% over the "
                    f"last two rungs — the throughput curve has plateaued")
    return None


def _cohort_shape(cohort: Cohort) -> dict | None:
    """The request shape this sweep ran, for the report's headline
    line — a saturation number is meaningless without it."""
    from .distributions import summarize
    from .personas import PERSONAS
    for pid in cohort.persona_weights:
        p = PERSONAS.get(pid)
        if p is None:
            continue
        return {
            "persona": pid,
            "input_tokens": summarize(p.input_tokens).get("median"),
            "output_tokens": summarize(p.output_tokens).get("median"),
            "ignore_eos": bool(getattr(p, "ignore_eos", False)),
        }
    return None


@dataclass
class _Acc:
    """Per-rung accumulators for client-side latency."""
    ttft: list[float] = field(default_factory=list)
    tpot: list[float] = field(default_factory=list)
    errors: int = 0

    def add(self, turns: list[dict]) -> None:
        for t in turns:
            if t.get("error"):
                self.errors += 1
                continue
            if t.get("ttft_ms") is not None:
                self.ttft.append(float(t["ttft_ms"]))
            if t.get("tpot_ms") is not None:
                self.tpot.append(float(t["tpot_ms"]))


async def run_headline_sweep(
    cfg: Config,
    cohort: Cohort,
    *,
    new_run: bool = False,
    run_dir: Path | None = None,
    max_concurrency: int | None = None,
    progress: dict | None = None,
) -> Path:
    """Sweep concurrency at saturation. Returns the summary JSON path."""
    sim = cfg.simulation
    if run_dir is None:
        run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ladder = [c for c in (sim.headline_sweep_ladder or DEFAULT_LADDER)
              if c <= (max_concurrency or 10**9)]
    if not ladder:
        ladder = [min(DEFAULT_LADDER)]

    cohort_run_id = uuid.uuid4().hex[:12]
    db = Database(run_dir / "run.db")
    started_at = datetime.now(timezone.utc).isoformat()
    db.insert_run(
        cohort_run_id=cohort_run_id, started_at=started_at,
        engine_type=cfg.engine.type, model_id=cfg.engine.model_id,
        cohort_id=cohort.id,
        cohort_definition={"name": cohort.name,
                           "persona_weights": cohort.persona_weights},
        config={"mode": "headline_sweep", "ladder": ladder},
    )
    db.update_cohort_run(cohort_run_id, {"mode": "headline_sweep"})

    preflight_check(cfg.engine.hardware_requirements)
    engine = make_engine(cfg.engine.type, cfg.engine)
    if progress is not None:
        progress.update({"phase": "launching engine",
                         "rungs": len(ladder), "done": False})
    BUS.publish("run", {
        "event": "started", "mode": "headline_sweep",
        "cohort_id": cohort.id, "engine": cfg.engine.type,
        "model": cfg.engine.model_id, "run_dir": str(run_dir),
    })
    log.info("headline sweep: ladder %s", ladder)
    await asyncio.to_thread(engine.launch, log_dir=run_dir)

    bind_str = (cfg.engine.cpu_bind
                or cfg.engine.vllm_extra_env.get("VLLM_CPU_OMP_THREADS_BIND")
                or "")
    telemetry = MeasurementTelemetry(
        cfg.telemetry, engine,
        bound_cpus=expand_thread_binding(bind_str) or None,
        engine_pid=getattr(engine, "pid", None),
        artifacts_dir=run_dir,
        host_telemetry=(cfg.engine.type != "remote"),
    )

    pool = WorkerPool(
        base_config={
            "persona_weights": cohort.persona_weights,
            "replica_urls": engine.replica_urls,
            "api_key": engine.api_key,
            "api_model_name": engine.api_model_name,
            "model_id": cfg.engine.model_id,
            "request_timeout_s": sim.request_timeout_s,
            "reasoning_effort": None,
            "seed": 0xC0FFEE,
        },
        log_dir=run_dir,
        max_workers=sim.open_loop_max_workers,
    )

    rungs: list[Rung] = []
    stop_reason = "ladder exhausted"
    status = "ok"

    async def _metrics() -> dict:
        try:
            return await asyncio.to_thread(engine.get_metrics)
        except Exception:  # noqa: BLE001
            return {}

    def _snapshot(phase: str, m: dict, offered: int) -> None:
        BUS.publish("snapshot", {
            "snapshot_at_ms": int(time.time() * 1000),
            "phase": phase,
            "pool_size": offered,
            "in_flight": int(m.get("num_running") or 0),
            "queue_depth": (int(m["queue_depth"])
                            if m.get("queue_depth") is not None else None),
            "requests_completed": 0, "errors": 0,
            "arrival_rate_per_min": None,
            "active_sessions": offered,
        })

    async def _chunk(phase: str, seconds: int, acc: _Acc) -> Chunk:
        m0 = await _metrics()
        t0 = time.monotonic()
        running: list[float] = []
        waiting: list[float] = []
        for _ in range(seconds):
            await asyncio.sleep(1.0)
            m = await _metrics()
            acc.add(pool.drain_turn_queue())
            if m.get("num_running") is not None:
                running.append(float(m["num_running"]))
            if m.get("queue_depth") is not None:
                waiting.append(float(m["queue_depth"]))
            _snapshot(phase, m, acc_offered[0])
        m1 = await _metrics()
        dt = max(1e-3, time.monotonic() - t0)
        gen = ((m1.get("generation_tokens_total") or 0)
               - (m0.get("generation_tokens_total") or 0))
        prm = ((m1.get("prompt_tokens_total") or 0)
               - (m0.get("prompt_tokens_total") or 0))
        return Chunk(
            running=statistics.fmean(running) if running else None,
            queue=statistics.fmean(waiting) if waiting else None,
            out_rate=gen / dt if gen else None,
            prompt_rate=prm / dt if prm else None,
        )

    acc_offered = [ladder[0]]   # current offered count, for snapshots

    try:
        await smoke_test_engine(engine)
        for idx, n in enumerate(ladder):
            acc_offered[0] = n
            if progress is not None:
                progress.update({"rung": idx + 1, "rungs": len(ladder),
                                 "concurrency": n, "done": False,
                                 "phase": "measuring"})
            log.info("headline rung %d/%d: %d concurrent streams",
                     idx + 1, len(ladder), n)
            # Enough worker processes to drive this many streams.
            import math
            await pool.scale_to(min(
                sim.open_loop_max_workers,
                max(1, math.ceil(n / sim.open_loop_inflight_per_worker))))
            await pool.set_outstanding(n)

            # Let the batch fill at the new pressure before measuring.
            settle_end = time.monotonic() + sim.headline_clear_s
            while time.monotonic() < settle_end:
                m = await _metrics()
                pool.drain_turn_queue()
                _snapshot(f"{n} streams — filling batch", m, n)
                await asyncio.sleep(1.0)

            pre_row = {
                "cohort_run_id": cohort_run_id, "step_index": idx,
                "target_pool_size": n,
                "measured_avg_pool_size": float(n),
                "measured_avg_in_flight": 0.0,
                "measurement_started_at": datetime.now(
                    timezone.utc).isoformat(),
                "measurement_duration_s": 0, "sample_size": 0,
                "ttft_violation_rate": 0.0, "tpot_violation_rate": 0.0,
                "combined_violation_rate": 0.0,
                "violation_rate_ci_lower": 0.0,
                "violation_rate_ci_upper": 0.0,
                "capacity_status": "pending", "target_status": "pending",
            }
            measurement_id = db.insert_measurement(pre_row)
            telemetry.start(measurement_id)

            acc = _Acc()
            chunks: list[Chunk] = []
            t_start = time.monotonic()
            steady = False
            while True:
                chunks.append(await _chunk(
                    f"{n} streams — measuring (chunk {len(chunks) + 1})",
                    sim.headline_measure_s, acc))
                if len(chunks) >= 2 and chunks_converged(chunks[-2],
                                                         chunks[-1]):
                    steady = True
                    break
                if time.monotonic() - t_start >= sim.headline_measure_max_s:
                    log.warning("rung %d hit the %ds cap before steady "
                                "state", n, sim.headline_measure_max_s)
                    break
            measure_s = round(time.monotonic() - t_start)
            last = chunks[-1]
            _, tele_rows, tele_agg = await telemetry.stop()
            db.insert_telemetry(tele_rows)
            kv_vals = [r["kv_cache_used_pct"] for r in tele_rows
                       if r.get("kv_cache_used_pct") is not None]
            gpu_vals = [r["gpu_power_w"] for r in tele_rows
                        if r.get("gpu_power_w") is not None]
            if tele_agg:
                agg_row = {k: v for k, v in tele_agg.items()
                           if k in AGGREGATE_COLUMN_NAMES and v is not None}
                if agg_row:
                    db.update_measurement(measurement_id, agg_row)

            out_rate = last.out_rate
            prm_rate = last.prompt_rate
            rung = Rung(
                concurrency=n,
                in_flight=(round(last.running, 1)
                           if last.running is not None else None),
                queue_depth=(round(last.queue, 1)
                             if last.queue is not None else None),
                out_tok_s=round(out_rate, 1) if out_rate else None,
                prompt_tok_s=round(prm_rate, 1) if prm_rate else None,
                total_tok_s=(round((out_rate or 0) + (prm_rate or 0), 1)
                             if (out_rate or prm_rate) else None),
                ttft_p50_ms=round(_percentile(acc.ttft, 0.50), 1) or None,
                ttft_p95_ms=round(_percentile(acc.ttft, 0.95), 1) or None,
                tpot_p50_ms=round(_percentile(acc.tpot, 0.50), 2) or None,
                tpot_p95_ms=round(_percentile(acc.tpot, 0.95), 2) or None,
                samples=len(acc.ttft), errors=acc.errors,
                kv_cache_pct=(round(sum(kv_vals) / len(kv_vals), 2)
                              if kv_vals else None),
                gpu_power_w=(round(sum(gpu_vals) / len(gpu_vals), 1)
                             if gpu_vals else None),
                steady_state=steady, measure_s=measure_s,
                held=engine_held(n, last.running),
            )
            rungs.append(rung)
            db.update_measurement(measurement_id, {
                "measured_avg_in_flight": float(rung.in_flight or 0.0),
                "measurement_duration_s": measure_s,
                "sample_size": rung.samples,
                "ttft_p50_ms": rung.ttft_p50_ms,
                "ttft_p95_ms": rung.ttft_p95_ms,
                "tpot_p50_ms": rung.tpot_p50_ms,
                "tpot_p95_ms": rung.tpot_p95_ms,
                "capacity_status": "measured",
                "target_status": "measured",
            })
            log.info("rung %d: %.0f running, %.0f out tok/s, TTFT p95 %.0fms",
                     n, rung.in_flight or 0, rung.out_tok_s or 0,
                     rung.ttft_p95_ms or 0)
            if progress is not None:
                pk = peak_rung(rungs)
                progress["peak"] = asdict(pk) if pk else None

            reason = should_stop(rungs, sim.headline_sweep_min_gain_pct)
            if reason:
                stop_reason = reason
                log.info("headline sweep stopping: %s", reason)
                break
    except EngineBrokenError as e:
        status = "engine_broken"
        stop_reason = str(e)
        log.error("headline sweep aborted: %s", e)
    finally:
        await pool.stop()
        await asyncio.to_thread(engine.shutdown)
        peak = peak_rung(rungs)
        summary = {
            "kind": "headline_sweep",
            "cohort_run_id": cohort_run_id,
            "cohort_id": cohort.id,
            "cohort_name": cohort.name,
            "model": cfg.engine.model_id,
            "engine": cfg.engine.type,
            "ladder": ladder,
            "shape": _cohort_shape(cohort),
            "rungs": [asdict(r) for r in rungs],
            "peak": asdict(peak) if peak else None,
            "stop_reason": stop_reason,
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": ("Saturation benchmark: streams are held in flight "
                     "with zero think time and EOS ignored, and no SLA is "
                     "enforced. Latency is reported as the price of the "
                     "headline, not as a gate. This says nothing about "
                     "how many real users the box serves."),
        }
        out_path = run_dir / "headline_sweep.json"
        out_path.write_text(json.dumps(summary, indent=2))
        db.finalise_run(cohort_run_id,
                        datetime.now(timezone.utc).isoformat(), status)
        db.close()
        if progress is not None:
            progress["done"] = True
            progress["peak"] = asdict(peak) if peak else None
        BUS.publish("run", {
            "event": "finished", "mode": "headline_sweep",
            "cohort_id": cohort.id, "final_status": status,
        })
        log.info("headline sweep done: peak %s",
                 summary.get("peak"))
    return out_path
