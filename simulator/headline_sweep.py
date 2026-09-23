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
from .headline_search import Chunk, chunks_converged, scrape_complete, whole_scrape
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
    samples: int = 0                 # completed turns with an answer
    errors: int = 0                  # failed turns
    no_content: int = 0              # reasoning-only completions
    scrape_gaps: int = 0             # chunks discarded for a missing replica
    kv_cache_pct: float | None = None
    gpu_power_w: float | None = None
    steady_state: bool = True
    measure_s: int = 0
    held: bool = True                # did the engine hold what we offered?
    # "counter": the engine's generated-token counter over the window;
    # "engine_gauge": SGLang's own generation-rate gauge, averaged over
    # the rung, used when the window saw too few completions for the
    # counter to be anything but a count of finishing waves.
    rate_source: str = "counter"


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
    """The headline: highest SUSTAINED output token rate.

    Sustained is the whole word. A rung measured before the running
    batch and the token rate stopped moving is a transient -- the
    engine discharging a backlog, or still filling one -- and it is
    routinely the LARGEST number in the sweep, which is exactly why it
    cannot be allowed to become the headline. The search already
    discarded unsettled candidates (``Candidate.usable``); this
    function said "sustained" in its docstring and then ranked on
    everything, so a per-run peak could still be an artifact.

    Unsettled rungs are used only when nothing settled at all, so a
    sweep that never converged still reports something rather than
    nothing -- and the rung it returns carries ``steady_state=False``
    for the caller to see.
    """
    scored = [r for r in rungs if r.out_tok_s]
    if not scored:
        return None
    # A rung whose answers mostly failed is not a rate anyone is
    # served at: Kimi-K2-Thinking on KTransformers at 512 offered
    # generated 97 tok/s while all 256 requests timed out, and it
    # beat the clean 128-stream rung (60). No answers back yet is not
    # failure -- a slow rung can end before its first lifetime does.
    served = [r for r in scored if rung_served(r)] or scored
    settled = [r for r in served if r.steady_state]
    return max(settled or served, key=lambda r: r.out_tok_s or 0.0)


# The roofline's publishing bar (roofline.MIN_SUCCESS).
RUNG_MIN_SUCCESS = 0.9


def rung_served(r: Rung) -> bool:
    """At least RUNG_MIN_SUCCESS of the rung's finished requests got an
    answer (reasoning-only completions count as served: the engine
    generated them), or none finished at all."""
    done = r.samples + r.errors + r.no_content
    return done == 0 or (r.samples + r.no_content) / done >= RUNG_MIN_SUCCESS


def engine_dead(rung: Rung) -> bool:
    """A rung in which the engine answered nothing but errors and its
    metrics were gone. DeepSeek-V3.1-NVFP4 at tp8 on the XE7740 hit
    CUDA out-of-memory in every scheduler rank four minutes into its
    first rung; the API process lived on, the sweep climbed three more
    rungs and spent fifteen minutes collecting 2.6 million errors,
    and the cell read "ladder exhausted" -- not the memory failure it
    was, which the roofline would have escalated."""
    return (rung.samples == 0 and rung.errors > 0
            and rung.in_flight is None and not rung.out_tok_s)


def should_stop(rungs: list[Rung], min_gain_pct: float,
                capacity: int | None = None) -> str | None:
    """Stop when the curve has plateaued or the engine is saturated.

    ``capacity`` is the engine's configured batch width across the box
    (max_num_seqs x replicas). A running count far below it is not the
    engine's ceiling: gpt-oss-20b's confirmation (8 x 2048 slots) held
    49 of 128 offered because thousands of reasoning-only answers and
    parse errors finished early and the load generator could not
    refill streams as fast as they ended -- and the sweep stopped
    there, at 9,389 tok/s, on an engine whose search had held 1,500+
    streams at 105k. Short of a tenth of capacity the shortfall is the
    client's, and the climb continues. (A KV-bound ceiling below the
    batch width -- Kimi-K2 NVFP4 held ~990 of 2,048 -- is well above a
    tenth and still stops the climb.)

    Returns a human reason, or None to keep climbing.
    """
    if not rungs:
        return None
    last = rungs[-1]
    # A batch ceiling means the engine STOPPED GROWING, not merely
    # that one rung came up short. Two extra conditions, both learned
    # the hard way: a rung that never settled says nothing (the
    # shortfall may just be fill time), and a rung whose running
    # count still beat every earlier rung plainly has not hit a
    # ceiling. Without these, a sweep aborted at 77-of-128 offered
    # and called it the ceiling of an engine already measured holding
    # 6,430 streams.
    prev_best_running = max(
        (r.in_flight or 0.0) for r in rungs[:-1]) if len(rungs) > 1 else 0.0
    still_growing = (last.in_flight or 0.0) > prev_best_running * 1.05
    far_below_capacity = bool(capacity) and (last.in_flight or 0.0) < 0.1 * float(capacity)
    if (last.steady_state
            and not still_growing
            and not far_below_capacity
            and not engine_held(last.concurrency, last.in_flight)):
        return (f"the engine held {last.in_flight:.0f} of "
                f"{last.concurrency} offered streams and stopped growing "
                f"— its own batch ceiling, so wider offers only add "
                f"queueing")
    if len(rungs) >= 3:
        best_before = max((r.out_tok_s or 0.0) for r in rungs[:-2])
        recent = max((r.out_tok_s or 0.0) for r in rungs[-2:])
        if best_before > 0 and \
                (recent - best_before) / best_before < min_gain_pct / 100.0:
            return (f"output rate gained under {min_gain_pct:.0f}% over the "
                    f"last two rungs — the throughput curve has plateaued")
    return None


def _batch_capacity(cfg) -> int | None:
    """max_num_seqs x replicas for the engine under test, or None."""
    eng = getattr(cfg, "engine", None)
    mns = getattr(eng, "max_num_seqs", None)
    if not mns:
        return None
    groups = getattr(eng, "replica_devices", None) or [None]
    return int(mns) * max(1, len(groups))


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
    """Per-rung accumulators for client-side latency and outcomes.

    ``errors`` are turns that failed: transport errors, timeouts,
    stalls, an engine's parse failure (gpt-oss's HarmonyError). A
    reasoning model that spent its whole budget thinking and never
    started an answer is filed by the client as ``no_content_tokens``;
    it generated real tokens the engine counted, so it is kept apart
    (``no_content``) rather than counted as a failure or a success.
    """
    ttft: list[float] = field(default_factory=list)
    tpot: list[float] = field(default_factory=list)
    completed: int = 0                                # turns finished
    errors: int = 0
    no_content: int = 0
    # SGLang's gen_throughput readings during the rung, from the first
    # one that changed (the value standing at the rung's start was the
    # previous rung's interval).
    gauge: list[float] = field(default_factory=list)
    gauge_start: float | None = None
    gauge_live: bool = False
    gauge_updates: int = 0            # fresh readings (value changes)

    def add_gauge(self, v: float | None) -> None:
        if v is None:
            return
        if not self.gauge_live:
            if self.gauge_start is None:
                self.gauge_start = v
                return
            if v == self.gauge_start:
                return
            self.gauge_live = True
        if not self.gauge or v != self.gauge[-1]:
            self.gauge_updates += 1
        # Per second, so each reading weighs as long as it stood --
        # i.e. by the length of the interval it covers.
        self.gauge.append(float(v))

    def add(self, turns: list[dict]) -> None:
        for t in turns:
            self.completed += 1
            if t.get("error"):
                if t["error"] == "no_content_tokens":
                    self.no_content += 1
                else:
                    self.errors += 1
                continue
            if t.get("ttft_ms") is not None:
                self.ttft.append(float(t["ttft_ms"]))
            if t.get("tpot_ms") is not None:
                self.tpot.append(float(t["tpot_ms"]))


# A chunk that saw fewer completions than this many per stream in the
# system reads the generated-token counter to within a wave: SGLang
# adds a request's tokens when it FINISHES, and a closed loop of
# equal-length requests on a slow engine finishes together. Kimi-K2-
# Thinking on KTransformers (64 streams, ~1 s a step) read 130, 265 or
# 264 tok/s from whole waves landing in a chunk while its scheduler
# logged 47-65. Below it, SGLang's gen_throughput gauge is the rate.
GAUGE_MIN_TURNS_PER_STREAM = 2
# One reading covers ~40 decode steps and swings 47-65 with whether a
# prefill fell inside it; a rung measured from one reading "settled"
# in 31 s at 58 with no answer back. Three is the least a rate is.
GAUGE_MIN_UPDATES = 3


def wave_bound(completions: int, population: float | None) -> bool:
    """Too few completions in a chunk for the finish-time counter."""
    return bool(population) and completions < GAUGE_MIN_TURNS_PER_STREAM * population


async def run_headline_sweep(
    cfg: Config,
    cohort: Cohort,
    *,
    new_run: bool = False,
    run_dir: Path | None = None,
    max_concurrency: int | None = None,
    progress: dict | None = None,
    # Override the concurrency ladder. A coarse ladder is enough to
    # RANK candidates in a joint engine/shape search; the winner then
    # earns a full-resolution sweep.
    ladder_override: list[int] | None = None,
) -> Path:
    """Sweep concurrency at saturation. Returns the summary JSON path."""
    sim = cfg.simulation
    if run_dir is None:
        run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ladder = [c for c in (ladder_override or sim.headline_sweep_ladder
                          or DEFAULT_LADDER)
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
        progress.update({
            "model": cfg.engine.model_id,
            "engine": cfg.engine.type,
            "shape": _cohort_shape(cohort),
            "ladder": ladder,
        })
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

    # Rolling latency window. A saturation sweep never publishes
    # per-turn events -- at eight thousand concurrent streams that is
    # tens of thousands of messages a second, which would cost more
    # than the measurement -- so the live latency charts had nothing
    # to draw. The percentiles ride on the once-a-second snapshot
    # instead, over a bounded tail so the cost stays flat no matter
    # how long a rung runs.
    LATENCY_WINDOW = 4000

    def _snapshot(phase: str, m: dict, offered: int,
                  acc: "_Acc | None" = None) -> None:
        lat: dict = {}
        if acc is not None:
            for name, series in (("ttft", acc.ttft), ("tpot", acc.tpot)):
                if not series:
                    continue
                tail = sorted(series[-LATENCY_WINDOW:])
                lat[f"{name}_p50_ms"] = round(_percentile(tail, 0.50), 2)
                lat[f"{name}_p95_ms"] = round(_percentile(tail, 0.95), 2)
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
            **lat,
        })

    async def _chunk(phase: str, seconds: int, acc: _Acc) -> Chunk:
        # The two boundary scrapes are the measurement (the rates are
        # their counter deltas): retried until whole, decisive if not.
        # A mid-chunk scrape that misses a replica only thins the
        # running/queue means and is skipped, not fatal.
        m0 = await whole_scrape(_metrics)
        whole = scrape_complete(m0)
        seen_before = acc.completed
        t0 = time.monotonic()
        running: list[float] = []
        waiting: list[float] = []
        for _ in range(seconds):
            await asyncio.sleep(1.0)
            m = await _metrics()
            acc.add(pool.drain_turn_queue())
            acc.add_gauge(m.get("gen_throughput"))
            if scrape_complete(m):
                if m.get("num_running") is not None:
                    running.append(float(m["num_running"]))
                if m.get("queue_depth") is not None:
                    waiting.append(float(m["queue_depth"]))
            _snapshot(phase, m, acc_offered[0], acc)
        m1 = await whole_scrape(_metrics)
        whole = whole and scrape_complete(m1)
        dt = max(1e-3, time.monotonic() - t0)
        gen = ((m1.get("generation_tokens_total") or 0)
               - (m0.get("generation_tokens_total") or 0))
        prm = ((m1.get("prompt_tokens_total") or 0)
               - (m0.get("prompt_tokens_total") or 0))
        run_mean = statistics.fmean(running) if running else None
        q_mean = statistics.fmean(waiting) if waiting else None
        out_rate = gen / dt if gen else None
        prompt_rate = prm / dt if prm else None
        source = "counter"
        if wave_bound(acc.completed - seen_before,
                      (run_mean or 0.0) + (q_mean or 0.0)):
            if acc.gauge_updates >= GAUGE_MIN_UPDATES:
                # Averaged over the rung, not the chunk: one reading
                # covers 40 decode steps, and consecutive ones swing
                # with whether a prefill landed inside them.
                ratio = (prompt_rate / out_rate) if (out_rate and prompt_rate) else None
                out_rate = statistics.fmean(acc.gauge)
                prompt_rate = out_rate * ratio if ratio else None
                source = "engine_gauge"
            elif m1.get("gen_throughput") is not None:
                # The gauge exists but has not moved often enough since
                # the rung began; a few-completion counter, or one
                # reading, could agree with the next chunk by accident.
                # Unmeasured keeps measuring.
                out_rate, prompt_rate = None, None
                source = "pending"
        return Chunk(
            running=run_mean,
            queue=q_mean,
            out_rate=out_rate,
            prompt_rate=prompt_rate,
            complete=whole,
            rate_source=source,
        )

    acc_offered = [ladder[0]]   # current offered count, for snapshots

    kv_capacity: float | None = None
    try:
        await smoke_test_engine(engine)
        # The KV pool the engine actually built. Recorded because the
        # memory knob is TRANSLATED per engine (see engines/vram.py):
        # two engines given the same share of VRAM should end up with
        # comparable room, and if they did not, a throughput
        # comparison between them is measuring allocation instead.
        try:
            kv_capacity = (await _metrics()).get("kv_cache_tokens")
            if kv_capacity:
                log.info("KV pool: %.0f tokens across the box "
                         "(engine %s)", kv_capacity, cfg.engine.type)
        except Exception:  # noqa: BLE001
            kv_capacity = None
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
            scrape_gaps = 0
            t_start = time.monotonic()
            steady = False
            while True:
                chunk = await _chunk(
                    f"{n} streams — measuring (chunk {len(chunks) + 1})",
                    sim.headline_measure_s, acc)
                if not chunk.complete:
                    # A replica did not answer: the counters this
                    # chunk's rates come from are not the box's. It
                    # is measured again, never compared or published.
                    scrape_gaps += 1
                    log.warning("rung %d: a replica scrape failed during "
                                "chunk %d; discarding it", n,
                                len(chunks) + scrape_gaps)
                    if time.monotonic() - t_start >= sim.headline_measure_max_s:
                        break
                    continue
                chunks.append(chunk)
                if len(chunks) >= 2 and chunks_converged(chunks[-2],
                                                         chunks[-1]):
                    steady = True
                    break
                if time.monotonic() - t_start >= sim.headline_measure_max_s:
                    log.warning("rung %d hit the %ds cap before steady "
                                "state", n, sim.headline_measure_max_s)
                    break
            measure_s = round(time.monotonic() - t_start)
            if not chunks:
                # Every chunk of the rung had a scrape gap. Nothing
                # here is a measurement; the rung records that and
                # the ladder moves on.
                log.warning("rung %d: no whole chunk in %ds; not measured",
                            n, measure_s)
                chunks.append(Chunk(running=None, queue=None,
                                    out_rate=None, prompt_rate=None,
                                    complete=False))
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
            rate_source = last.rate_source
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
                no_content=acc.no_content, scrape_gaps=scrape_gaps,
                kv_cache_pct=(round(sum(kv_vals) / len(kv_vals), 2)
                              if kv_vals else None),
                gpu_power_w=(round(sum(gpu_vals) / len(gpu_vals), 1)
                             if gpu_vals else None),
                steady_state=steady, measure_s=measure_s,
                held=engine_held(n, last.running),
                rate_source=rate_source,
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
                progress["rungs_done"] = [asdict(r) for r in rungs]
                progress["kv_cache_tokens"] = kv_capacity
                pk = peak_rung(rungs)
                progress["peak"] = asdict(pk) if pk else None

            if engine_dead(rung):
                cause = None
                probe = getattr(engine, "_fatal_in_log", None)
                if callable(probe):
                    try:
                        cause = probe(None)      # the whole log, not its tail
                    except Exception:  # noqa: BLE001
                        cause = None
                raise EngineBrokenError(
                    f"engine stopped serving at {n} streams"
                    f"{': ' + cause if cause else ' (no metrics, every request failed)'}")
            reason = should_stop(rungs, sim.headline_sweep_min_gain_pct,
                                 capacity=_batch_capacity(cfg))
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
            # Whole-box KV capacity in tokens: what makes a
            # cross-engine result comparable rather than merely
            # adjacent. None when the engine does not report it.
            "kv_cache_tokens": kv_capacity,
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
