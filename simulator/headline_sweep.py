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

from .benchmarks.storage import redact
from .bus import BUS
from .config import Config
from .cpu_binding import expand_thread_binding
from .database import AGGREGATE_COLUMN_NAMES, Database
from .engines import make_engine
from .headline_search import Chunk, chunks_converged, scrape_complete, whole_scrape
from .measurement import _percentile
from .open_loop import EngineBrokenError, WorkerPool, smoke_test_engine
from .personas import Cohort
from .prefix_policy import headline_reuse
from .preflight import preflight_check
from .runs import resolve_run_dir
from .telemetry import MeasurementTelemetry

log = logging.getLogger(__name__)

DEFAULT_LADDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
# A fixed batch that completes no request for this long is abandoned
# (an engine that died mid-batch would otherwise be waited on for ever).
BATCH_STALL_S = 180.0


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
    error_counts: dict[str, int] = field(default_factory=dict)
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
    # Prompt tokens served from the prefix cache over the published chunk,
    # and whether that is what the chat template explains ("template") or
    # more ("above_template"); see prefix_policy.
    prefix_hit_rate: float | None = None
    prefix_reuse: str | None = None
    # Fixed-duration rung (``headline_hold_s``): the rate is every
    # replica's generated-token counter growth over the whole hold, the
    # raw readings are in ``counters``, and ``window_rates`` are the
    # hold's consecutive chunk rates (how far the rate swings inside it).
    hold_s: float | None = None
    counters: dict | None = None
    window_rates: list[float] = field(default_factory=list)
    # Client-side totals over the rung's completed requests: the server's
    # own usage counts per request (``stream_usage``), and streamed chunks.
    client_requests: int = 0
    client_completion_tokens: int = 0
    client_prompt_tokens: int = 0
    client_usage_requests: int = 0
    client_content_chunks: int = 0
    # Fixed batch (``headline_batch_requests``): what was sent, what came
    # back, and how the client's token total compares with the engine's.
    batch: dict | None = None


def counter_growth(start: dict, end: dict) -> dict:
    """Growth of every replica's token counters between two
    ``get_replica_counters`` readings, with the raw readings kept.

    ``ok`` is False, with the reason, when the two readings cannot give a
    rate: a replica missing at either end (its whole history would count
    as new generation at the other), or a counter that went backwards (an
    engine restart). Nothing is summed in that case."""
    out: dict = {"ok": False, "reason": None, "replicas": [],
                 "generation_tokens": None, "prompt_tokens": None,
                 "prefix_cache_queries": None, "prefix_cache_hits": None,
                 "started_at": start.get("at"), "ended_at": end.get("at")}
    total = int(end.get("replicas_total") or start.get("replicas_total") or 0)
    a = {r["replica"]: r for r in start.get("replicas") or []}
    b = {r["replica"]: r for r in end.get("replicas") or []}
    if not total or len(a) < total or len(b) < total or set(a) != set(b):
        out["reason"] = (f"a replica did not answer: {len(a)} of {total} at "
                         f"the start, {len(b)} of {total} at the end")
        return out
    sums = {"generation_tokens": 0.0, "prompt_tokens": 0.0,
            "prefix_cache_queries": 0.0, "prefix_cache_hits": 0.0}
    seen = {k: False for k in sums}
    for i in sorted(a):
        row = {"replica": i, "port": a[i].get("port")}
        for key, name in (("generation_tokens_total", "generation_tokens"),
                          ("prompt_tokens_total", "prompt_tokens"),
                          ("prefix_cache_queries", "prefix_cache_queries"),
                          ("prefix_cache_hits", "prefix_cache_hits")):
            v0, v1 = a[i].get(key), b[i].get(key)
            row[f"{name}_start"], row[f"{name}_end"] = v0, v1
            if v0 is None or v1 is None:
                if name == "generation_tokens":
                    out["reason"] = f"replica {i} reported no generated-token counter"
                    out["replicas"].append(row)
                    return out
                continue
            if v1 < v0:
                out["reason"] = (f"replica {i}'s {key} went backwards "
                                 f"({v0:.0f} to {v1:.0f}): the engine restarted")
                out["replicas"].append(row)
                return out
            row[name] = v1 - v0
            sums[name] += v1 - v0
            seen[name] = True
        out["replicas"].append(row)
    for name, v in sums.items():
        out[name] = v if seen[name] else None
    out["ok"] = True
    return out


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
    error_counts: dict[str, int] = field(default_factory=dict)
    no_content: int = 0
    # SGLang's gen_throughput readings during the rung, from the first
    # one that changed (the value standing at the rung's start was the
    # previous rung's interval).
    gauge: list[float] = field(default_factory=list)
    gauge_start: float | None = None
    gauge_live: bool = False
    gauge_updates: int = 0            # fresh readings (value changes)
    # Client-side token totals over completed turns (see Rung).
    usage_turns: int = 0
    usage_completion_tokens: int = 0
    usage_prompt_tokens: int = 0
    content_chunks: int = 0

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
            self.content_chunks += int(t.get("content_chunks") or 0)
            if t.get("usage_completion_tokens") is not None:
                self.usage_turns += 1
                self.usage_completion_tokens += int(t["usage_completion_tokens"])
                self.usage_prompt_tokens += int(t.get("usage_prompt_tokens") or 0)
            if t.get("error"):
                if t["error"] == "no_content_tokens":
                    self.no_content += 1
                else:
                    self.errors += 1
                    label = str(t["error"])
                    self.error_counts[label] = self.error_counts.get(label, 0) + 1
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


def prefix_hit_rate(m0: dict, m1: dict, gauge: list[float]) -> float | None:
    """Share of the chunk's prompt tokens the engine served from its
    prefix cache: the rise in the hit counter over the rise in the query
    counter, both already summed over replicas. Engines without the
    counters (SGLang) fall back to their own hit-rate gauge, averaged."""
    q0, q1 = m0.get("prefix_cache_queries"), m1.get("prefix_cache_queries")
    h0, h1 = m0.get("prefix_cache_hits"), m1.get("prefix_cache_hits")
    if None not in (q0, q1, h0, h1) and q1 > q0:
        return round(max(0.0, h1 - h0) / (q1 - q0), 4)
    return round(statistics.fmean(gauge), 4) if gauge else None


def _prefix_plan(cfg: Config, cohort: Cohort) -> dict:
    """How this sweep reaches its prefix-cache reuse level
    (``simulation.prefix_reuse``; docs/prefix-cache-policy.md): the length
    of the system prompt every stream shares, the unique user text left of
    the shape's prompt length, and the hit rate that split gives with this
    model's chat template (``template_floor``, which every rung is judged
    against)."""
    import random

    from .prefix_policy import (
        MIN_UNIQUE_USER_TOKENS,
        OFF,
        block_tokens,
        engine_cache_off_flags,
        choose_system_tokens,
        resolve_reuse,
        template_floor,
    )
    from .tokenizer_corpus import TokenCorpus, shared_system_prompt

    shape = _cohort_shape(cohort) or {}
    n_in = int(shape.get("input_tokens") or 0)
    pid = next(iter(cohort.persona_weights), "")
    explicit = (cfg.simulation.system_prompt_tokens or {}).get(pid)
    level = resolve_reuse(cfg.simulation.prefix_reuse)
    block = block_tokens(cfg.engine.type)
    out = {"setting": level["setting"], "target": level["target"],
           "source": level["source"],
           "within_production": level["within_production"],
           "block_tokens": block, "system_prompt_tokens": 0,
           "system_prompt_actual_tokens": 0, "user_tokens": n_in,
           "template_floor": None}
    if explicit is not None:
        out.update(setting="explicit", target=None,
                   source="simulation.system_prompt_tokens")
    if level["setting"] == OFF and explicit is None:
        # No shared prompt, and the engine's cache is off at launch, so
        # nothing is expected from cache: not even the chat template.
        out.update(template_floor=0.0,
                   engine_cache_off_flags=engine_cache_off_flags(cfg.engine.type))
        return out
    if not n_in:
        return out
    try:
        corpus = TokenCorpus(cfg.engine.model_id)
        floors: dict[int, tuple] = {}

        def split(sys_n: int) -> tuple:
            """(floor, system prompt's real tokens, user tokens) at sys_n."""
            if sys_n not in floors:
                system = shared_system_prompt(corpus, pid, sys_n)
                user_n = max(1, n_in - (system[1] if system else 0))
                texts = [corpus.make_text(user_n, random.Random(seed))
                         for seed in (1, 2)]
                floors[sys_n] = (
                    template_floor(corpus._tokenizer, texts, block,
                                   system_prompt=system[0] if system else None),
                    system[1] if system else 0, user_n)
            return floors[sys_n]

        if explicit is not None:
            sys_n = max(0, int(explicit))
        else:
            sys_n = choose_system_tokens(
                lambda n: split(n)[0], n_in - MIN_UNIQUE_USER_TOKENS,
                level["target"])
        floor, actual, user_n = split(sys_n)
        out.update(system_prompt_tokens=sys_n,
                   system_prompt_actual_tokens=actual, user_tokens=user_n,
                   template_floor=floor)
        if level["target"] is not None and explicit is None and (
                floor is None or abs(floor - level["target"]) > 0.1):
            log.warning(
                "prefix reuse: aimed for %.0f%%, this model and a %d-token "
                "prompt give %s", 100 * level["target"], n_in,
                "an unknown rate (no usable chat template)" if floor is None
                else f"{100 * floor:.0f}%")
    except Exception as e:  # noqa: BLE001
        log.warning("prefix reuse plan unavailable: %s", e)
    return out


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
    existing_engine=None,
    persona_specs: dict | None = None,
    # Hold every rung this many seconds and report the whole-period rate
    # from each replica's counter growth (None: simulation.headline_hold_s;
    # 0: the chunk-convergence method).
    hold_s: float | None = None,
    # Send exactly this many requests at each rung, wait for all of them,
    # and report client and engine token totals (None:
    # simulation.headline_batch_requests; 0: off).
    batch_requests: int | None = None,
) -> Path:
    """Sweep concurrency at saturation. Returns the summary JSON path."""
    sim = cfg.simulation
    hold_s = float(sim.headline_hold_s if hold_s is None else hold_s or 0)
    batch_requests = int(sim.headline_batch_requests
                         if batch_requests is None else batch_requests or 0)
    from .prefix_policy import cache_off, engine_cache_off_flags
    if existing_engine is None and cache_off(sim.prefix_reuse):
        # Reuse "off" is off in the engine too, not only in the prompts.
        flags = engine_cache_off_flags(cfg.engine.type)
        field_name = ("sglang_extra_flags" if cfg.engine.type == "sglang_cuda"
                      else "vllm_extra_flags")
        have = list(getattr(cfg.engine, field_name, None) or [])
        setattr(cfg.engine, field_name,
                have + [f for f in flags if f not in have])
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
        config={"mode": "headline_sweep", "ladder": ladder,
                "resolved_config": redact(asdict(cfg))},
    )
    db.update_cohort_run(cohort_run_id, {"mode": "headline_sweep"})

    preflight_check(cfg.engine.hardware_requirements)
    engine = existing_engine or make_engine(cfg.engine.type, cfg.engine)
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
    if existing_engine is None:
        await asyncio.to_thread(engine.launch, log_dir=run_dir)

    from .benchmarks.assets import fingerprint
    from .benchmarks.storage import write_json
    assets = await asyncio.to_thread(fingerprint, cfg.engine)
    write_json(run_dir / "resolved-assets.json", assets)


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

    prefix_cache = await asyncio.to_thread(_prefix_plan, cfg, cohort)
    log.info("prefix reuse: %s -> %d-token shared system prompt + %d unique "
             "user tokens, expected hit rate %s",
             prefix_cache["setting"], prefix_cache["system_prompt_actual_tokens"],
             prefix_cache["user_tokens"],
             "unknown" if prefix_cache["template_floor"] is None
             else f"{100 * prefix_cache['template_floor']:.1f}%")
    shared_prompt = ({pid: prefix_cache["system_prompt_tokens"]
                      for pid in cohort.persona_weights}
                     if prefix_cache["system_prompt_tokens"] else None)

    pool = WorkerPool(
        base_config={
            "persona_weights": cohort.persona_weights,
            "persona_specs": persona_specs,
            # The shared system prompt comes out of the shape's prompt
            # length, so the request stays the size the sweep reports.
            "system_prompt_tokens": shared_prompt,
            "system_prompt_in_input": True,
            # vLLM and SGLang close a stream with its usage record when
            # asked; other engines are not asked (an unknown field is a
            # 400 on some).
            "stream_usage": cfg.engine.type.startswith(("vllm", "sglang", "mock")),
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
        hit_gauge: list[float] = []
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
                if m.get("prefix_cache_queries") is None and \
                        m.get("prefix_cache_hit_rate") is not None:
                    hit_gauge.append(float(m["prefix_cache_hit_rate"]))
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
        hit_rate = prefix_hit_rate(m0, m1, hit_gauge)
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
            prefix_hit_rate=hit_rate,
        )

    async def _counters() -> dict:
        """Every replica's own counters, retried until all answer. An
        engine that is one process is its own single replica."""
        fetch = getattr(engine, "get_replica_counters", None)
        c: dict = {}
        for i in range(3):
            if fetch is not None:
                try:
                    c = await asyncio.to_thread(fetch)
                except Exception:  # noqa: BLE001
                    c = {}
            else:
                m = await _metrics()
                c = {"replicas_total": 1, "at": time.time(),
                     "replicas": ([{
                         "replica": 0, "port": None,
                         **{k: m.get(k) for k in (
                             "generation_tokens_total", "prompt_tokens_total",
                             "prefix_cache_queries", "prefix_cache_hits")},
                     }] if scrape_complete(m) else [])}
            if c and len(c.get("replicas") or []) >= int(
                    c.get("replicas_total") or 1):
                return c
            if i < 2:
                await asyncio.sleep(0.5)
        return c

    async def _wait_idle(timeout_s: float) -> bool:
        """Wait until the engine holds no request (running or queued)."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            m = await _metrics()
            pool.drain_turn_queue()
            if scrape_complete(m) and not (m.get("num_running") or 0) \
                    and not (m.get("queue_depth") or 0):
                return True
            await asyncio.sleep(1.0)
        return False

    async def _hold(n: int, acc: _Acc) -> tuple[dict, list[Chunk], int]:
        """Hold ``n`` streams for ``hold_s`` seconds. The rate is every
        replica's counter growth over the whole hold; the chunks along
        the way give the running count and how far the rate swings."""
        c0 = await _counters()
        t0 = time.monotonic()
        chunks: list[Chunk] = []
        gaps = 0
        while True:
            done_s = time.monotonic() - t0
            left = hold_s - done_s
            if left < 1.0:
                # Under a second left: wait it out so the hold is at
                # least as long as asked.
                await asyncio.sleep(max(0.0, left))
                break
            chunk = await _chunk(
                f"{n} streams — holding ({done_s:.0f}s of {hold_s:.0f}s)",
                max(1, min(sim.headline_measure_s, int(left))), acc)
            if chunk.complete:
                chunks.append(chunk)
            else:
                gaps += 1
        c1 = await _counters()
        elapsed = time.monotonic() - t0
        growth = counter_growth(c0, c1)
        growth["elapsed_s"] = round(elapsed, 3)
        if not growth["ok"]:
            log.warning("rung %d: hold not measured: %s", n, growth["reason"])
        return growth, chunks, gaps

    async def _batch(n: int, acc: _Acc) -> tuple[dict, Chunk]:
        """Send exactly ``batch_requests`` requests, ``n`` at a time, and
        wait for all of them. Engine counters are read with the engine
        idle before and after, so their growth is these requests' tokens
        and nothing else, and can be set against the client's total."""
        c0 = await _counters()
        t0 = time.monotonic()
        await pool.set_outstanding(n, budget=batch_requests)
        running: list[float] = []
        seen, last_progress = 0, t0
        stalled = False
        while acc.completed < batch_requests:
            await asyncio.sleep(1.0)
            m = await _metrics()
            acc.add(pool.drain_turn_queue())
            if scrape_complete(m) and m.get("num_running") is not None:
                running.append(float(m["num_running"]))
            _snapshot(f"{n} streams — batch {acc.completed} of "
                      f"{batch_requests}", m, n, acc)
            now = time.monotonic()
            if acc.completed > seen:
                seen, last_progress = acc.completed, now
            elif now - last_progress > BATCH_STALL_S:
                stalled = True
                log.warning("rung %d: batch stalled at %d of %d requests",
                            n, acc.completed, batch_requests)
                break
        elapsed = time.monotonic() - t0
        idle = await _wait_idle(60.0)
        acc.add(pool.drain_turn_queue())
        c1 = await _counters()
        growth = counter_growth(c0, c1)
        gen = growth.get("generation_tokens") if growth["ok"] else None
        workers = pool.aggregate() or {}
        answered = acc.completed - acc.errors - acc.no_content
        batch = {
            "requests": batch_requests,
            "completed": acc.completed,
            "answered": answered,
            "errors": acc.errors,
            "error_counts": dict(acc.error_counts),
            "no_content": acc.no_content,
            "cancelled": workers.get("cancelled"),
            "stalled": stalled,
            "engine_idle_after": idle,
            "elapsed_s": round(elapsed, 3),
            "client_usage_requests": acc.usage_turns,
            "client_completion_tokens": acc.usage_completion_tokens,
            "client_prompt_tokens": acc.usage_prompt_tokens,
            "client_content_chunks": acc.content_chunks,
            "engine_generation_tokens": gen,
            "engine_prompt_tokens": (growth.get("prompt_tokens")
                                     if growth["ok"] else None),
            # Client total over engine total: 1.0 when every generated
            # token belongs to a request whose usage the client read.
            "client_over_engine": (round(acc.usage_completion_tokens / gen, 5)
                                   if gen else None),
            "counters": growth,
        }
        chunk = Chunk(
            running=statistics.fmean(running) if running else None,
            queue=None,
            out_rate=(gen / elapsed) if gen else None,
            prompt_rate=((growth["prompt_tokens"] / elapsed)
                         if growth["ok"] and growth.get("prompt_tokens") else None),
            complete=bool(growth["ok"]),
            rate_source="counter_batch",
            prefix_hit_rate=(
                round(growth["prefix_cache_hits"] / growth["prefix_cache_queries"], 4)
                if growth["ok"] and growth.get("prefix_cache_queries") else None),
        )
        return batch, chunk

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
            if batch_requests:
                # A fixed batch starts from an idle engine: stop
                # replacing the previous rung's requests and let them end.
                await pool.set_outstanding(0)
                await _wait_idle(float(sim.headline_refill_max_s))
            else:
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
            hold: dict | None = None
            batch: dict | None = None
            if batch_requests:
                batch, batch_chunk = await _batch(n, acc)
                chunks.append(batch_chunk)
                steady = batch_chunk.complete and not batch["stalled"]
            elif hold_s:
                hold, chunks, scrape_gaps = await _hold(n, acc)
                steady = bool(hold["ok"])
            else:
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
            hit_rate = last.prefix_hit_rate
            window_rates: list[float] = []
            if hold is not None:
                # The whole hold, from each replica's counters. Its chunks
                # are kept only for the running count and the swing.
                window_rates = [round(c.out_rate, 1) for c in chunks
                                if c.out_rate]
                span = hold["elapsed_s"]
                out_rate = (hold["generation_tokens"] / span
                            if hold["ok"] and hold["generation_tokens"] else None)
                prm_rate = (hold["prompt_tokens"] / span
                            if hold["ok"] and hold["prompt_tokens"] else None)
                rate_source = "counter_hold"
                if hold["ok"] and hold.get("prefix_cache_queries"):
                    hit_rate = round(hold["prefix_cache_hits"]
                                     / hold["prefix_cache_queries"], 4)
                runs = [c.running for c in chunks if c.running is not None]
                queues = [c.queue for c in chunks if c.queue is not None]
                last = Chunk(
                    running=statistics.fmean(runs) if runs else None,
                    queue=statistics.fmean(queues) if queues else None,
                    out_rate=out_rate, prompt_rate=prm_rate,
                    complete=bool(hold["ok"]), rate_source=rate_source,
                    prefix_hit_rate=hit_rate)
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
                error_counts=dict(acc.error_counts),
                no_content=acc.no_content, scrape_gaps=scrape_gaps,
                kv_cache_pct=(round(sum(kv_vals) / len(kv_vals), 2)
                              if kv_vals else None),
                gpu_power_w=(round(sum(gpu_vals) / len(gpu_vals), 1)
                             if gpu_vals else None),
                steady_state=steady, measure_s=measure_s,
                held=engine_held(n, last.running),
                rate_source=rate_source,
                prefix_hit_rate=hit_rate,
                prefix_reuse=headline_reuse(hit_rate,
                                            prefix_cache["template_floor"]),
                hold_s=hold["elapsed_s"] if hold is not None else None,
                counters=hold,
                window_rates=window_rates,
                client_requests=acc.completed,
                client_completion_tokens=acc.usage_completion_tokens,
                client_prompt_tokens=acc.usage_prompt_tokens,
                client_usage_requests=acc.usage_turns,
                client_content_chunks=acc.content_chunks,
                batch=batch,
            )
            if rung.prefix_reuse == "above_template":
                log.warning("rung %d: %.0f%% of prompt tokens came from the "
                            "prefix cache; the chat template and shared "
                            "system prompt explain %.0f%%",
                            n, 100 * rung.prefix_hit_rate,
                            100 * prefix_cache["template_floor"])
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
        if existing_engine is None:
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
            # The reuse level this sweep was built to reach, how (shared
            # system prompt + unique user text), and the hit rate that
            # gives (template_floor), against which every rung's
            # prefix_hit_rate is judged.
            "prefix_cache": prefix_cache,
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
