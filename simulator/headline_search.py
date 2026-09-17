"""Headline shape search — which (input, output) firehose shape jointly
maximizes concurrency AND output throughput?

The two headline numbers pull in opposite directions: shrinking
sequences shrinks per-request KV so more requests fit in flight
(concurrency ↑), while lengthening outputs amortizes prefill and
scheduling so generated tokens/sec climbs (throughput ↑) — until KV
capacity and the batch cap bite. Somewhere on that surface is the
shape that makes BOTH numbers as large as they can jointly be.

The search is FAST because ranking shapes needs saturation, not a
capacity search. One engine launch; the generator holds a fixed
number of outstanding zero-think requests (classic max-throughput
closed loop — self-throttling is exactly right here, it keeps the
engine perfectly fed); shapes change ON THE FLY: the cell persona
overlay is rewritten, workers reload their registry, and every
in-flight session is ABORTED — the engine cancels aborted requests
and each session respawns instantly with the new shape, so no
old-shape straggler contaminates the next cell. Throughput and
running-batch size are read from the ENGINE'S OWN counters (token
totals + num_running), so client-side lag cannot distort the
measurement.

Measurement runs in chunks until consecutive chunks agree (steady
state). This guards the YOUNG-KV BIAS: a long-output shape looks
great in its first half-minute because no sequence has grown its KV
yet — the running batch only sags once the pool fills. Short-output
cells converge in ~2 chunks (~40s); long-output cells measure until
the surface stops moving, capped at headline_measure_max_s. Cells
that hit the cap are scored from their last chunk and flagged
steady_state=false in the summary.

The winning shape becomes the DEFAULT of the "Headline: Generation"
workload (its persona overlay is rewritten in place) and is recorded
per MODEL FAMILY in ``config/headline_shapes.json`` — pick a sibling
model later and the UI offers to load the family's optimum. A summary
of every cell lands in ``run_NN/headline_search.json``.

Caveat the summary states explicitly: concurrency is capped by the
ENGINE shape (max_num_seqs × replicas) — this search finds the best
workload shape GIVEN the engine it runs against.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .bus import BUS
from .config import Config
from .engines import make_engine
from .personas import reload_personas
from .preflight import preflight_check
from .runs import resolve_run_dir

log = logging.getLogger(__name__)

LATTICE_IN = [32, 64, 128, 256, 512, 1024, 2048, 4096]
LATTICE_OUT = [64, 128, 256, 512, 1024, 2048, 4096]

CELL_PERSONA_ID = "headline_cell"

# Saturation-pressure controller bounds.
INITIAL_OUTSTANDING = 512
MAX_OUTSTANDING = 8192
# Event-loop lag above this means the WORKER, not the engine, is the
# thing limiting how many streams stay resident — the cell is then
# measuring our own client and must say so.
CLIENT_LAG_LIMIT_MS = 250.0
# How many times one cell may double its offered pressure while
# hunting for the point where the engine stops keeping up.
MAX_PRESSURE_STEPS = 4


@dataclass
class CellResult:
    input_tokens: int
    output_tokens: int
    out_tok_s: float | None
    prompt_tok_s: float | None
    in_flight: float | None          # engine num_running, mean
    queue_depth: float | None        # engine waiting, mean
    objective: float
    steady_state: bool = True        # converged before the time cap?
    measure_s: int = 0               # wall time spent measuring
    client_limited: bool = False     # the LOAD GENERATOR was the ceiling
    client_lag_ms: float = 0.0


@dataclass
class Chunk:
    """One measurement chunk read from the engine's own counters."""
    running: float | None            # num_running mean
    queue: float | None              # waiting mean
    out_rate: float | None           # generation tok/s
    prompt_rate: float | None        # prompt tok/s


def chunks_converged(prev: Chunk, cur: Chunk, *,
                     running_tol: float = 0.03,
                     rate_tol: float = 0.05) -> bool:
    """Two consecutive chunks agree — the running batch has stopped
    sliding (KV occupancy converged) and output rate is steady. This
    is the guard against the young-KV bias: a long-output shape looks
    great in its first half-minute because sequences haven't grown
    their KV yet; we keep measuring until the surface stops moving.

    An IDLE engine is not a settled one. Zero running batch and no
    generated tokens means the session population has not rebuilt
    after the shape swap — two such chunks used to "agree" and score
    the cell a legitimate zero, which then pruned a perfectly good
    region out of the climb. Waiting is right; scoring is not."""
    if not cur.out_rate and not cur.running:
        return False
    if prev.running and cur.running:
        if abs(cur.running - prev.running) > \
                running_tol * max(prev.running, 1.0):
            return False
    if prev.out_rate and cur.out_rate:
        if abs(cur.out_rate - prev.out_rate) > \
                rate_tol * max(prev.out_rate, 1.0):
            return False
    return cur.running is not None or cur.out_rate is not None


def objective(in_flight: float | None, out_tok_s: float | None) -> float:
    """√(concurrency × output tok/s) — the symmetric joint score. A
    cell that trades all of one for the other scores worse than a
    balanced one; a cell that produced nothing scores zero."""
    if not in_flight or not out_tok_s or in_flight <= 0 or out_tok_s <= 0:
        return 0.0
    return (in_flight * out_tok_s) ** 0.5


def _neighbors(cell: tuple[int, int],
               lattice_in: list[int] | None = None,
               lattice_out: list[int] | None = None,
               ) -> list[tuple[int, int]]:
    """One-step neighbours on the lattice. With a single-valued
    ``lattice_in`` (input pinned) this degenerates to a 1-D walk along
    output length, which is the only axis worth searching for a
    generation headline — see run_headline_search."""
    lin = lattice_in if lattice_in is not None else LATTICE_IN
    lout = lattice_out if lattice_out is not None else LATTICE_OUT
    i_idx = lin.index(cell[0])
    o_idx = lout.index(cell[1])
    out: list[tuple[int, int]] = []
    for di, do in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, no = i_idx + di, o_idx + do
        if 0 <= ni < len(lin) and 0 <= no < len(lout):
            out.append((lin[ni], lout[no]))
    return out


class ShapeClimb:
    """Greedy hill-climb on the shape lattice with an eval budget.

    ``propose()`` yields the next un-evaluated cell (the start point,
    then neighbors of the current best); ``record()`` feeds a score
    back. Terminates when the best cell's neighborhood is exhausted
    without improvement or the budget runs out.
    """

    def __init__(self, start: tuple[int, int] = (128, 512),
                 budget: int = 12,
                 lattice_in: list[int] | None = None,
                 lattice_out: list[int] | None = None):
        self.lattice_in = lattice_in if lattice_in is not None else LATTICE_IN
        self.lattice_out = (lattice_out if lattice_out is not None
                            else LATTICE_OUT)
        assert start[0] in self.lattice_in and start[1] in self.lattice_out
        self.start = start
        self.budget = budget
        self.scores: dict[tuple[int, int], float] = {}
        self._queue: list[tuple[int, int]] = [start]

    def best(self) -> tuple[int, int] | None:
        if not self.scores:
            return None
        return max(self.scores, key=lambda c: self.scores[c])

    def propose(self) -> tuple[int, int] | None:
        if len(self.scores) >= self.budget:
            return None
        while True:
            while self._queue:
                cell = self._queue.pop(0)
                if cell not in self.scores:
                    return cell
            best = self.best()
            if best is None:
                return None
            fresh = [n for n in _neighbors(
                best, self.lattice_in, self.lattice_out)
                if n not in self.scores]
            if not fresh:
                return None  # local optimum: neighborhood exhausted
            self._queue.extend(fresh)

    def record(self, cell: tuple[int, int], score: float) -> None:
        self.scores[cell] = score


def _cell_persona_spec(inp: int, out: int, *, name: str,
                       description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "ignore_eos": True,
        "input_tokens": {"constant": inp},
        "output_tokens": {"constant": out},
        "turns_per_session": {"constant": 1},
        "sessions_before_leaving": {"constant": 1},
        "inter_session_gap_seconds": {"constant": 1},
        "read_time_seconds": {"constant": 0},
        "active_think_seconds": {"constant": 0},
        "sla": {
            "ttft_target_seconds": 30.0,
            "ttft_failure_seconds": 120.0,
            "tpot_target_ms": 500.0,
            "tpot_failure_ms": 2000.0,
        },
    }


def _write_persona_overlay(catalog_dir: Path, pid: str, spec: dict) -> Path:
    """Overlay file + coordinator registry reload. The overlay (not an
    in-memory persona) matters because load-generator WORKER
    subprocesses resolve personas from the catalog — they get a
    ``reload_personas`` command after each rewrite and pick the new
    shape up at their next spawn."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / f"{pid}.yaml"
    path.write_text(yaml.safe_dump(
        {"personas": {pid: spec}}, sort_keys=False))
    reload_personas()
    return path


async def run_headline_search(
    cfg: Config,
    *,
    new_run: bool = False,
    run_dir: Path | None = None,
    catalog_dir: Path | None = None,
    # Mutable holder the service exposes via /api/status — the UI's
    # completion bar reads {cell, budget, shape, best, done} from it.
    progress: dict | None = None,
    # Prompt length, PINNED rather than searched. Input tokens cannot
    # improve this objective — they add prefill work and KV pressure
    # while contributing nothing to output tokens/sec or to
    # concurrency — so searching that axis only ever walks to the
    # smallest prompt on the lattice and produces a number nobody can
    # quote (the winning cell did 7.6 prefill tokens/sec). Pinning it
    # at the vendor convention keeps the result comparable with
    # published tables and spends the whole budget on the axis that
    # genuinely trades off.
    input_tokens: int | None = None,
) -> Path:
    """Run the shape search end-to-end. Returns the summary JSON path."""
    from .open_loop import WorkerPool
    from .persona_loader import USER_CATALOG_DIR

    catalog_dir = Path(catalog_dir) if catalog_dir else USER_CATALOG_DIR
    if run_dir is None:
        run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sim = cfg.simulation

    preflight_check(cfg.engine.hardware_requirements)
    engine = make_engine(cfg.engine.type, cfg.engine)
    # Say what we're doing before the engine launch, which is minutes
    # of silence — an empty progress dict renders as "cell 1 of ?"
    # and reads exactly like a hang.
    if progress is not None:
        progress.update({"phase": "launching engine",
                         "budget": sim.headline_cell_budget, "done": False})
    BUS.publish("run", {
        "event": "started", "mode": "headline_search",
        "cohort_id": "headline_search", "engine": cfg.engine.type,
        "model": cfg.engine.model_id, "run_dir": str(run_dir),
    })
    log.info("headline shape search: launching engine once for all cells")
    await asyncio.to_thread(engine.launch, log_dir=run_dir)

    pinned_in = int(input_tokens or sim.headline_input_tokens)
    start_out = 512 if 512 in LATTICE_OUT else LATTICE_OUT[0]
    climb = ShapeClimb(start=(pinned_in, start_out),
                       budget=sim.headline_cell_budget,
                       lattice_in=[pinned_in])
    log.info("shape search: prompt pinned at %d tokens, searching "
             "output length only", pinned_in)
    cells: list[CellResult] = []
    cell_overlay: Path | None = None
    outstanding = INITIAL_OUTSTANDING
    started = time.monotonic()

    async def _metrics() -> dict:
        try:
            return await asyncio.to_thread(engine.get_metrics)
        except Exception:  # noqa: BLE001
            return {}

    def _snapshot(phase: str, m: dict, active: int) -> None:
        BUS.publish("snapshot", {
            "snapshot_at_ms": int(time.time() * 1000),
            "phase": phase,
            "pool_size": active,
            "in_flight": int(m.get("num_running") or 0),
            "queue_depth": (int(m["queue_depth"])
                            if m.get("queue_depth") is not None else None),
            "requests_completed": 0, "errors": 0,
            "arrival_rate_per_min": None,
            "active_sessions": active,
        })

    pool = WorkerPool(
        base_config={
            "persona_weights": {CELL_PERSONA_ID: 1.0},
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

    async def _apply_pressure(n: int) -> None:
        import math
        # Size the worker fleet the way the capacity runner does. A
        # single asyncio loop cannot hold hundreds of streaming
        # responses without falling behind: at 512 streams per worker
        # the measured "concurrency" plateaued in exact multiples of a
        # per-worker ceiling, which is the client's limit, not the
        # engine's.
        await pool.scale_to(
            min(sim.open_loop_max_workers,
                max(1, math.ceil(n / sim.open_loop_inflight_per_worker))))
        await pool.set_outstanding(n)

    async def _measure_chunk(phase: str, seconds: int) -> Chunk:
        m0 = await _metrics()
        t0 = time.monotonic()
        running: list[float] = []
        waiting: list[float] = []
        for _ in range(seconds):
            await asyncio.sleep(1.0)
            m = await _metrics()
            pool.drain_turn_queue()
            if m.get("num_running") is not None:
                running.append(float(m["num_running"]))
            if m.get("queue_depth") is not None:
                waiting.append(float(m["queue_depth"]))
            _snapshot(phase, m, pool.aggregate().get("sessions_active", 0))
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

    async def _measure_cell(inp: int, out: int, phase: str) -> CellResult:
        nonlocal outstanding
        pressure_steps = 0
        for attempt in range(MAX_PRESSURE_STEPS + 2):
            if attempt == 0:
                # Instant shape swap: abort every in-flight session —
                # the engine cancels aborted requests, and each
                # session respawns at once with the NEW shape. No
                # cross-cell contamination from long-output stragglers.
                await pool.restart_sessions()
            # Wait for the batch to actually REBUILD, rather than a
            # fixed few seconds. Aborting hundreds of sessions and
            # respawning them takes as long as it takes (each respawn
            # builds a prompt and opens a request); measuring before
            # the engine is full reads an idle box and scores the
            # shape zero. Poll until the running batch stops climbing.
            refill_end = time.monotonic() + sim.headline_refill_max_s
            prev_running = -1.0
            settled_for = 0
            while time.monotonic() < refill_end:
                m = await _metrics()
                pool.drain_turn_queue()
                running = float(m.get("num_running") or 0)
                _snapshot(f"{phase} — refilling batch", m,
                          pool.aggregate().get("sessions_active", 0))
                if running > 0 and running <= prev_running * 1.05:
                    settled_for += 1
                    if settled_for >= 3:
                        break
                else:
                    settled_for = 0
                prev_running = max(prev_running, running)
                await asyncio.sleep(1.0)
            else:
                log.warning("batch did not refill within %ds after the "
                            "swap to %d→%d", sim.headline_refill_max_s,
                            inp, out)
            # Measure in chunks until the engine's own counters stop
            # trending — the running batch keeps sliding down while
            # sequences grow into their KV, and scoring before it
            # settles inflates long-output shapes (the young-KV bias).
            # Short-output cells converge in two chunks; long-output
            # cells take as long as they take, capped.
            chunks: list[Chunk] = []
            t_start = time.monotonic()
            steady = False
            max_lag = 0.0
            while True:
                n = len(chunks) + 1
                max_lag = max(max_lag,
                              float(pool.aggregate().get("loop_lag_ms") or 0.0))
                chunks.append(await _measure_chunk(
                    f"{phase} — measuring (chunk {n})",
                    sim.headline_measure_s))
                if len(chunks) >= 2 and chunks_converged(
                        chunks[-2], chunks[-1]):
                    steady = True
                    break
                if time.monotonic() - t_start >= sim.headline_measure_max_s:
                    log.warning(
                        "shape %d→%d hit the %ds measurement cap before "
                        "steady state — scoring the last chunk",
                        inp, out, sim.headline_measure_max_s)
                    break
            measure_s = round(time.monotonic() - t_start)
            max_lag = max(max_lag,
                          float(pool.aggregate().get("loop_lag_ms") or 0.0))
            last = chunks[-1]
            # Under-pressure check. If the engine is running nearly
            # everything we offered, we have measured OUR pressure,
            # not its capacity — the cell's concurrency is then a
            # constant across shapes and stops discriminating. Keep
            # doubling until the engine visibly stops keeping up.
            # The queue tolerance is RELATIVE: a couple of waiting
            # requests against a thousand running is noise, and an
            # absolute "< 1" test left the engine at a quarter of its
            # batch capacity.
            queue_ok = (last.queue is None
                        or last.queue < max(4.0, 0.02 * outstanding))
            underfed = (
                pressure_steps < MAX_PRESSURE_STEPS
                and queue_ok
                and last.running is not None
                and last.running >= 0.9 * outstanding
                and outstanding < MAX_OUTSTANDING
            )
            if underfed:
                pressure_steps += 1
                outstanding = min(MAX_OUTSTANDING, outstanding * 2)
                log.info("engine held %.0f of %d offered — raising "
                         "pressure to %d and re-measuring",
                         last.running, outstanding // 2, outstanding)
                await _apply_pressure(outstanding)
                continue
            # Nothing generated at all: the population never came
            # back. That is a FAILED measurement, not a shape that
            # deserves a zero — a zero here prunes the region from
            # the climb. Re-pressurize and try once more.
            if not last.out_rate and attempt == 0:
                log.warning("shape %d→%d measured an idle engine — "
                            "re-applying pressure and retrying", inp, out)
                await _apply_pressure(outstanding)
                continue
            if not last.out_rate:
                log.error("shape %d→%d produced no tokens on either "
                          "attempt; recording it as unmeasured", inp, out)
            return CellResult(
                input_tokens=inp, output_tokens=out,
                out_tok_s=(round(last.out_rate, 1)
                           if last.out_rate is not None else None),
                prompt_tok_s=(round(last.prompt_rate, 1)
                              if last.prompt_rate is not None else None),
                in_flight=(round(last.running, 1)
                           if last.running is not None else None),
                queue_depth=(round(last.queue, 1)
                             if last.queue is not None else None),
                objective=objective(last.running, last.out_rate),
                steady_state=steady,
                measure_s=measure_s,
                client_limited=max_lag > CLIENT_LAG_LIMIT_MS,
                client_lag_ms=round(max_lag, 1),
            )
        raise AssertionError("unreachable")

    try:
        # Initial persona + pressure before the first cell.
        first = climb.propose()
        assert first is not None
        cell_overlay = _write_persona_overlay(
            catalog_dir, CELL_PERSONA_ID,
            _cell_persona_spec(*first, name="Headline cell",
                               description="shape-search cell"))
        await _apply_pressure(outstanding)
        shape: tuple[int, int] | None = first
        while shape is not None:
            inp, out = shape
            n = len(climb.scores) + 1
            if progress is not None:
                progress.update({
                    "cell": n, "budget": sim.headline_cell_budget,
                    "shape": [inp, out], "done": False, "phase": "measuring",
                })
            log.info("headline cell %d/%d: shape %d→%d (outstanding=%d)",
                     n, sim.headline_cell_budget, inp, out, outstanding)
            cell_overlay = _write_persona_overlay(
                catalog_dir, CELL_PERSONA_ID,
                _cell_persona_spec(
                    inp, out,
                    name=f"Headline cell {inp}→{out}",
                    description=(f"shape-search cell {n}: {inp} tokens "
                                 f"in, exactly {out} out, zero think"),
                ))
            await pool.reload_personas()
            result = await _measure_cell(
                inp, out, f"shape search {inp}→{out} (cell {n})")
            cells.append(result)
            climb.record(shape, result.objective)
            if progress is not None:
                b = climb.best()
                progress["best"] = {
                    "shape": list(b) if b else None,
                    "objective": round(climb.scores.get(b, 0)) if b else 0,
                }
            log.info(
                "headline cell %d→%d: C=%.0f running, T=%.0f out tok/s "
                "→ score %.0f", inp, out,
                result.in_flight or 0, result.out_tok_s or 0,
                result.objective,
            )
            shape = climb.propose()
    finally:
        await pool.stop()
        await asyncio.to_thread(engine.shutdown)
        if cell_overlay is not None:
            cell_overlay.unlink(missing_ok=True)

        best = climb.best()
        winner = next(
            (c for c in cells
             if (c.input_tokens, c.output_tokens) == best), None,
        ) if best else None
        family = None
        if winner and winner.objective > 0:
            # The winner becomes the Headline: Generation default AND
            # the model family's stored optimum, so siblings of this
            # model can load it later without re-searching.
            from .headline_shapes import (
                apply_shape_to_generation,
                save_shape,
                shapes_path,
            )
            apply_shape_to_generation(
                catalog_dir, winner.input_tokens, winner.output_tokens)
            family = save_shape(
                shapes_path(catalog_dir), cfg.engine.model_id, {
                    "input_tokens": winner.input_tokens,
                    "output_tokens": winner.output_tokens,
                    "in_flight": winner.in_flight,
                    "out_tok_s": winner.out_tok_s,
                    "objective": round(winner.objective, 1),
                    "found_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
        else:
            reload_personas()
        summary = {
            "winner": asdict(winner) if winner else None,
            "applied_to": (
                {"persona": "headline_generation", "family": family}
                if winner and winner.objective > 0 else None),
            "input_tokens_pinned": pinned_in,
            "outstanding": outstanding,
            "cells": [asdict(c) for c in cells],
            "duration_s": round(time.monotonic() - started),
            "note": ("concurrency is capped by the engine shape "
                     "(max_num_seqs × replicas) — this is the best "
                     "workload shape for the engine it ran against"),
        }
        out_path = run_dir / "headline_search.json"
        out_path.write_text(json.dumps(summary, indent=2))
        if progress is not None:
            progress["done"] = True
            progress["winner"] = summary["winner"]
        BUS.publish("run", {
            "event": "finished", "mode": "headline_search",
            "cohort_id": "headline_search",
            "final_status": "ok" if winner else "no_result",
        })
        log.info("headline search done in %ds: %s",
                 summary["duration_s"], summary.get("winner"))
    return out_path
