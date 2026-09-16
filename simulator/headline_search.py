"""Headline shape search — which (input, output) firehose shape jointly
maximizes concurrency AND output throughput?

The two headline numbers pull in opposite directions: shrinking
sequences shrinks per-request KV so more requests fit in flight
(concurrency ↑), while lengthening outputs amortizes prefill and
scheduling so generated tokens/sec climbs (throughput ↑) — until KV
capacity and the batch cap bite. Somewhere on that surface is the
shape that makes BOTH numbers as large as they can jointly be.

This module hill-climbs a power-of-two lattice of shapes. Each cell:

  1. writes an ephemeral zero-think, EOS-pinned persona for the shape
     (as a catalog overlay, so load-generator worker subprocesses see
     it too),
  2. runs a COARSE open-loop rate search against the already-running
     engine (short windows, ~15% bracket — the point is ranking cells,
     not pinning them),
  3. reads back the stability boundary's concurrency (mean in-flight)
     and output tokens/sec, scored as √(C × T).

Climbing stops when no lattice neighbor improves the score or the
cell budget runs out. The winning shape is saved as the persona
``headline_best`` so it can be re-run at full 5% resolution like any
other workload, and a summary lands in ``run_NN/headline_search.json``.

Caveat the summary states explicitly: concurrency is capped by the
ENGINE shape (max_num_seqs × replicas) — this search finds the best
workload shape GIVEN the engine it runs against.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .bus import BUS
from .config import Config
from .engines import make_engine
from .personas import Cohort, reload_personas
from .preflight import preflight_check
from .runs import resolve_run_dir

log = logging.getLogger(__name__)

LATTICE_IN = [32, 64, 128, 256, 512, 1024, 2048, 4096]
LATTICE_OUT = [64, 128, 256, 512, 1024, 2048, 4096]

CELL_PERSONA_ID = "headline_cell"
WINNER_PERSONA_ID = "headline_best"


@dataclass
class CellResult:
    input_tokens: int
    output_tokens: int
    rate_max_per_min: float | None
    out_tok_s: float | None
    in_flight: float | None
    objective: float
    cohort_run_id: str | None = None


def objective(in_flight: float | None, out_tok_s: float | None) -> float:
    """√(concurrency × output tok/s) — the symmetric joint score. A
    cell that trades all of one for the other scores worse than a
    balanced one; a cell with no stable boundary scores zero."""
    if not in_flight or not out_tok_s or in_flight <= 0 or out_tok_s <= 0:
        return 0.0
    return (in_flight * out_tok_s) ** 0.5


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    i_idx = LATTICE_IN.index(cell[0])
    o_idx = LATTICE_OUT.index(cell[1])
    out: list[tuple[int, int]] = []
    for di, do in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, no = i_idx + di, o_idx + do
        if 0 <= ni < len(LATTICE_IN) and 0 <= no < len(LATTICE_OUT):
            out.append((LATTICE_IN[ni], LATTICE_OUT[no]))
    return out


class ShapeClimb:
    """Greedy hill-climb on the shape lattice with an eval budget.

    ``propose()`` yields the next un-evaluated cell (the start point,
    then neighbors of the current best); ``record()`` feeds a score
    back. Terminates when the best cell's neighborhood is exhausted
    without improvement or the budget runs out.
    """

    def __init__(self, start: tuple[int, int] = (128, 512),
                 budget: int = 12):
        assert start[0] in LATTICE_IN and start[1] in LATTICE_OUT
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
            fresh = [n for n in _neighbors(best) if n not in self.scores]
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
    """Overlay file + registry reload — an overlay (not an in-memory
    persona) because the load-generator WORKER SUBPROCESSES resolve
    personas from the catalog at startup."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / f"{pid}.yaml"
    path.write_text(yaml.safe_dump(
        {"personas": {pid: spec}}, sort_keys=False))
    reload_personas()
    return path


def _read_cell_metrics(db_path: Path, cohort_id: str) -> dict:
    """Boundary metrics for the newest cohort_run with this id:
    concurrency = mean in-flight at the highest stable rate, output
    tok/s summed from that window's turns."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT cohort_run_id FROM cohort_run WHERE cohort_id = ? "
            "ORDER BY started_at DESC LIMIT 1", (cohort_id,),
        ).fetchone()
        if run is None:
            return {}
        m = conn.execute(
            "SELECT measurement_id, arrival_rate_per_min, "
            "measured_avg_in_flight, measurement_duration_s "
            "FROM cohort_measurements WHERE cohort_run_id = ? "
            "AND stability = 'stable' "
            "ORDER BY arrival_rate_per_min DESC LIMIT 1",
            (run["cohort_run_id"],),
        ).fetchone()
        if m is None:
            return {"cohort_run_id": run["cohort_run_id"]}
        tok = conn.execute(
            "SELECT COALESCE(SUM(output_tokens + reasoning_tokens), 0) AS t "
            "FROM turn_events WHERE measurement_id = ?",
            (m["measurement_id"],),
        ).fetchone()
        dur = m["measurement_duration_s"] or 0
        return {
            "cohort_run_id": run["cohort_run_id"],
            "rate_max_per_min": m["arrival_rate_per_min"],
            "in_flight": m["measured_avg_in_flight"],
            "out_tok_s": (tok["t"] / dur) if dur else None,
        }
    finally:
        conn.close()


async def run_headline_search(
    cfg: Config,
    *,
    new_run: bool = False,
    run_dir: Path | None = None,
    catalog_dir: Path | None = None,
) -> Path:
    """Run the shape search end-to-end. Returns the summary JSON path."""
    from .open_loop import run_cohort_open_loop
    from .persona_loader import USER_CATALOG_DIR

    catalog_dir = Path(catalog_dir) if catalog_dir else USER_CATALOG_DIR
    if run_dir is None:
        run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    sim = cfg.simulation

    preflight_check(cfg.engine.hardware_requirements)
    engine = make_engine(cfg.engine.type, cfg.engine)
    BUS.publish("run", {
        "event": "started", "mode": "headline_search",
        "cohort_id": "headline_search", "engine": cfg.engine.type,
        "model": cfg.engine.model_id, "run_dir": str(run_dir),
    })
    log.info("headline shape search: launching engine once for all cells")
    await asyncio.to_thread(engine.launch, log_dir=run_dir)

    climb = ShapeClimb(budget=sim.headline_cell_budget)
    cells: list[CellResult] = []
    cell_overlay: Path | None = None
    prev: CellResult | None = None
    started = time.monotonic()
    try:
        while (shape := climb.propose()) is not None:
            inp, out = shape
            n = len(climb.scores) + 1
            log.info("headline cell %d/%d: shape %d→%d",
                     n, sim.headline_cell_budget, inp, out)
            cell_overlay = _write_persona_overlay(
                catalog_dir, CELL_PERSONA_ID,
                _cell_persona_spec(
                    inp, out,
                    name=f"Headline cell {inp}→{out}",
                    description=(f"shape-search cell {n}: {inp} tokens "
                                 f"in, exactly {out} out, zero think"),
                ))
            cohort = Cohort(
                id=f"headline_{inp}x{out}",
                name=f"Headline shape {inp}→{out}",
                description=f"shape-search cell: {inp} in / {out} out",
                persona_weights={CELL_PERSONA_ID: 1.0},
                category="persona",
            )
            cfg2 = copy.deepcopy(cfg)
            s2 = cfg2.simulation
            s2.open_loop_window_s = sim.headline_cell_window_s
            s2.open_loop_refine_window_s = sim.headline_cell_window_s
            s2.open_loop_warmup_s = 25
            s2.open_loop_resolution_pct = sim.headline_resolution_pct
            s2.open_loop_drain_timeout_s = 60
            # Seed the rate search near the expected boundary: the
            # previous cell's boundary scaled by output-length ratio
            # (decode-bound λ ∝ 1/output_len) — saves 2-4 doubling
            # windows per cell.
            if prev and prev.rate_max_per_min:
                seed = (prev.rate_max_per_min / 60.0
                        * (prev.output_tokens / out) * 0.5)
                s2.open_loop_initial_rate_per_s = min(64.0, max(0.5, seed))
            else:
                s2.open_loop_initial_rate_per_s = 2.0
            db_path = await run_cohort_open_loop(
                cfg2, cohort, engine=engine, run_dir=run_dir,
            )
            metrics = _read_cell_metrics(Path(db_path), cohort.id)
            result = CellResult(
                input_tokens=inp, output_tokens=out,
                rate_max_per_min=metrics.get("rate_max_per_min"),
                out_tok_s=metrics.get("out_tok_s"),
                in_flight=metrics.get("in_flight"),
                objective=objective(metrics.get("in_flight"),
                                    metrics.get("out_tok_s")),
                cohort_run_id=metrics.get("cohort_run_id"),
            )
            cells.append(result)
            climb.record(shape, result.objective)
            prev = result
            log.info(
                "headline cell %d→%d: C=%.0f in-flight, T=%.0f out tok/s "
                "→ score %.0f", inp, out,
                result.in_flight or 0, result.out_tok_s or 0,
                result.objective,
            )
    finally:
        await asyncio.to_thread(engine.shutdown)
        if cell_overlay is not None:
            cell_overlay.unlink(missing_ok=True)

        best = climb.best()
        winner = next(
            (c for c in cells
             if (c.input_tokens, c.output_tokens) == best), None,
        ) if best else None
        if winner and winner.objective > 0:
            _write_persona_overlay(
                catalog_dir, WINNER_PERSONA_ID,
                _cell_persona_spec(
                    winner.input_tokens, winner.output_tokens,
                    name=(f"Headline: best shape "
                          f"({winner.input_tokens}→{winner.output_tokens})"),
                    description=(
                        f"Shape found by the headline search: jointly "
                        f"maximizes concurrency (~{winner.in_flight:.0f} "
                        f"in flight) and output throughput "
                        f"(~{winner.out_tok_s:.0f} tok/s) on this engine. "
                        f"Re-run this workload for the full-resolution "
                        f"headline numbers. Marketing stress test — says "
                        f"nothing about real users."),
                ))
        else:
            reload_personas()
        summary = {
            "winner": asdict(winner) if winner else None,
            "saved_persona": WINNER_PERSONA_ID if winner else None,
            "cells": [asdict(c) for c in cells],
            "duration_s": round(time.monotonic() - started),
            "note": ("concurrency is capped by the engine shape "
                     "(max_num_seqs × replicas) — this is the best "
                     "workload shape for the engine it ran against"),
        }
        out_path = Path(run_dir) / "headline_search.json"
        out_path.write_text(json.dumps(summary, indent=2))
        BUS.publish("run", {
            "event": "finished", "mode": "headline_search",
            "cohort_id": "headline_search",
            "final_status": "ok" if winner else "no_result",
        })
        log.info("headline search done: %s", summary.get("winner"))
    return out_path
