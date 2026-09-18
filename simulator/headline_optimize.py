"""Joint engine + shape search for the headline number.

The engine config and the request shape are COUPLED, and optimizing
them separately walks in circles. Today's evidence: at max_num_seqs
128 the best shape looked like 4096 output tokens; at 512 it was
2048; raising the cap to 1024 and halving the output produced the
same throughput by a different route. Each answer was correct for
the engine it was measured against and wrong for the next one.

So this searches the product. For each engine candidate it runs the
ordinary saturation sweep — the same code path, with the same
honesty guards — over a COARSE concurrency ladder, which is enough
to rank. The winner then earns a full-resolution sweep, and that
run is what gets published.

The SERVER is part of that product too. vLLM and TensorRT-LLM make
different scheduling and KV-layout choices, and which one wins is a
property of the model and the shape, not a standing fact — so it is
measured, not assumed. Held constant across engines: the model, the
request shape, the concurrency ladder and the stopping rule, which
is what makes the comparison mean anything.

Deliberately NOT searched: replicas and tensor parallelism. The
engine optimizer already settled 8×tp1 for this box, and a 3B-active
MoE has no reason to pay an all-reduce. Ranking is by sustained
output tokens/sec, after discarding candidates the measurement
itself flagged as untrustworthy (see `usable`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bus import BUS
from .config import Config
from .personas import Cohort

log = logging.getLogger(__name__)

# Coarse ladder for ranking: the peak always sits at the top of the
# curve, so the low rungs only cost time. The winner gets the full one.
SEARCH_LADDER = [512, 2048, 4096, 8192]

# The engine axis defaults to vLLM alone so an unqualified search
# costs what it always did; asking for both engines is an explicit
# choice that doubles the grid.
DEFAULT_ENGINES = ["vllm_cuda_multi"]

PRESETS: dict[str, dict] = {
    "quick": {"max_num_seqs": [512, 1024], "output_tokens": [1024, 2048]},
    "standard": {"max_num_seqs": [512, 1024, 2048],
                 "output_tokens": [512, 1024, 2048]},
    "thorough": {"max_num_seqs": [256, 512, 1024, 2048],
                 "output_tokens": [512, 1024, 2048, 4096]},
}


@dataclass
class Candidate:
    """One (engine, shape) point and what the sweep measured for it."""
    engine: str
    max_num_seqs: int
    output_tokens: int
    input_tokens: int
    out_tok_s: float | None = None
    total_tok_s: float | None = None
    in_flight: float | None = None
    queue_depth: float | None = None
    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    kv_cache_pct: float | None = None
    gpu_power_w: float | None = None
    steady_state: bool = True
    held: bool = True
    error: str | None = None
    run_dir: str | None = None

    @property
    def usable(self) -> bool:
        """Did this candidate produce a number worth ranking?

        A peak measured before the engine settled, or one the engine
        was not actually sustaining, is not a throughput result — it
        is an artifact. Ranking on it is how a search talks itself
        into the wrong answer.
        """
        return (self.error is None and self.out_tok_s is not None
                and self.out_tok_s > 0 and self.steady_state)


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Usable candidates, best sustained output rate first."""
    return sorted([c for c in candidates if c.usable],
                  key=lambda c: c.out_tok_s or 0.0, reverse=True)


def grid(preset: str = "standard",
         max_num_seqs: list[int] | None = None,
         output_tokens: list[int] | None = None,
         engines: list[str] | None = None) -> list[tuple[str, int, int]]:
    """(engine, max_num_seqs, output_tokens) points to evaluate.

    Ordered engine-major, then by ascending cost: an interrupted
    search still leaves a usable ranking, and one engine's whole
    grid completes before the other starts, so a partial result is
    a complete answer about one server rather than half an answer
    about two."""
    base = PRESETS.get(preset) or PRESETS["standard"]
    mns = sorted(max_num_seqs or base["max_num_seqs"])
    outs = sorted(output_tokens or base["output_tokens"])
    engs = list(engines or DEFAULT_ENGINES)
    return [(e, m, o) for e in engs for m in mns for o in outs]


def estimate_minutes(pairs: list, *,
                     launch_min: float = 5.0,
                     rung_min: float = 1.2,
                     rungs: int = len(SEARCH_LADDER),
                     final_sweep_min: float = 14.0) -> int:
    """Rough wall clock, for the UI to show before committing. Every
    pair costs an engine launch because vLLM cannot change
    max_num_seqs in place."""
    return int(len(pairs) * (launch_min + rungs * rung_min)
               + final_sweep_min)


def engines_in(pairs: list) -> list[str]:
    """Distinct engines the grid covers, in order."""
    out: list[str] = []
    for p in pairs:
        if p[0] not in out:
            out.append(p[0])
    return out


def best_per_engine(candidates: list) -> dict:
    """Each engine's best usable candidate — the head-to-head the
    operator actually wants to see, which a single ranked list hides
    as soon as one engine sweeps the top."""
    out: dict = {}
    for c in rank(candidates):
        out.setdefault(c.engine, c)
    return out


@dataclass
class _Progress:
    """Mutable holder the service exposes via /api/status."""
    pair: int = 0
    pairs: int = 0
    phase: str = "starting"
    current: dict = field(default_factory=dict)
    best: dict | None = None
    done: bool = False


def _peak_from_summary(path: Path) -> tuple[dict | None, str | None]:
    """The peak rung a sweep recorded, plus its stop reason."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, f"unreadable sweep summary: {e}"
    if doc.get("status") != "ok":
        return doc.get("peak"), doc.get("stop_reason") or doc.get("status")
    return doc.get("peak"), None


async def run_headline_optimize(
    cfg: Config,
    cohort: Cohort,
    *,
    preset: str = "standard",
    max_num_seqs: list[int] | None = None,
    output_tokens: list[int] | None = None,
    engines: list[str] | None = None,
    input_tokens: int = 128,
    build_config,
    runs_base: Path,
    progress: dict | None = None,
) -> Path:
    """Search (max_num_seqs × output length), then sweep the winner.

    ``build_config`` is injected (the service owns config generation):
    it takes a dict of engine overrides and returns a config path.
    """
    from .headline_shapes import apply_shape_to_generation
    from .headline_sweep import run_headline_sweep
    from .persona_loader import USER_CATALOG_DIR

    pairs = grid(preset, max_num_seqs, output_tokens, engines)
    started = time.monotonic()
    results: list[Candidate] = []
    BUS.publish("run", {
        "event": "started", "mode": "headline_optimize",
        "cohort_id": cohort.id, "model": cfg.engine.model_id,
        "pairs": len(pairs), "engines": engines_in(pairs),
    })
    log.info("headline optimize: %d (engine, shape) points over %s, ~%d min",
             len(pairs), ", ".join(engines_in(pairs)),
             estimate_minutes(pairs))

    def _emit(**kw):
        if progress is not None:
            progress.update(kw)

    _emit(pairs=len(pairs), pair=0, phase="searching", done=False,
          estimate_min=estimate_minutes(pairs),
          engines=engines_in(pairs), model=cfg.engine.model_id)

    for i, (eng, mns, out_tok) in enumerate(pairs, 1):
        cand = Candidate(engine=eng, max_num_seqs=mns,
                         output_tokens=out_tok, input_tokens=input_tokens)
        _emit(pair=i, phase="searching",
              current={"engine": eng, "max_num_seqs": mns,
                       "output_tokens": out_tok})
        log.info("candidate %d/%d: %s mns=%d shape=%d→%d",
                 i, len(pairs), eng, mns, input_tokens, out_tok)
        try:
            # The shape lives in the persona; the engine lives in the
            # generated config. Both must be set before the sweep.
            apply_shape_to_generation(USER_CATALOG_DIR, input_tokens, out_tok)
            cfg_path = build_config({"engine": eng, "max_num_seqs": mns})
            from .config import load_config
            sub = load_config(cfg_path)
            sub.output.db_directory = str(runs_base)
            summary = await run_headline_sweep(
                sub, cohort, new_run=True,
                ladder_override=SEARCH_LADDER,
            )
            peak, err = _peak_from_summary(summary)
            cand.run_dir = str(Path(summary).parent)
            if err and peak is None:
                cand.error = err
            elif peak:
                cand.out_tok_s = peak.get("out_tok_s")
                cand.total_tok_s = peak.get("total_tok_s")
                cand.in_flight = peak.get("in_flight")
                cand.queue_depth = peak.get("queue_depth")
                cand.ttft_p95_ms = peak.get("ttft_p95_ms")
                cand.tpot_p95_ms = peak.get("tpot_p95_ms")
                cand.kv_cache_pct = peak.get("kv_cache_pct")
                cand.gpu_power_w = peak.get("gpu_power_w")
                cand.steady_state = bool(peak.get("steady_state", True))
                cand.held = bool(peak.get("held", True))
            else:
                cand.error = "sweep produced no peak"
        except Exception as e:  # noqa: BLE001
            # One bad candidate (an engine shape that will not launch,
            # say) must not abandon the search.
            cand.error = f"{type(e).__name__}: {e}"
            log.warning("candidate %s mns=%d shape=%d→%d failed: %s",
                        eng, mns, input_tokens, out_tok, e)
        results.append(cand)
        best = rank(results)
        # The head-to-head, live. A single "best" hides the loser the
        # moment one engine sweeps the top, which is the comparison the
        # operator started the search to see.
        _emit(best=asdict(best[0]) if best else None,
              best_per_engine={k: asdict(v) for k, v
                               in best_per_engine(results).items()},
              done_candidates=[asdict(c) for c in results])
        log.info("  -> %s", "error: " + cand.error if cand.error
                 else f"{cand.out_tok_s:.0f} out tok/s at "
                      f"{cand.in_flight:.0f} streams")

    ranked = rank(results)
    winner = ranked[0] if ranked else None
    final_dir = None
    if winner is not None:
        # The winner earns a full-resolution sweep; THAT is the run to
        # publish, and it leaves the box configured as it was measured.
        _emit(phase="confirming winner",
              current={"engine": winner.engine,
                       "max_num_seqs": winner.max_num_seqs,
                       "output_tokens": winner.output_tokens})
        log.info("winner: %s mns=%d shape=%d→%d — full sweep",
                 winner.engine, winner.max_num_seqs, input_tokens,
                 winner.output_tokens)
        try:
            apply_shape_to_generation(
                USER_CATALOG_DIR, input_tokens, winner.output_tokens)
            cfg_path = build_config({"engine": winner.engine,
                                     "max_num_seqs": winner.max_num_seqs})
            from .config import load_config
            sub = load_config(cfg_path)
            sub.output.db_directory = str(runs_base)
            final = await run_headline_sweep(sub, cohort, new_run=True)
            final_dir = str(Path(final).parent)
        except Exception as e:  # noqa: BLE001
            log.error("winner confirmation sweep failed: %s", e)

    summary_doc = {
        "kind": "headline_optimize",
        "model": cfg.engine.model_id,
        "cohort_id": cohort.id,
        "input_tokens": input_tokens,
        "preset": preset,
        "engines": engines_in(pairs),
        "best_per_engine": {k: asdict(v)
                            for k, v in best_per_engine(results).items()},
        "candidates": [asdict(c) for c in results],
        "ranked": [asdict(c) for c in ranked],
        "winner": asdict(winner) if winner else None,
        "final_run_dir": final_dir,
        "duration_s": round(time.monotonic() - started),
        "note": ("Search ladder is coarse — it ranks candidates. The "
                 "winner's full-resolution sweep in final_run_dir is "
                 "the run to publish."),
    }
    out = Path(runs_base) / "headline_optimize.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary_doc, indent=2))
    _emit(phase="finished", done=True,
          best=asdict(winner) if winner else None)
    BUS.publish("run", {
        "event": "finished", "mode": "headline_optimize",
        "cohort_id": cohort.id,
        "final_status": "ok" if winner else "no_result",
    })
    log.info("headline optimize done in %ds: %s",
             summary_doc["duration_s"], summary_doc.get("winner"))
    return out
