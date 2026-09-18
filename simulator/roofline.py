"""Roofline autopilot — what is the most this hardware can do?

A capacity benchmark asks "how many users does THIS deployment hold".
A roofline asks the opposite: given the box, which model and which
engine and which shape produce the largest sustained token rate at
all. It is a search over the whole product — models x engines x
launch shapes — and it takes hours, so everything here is built
around that fact rather than apologising for it.

Three consequences, and they shape the whole module:

* **Every step is written to disk before the next begins.** The
  operator will not be watching; they will close a laptop and come
  back over a VPN to a page that must simply be correct. There is no
  in-memory state that matters, so a reconnect is a GET, not a replay.

* **It resumes.** A run that dies at cell 23 of 40 must not repeat the
  first 22, because each of those cost minutes of engine launch. Cells
  are keyed by (model, engine, shape) and completed ones are skipped.

* **It reports a matrix, not a winner.** "Fastest on this box" is one
  number; which engine wins for which model is the finding that
  survives, because it is what tells you what to do with the NEXT
  model. The winner is a cell of that matrix, not a replacement for it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .bus import BUS

log = logging.getLogger(__name__)

STATE_NAME = "roofline.json"

# Default shape grid. Short outputs on purpose: a KV-heavy model is
# bound by cache traffic long before it is bound by compute, so long
# generations only move the peak down the concurrency ladder.
DEFAULT_SHAPES = {"max_num_seqs": [1024, 2048], "output_tokens": [128, 256]}
DEFAULT_INPUT_TOKENS = 128


# ── Model selection ───────────────────────────────────────────────────

def kv_bytes_per_token(model_id: str, cache: Path | None = None) -> Optional[int]:
    """KV cache cost of one token, read from the model's own config.

    2 x layers x kv_heads x head_dim x dtype_bytes. This is THE number
    that decides a headline: at large batch the KV traffic dwarfs the
    weight reads, so a model with 16x the KV cost per token cannot be
    rescued by having fewer active parameters.

    None when the model is not staged -- the config has to be on disk
    to be read, and guessing it from parameter count is how a 70B
    dense model gets mistaken for a cheap one.
    """
    from .models import hf_cache_dir
    base = Path(cache) if cache else hf_cache_dir()
    d = base / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return None
    for snap in sorted(d.iterdir()):
        cfg = snap / "config.json"
        if not cfg.is_file():
            continue
        try:
            c = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = c.get("text_config") or c
        layers = text.get("num_hidden_layers")
        heads = text.get("num_attention_heads")
        kv_heads = text.get("num_key_value_heads") or heads
        hidden = text.get("hidden_size")
        head_dim = text.get("head_dim") or (
            hidden // heads if hidden and heads else None)
        if not (layers and kv_heads and head_dim):
            continue
        # One byte per element: every model we rank here runs an fp8 or
        # nvfp4 KV cache. Ranking is by RATIO, so a constant factor
        # cannot change the order.
        return int(2 * layers * kv_heads * head_dim)
    return None


@dataclass
class Candidate:
    """One model, scored for its roofline potential."""
    id: str
    quant: str = ""
    params_b: Optional[float] = None
    active_b: Optional[float] = None
    moe: bool = False
    size_gb: Optional[float] = None
    kv_bytes: Optional[int] = None
    cached: bool = False
    fits: bool = True
    score: float = 0.0
    why: str = ""


def score_models(catalog: list[dict], *, vram_per_gpu_gb: float | None,
                 cache: Path | None = None) -> list[Candidate]:
    """Rank catalog models by how fast they could plausibly go here.

    At saturation the box is reading, per generated token, the KV of
    every live sequence plus a share of the weights. The KV term wins
    by an order of magnitude at the batch sizes a headline runs at, so
    that is the primary key; weight footprint breaks ties because a
    smaller model leaves more VRAM for the cache that is the actual
    constraint.

    This is a PREDICTION and is labelled as one. It orders the search;
    it does not decide the answer.
    """
    out: list[Candidate] = []
    for e in catalog:
        mid = str(e.get("id") or "")
        if not mid:
            continue
        from .models import model_status
        st = model_status(mid, cache) if cache else model_status(mid)
        size = e.get("approx_size_gb")
        need = e.get("min_vram_gb") or size
        fits = True
        if vram_per_gpu_gb and need:
            # Whole-box: eight cards. A model needing more than one
            # card can still run at tp>1, so this only excludes what
            # will not fit the box at all.
            fits = float(need) <= float(vram_per_gpu_gb) * 8
        c = Candidate(
            id=mid, quant=str(e.get("quant") or ""),
            params_b=e.get("params_b"), moe=bool(e.get("moe")),
            size_gb=size, cached=bool(st.get("cached")),
            kv_bytes=kv_bytes_per_token(mid, cache), fits=fits,
        )
        bits = []
        # Primary: KV bytes per token, when we can read it.
        if c.kv_bytes:
            c.score = 1_000_000.0 / c.kv_bytes
            bits.append(f"{c.kv_bytes // 1024} KiB of KV per token")
        elif c.params_b:
            # Unstaged: fall back to parameter count, and say so. A
            # dense model's KV scales with its layer count, which
            # tracks size loosely enough to order a shortlist.
            c.score = 200.0 / max(1.0, float(c.params_b)) ** 0.5
            bits.append("KV cost estimated from parameter count "
                        "(weights not staged yet)")
        if c.moe:
            c.score *= 1.35
            bits.append("MoE — fewer active parameters per token")
        if c.quant in ("nvfp4", "fp4", "mxfp4"):
            c.score *= 1.25
            bits.append(f"{c.quant} weights")
        elif c.quant == "fp8":
            c.score *= 1.1
            bits.append("fp8 weights")
        if size and vram_per_gpu_gb and float(size) <= float(vram_per_gpu_gb):
            c.score *= 1.15
            bits.append("fits one GPU — no tensor-parallel all-reduce")
        if not c.fits:
            c.score = 0.0
            bits = ["does not fit this box"]
        c.why = "; ".join(bits)
        out.append(c)
    return sorted(out, key=lambda x: -x.score)


def pick_models(catalog: list[dict], *, vram_per_gpu_gb: float | None,
                limit: int = 3, cached_only: bool = False,
                cache: Path | None = None) -> list[Candidate]:
    ranked = [c for c in score_models(
        catalog, vram_per_gpu_gb=vram_per_gpu_gb, cache=cache) if c.fits]
    if cached_only:
        ranked = [c for c in ranked if c.cached]
    return ranked[:max(1, limit)]


# ── Plan ──────────────────────────────────────────────────────────────

def cells(models: list[str], engines: list[str], shapes: dict) -> list[dict]:
    """Every (model, engine, shape) the run will measure, model-major.

    Model-major because switching model costs a full weight load while
    switching engine does not, and because a partial run then holds a
    COMPLETE answer for the models it reached rather than a fragment
    of each.
    """
    mns = sorted(shapes.get("max_num_seqs") or DEFAULT_SHAPES["max_num_seqs"])
    outs = sorted(shapes.get("output_tokens") or DEFAULT_SHAPES["output_tokens"])
    return [{"model": m, "engine": e, "max_num_seqs": n, "output_tokens": o}
            for m in models for e in engines for n in mns for o in outs]


def estimate_minutes(n_cells: int, *, launch_min: float = 6.0,
                     rungs: int = 4, rung_min: float = 1.5,
                     final_sweep_min: float = 18.0,
                     n_models: int = 1) -> int:
    """Wall clock, shown before anything is committed. Every cell pays
    an engine launch; each model additionally pays one confirmation
    sweep of its winner."""
    return int(n_cells * (launch_min + rungs * rung_min)
               + n_models * final_sweep_min)


def cell_key(c: dict) -> str:
    return (f"{c['model']}|{c['engine']}|{c['max_num_seqs']}"
            f"|{c['output_tokens']}")


# ── State (the only thing that matters across a disconnect) ───────────

@dataclass
class State:
    status: str = "planning"
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    input_tokens: int = DEFAULT_INPUT_TOKENS
    plan: dict = field(default_factory=dict)
    estimate_min: int = 0
    models: list = field(default_factory=list)
    results: list = field(default_factory=list)      # completed cells
    current: dict | None = None
    error: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = "roofline"
        d["done"] = self.status in ("finished", "failed", "stopped")
        d["summary"] = summarize(self.results)
        return d


def summarize(results: list[dict]) -> dict:
    """The matrix, plus the cell that won it.

    Deliberately reports best-per-model and best-per-engine alongside
    the overall winner: a single number tells you what to publish,
    while the matrix tells you what to do with the next model you try.
    """
    usable = [r for r in results
              if r.get("out_tok_s") and r.get("steady_state", True)]
    by_model: dict[str, dict] = {}
    by_engine: dict[str, dict] = {}
    for r in sorted(usable, key=lambda x: -(x["out_tok_s"] or 0)):
        by_model.setdefault(r["model"], r)
        by_engine.setdefault(r["engine"], r)
    best = max(usable, key=lambda r: r["out_tok_s"], default=None)
    return {
        "best": best,
        "best_per_model": by_model,
        "best_per_engine": by_engine,
        "measured": len(usable),
        "attempted": len(results),
        "failed": [r for r in results if r.get("error")],
    }


def load_state(path: Path) -> Optional[State]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    known = {f for f in State.__dataclass_fields__}
    return State(**{k: v for k, v in d.items() if k in known})


def save_state(path: Path, st: State) -> None:
    """Write atomically. A half-written state read by a reconnecting
    browser is worse than a slightly stale one."""
    st.updated_at = time.time()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st.to_dict(), indent=2))
    tmp.replace(p)


# ── Staging ───────────────────────────────────────────────────────────

async def ensure_staged(models: list[str], st: State, state_path: Path,
                        *, log_dir: Path) -> list[str]:
    """Download anything missing, and report progress while it happens.

    Weights are tens of gigabytes. Doing this inside the run rather
    than demanding it beforehand is the difference between an autopilot
    and a checklist -- but a model that will not download must not take
    the whole run down with it, so failures drop the model and carry
    on with the rest.
    """
    import asyncio
    import subprocess

    from .models import download_command, model_status

    ready: list[str] = []
    for mid in models:
        entry = next((m for m in st.models if m["id"] == mid), None)
        if model_status(mid).get("cached"):
            if entry:
                entry["status"] = "cached"
            ready.append(mid)
            save_state(state_path, st)
            continue
        if entry:
            entry["status"] = "downloading"
        st.note = f"staging weights for {mid}"
        save_state(state_path, st)
        log.info("roofline: downloading %s", mid)
        try:
            argv, env = download_command(mid)
            import os
            log_dir.mkdir(parents=True, exist_ok=True)
            lf = open(log_dir / f"download_{mid.replace('/', '--')}.log", "w")
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=lf, stderr=subprocess.STDOUT,
                env={**os.environ, **env})
            rc = await proc.wait()
        except Exception as e:  # noqa: BLE001
            rc, e_msg = 1, str(e)
            log.warning("roofline: download of %s failed: %s", mid, e)
        else:
            e_msg = f"downloader exited {rc}"
        if rc == 0 and model_status(mid).get("cached"):
            if entry:
                entry["status"] = "cached"
            ready.append(mid)
        else:
            if entry:
                entry["status"] = "unavailable"
                entry["error"] = e_msg
            log.warning("roofline: skipping %s — %s", mid, e_msg)
        save_state(state_path, st)
    return ready


# ── The run ───────────────────────────────────────────────────────────

async def run_roofline(
    *,
    models: list[str],
    engines: list[str],
    shapes: dict | None = None,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    build_config,
    runs_base: Path,
    state_path: Path | None = None,
    resume: bool = True,
    confirm_winners: bool = True,
) -> Path:
    """Stage, search the product, confirm each model's winner, report.

    ``build_config`` is injected exactly as the joint search does it:
    it takes engine overrides and returns a config path, so this module
    never learns how configs are generated.
    """
    from .headline_shapes import apply_shape_to_generation
    from .persona_loader import USER_CATALOG_DIR

    shapes = shapes or dict(DEFAULT_SHAPES)
    path = Path(state_path or (Path(runs_base) / STATE_NAME))

    st = load_state(path) if resume else None
    done: dict[str, dict] = {}
    if st and st.plan.get("cells"):
        done = {cell_key(r): r for r in st.results if not r.get("error")}
        log.info("roofline: resuming with %d cells already measured", len(done))
    else:
        st = State()

    plan_cells = cells(models, engines, shapes)
    st.status = "staging"
    st.input_tokens = input_tokens
    st.plan = {"models": models, "engines": engines, "shapes": shapes,
               "cells": plan_cells}
    st.estimate_min = estimate_minutes(len(plan_cells) - len(done),
                                       n_models=len(models))
    if not st.models:
        st.models = [{"id": m, "status": "pending"} for m in models]
    st.error = None
    save_state(path, st)
    BUS.publish("run", {"event": "started", "mode": "roofline",
                        "models": models, "engines": engines,
                        "cells": len(plan_cells)})

    staged = await ensure_staged(models, st, path,
                                 log_dir=Path(runs_base) / "roofline")
    if not staged:
        st.status = "failed"
        st.error = "no model could be staged"
        save_state(path, st)
        return path

    from .config import load_config
    from .headline_sweep import run_headline_sweep
    from .personas import cohort_from_persona

    st.status = "searching"
    st.note = ""
    save_state(path, st)

    for cell in plan_cells:
        if cell["model"] not in staged:
            continue
        key = cell_key(cell)
        if key in done:
            continue
        st.current = dict(cell)
        save_state(path, st)
        log.info("roofline: %s / %s mns=%d out=%d", cell["model"],
                 cell["engine"], cell["max_num_seqs"], cell["output_tokens"])
        row = dict(cell)
        try:
            apply_shape_to_generation(USER_CATALOG_DIR, input_tokens,
                                      cell["output_tokens"])
            cfg_path = build_config({
                "model_id": cell["model"], "engine": cell["engine"],
                "max_num_seqs": cell["max_num_seqs"],
            })
            sub = load_config(cfg_path)
            sub.output.db_directory = str(runs_base)
            # Coarse ladder while RANKING: the peak sits at the top
            # of the curve, so the low rungs cost minutes and teach
            # nothing. The winner earns the full ladder below.
            from .headline_optimize import SEARCH_LADDER
            summary = await run_headline_sweep(
                sub, cohort_from_persona("headline_generation"),
                new_run=True, ladder_override=SEARCH_LADDER)
            row.update(_peak_of(summary))
        except Exception as e:  # noqa: BLE001
            # One dead cell must not end a six-hour run.
            row["error"] = f"{type(e).__name__}: {e}"
            log.warning("roofline cell failed: %s", e)
        st.results.append(row)
        st.current = None
        save_state(path, st)

    if confirm_winners:
        st.status = "confirming"
        save_state(path, st)
        for mid, best in (summarize(st.results).get("best_per_model")
                          or {}).items():
            st.current = {**best, "phase": "confirming"}
            save_state(path, st)
            try:
                apply_shape_to_generation(USER_CATALOG_DIR, input_tokens,
                                          best["output_tokens"])
                cfg_path = build_config({
                    "model_id": mid, "engine": best["engine"],
                    "max_num_seqs": best["max_num_seqs"],
                })
                sub = load_config(cfg_path)
                sub.output.db_directory = str(runs_base)
                final = await run_headline_sweep(
                    sub, cohort_from_persona("headline_generation"),
                    new_run=True)
                row = {**best, "confirmed": True, **_peak_of(final)}
                st.results.append(row)
            except Exception as e:  # noqa: BLE001
                log.error("roofline confirmation for %s failed: %s", mid, e)
            st.current = None
            save_state(path, st)

    st.status = "finished"
    st.note = ("Search rungs rank candidates; the confirmed rows are the "
               "ones to publish.")
    save_state(path, st)
    BUS.publish("run", {"event": "finished", "mode": "roofline",
                        "final_status": "ok"})
    log.info("roofline done: %s", summarize(st.results).get("best"))
    return path


def _peak_of(summary_path) -> dict:
    """The peak a sweep recorded, flattened into a result row."""
    try:
        doc = json.loads(Path(summary_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"unreadable sweep summary: {e}"}
    pk = doc.get("peak") or {}
    if not pk:
        return {"error": doc.get("stop_reason") or "sweep produced no peak"}
    return {
        "out_tok_s": pk.get("out_tok_s"),
        "total_tok_s": pk.get("total_tok_s"),
        "in_flight": pk.get("in_flight"),
        "concurrency": pk.get("concurrency"),
        "queue_depth": pk.get("queue_depth"),
        "ttft_p95_ms": pk.get("ttft_p95_ms"),
        "tpot_p95_ms": pk.get("tpot_p95_ms"),
        "kv_cache_pct": pk.get("kv_cache_pct"),
        "gpu_power_w": pk.get("gpu_power_w"),
        "steady_state": pk.get("steady_state", True),
        "kv_cache_tokens": doc.get("kv_cache_tokens"),
        "run_dir": str(Path(summary_path).parent),
        "tokens_per_watt": (
            round(pk["out_tok_s"] / pk["gpu_power_w"], 2)
            if pk.get("out_tok_s") and pk.get("gpu_power_w") else None),
    }
