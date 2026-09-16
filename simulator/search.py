"""Guided launch-shape search — coarse-to-fine over a parameter space.

The hand-curated optimizer registry answers "which of these seven
shapes wins?". This module answers the broader question: over models ×
precision × TP × DP × placement × batch shape, where is peak
performance? The strategy is deliberately explainable rather than a
black-box optimizer:

    1. **Coverage stage** — a greedy-coverage sample of the valid
       space (every dimension value represented at least once where
       feasible), sized ``initial_samples``.
    2. **Rank** — every candidate runs the standard optimizer cells;
       an SLA-aware objective reduces the cells to one score.
    3. **Refine** — take the ``top_k`` leaders, propose neighbors that
       change exactly ONE dimension by one step (ordinal dims step to
       adjacent values; categorical dims swap), dedupe against
       everything already measured, evaluate.
    4. Repeat 3 until the total ``budget`` is spent, ``max_iterations``
       refinements have run, or the best score improved less than
       ``min_improvement`` in an iteration.

Everything is deterministic under ``seed`` and serializable: the
driver (scripts/engine_optimizer.py ``--search``) persists a
SearchState after every evaluation, and ``next_batch`` re-emits any
unevaluated candidates of the current iteration first, so a crash or
stop resumes mid-iteration.

Spaces are YAML (``config/search/*.yaml``) — see ``load_space`` for
the schema. Precision is expressed as a *model variant* (a different
HF artifact plus optional engine args): FP8 is not a flag, it's a
different set of weights.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Dimensions the candidate→engine-config builder understands, in
# canonical order. ``ordinal`` dims refine by stepping to adjacent
# values in the listed order; ``categorical`` dims refine by swapping.
KNOWN_DIMENSIONS: dict[str, str] = {
    "model_variant": "categorical",
    "tp": "ordinal",
    "dp": "ordinal",
    "gpu_memory_utilization": "ordinal",
    "max_num_seqs": "ordinal",
    "max_num_batched_tokens": "ordinal",
    "placement": "categorical",
    # FP8 KV cache halves KV memory/bandwidth (Blackwell-friendly).
    "kv_cache_dtype": "categorical",         # auto | fp8 | nvfp4
    # Expert parallelism for MoE models — only meaningful when the
    # variant is MoE and tp>1 (normalize() forces "off" otherwise, so
    # infeasible combinations dedupe instead of wasting evaluations).
    "expert_parallel": "categorical",         # off | on
}

# Sentinel meaning "don't pass the flag; let the engine pick".
DEFAULT = "default"


class SearchSpaceError(ValueError):
    """A search-space YAML that doesn't parse or validate."""


@dataclass
class Objective:
    kind: str = "sla_throughput"        # sla_throughput | throughput | latency
    ttft_p95_cap_ms: float = 10_000.0
    tpot_p95_cap_ms: float = 100.0
    # Multiplier applied to a cell's contribution when it blows a cap
    # (sla_throughput only). 0 would erase the signal entirely; 0.25
    # keeps a gradient so "fast but slightly over" still beats "slow".
    penalty: float = 0.25


@dataclass
class Measurement:
    """How each candidate is measured: one workload shape climbed
    over a concurrency ladder with SLA early-exit.

    Fixed-concurrency cells rank configs at an arbitrary point on
    their throughput curves — dp2 and tp2 can tie at concurrency 32
    (neither saturated) while dp2 wins 2× at 256. The ladder finds
    each config's own SLA-capacity and scores THAT, which is what the
    capacity benchmark ultimately cares about. Climbing stops at the
    first rung that blows the objective's p95 caps — higher rungs are
    strictly worse for latency, so they can't produce a better
    SLA-passing point."""
    input_tokens: int = 512
    output_tokens: int = 256
    ladder: list[int] = field(default_factory=lambda: [8, 32, 128, 512])


@dataclass
class SearchParams:
    seed: int = 42
    initial_samples: int = 12
    top_k: int = 3
    neighbors_per_iteration: int = 8
    max_iterations: int = 4
    min_improvement: float = 0.03
    budget: int = 40


@dataclass
class SearchSpace:
    name: str
    engine: str
    device_groups: list[list[int]]
    model_variants: dict[str, dict]     # name -> {model, served_name,
                                        #   extra_args?, min_vram_gb?, moe?}
    dimensions: dict[str, list]         # dim -> ordered values
    objective: Objective
    search: SearchParams
    source: str = ""                    # file path, informational
    # Per-GPU VRAM (GB). When set, a candidate is invalid unless
    # tp × vram covers the variant's min_vram_gb — so a 235B model
    # never wastes an evaluation trying to load at tp=1.
    vram_per_gpu_gb: Optional[float] = None
    measurement: Measurement = field(default_factory=Measurement)
    # Engine container image (vllm_cuda). None -> the driver's default
    # GPU image. Part of the fingerprint: a different image is a
    # different engine.
    gpu_image: Optional[str] = None

    @property
    def total_devices(self) -> int:
        return sum(len(g) for g in self.device_groups)

    def space_hash(self) -> str:
        """Stable fingerprint of everything that affects candidate
        identity — used to refuse resuming a state file produced by a
        different space."""
        doc = {
            "engine": self.engine,
            "device_groups": self.device_groups,
            "model_variants": self.model_variants,
            "dimensions": self.dimensions,
            "vram_per_gpu_gb": self.vram_per_gpu_gb,
            # Objective + measurement change what scores MEAN, so a
            # resumed state with different ones would silently mix
            # incomparable numbers — hash them like candidate identity.
            "objective": dataclasses.asdict(self.objective),
            "measurement": dataclasses.asdict(self.measurement),
            "gpu_image": self.gpu_image,
        }
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True).encode()
        ).hexdigest()[:16]


def load_space(path: str | Path) -> SearchSpace:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise SearchSpaceError(f"{path}: {e}") from e
    if not isinstance(raw, dict):
        raise SearchSpaceError(f"{path}: top level must be a mapping")

    engine = raw.get("engine", "vllm_cuda")
    if engine != "vllm_cuda":
        raise SearchSpaceError(
            f"{path}: engine '{engine}' not supported by the search "
            f"driver yet — vllm_cuda only for now"
        )

    groups = raw.get("device_groups")
    if not (isinstance(groups, list) and groups
            and all(isinstance(g, list) and g for g in groups)):
        raise SearchSpaceError(
            f"{path}: device_groups must be a non-empty list of "
            f"non-empty GPU-id lists (one list per PCIe/NUMA domain)"
        )

    variants = dict(raw.get("model_variants") or {})
    for vname, v in variants.items():
        if not isinstance(v, dict) or "model" not in v:
            raise SearchSpaceError(
                f"{path}: model_variant '{vname}' needs at least a "
                f"'model' (HF id or /models path)"
            )
    # Families pulled from the model catalog expand into variants named
    # ``{family}-{quant}``. NOTE: the expansion is part of space_hash,
    # so adding a model to a listed family deliberately invalidates a
    # half-finished search state (done:space_changed) — the space is
    # genuinely different.
    families = raw.get("catalog_families")
    if families:
        if not (isinstance(families, list) and
                all(isinstance(f, str) for f in families)):
            raise SearchSpaceError(
                f"{path}: catalog_families must be a list of family names"
            )
        from .model_catalog import CatalogError, catalog_families
        try:
            by_family = catalog_families()
        except CatalogError as e:
            raise SearchSpaceError(f"{path}: model catalog broken: {e}") from e
        unknown_fams = [f for f in families if f not in by_family]
        if unknown_fams:
            raise SearchSpaceError(
                f"{path}: catalog_families {unknown_fams} not in the model "
                f"catalog — known: {sorted(by_family)}"
            )
        for fam in families:
            for entry in by_family[fam]:
                vname = f"{fam}-{entry['quant']}"
                variants.setdefault(vname, {
                    "model": entry["id"],
                    "served_name": vname,
                    "extra_args": list(entry.get("engine_args") or []),
                    "min_vram_gb": entry.get("min_vram_gb"),
                    "moe": bool(entry.get("moe")),
                })
    if not variants:
        raise SearchSpaceError(
            f"{path}: model_variants map or catalog_families required"
        )

    dims_raw = raw.get("dimensions")
    if not isinstance(dims_raw, dict) or not dims_raw:
        raise SearchSpaceError(f"{path}: dimensions map required")
    unknown = set(dims_raw) - set(KNOWN_DIMENSIONS)
    if unknown:
        raise SearchSpaceError(
            f"{path}: unknown dimensions {sorted(unknown)} — known: "
            f"{sorted(KNOWN_DIMENSIONS)}"
        )
    dims: dict[str, list] = {}
    for name in KNOWN_DIMENSIONS:           # canonical order
        if name not in dims_raw:
            continue
        vals = dims_raw[name]
        if not isinstance(vals, list) or not vals:
            raise SearchSpaceError(f"{path}: dimension {name} must be a non-empty list")
        if len(set(map(str, vals))) != len(vals):
            raise SearchSpaceError(f"{path}: dimension {name} has duplicate values")
        dims[name] = vals
    if "model_variant" in dims:
        missing = set(dims["model_variant"]) - set(variants)
        if missing:
            raise SearchSpaceError(
                f"{path}: model_variant values {sorted(missing)} not in model_variants"
            )
    else:
        dims = {"model_variant": list(variants), **dims}

    objective = Objective(**(raw.get("objective") or {}))
    if objective.kind not in ("sla_throughput", "throughput", "latency"):
        raise SearchSpaceError(f"{path}: objective.kind '{objective.kind}' unknown")
    search = SearchParams(**(raw.get("search") or {}))
    measurement = Measurement(**(raw.get("measurement") or {}))
    if not measurement.ladder or \
            sorted(measurement.ladder) != list(measurement.ladder):
        raise SearchSpaceError(
            f"{path}: measurement.ladder must be an ascending "
            f"concurrency list"
        )

    vram = raw.get("vram_per_gpu_gb")
    return SearchSpace(
        name=str(raw.get("name", path.stem)),
        engine=engine,
        device_groups=[[int(d) for d in g] for g in groups],
        model_variants=variants,
        dimensions=dims,
        objective=objective,
        search=search,
        source=str(path),
        vram_per_gpu_gb=float(vram) if vram is not None else None,
        measurement=measurement,
        gpu_image=raw.get("gpu_image") or None,
    )


# ── Candidates ───────────────────────────────────────────────────────


def _dim_value(params: dict, dim: str, space: SearchSpace):
    """Value of ``dim`` for a candidate, defaulting to the dimension's
    first listed value (or a sensible constant when the dimension was
    omitted from the space)."""
    if dim in params:
        return params[dim]
    if dim in space.dimensions:
        return space.dimensions[dim][0]
    return {"tp": 1, "dp": 1, "gpu_memory_utilization": 0.90,
            "max_num_seqs": DEFAULT, "max_num_batched_tokens": DEFAULT,
            "placement": "pack", "kv_cache_dtype": "auto",
            "expert_parallel": "off"}.get(dim)


def normalize(params: dict, space: SearchSpace) -> dict:
    """Canonical form so equivalent candidates dedupe: placement is
    meaningless (forced to the first placement value) when the
    candidate uses a single device, and expert_parallel is meaningless
    (forced "off") unless the variant is MoE with tp>1 — vLLM's EP
    splits experts across the TP group."""
    out = {d: _dim_value(params, d, space) for d in KNOWN_DIMENSIONS
           if d in space.dimensions or d in params}
    tp, dp = int(_dim_value(out, "tp", space)), int(_dim_value(out, "dp", space))
    if tp * dp <= 1 and "placement" in out:
        out["placement"] = space.dimensions.get("placement", ["pack"])[0]
    if "expert_parallel" in out:
        variant = space.model_variants.get(
            str(_dim_value(out, "model_variant", space))) or {}
        if not variant.get("moe") or tp <= 1:
            out["expert_parallel"] = "off"
    return out


def canonical_key(params: dict, space: SearchSpace) -> str:
    n = normalize(params, space)
    return "|".join(f"{d}={n[d]}" for d in KNOWN_DIMENSIONS if d in n)


def assign_devices(
    tp: int, dp: int, placement: str, device_groups: list[list[int]],
) -> Optional[list[list[int]]]:
    """Device ids for each of the ``dp`` replicas (each ``tp`` wide),
    or None when the shape doesn't fit.

    pack   — fill one group before starting the next: TP peers stay on
             the same PCIe/NUMA domain (fast P2P), replicas cluster.
    spread — deal replicas round-robin across groups: replicas land on
             different domains (balanced host bandwidth). A TP replica
             never spans groups in either mode — cross-domain
             all-reduce is the one shape that's never the answer.
    """
    total = sum(len(g) for g in device_groups)
    if tp * dp > total or tp > max(len(g) for g in device_groups):
        return None
    pools = [list(g) for g in device_groups]
    replicas: list[list[int]] = []
    gi = 0
    for _ in range(dp):
        if placement == "spread":
            # Next group (round-robin) with room for a full TP set.
            for off in range(len(pools)):
                cand = (gi + off) % len(pools)
                if len(pools[cand]) >= tp:
                    gi = cand
                    break
            else:
                return None
            replicas.append([pools[gi].pop(0) for _ in range(tp)])
            gi = (gi + 1) % len(pools)
        else:  # pack
            placed = False
            for pool in pools:
                if len(pool) >= tp:
                    replicas.append([pool.pop(0) for _ in range(tp)])
                    placed = True
                    break
            if not placed:
                return None
    return replicas


def validate_candidate(params: dict, space: SearchSpace) -> tuple[bool, str]:
    n = normalize(params, space)
    for dim, val in n.items():
        if dim in space.dimensions and val not in space.dimensions[dim]:
            return False, f"{dim}={val} not in space"
    tp = int(_dim_value(n, "tp", space))
    dp = int(_dim_value(n, "dp", space))
    placement = str(_dim_value(n, "placement", space))
    if assign_devices(tp, dp, placement, space.device_groups) is None:
        return False, f"tp={tp} dp={dp} placement={placement} does not fit devices"
    # VRAM fit: don't burn an evaluation on a shape that can't load
    # the weights (a 235B bf16 model at tp=1 fails after minutes of
    # downloading/loading — prune it here instead).
    if space.vram_per_gpu_gb:
        variant = space.model_variants.get(
            str(_dim_value(n, "model_variant", space))) or {}
        need = variant.get("min_vram_gb")
        if need and tp * space.vram_per_gpu_gb < float(need):
            return False, (
                f"{_dim_value(n, 'model_variant', space)} needs "
                f"{need} GB, tp={tp} provides "
                f"{tp * space.vram_per_gpu_gb:.0f} GB"
            )
    return True, ""


def _random_candidate(space: SearchSpace, rng: random.Random) -> dict:
    return normalize(
        {d: rng.choice(vals) for d, vals in space.dimensions.items()}, space,
    )


def propose_initial(
    space: SearchSpace, rng: random.Random,
    evaluated_keys: frozenset = frozenset(),
) -> list[dict]:
    """Greedy-coverage sample: draw a pool of valid candidates, then
    repeatedly pick the one covering the most not-yet-covered
    (dimension, value) pairs — every value of every dimension shows up
    in stage 0 when the budget allows, which is what makes the first
    ranking a *representative* picture rather than a lucky corner.

    ``evaluated_keys`` (seeded results from a prior run of the same
    investigation) are never re-proposed — the budget spends on NEW
    territory — but they still count as covering their (dim, value)
    pairs, so coverage tops up around them instead of re-measuring."""
    want = space.search.initial_samples
    pool: list[dict] = []
    seen: set[str] = set(evaluated_keys)
    for _ in range(max(400, want * 40)):
        cand = _random_candidate(space, rng)
        ok, _reason = validate_candidate(cand, space)
        key = canonical_key(cand, space)
        if ok and key not in seen:
            seen.add(key)
            pool.append(cand)
    uncovered = {(d, str(v)) for d, vals in space.dimensions.items() for v in vals}
    picked: list[dict] = []
    for key in evaluated_keys:
        # "dim=value|dim=value" canonical keys → mark covered.
        for part in key.split("|"):
            d, _, v = part.partition("=")
            uncovered.discard((d, v))
    while pool and len(picked) < want:
        snapshot = frozenset(uncovered)
        pool.sort(
            key=lambda c, _u=snapshot: sum((d, str(v)) in _u for d, v in c.items()),
            reverse=True,
        )
        best = pool.pop(0)
        picked.append(best)
        uncovered -= {(d, str(v)) for d, v in best.items()}
    return picked


def propose_neighbors(
    space: SearchSpace,
    tops: list[dict],
    evaluated_keys: set[str],
    limit: int,
) -> list[dict]:
    """One-dimension-changed neighbors of the leaders, deduped against
    everything already measured. Ordinal dims step ±1 in listed order;
    categorical dims swap to each alternative. Interleaved across
    leaders so the batch isn't all mutations of one parent."""
    per_parent: list[list[dict]] = []
    for parent in tops:
        muts: list[dict] = []
        for dim, vals in space.dimensions.items():
            cur = _dim_value(parent, dim, space)
            kind = KNOWN_DIMENSIONS[dim]
            if kind == "ordinal":
                idx = vals.index(cur) if cur in vals else 0
                steps = [i for i in (idx - 1, idx + 1) if 0 <= i < len(vals)]
                alts = [vals[i] for i in steps]
            else:
                alts = [v for v in vals if v != cur]
            for alt in alts:
                muts.append(normalize({**parent, dim: alt}, space))
        per_parent.append(muts)
    out: list[dict] = []
    seen = set(evaluated_keys)
    i = 0
    while len(out) < limit and any(per_parent):
        lane = per_parent[i % len(per_parent)]
        i += 1
        while lane:
            cand = lane.pop(0)
            key = canonical_key(cand, space)
            ok, _ = validate_candidate(cand, space)
            if ok and key not in seen:
                seen.add(key)
                out.append(cand)
                break
    return out


# ── Scoring ──────────────────────────────────────────────────────────


def score_cells(cells: list[dict], objective: Objective) -> Optional[float]:
    """Reduce a config's CellResults to one scalar (higher = better).

    sla_throughput — summed output tok/s across cells, with a cell's
        contribution multiplied by ``penalty`` when its TTFT/TPOT p95
        blows the caps, and by the fraction of requests that actually
        completed (errors/timeouts bleed score naturally).
    throughput     — the same sum, no SLA caps.
    latency        — negative mean TTFT p95 (for pure-latency tuning).
    """
    if not cells:
        return None
    if objective.kind == "latency":
        vals = [c["ttft_p95_ms"] for c in cells if c.get("ttft_p95_ms") is not None]
        return -sum(vals) / len(vals) if vals else None
    total = 0.0
    any_signal = False
    for c in cells:
        tps = c.get("throughput_out_tok_s")
        if tps is None:
            continue
        any_signal = True
        contribution = float(tps)
        attempts = (c.get("samples") or 0) + (c.get("errors") or 0) + (c.get("timeouts") or 0)
        if attempts:
            contribution *= (c.get("samples") or 0) / attempts
        if objective.kind == "sla_throughput":
            ttft = c.get("ttft_p95_ms")
            tpot = c.get("tpot_p95_ms")
            if (ttft is not None and ttft > objective.ttft_p95_cap_ms) or \
               (tpot is not None and tpot > objective.tpot_p95_cap_ms):
                contribution *= objective.penalty
        total += contribution
    return total if any_signal else None


def _restart_order(batch: list[dict], space: SearchSpace) -> list[dict]:
    """Order a batch to minimize expensive engine transitions: every
    evaluation restarts the engine, but consecutive candidates on the
    SAME model reuse hot weights (page cache + HF cache) — a ~60 GB
    reload versus seconds. Group by variant, then by tp/dp so shape
    changes cluster too. Stable within groups, so proposal order (and
    determinism under seed) is preserved."""
    return sorted(batch, key=lambda p: (
        str(_dim_value(p, "model_variant", space)),
        int(_dim_value(p, "tp", space)),
        int(_dim_value(p, "dp", space)),
    ))


def _rung_contribution(cell: dict, objective: Objective) -> Optional[float]:
    """One ladder rung's scored throughput: output tok/s × completion
    fraction, × penalty when its p95s blow the caps."""
    tps = cell.get("throughput_out_tok_s")
    if tps is None:
        return None
    contribution = float(tps)
    attempts = (cell.get("samples") or 0) + (cell.get("errors") or 0) \
        + (cell.get("timeouts") or 0)
    if attempts:
        contribution *= (cell.get("samples") or 0) / attempts
    if objective.kind == "sla_throughput" and not rung_sla_ok(cell, objective):
        contribution *= objective.penalty
    return contribution


def rung_sla_ok(cell: dict, objective: Objective) -> bool:
    ttft = cell.get("ttft_p95_ms")
    tpot = cell.get("tpot_p95_ms")
    return not ((ttft is not None and ttft > objective.ttft_p95_cap_ms) or
                (tpot is not None and tpot > objective.tpot_p95_cap_ms))


def score_ladder(cells: list[dict], objective: Objective) -> Optional[float]:
    """Reduce a candidate's ladder rungs to one scalar: the BEST rung,
    not the sum — every rung is the same workload at a different
    concurrency, so summing would reward ladder length, while the max
    is the config's demonstrated (SLA-penalized) capacity. ``latency``
    objectives score the gentlest rung's TTFT instead."""
    if not cells:
        return None
    if objective.kind == "latency":
        vals = [c["ttft_p95_ms"] for c in cells
                if c.get("ttft_p95_ms") is not None]
        return -min(vals) if vals else None
    scores = [s for c in cells
              if (s := _rung_contribution(c, objective)) is not None]
    return max(scores) if scores else None


def best_rung(cells: list[dict], objective: Objective) -> Optional[dict]:
    """The rung the score came from — the operator-facing 'this shape
    delivers N tok/s at concurrency C inside SLA' statement."""
    best, best_score = None, None
    for c in cells:
        s = _rung_contribution(c, objective)
        if s is not None and (best_score is None or s > best_score):
            best, best_score = c, s
    if best is None:
        return None
    return {
        "cell_name": best.get("cell_name"),
        "throughput_out_tok_s": best.get("throughput_out_tok_s"),
        "ttft_p95_ms": best.get("ttft_p95_ms"),
        "tpot_p95_ms": best.get("tpot_p95_ms"),
        "sla_ok": rung_sla_ok(best, objective),
    }


# ── Search state machine ─────────────────────────────────────────────


@dataclass
class Evaluation:
    params: dict
    iteration: int
    status: str                       # ok | launch_failed | no_samples
    score: Optional[float] = None
    config_name: str = ""
    cells: list = field(default_factory=list)


@dataclass
class SearchState:
    space_hash: str
    iterations: list[dict] = field(default_factory=list)   # {index, kind, keys, params}
    evaluated: dict[str, Evaluation] = field(default_factory=dict)   # key -> Evaluation
    done_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "space_hash": self.space_hash,
            "iterations": self.iterations,
            "evaluated": {
                k: {"params": e.params, "iteration": e.iteration,
                    "status": e.status, "score": e.score,
                    "config_name": e.config_name, "cells": e.cells}
                for k, e in self.evaluated.items()
            },
            "done_reason": self.done_reason,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "SearchState":
        st = cls(space_hash=doc["space_hash"],
                 iterations=list(doc.get("iterations", [])),
                 done_reason=doc.get("done_reason", ""))
        for k, e in (doc.get("evaluated") or {}).items():
            st.evaluated[k] = Evaluation(
                params=e["params"], iteration=e["iteration"],
                status=e["status"], score=e.get("score"),
                config_name=e.get("config_name", ""),
                cells=e.get("cells", []),
            )
        return st

    # -- queries ------------------------------------------------------

    def ranked(self) -> list[tuple[str, Evaluation]]:
        return sorted(
            ((k, e) for k, e in self.evaluated.items()
             if e.status == "ok" and e.score is not None),
            key=lambda kv: kv[1].score, reverse=True,
        )

    def best(self) -> Optional[tuple[str, Evaluation]]:
        r = self.ranked()
        return r[0] if r else None

    def best_score_through(self, iteration: int) -> Optional[float]:
        scores = [e.score for e in self.evaluated.values()
                  if e.status == "ok" and e.score is not None
                  and e.iteration <= iteration]
        return max(scores) if scores else None


def record_evaluation(
    state: SearchState, space: SearchSpace, params: dict, *,
    status: str, score: Optional[float], config_name: str,
    cells: list, iteration: int,
) -> None:
    state.evaluated[canonical_key(params, space)] = Evaluation(
        params=normalize(params, space), iteration=iteration,
        status=status, score=score, config_name=config_name, cells=cells,
    )


def next_batch(
    state: SearchState, space: SearchSpace, rng: random.Random,
) -> tuple[str, list[dict]]:
    """Advance the state machine. Returns (kind, candidates):

    kind "initial"/"refine" — evaluate these, record each with
    ``record_evaluation``, then call again. A partially evaluated
    iteration (resume after crash/stop) re-emits only its pending
    candidates. kind "done:<reason>" ends the search.
    """
    sp = state.space_hash
    if sp != space.space_hash():
        return ("done:space_changed — start a new run (the state file "
                "was produced by a different space)", [])

    # Resume: pending candidates of the current iteration first.
    if state.iterations:
        last = state.iterations[-1]
        pending = [p for p in last["params"]
                   if canonical_key(p, space) not in state.evaluated]
        if pending:
            return last["kind"], pending

    if state.done_reason:
        return f"done:{state.done_reason}", []

    budget_left = space.search.budget - len(state.evaluated)
    if budget_left <= 0:
        state.done_reason = "budget"
        return "done:budget", []

    if not state.iterations:
        batch = _restart_order(
            propose_initial(space, rng,
                            evaluated_keys=frozenset(state.evaluated),
                            )[:budget_left], space)
        state.iterations.append({
            "index": 0, "kind": "initial",
            "params": batch,
        })
        return "initial", batch

    refinements_run = len(state.iterations) - 1
    if refinements_run >= space.search.max_iterations:
        state.done_reason = "max_iterations"
        return "done:max_iterations", []

    if refinements_run >= 1:
        prev_iter = state.iterations[-2]["index"]
        before = state.best_score_through(prev_iter)
        after = state.best_score_through(state.iterations[-1]["index"])
        if before is not None and after is not None and before > 0:
            if (after - before) / before < space.search.min_improvement:
                state.done_reason = (
                    f"converged (<{space.search.min_improvement:.0%} "
                    f"improvement in last iteration)"
                )
                return f"done:{state.done_reason}", []

    tops = [e.params for _, e in state.ranked()[: space.search.top_k]]
    if not tops:
        state.done_reason = "no_successful_candidates"
        return "done:no_successful_candidates", []
    batch = _restart_order(propose_neighbors(
        space, tops, set(state.evaluated),
        min(space.search.neighbors_per_iteration, budget_left),
    ), space)
    if not batch:
        state.done_reason = "neighborhood_exhausted"
        return "done:neighborhood_exhausted", []
    state.iterations.append({
        "index": state.iterations[-1]["index"] + 1,
        "kind": "refine",
        "params": batch,
    })
    return "refine", batch


def list_spaces(directory: str | Path = Path("config/search")) -> dict[str, str]:
    """Available search-space files: name -> path."""
    d = Path(directory)
    if not d.exists():
        return {}
    return {p.stem: str(p) for p in sorted(d.glob("*.yaml"))}


def summarize(state: SearchState, space: Optional[SearchSpace] = None) -> dict:
    """Compact JSON summary for the UI/CLI: ranking + per-iteration
    best-so-far progression. With ``space``, each ranked entry also
    carries its best ladder rung (where the score was demonstrated)."""
    ranked = state.ranked()

    def _rung(e: Evaluation) -> Optional[dict]:
        return best_rung(e.cells, space.objective) if space else None
    progression = []
    for it in state.iterations:
        b = state.best_score_through(it["index"])
        progression.append({
            "iteration": it["index"], "kind": it["kind"],
            "candidates": len(it["params"]),
            "best_score_so_far": b,
        })
    return {
        "evaluated": len(state.evaluated),
        "ok": sum(1 for e in state.evaluated.values() if e.status == "ok"),
        "failed": sum(1 for e in state.evaluated.values() if e.status != "ok"),
        "iterations": progression,
        "done_reason": state.done_reason,
        "best": (
            {"key": ranked[0][0], "params": ranked[0][1].params,
             "score": ranked[0][1].score,
             "config_name": ranked[0][1].config_name,
             "best_rung": _rung(ranked[0][1])}
            if ranked else None
        ),
        "top": [
            {"key": k, "params": e.params, "score": e.score,
             "iteration": e.iteration, "config_name": e.config_name,
             "best_rung": _rung(e)}
            for k, e in ranked[:10]
        ],
    }


def build_replica_devices(params: dict, space: SearchSpace) -> list[list[int]]:
    """Public helper for the driver: device assignment for a validated
    candidate."""
    n = normalize(params, space)
    out = assign_devices(
        int(_dim_value(n, "tp", space)), int(_dim_value(n, "dp", space)),
        str(_dim_value(n, "placement", space)), space.device_groups,
    )
    if out is None:
        raise SearchSpaceError(f"candidate does not fit devices: {n}")
    return out


def candidate_summary(params: dict, space: SearchSpace) -> dict[str, Any]:
    """Engine-facing view of a candidate (the driver builds the actual
    EngineConfig from this): variant details + flags + devices."""
    n = normalize(params, space)
    variant = space.model_variants[str(_dim_value(n, "model_variant", space))]
    tp = int(_dim_value(n, "tp", space))
    args: list[str] = ["--gpu-memory-utilization",
                       str(_dim_value(n, "gpu_memory_utilization", space))]
    if tp > 1:
        args += ["--tensor-parallel-size", str(tp)]
    mns = _dim_value(n, "max_num_seqs", space)
    if mns not in (None, DEFAULT):
        args += ["--max-num-seqs", str(mns)]
    mbt = _dim_value(n, "max_num_batched_tokens", space)
    if mbt not in (None, DEFAULT):
        args += ["--max-num-batched-tokens", str(mbt)]
    kv = _dim_value(n, "kv_cache_dtype", space)
    if kv not in (None, "auto"):
        args += ["--kv-cache-dtype", str(kv)]
    if _dim_value(n, "expert_parallel", space) == "on":
        args += ["--enable-expert-parallel"]
    args += list(variant.get("extra_args") or [])
    return {
        "params": n,
        "model": variant["model"],
        "served_name": variant.get("served_name") or "search-model",
        "replica_devices": build_replica_devices(n, space),
        "engine_args": args,
        "tp": tp,
    }
