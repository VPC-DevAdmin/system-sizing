"""The test arena — every launch shape this installation can run.

The old model was inverted: hand-picked configs presented as the
choices. The arena starts from the other end — the hardware (detected
GPUs) and the model catalog define the FULL feasible space, the UI
shows every dimension with all of its values in play by default, and
the operator subtracts rather than adds. The guided search then
explores whatever remains.

Feasibility is derived, not declared:
  * TP values are powers of two up to the largest PCIe/NUMA domain.
  * A (model, tp) pair is feasible when tp × per-GPU VRAM covers the
    catalog's min_vram_gb — a 235B bf16 model simply never appears at
    tp<8 on 96 GB cards.
  * DP fills the remaining devices (tp × dp ≤ total GPUs).
  * expert_parallel only pairs with MoE variants at tp>1 (normalize()
    in search.py collapses the rest, so they dedupe, not waste).

Restart cost: every candidate is an engine relaunch, but relaunches on
the SAME model reuse hot weights (~seconds vs ~minutes for 60 GB).
The search orders each batch by model variant (see
search._restart_order); the arena reports estimated relaunch counts so
the operator sees the cost model, not just the combinatorics.

Device groups (PCIe/NUMA domains) can't be reliably auto-detected, so
``config/arena.yaml`` may pin them; otherwise all GPUs form one group.
That file is per-host and gitignored — ``config/arena.example.yaml``
is the template (the XE7740's map). It is read only when present, and
when it claims more GPUs than detection finds the result is flagged
``source: "config (unverified: detected N)"`` rather than trusted.
"""

from __future__ import annotations

import math
import subprocess
from itertools import product
from pathlib import Path
from typing import Optional

import yaml

# Per-host, optional, gitignored. config/arena.example.yaml documents
# the format; nothing is loaded unless the operator copies it here.
ARENA_CONFIG = Path("config/arena.yaml")

# Dimension values offered beyond the hardware-derived ones. Batch
# knobs are coarse on purpose: the search refines around leaders.
BATCH_DIMS: dict[str, list] = {
    "max_num_seqs": [64, 128, 256, 512],
    "max_num_batched_tokens": ["default", 2048, 8192],
    # nvfp4 KV needs the FlashInfer/TRT-LLM attention path (Blackwell)
    # — vLLM falls back or errors on older stacks, and the search
    # simply scores such launches as failed.
    "kv_cache_dtype": ["auto", "fp8", "nvfp4"],
    "expert_parallel": ["off", "on"],
    "placement": ["pack", "spread"],
}

# gpu_memory_utilization is deliberately NOT a dimension: on large-VRAM
# cards it only nudges the KV pool — a capacity dial, not a perf knob.
FIXED_GMU = 0.92


def detect_gpus() -> list[float]:
    """Per-GPU VRAM in GB via nvidia-smi; [] on non-GPU hosts."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(round(int(line) / 1024, 1))   # MiB -> GB
            except ValueError:
                continue
    return out


def _arena_config() -> dict:
    if ARENA_CONFIG.exists():
        try:
            return yaml.safe_load(ARENA_CONFIG.read_text()) or {}
        except yaml.YAMLError:
            return {}
    return {}


def hardware() -> dict:
    """{count, vram_per_gpu_gb, device_groups, detected_count, source}
    — detected, with config/arena.yaml (when present) able to pin
    device_groups (PCIe/NUMA domains aren't reliably auto-detectable).

    Config wins on SHAPE: containerized nvidia-smi sometimes sees a
    subset of the box, so a pinned map is kept even when detection
    finds fewer devices. But that must never pass silently — a copied
    arena.yaml on a GPU-less laptop would otherwise announce eight
    GPUs — so ``source`` says "config (unverified: detected N)" and
    ``detected_count`` carries what nvidia-smi actually saw."""
    vrams = detect_gpus()
    cfg = _arena_config()
    groups = cfg.get("device_groups")
    detected = len(vrams)
    if groups and all(isinstance(g, list) for g in groups):
        groups = [[int(d) for d in g] for g in groups]
        count = sum(len(g) for g in groups)
        source = ("config" if detected >= count
                  else f"config (unverified: detected {detected})")
    else:
        count = detected
        groups = [list(range(count))] if count else []
        source = "detected"
    vram = min(vrams) if vrams else cfg.get("vram_per_gpu_gb")
    return {
        "count": count,
        "detected_count": detected,
        "vram_per_gpu_gb": float(vram) if vram else None,
        "device_groups": groups,
        "source": source,
    }


# Size grouping for the UI's size dropdown: 30B and 32B are the same
# operator-level choice. (lo, hi, label) over total params in B.
SIZE_CLASSES = [
    (0, 15, "≤ 15B"),
    (15, 45, "16–45B"),
    (45, 90, "46–90B"),
    (90, 10_000, "> 90B"),
]


def size_class(params_b) -> str:
    if params_b is None:
        return "unknown"
    for lo, hi, label in SIZE_CLASSES:
        if lo < float(params_b) <= hi:
            return label
    return "unknown"


def group_for_models(model_ids: list[str]) -> Optional[dict]:
    """Investigation-grouping identity for a set of models: the
    (series, size-range) pairs they cover — "Qwen3 16–45B". Two runs
    over Qwen3 mid-size models are the same investigation even if one
    added a coder variant; Qwen3.6 is a DIFFERENT series and never
    groups with Qwen3. Unknown models fall into a stable 'custom'
    bucket keyed by the exact id set."""
    if not model_ids:
        return None
    import hashlib

    from .model_catalog import load_model_catalog
    try:
        by_id = {e["id"]: e for e in load_model_catalog()}
    except Exception:  # noqa: BLE001
        by_id = {}
    pairs: set[tuple[str, str]] = set()
    unknown: list[str] = []
    for mid in model_ids:
        e = by_id.get(mid)
        if e and e.get("series"):
            pairs.add((e["series"], size_class(e.get("params_b"))))
        else:
            unknown.append(mid)
    if pairs:
        key_src = "\n".join(sorted(f"{s}|{z}" for s, z in pairs))
        by_series: dict[str, list[str]] = {}
        for s, z in sorted(pairs):
            by_series.setdefault(s, []).append(z)
        label = " + ".join(
            f"{s} {'/'.join(zs)}" for s, zs in sorted(by_series.items()))
        if unknown:
            label += f" (+{len(unknown)} custom)"
            key_src += "\n" + "\n".join(sorted(unknown))
    else:
        key_src = "\n".join(sorted(unknown))
        label = f"custom ({len(unknown)} models)"
    return {
        "key": hashlib.sha256(key_src.encode()).hexdigest()[:10],
        "label": label,
        "models": sorted(model_ids),
    }


def _tp_values(max_group: int) -> list[int]:
    out, t = [], 1
    while t <= max_group:
        out.append(t)
        t *= 2
    return out


def feasible_tps(entry: dict, tp_values: list[int],
                 vram_per_gpu: Optional[float]) -> list[int]:
    """TP values that can load this model. Unknown VRAM or unknown
    model size -> everything is allowed (validated at launch)."""
    need = entry.get("min_vram_gb")
    if not need or not vram_per_gpu:
        return list(tp_values)
    return [t for t in tp_values if t * vram_per_gpu >= float(need)]


def full_arena(catalog: Optional[list[dict]] = None) -> dict:
    """The complete arena for this host: hardware, every catalog model
    with its feasible TP set, and every dimension with all values —
    the UI renders this with everything selected by default."""
    from .model_catalog import load_model_catalog
    hw = hardware()
    catalog = catalog if catalog is not None else load_model_catalog()
    max_group = max((len(g) for g in hw["device_groups"]), default=0)
    tp_all = _tp_values(max_group) if max_group else []
    dp_all = [d for d in _tp_values(hw["count"])] if hw["count"] else []

    models = []
    for e in catalog:
        tps = feasible_tps(e, tp_all, hw["vram_per_gpu_gb"])
        models.append({
            "id": e["id"], "family": e["family"], "quant": e["quant"],
            "series": e.get("series", ""),
            "params_b": e.get("params_b"),
            "size_class": size_class(e.get("params_b")),
            "specialty": e.get("specialty", "instruct"),
            "moe": e.get("moe", False), "gated": e.get("gated", False),
            "approx_size_gb": e.get("approx_size_gb"),
            "min_vram_gb": e.get("min_vram_gb"),
            "notes": e.get("notes", ""),
            "feasible_tps": tps,
            "feasible": bool(tps) and bool(hw["count"]),
        })
    dims = {"tp": tp_all, "dp": dp_all,
            **{k: list(v) for k, v in BATCH_DIMS.items()}}
    # Feasibility is derived, not declared -- same rule as TP. An
    # engine whose image is not staged is not a choice the operator
    # has, and offering it would spend the first candidate
    # discovering a tens-of-GB download it cannot do mid-search.
    # A host with one staged runtime gets no engine dimension at all,
    # so the arena's combinatorics are unchanged until a second one
    # is actually pulled.
    from .engine_runtimes import available_engines
    engines = available_engines()
    if len(engines) > 1:
        dims["engine"] = engines
    # Engine-specific levers become dimensions only when that engine is
    # staged. Each carries what it MEASURED here (engine_notes.py), so
    # the arena can offer a knob without inviting the same afternoon to
    # be spent discovering its cost twice.
    from .engine_notes import searchable_dimensions
    dims.update(searchable_dimensions(engines))
    from .engine_notes import ENGINE_NOTES, as_dicts
    return {
        "hardware": hw,
        "models": models,
        "dimensions": dims,
        "fixed": {"gpu_memory_utilization": FIXED_GMU},
        # Narrative + evidence for the optimize cards.
        "engine_notes": {e: ENGINE_NOTES.get(e, "") for e in engines},
        "levers": [d for d in as_dicts() if d["engine"] in engines],
    }


def _slug(model_id: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in model_id.split("/")[-1]).strip("-").lower()


def build_space_doc(
    selection: dict,
    catalog: Optional[list[dict]] = None,
    budget: Optional[int] = None,
) -> dict:
    """Search-space YAML doc from an arena selection:
    ``{"models": [ids...], "dims": {dim: [values...]}}`` — empty/absent
    dims mean "the full arena values". Raises ValueError when the
    selection leaves nothing runnable."""
    from .model_catalog import load_model_catalog
    catalog = catalog if catalog is not None else load_model_catalog()
    arena = full_arena(catalog)
    hw = arena["hardware"]
    if not hw["count"]:
        raise ValueError("no GPUs detected — the arena needs a GPU host "
                         "(or device_groups in config/arena.yaml — see "
                         "config/arena.example.yaml)")

    by_id = {m["id"]: m for m in arena["models"]}
    chosen_ids = selection.get("models") or [
        m["id"] for m in arena["models"] if m["feasible"]]
    variants: dict[str, dict] = {}
    cat_by_id = {e["id"]: e for e in catalog}
    for mid in chosen_ids:
        m = by_id.get(mid)
        if m is None:
            raise ValueError(f"'{mid}' is not in the model catalog")
        if not m["feasible"]:
            raise ValueError(
                f"'{mid}' cannot run here — needs {m['min_vram_gb']} GB, "
                f"largest domain provides "
                f"{max(len(g) for g in hw['device_groups']) * (hw['vram_per_gpu_gb'] or 0):.0f} GB"
            )
        vname = f"{m['family']}-{m['quant']}"
        if vname in variants:                    # two entries, same family+quant
            vname = f"{vname}-{_slug(mid)}"
        entry = cat_by_id[mid]
        variants[vname] = {
            "model": mid, "served_name": vname,
            "extra_args": list(entry.get("engine_args") or []),
            "min_vram_gb": entry.get("min_vram_gb"),
            "moe": bool(entry.get("moe")),
        }

    dims_sel = selection.get("dims") or {}
    dims: dict[str, list] = {}
    for name, all_vals in arena["dimensions"].items():
        vals = dims_sel.get(name) or all_vals
        bad = [v for v in vals if v not in all_vals]
        if bad:
            raise ValueError(f"dimension {name}: {bad} not in the arena")
        if vals:
            dims[name] = list(vals)
    if not dims.get("tp") or not dims.get("dp"):
        raise ValueError("tp and dp must keep at least one value each")

    doc = {
        "name": "arena",
        "engine": "vllm_cuda",
        # Explicit, not defaulted: the CPU-image field incident showed
        # what an implicit image costs (every candidate silently ran
        # vLLM on the Xeon until the health gate timed out).
        "gpu_image": "vllm/vllm-openai:latest",
        "device_groups": hw["device_groups"],
        "vram_per_gpu_gb": hw["vram_per_gpu_gb"],
        "model_variants": variants,
        "dimensions": {"model_variant": list(variants), **dims},
        "objective": {"kind": "sla_throughput"},
        # Explicit so the generated YAML self-documents how candidates
        # are scored: one workload shape climbed over a concurrency
        # ladder with SLA early-exit, scored at the best rung.
        "measurement": {"input_tokens": 512, "output_tokens": 256,
                        "ladder": [8, 32, 128, 512]},
    }
    # Search params sized to the chosen budget (default: the
    # statistical recommendation): the coverage stage takes what the
    # refinement allowance doesn't, so a bigger budget widens the
    # screen instead of just adding refinement rounds.
    rec = recommend_search(doc["dimensions"])
    b = int(budget) if budget else rec["recommended"]
    doc["search"] = {
        "budget": b,
        "initial_samples": max(8, min(b - 8, b - rec["refinement_stage"],
                                      rec["coverage_stage"] * 2)),
        "top_k": 3,
        "neighbors_per_iteration": 8,
        "max_iterations": 4,
    }
    return doc


def recommend_search(dimensions: dict[str, list]) -> dict:
    """Statistically-grounded budget recommendation for a space.

    The floor is the one-factor-at-a-time bound: Σ(|values|-1)+1 —
    the smallest design where every value of every dimension is
    measured at least once against a baseline. Screening designs pad
    that ~25% so values are seen in more than one context (the greedy
    coverage sampler spreads them across combinations), with a floor
    of 2× the widest dimension so every model appears at least twice.
    Refinement then needs its own allowance: 3 rounds × 8 one-knob
    neighbors of the leaders. Beyond ~2× the recommendation the
    search usually stops itself first (<3% improvement per round).
    """
    cards = {d: len(v) for d, v in dimensions.items() if len(v) > 1}
    max_v = max(cards.values(), default=1)
    ofat = sum(v - 1 for v in cards.values()) + 1
    coverage = max(2 * max_v, math.ceil(1.25 * ofat))
    refinement = 24
    rec = coverage + refinement
    return {
        "dimensions": len(cards),
        "ofat_min": ofat,
        "coverage_stage": coverage,
        "refinement_stage": refinement,
        "recommended": rec,
        "screening": max(ofat + 8, math.ceil(rec * 0.6)),
        "thorough": rec * 2,
    }


def _estimate_hours(budget: int, total_combos: int, n_models: int,
                    max_iterations: int) -> float:
    evals = min(budget, total_combos)
    cold = min(evals, n_models * (max_iterations + 1))
    return round((evals * 9 + cold * 4) / 60, 1)


def summarize_space_doc(doc: dict) -> dict:
    """Feasible-shape counting + restart-cost estimate for a space doc.
    'Launch shapes' are the distinct (variant, tp, dp, placement, ep)
    combinations that pass device+VRAM fit; batch knobs multiply on
    top but never change feasibility."""
    import tempfile

    from .search import load_space, validate_candidate
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        tmp = f.name
    try:
        space = load_space(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)

    dims = space.dimensions
    shape_dims = [d for d in ("model_variant", "tp", "dp", "placement",
                              "expert_parallel") if d in dims]
    shapes = set()
    for combo in product(*(dims[d] for d in shape_dims)):
        params = dict(zip(shape_dims, combo, strict=True))
        ok, _ = validate_candidate(params, space)
        if ok:
            from .search import canonical_key, normalize
            n = normalize(params, space)
            shapes.add(canonical_key(
                {d: n[d] for d in shape_dims if d in n}, space))
    batch_mult = 1
    for d in ("max_num_seqs", "max_num_batched_tokens", "kv_cache_dtype"):
        if d in dims:
            batch_mult *= len(dims[d])
    budget = space.search.budget
    n_models = len(space.model_variants)
    total = len(shapes) * batch_mult
    rec = recommend_search(space.dimensions)
    tiers = [
        {"name": "screening", "budget": rec["screening"],
         "hours": _estimate_hours(rec["screening"], total, n_models,
                                  space.search.max_iterations)},
        {"name": "recommended", "budget": rec["recommended"],
         "hours": _estimate_hours(rec["recommended"], total, n_models,
                                  space.search.max_iterations)},
        {"name": "thorough", "budget": rec["thorough"],
         "hours": _estimate_hours(rec["thorough"], total, n_models,
                                  space.search.max_iterations)},
    ]
    return {
        "recommendation": {**rec, "tiers": tiers},
        "ladder": list(space.measurement.ladder),
        "measurement_tokens": [space.measurement.input_tokens,
                               space.measurement.output_tokens],
        "launch_shapes": len(shapes),
        "total_combinations": len(shapes) * batch_mult,
        "budget": budget,
        "models": n_models,
        # Every evaluation relaunches the engine; ordering by model
        # bounds cold weight loads at ~one per model per iteration.
        "estimated_engine_restarts": min(budget, len(shapes) * batch_mult),
        "estimated_cold_weight_loads": min(
            budget, n_models * (space.search.max_iterations + 1)),
        # Rough wall clock: ~9 min per evaluation (relaunch + ladder
        # climb with early exit), plus ~4 min extra per cold model
        # swap. An estimate for planning, not a promise.
        "estimated_hours": _estimate_hours(
            budget, total, n_models, space.search.max_iterations),
        "space_hash": space.space_hash(),
    }
