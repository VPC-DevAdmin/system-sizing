"""The catalog: hardware profiles, personas and cohorts (with the
editor that writes their overlay files), the per-family headline
shape store, and the hardware probe the benchmark form reads."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from .schemas import SaveSpecRequest
from .state import Paths

router = APIRouter()


@router.get("/api/profiles")
async def profiles() -> dict:
    """Profiles with operator-facing metadata: what model/shape
    each one runs, whether it matches THIS machine's hardware,
    and whether it came from the optimizer — so the Benchmark
    picker can lead with relevant, plainly-labeled choices
    instead of a flat list of filenames."""
    import yaml as _yaml

    from ..arena import hardware
    from ..config import list_profiles

    hw = await asyncio.to_thread(hardware)
    has_gpu = bool(hw["count"])
    _CPU_ENGINES = ("vllm", "sglang", "vllm_dual_socket")

    def _describe(name: str, path) -> Optional[dict]:
        engine_type, model, label, detail = "?", "", name, ""
        try:
            raw = _yaml.safe_load(Path(path).read_text()) or {}
            eng = raw.get("engine") or {}
            if not eng:
                # Not a benchmark profile (e.g. config/arena.yaml,
                # the hardware-hints file) — keep it out of the
                # picker entirely.
                return None
            engine_type = str(eng.get("type", "?"))
            model = str(eng.get("model_id") or eng.get("model") or "")
            short = model.split("/")[-1] if model else ""
            if engine_type == "vllm_cuda_multi":
                reps = eng.get("replica_devices") or []
                tp = max((len(g) for g in reps), default=1)
                label = f"{short} — whole box, {len(reps)} replicas"
                detail = (f"tp{tp} per replica"
                          if tp > 1 else "one GPU per replica")
            elif engine_type in ("trtllm", "sglang_cuda",
                                 "ktransformers"):
                from ..engines.knobs import ENGINE_LABELS
                reps = eng.get("replica_devices") or []
                tp = max((len(g) for g in reps), default=1)
                label = (f"{short} — "
                         f"{ENGINE_LABELS.get(engine_type, engine_type)}"
                         f", {len(reps)} replicas")
                detail = (f"tp{tp} per replica"
                          if tp > 1 else "one GPU per replica")
            elif engine_type == "vllm_cuda":
                tp = eng.get("tensor_parallel_size", 1)
                label = f"{short} — single engine"
                detail = f"tp{tp}" if tp and tp > 1 else "one GPU"
            elif engine_type in _CPU_ENGINES:
                label = f"{short} — CPU engine"
                detail = engine_type
            elif engine_type == "mock":
                label = "Self-test (mock engine)"
                detail = "no hardware needed — verifies the pipeline"
            elif engine_type == "remote":
                label = f"Remote endpoint{' — ' + short if short else ''}"
        except Exception:  # noqa: BLE001
            pass
        from ..engines.knobs import GPU_ENGINES
        gpu_engine = engine_type in GPU_ENGINES or engine_type == "vllm_cuda"
        # The searched engine dimensions, extracted so the
        # benchmark form can prefill its Advanced settings with
        # exactly what the optimization landed on.
        params: dict = {}
        try:
            flags = [str(f) for f in (eng.get("vllm_extra_flags") or [])]
            def _flag(key):
                return (flags[flags.index(key) + 1]
                        if key in flags
                        and flags.index(key) + 1 < len(flags) else None)
            reps = eng.get("replica_devices") or []
            params = {
                # Which server the profile was measured on, so the
                # benchmark form prefills the engine too and does
                # not silently re-measure on a different one.
                "engine": (engine_type
                           if engine_type in GPU_ENGINES
                           else "vllm_cuda_multi"),
                "replicas": len(reps) if reps else 1,
                "tp": (max((len(g) for g in reps), default=1) if reps
                       else int(eng.get("tensor_parallel_size") or 1)),
                "gpu_memory_utilization":
                    eng.get("gpu_memory_utilization"),
                "max_model_len": eng.get("max_model_len"),
                "max_num_seqs": _flag("--max-num-seqs"),
                "max_num_batched_tokens":
                    _flag("--max-num-batched-tokens"),
                "kv_cache_dtype": _flag("--kv-cache-dtype") or "auto",
                "expert_parallel": "--enable-expert-parallel" in flags,
            }
        except Exception:  # noqa: BLE001
            params = {}
        return {
            "path": str(path),
            "engine_type": engine_type,
            "model_id": model,
            "label": label,
            "detail": detail,
            "optimized": name.startswith("optimized-"),
            "engine": params,
            # Does this profile match THIS machine?
            "fits_hardware": (
                has_gpu if gpu_engine
                else (not has_gpu) if engine_type in _CPU_ENGINES
                else True   # mock / remote / unknown: always usable
            ),
        }

    out = {}
    for name, path in list_profiles().items():
        entry = await asyncio.to_thread(_describe, name, path)
        if entry is not None:
            out[name] = entry
    return out


def _persona_summary(p) -> dict:
    """The numbers a human needs to know what a persona MEANS:
    question/answer sizes, session length, and the read/think gap
    between turns — analytic, not sampled."""
    from ..distributions import summarize
    inp = summarize(p.input_tokens)
    out = summarize(p.output_tokens)
    turns = summarize(p.turns_per_session)
    read = summarize(p.read_time_seconds)
    think = summarize(p.active_think_seconds)
    gap = {
        k: ((read[k] or 0) + (think[k] or 0))
        if read[k] is not None and think[k] is not None else None
        for k in ("median", "mean", "p90")
    }
    return {"input_tokens": inp, "output_tokens": out,
            "turns_per_session": turns, "think_gap_s": gap}


# Working personas of the shape search — real registry entries
# (workers must resolve them) but not workloads a user picks.
_INTERNAL_PERSONAS = {"headline_cell", "headline_best"}


@router.get("/api/personas")
async def personas() -> list[dict]:
    from ..personas import PERSONAS
    return [
        {
            "id": p.id,
            "name": getattr(p, "name", "") or p.id,
            "description": p.description,
            "ttft_target_s": p.ttft_target_seconds,
            "ttft_failure_s": p.ttft_failure_seconds,
            "tpot_target_ms": p.tpot_target_ms,
            "tpot_failure_ms": p.tpot_failure_ms,
            "summary": _persona_summary(p),
        }
        for p in PERSONAS.values()
        if p.id not in _INTERNAL_PERSONAS
    ]


@router.get("/api/cohorts")
async def cohorts() -> list[dict]:
    from ..personas import COHORTS, PERSONAS

    def _blended(c) -> Optional[dict]:
        """Weight-averaged medians across the mix — 'what a
        typical turn of this team looks like'."""
        total = sum(c.persona_weights.values()) or 1.0
        acc = {"input_tokens": 0.0, "output_tokens": 0.0,
               "turns_per_session": 0.0, "think_gap_s": 0.0}
        for pid, w in c.persona_weights.items():
            p = PERSONAS.get(pid)
            if p is None:
                return None
            s = _persona_summary(p)
            acc["input_tokens"] += (s["input_tokens"]["median"] or 0) * w
            acc["output_tokens"] += (s["output_tokens"]["median"] or 0) * w
            acc["turns_per_session"] += (
                s["turns_per_session"]["mean"] or 0) * w
            acc["think_gap_s"] += (s["think_gap_s"]["median"] or 0) * w
        return {k: round(v / total, 1) for k, v in acc.items()}

    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "persona_weights": c.persona_weights,
            "blended": _blended(c),
        }
        for c in COHORTS.values()
    ]


# ── persona/cohort editor (Phase 4) ───────────────────────────
# The catalog is data: packaged defaults + config/personas/*.yaml
# overlays. The editor GETs a spec as YAML, PUTs it back; the
# server validates against the full merged catalog before the
# overlay file lands, so a bad save can't wedge the registry.

def _catalog_dir(paths: Paths) -> Path:
    from ..persona_loader import USER_CATALOG_DIR
    return paths.catalog_dir if paths.catalog_dir is not None else USER_CATALOG_DIR


def _save_catalog_entry(
    paths: Paths, kind: str, entry_id: str,
    spec_yaml: Optional[str] = None, spec: Optional[dict] = None,
) -> None:
    import yaml as _yaml

    from ..persona_loader import PersonaSpecError, load_catalog
    from ..personas import reload_personas

    if not entry_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(422, "id must be alphanumeric/_/-")
    if spec is None:
        if spec_yaml is None:
            raise HTTPException(422, "pass spec or yaml")
        try:
            spec = _yaml.safe_load(spec_yaml)
        except _yaml.YAMLError as e:
            raise HTTPException(422, f"invalid YAML: {e}") from e
    if not isinstance(spec, dict):
        raise HTTPException(422, "spec must be a mapping")

    target = _catalog_dir(paths) / f"{entry_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.read_text() if target.exists() else None
    target.write_text(_yaml.safe_dump(
        {kind: {entry_id: spec}}, sort_keys=False, allow_unicode=True,
    ))
    try:
        load_catalog(user_dir=_catalog_dir(paths))   # validate with the new file
    except PersonaSpecError as e:
        if previous is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(previous)
        raise HTTPException(422, str(e)) from e
    reload_personas(user_dir=_catalog_dir(paths))


@router.get("/api/personas/{persona_id}")
async def persona_detail(persona_id: str, request: Request) -> dict:
    paths = request.app.state.paths
    import yaml as _yaml

    from ..persona_loader import serialize_persona
    from ..personas import PERSONAS
    if persona_id not in PERSONAS:
        raise HTTPException(404, f"unknown persona '{persona_id}'")
    spec = serialize_persona(PERSONAS[persona_id])
    return {
        "id": persona_id,
        "spec": spec,
        "yaml": _yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        "editable_file": str(_catalog_dir(paths) / f"{persona_id}.yaml"),
    }


@router.put("/api/personas/{persona_id}")
async def persona_save(
        persona_id: str, req: SaveSpecRequest, request: Request) -> dict:
    _save_catalog_entry(
        request.app.state.paths, "personas", persona_id, req.yaml, req.spec)
    return {"saved": persona_id}


@router.get("/api/cohorts/{cohort_id}")
async def cohort_detail(cohort_id: str, request: Request) -> dict:
    paths = request.app.state.paths
    import yaml as _yaml

    from ..persona_loader import serialize_cohort
    from ..personas import COHORTS
    if cohort_id not in COHORTS:
        raise HTTPException(404, f"unknown cohort '{cohort_id}'")
    spec = serialize_cohort(COHORTS[cohort_id])
    return {
        "id": cohort_id,
        "spec": spec,
        "yaml": _yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        "editable_file": str(_catalog_dir(paths) / f"{cohort_id}.yaml"),
    }


@router.put("/api/cohorts/{cohort_id}")
async def cohort_save(
        cohort_id: str, req: SaveSpecRequest, request: Request) -> dict:
    _save_catalog_entry(
        request.app.state.paths, "cohorts", cohort_id, req.yaml, req.spec)
    return {"saved": cohort_id}


# ── headline shape store ──────────────────────────────────────
# The shape search's winner is stored per MODEL FAMILY; the UI
# asks whether the selected model's family has one, and can apply
# it to the Headline: Generation workload with one click.

@router.get("/api/headline-shape")
async def headline_shape(model: str, request: Request) -> dict:
    paths = request.app.state.paths
    from ..headline_shapes import (
        generation_shape,
        model_family,
        shape_for,
        shapes_path,
    )
    stored = shape_for(shapes_path(_catalog_dir(paths)), model)
    current = generation_shape()
    return {
        "family": model_family(model),
        "shape": stored,
        "active": bool(
            stored and current
            and current == (stored.get("input_tokens"),
                            stored.get("output_tokens"))),
    }


@router.post("/api/headline-shape/apply")
async def headline_shape_apply(req: dict, request: Request) -> dict:
    paths = request.app.state.paths
    from ..headline_shapes import (
        apply_shape_to_generation,
        shape_for,
        shapes_path,
    )
    model = str(req.get("model") or "")
    stored = shape_for(shapes_path(_catalog_dir(paths)), model)
    if not stored:
        raise HTTPException(
            404, f"no optimized shape stored for '{model}' — run "
                 "the shape search first")
    await asyncio.to_thread(
        apply_shape_to_generation, _catalog_dir(paths),
        stored["input_tokens"], stored["output_tokens"])
    return {"applied": True, "shape": stored}


@router.get("/api/hardware")
async def hardware_summary() -> dict:
    """Tiny hardware probe for the UI — GPU count/names so the
    benchmark form can gray out the CPU/GPU toggle honestly."""
    from ..arena import hardware as _hw
    try:
        hw = await asyncio.to_thread(_hw)
    except Exception:  # noqa: BLE001
        hw = {}
    # `gpus` keeps the arena's view (config may pin the shape);
    # `detected_gpus` is what nvidia-smi saw, which is what decides
    # whether a GPU engine can actually launch here.
    return {
        "gpus": hw.get("count", 0) or 0,
        "detected_gpus": hw.get("detected_count", hw.get("count", 0)) or 0,
        "source": hw.get("source", "detected"),
        "vram_per_gpu_gb": hw.get("vram_per_gpu_gb"),
    }
