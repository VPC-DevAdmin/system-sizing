"""Roofline autopilot: the persisted state of a running or finished
roofline and the ranked model shortlist it starts from."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

router = APIRouter()


# ── roofline autopilot ───────────────────────────────────────
# A roofline run takes hours and the operator will not be watching.
# Everything it knows lives in runs/roofline.json, written after
# every step, so reconnecting is a GET rather than a replay of
# events nobody was listening for.

@router.get("/api/roofline")
async def roofline_state(request: Request) -> dict:
    from ..roofline import load_state
    st = await asyncio.to_thread(
        load_state, request.app.state.paths.roofline_state)
    if st is None:
        return {"kind": "roofline", "status": "none", "done": True,
                "results": [], "summary": {"best": None,
                                           "best_per_model": {},
                                           "best_per_engine": {}}}
    doc = st.to_dict()
    # Whether THIS service is still driving it. A state file left
    # at "searching" by a killed process must not read as running,
    # or the page will sit forever waiting for a dead run.
    active = request.app.state.active
    desc = active.describe() if active else None
    doc["live"] = bool(desc and desc.get("running")
                       and (desc.get("workload") or {}).get("kind")
                       == "roofline")
    return doc


@router.get("/api/roofline/candidates")
async def roofline_candidates(limit: int = 12, diverse: bool = True,
                              cached_only: bool = False) -> dict:
    """The model shortlist in PICK ORDER, with the reasoning shown.

    The Roofline tab's auto mode plans the first N of this list, so
    it has to be the same list ``pick_models`` would run -- with
    ``diverse`` (the default) that is a series round-robin, and each
    candidate carries ``series``, ``family`` and the ``pick_round``
    that chose it. Models that do not fit the box are appended after
    the picks so the table can still show, greyed, why they are out.
    """
    from dataclasses import asdict as _asdict

    from ..arena import hardware
    from ..model_catalog import load_model_catalog
    from ..roofline import pick_models, score_models

    hw = await asyncio.to_thread(hardware)
    cat = await asyncio.to_thread(load_model_catalog)
    vram = hw.get("vram_per_gpu_gb")
    limit = max(1, limit)
    picked = await asyncio.to_thread(
        pick_models, cat, vram_per_gpu_gb=vram, limit=limit, diverse=diverse,
        cached_only=cached_only)
    chosen = {c.id for c in picked}
    rest = [c for c in await asyncio.to_thread(
        score_models, cat, vram_per_gpu_gb=vram) if c.id not in chosen]
    return {"hardware": hw, "diverse": diverse, "cached_only": cached_only,
            "candidates": [_asdict(c) for c in (picked + rest)[:limit]]}
