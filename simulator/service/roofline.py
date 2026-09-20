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
                "results": [], "summary": {"best": None, "fastest": None,
                                           "largest_served": None,
                                           "largest_attempted": None,
                                           "spectrum": [],
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
async def roofline_candidates(limit: int = 8, diverse: bool = True,
                              cached_only: bool = False,
                              spectrum: bool = True, large_limit: int = 3,
                              beyond_limit: int = 2, extra: int = 0) -> dict:
    """The model shortlist in PICK ORDER, with the reasoning shown.

    The Roofline tab's auto mode plans exactly this list, so it has to
    be the one ``pick_models`` would run: with ``spectrum`` (the
    default) that is the FAST vendor round-robin plus the LARGE and
    BEYOND_VRAM tiers, and each candidate carries its ``tier``,
    ``fits_gpu`` / ``kt_eligible`` (which decide its engines), the
    ``tp`` that holds it, and the ``pick_round`` that chose it. Up to
    ``extra`` unpicked models follow the picks (tier "") so the table
    can still show, greyed, why they are out.
    """
    from dataclasses import asdict as _asdict

    from ..arena import hardware
    from ..model_catalog import load_model_catalog
    from ..roofline import pick_models, score_models

    hw = await asyncio.to_thread(hardware)
    cat = await asyncio.to_thread(load_model_catalog)
    vram = hw.get("vram_per_gpu_gb")
    ram = hw.get("host_ram_gb")
    gpus = int(hw.get("count") or 8)
    limit = max(1, limit)
    picked = await asyncio.to_thread(
        pick_models, cat, vram_per_gpu_gb=vram, host_ram_gb=ram,
        gpu_count=gpus, limit=limit, diverse=diverse,
        cached_only=cached_only, spectrum=spectrum,
        large_limit=large_limit, beyond_limit=beyond_limit)
    chosen = {c.id for c in picked}
    rest = [c for c in await asyncio.to_thread(
        score_models, cat, vram_per_gpu_gb=vram, host_ram_gb=ram,
        gpu_count=gpus) if c.id not in chosen]
    return {"hardware": hw, "diverse": diverse, "cached_only": cached_only,
            "spectrum": spectrum,
            "candidates": [_asdict(c) for c in
                           (picked + rest)[:limit + max(0, extra)]]}
