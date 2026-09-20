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
async def roofline_candidates(limit: int = 12) -> dict:
    """The ranked model shortlist, with the reasoning shown."""
    from dataclasses import asdict as _asdict

    from ..arena import hardware
    from ..model_catalog import load_model_catalog
    from ..roofline import score_models

    hw = await asyncio.to_thread(hardware)
    cat = await asyncio.to_thread(load_model_catalog)
    ranked = await asyncio.to_thread(
        score_models, cat, vram_per_gpu_gb=hw.get("vram_per_gpu_gb"))
    return {"hardware": hw,
            "candidates": [_asdict(c) for c in ranked[:max(1, limit)]]}
