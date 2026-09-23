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


# ── sizing export ────────────────────────────────────────────
# The roofline's measurements in the sizing tool's document shape
# (export_sizing.py), built from the state file on every request so a
# download is never older than the page it was clicked on.

_SYSTEM: dict | None = None


def _export_docs(request: Request, collapse: bool) -> list[dict]:
    import json
    from pathlib import Path

    from ..export_sizing import build, detect_system
    from ..roofline import load_state
    global _SYSTEM
    state_path = Path(request.app.state.paths.roofline_state)
    st = load_state(state_path)
    if st is None:
        return []
    if _SYSTEM is None:
        _SYSTEM = detect_system()
    placement = None
    p = state_path.parent / "kt_placement" / "kt_placement.json"
    if p.is_file():
        try:
            placement = json.loads(p.read_text())
        except ValueError:
            placement = None
    return build(st.to_dict(), state_path.parent, system=_SYSTEM,
                 placement=placement, collapse=collapse)


def _export_name(ext: str) -> str:
    import time
    host = ((_SYSTEM or {}).get("platform") or "capsim").replace("PowerEdge ", "")
    return f"{host.lower().replace(' ', '-')}_sizing_{time.strftime('%Y-%m-%d')}.{ext}"


@router.get("/api/roofline/export/index")
async def roofline_export_index(request: Request, all_attempts: bool = False) -> dict:
    from ..export_sizing import summary
    docs = await asyncio.to_thread(_export_docs, request, not all_attempts)
    return {**summary(docs), "json_name": _export_name("json"),
            "zip_name": _export_name("zip")}


@router.get("/api/roofline/export")
async def roofline_export(request: Request, all_attempts: bool = False):
    import json

    from fastapi.responses import Response
    docs = await asyncio.to_thread(_export_docs, request, not all_attempts)
    return Response(json.dumps(docs, indent=1), media_type="application/json",
                    headers={"Content-Disposition":
                             f'attachment; filename="{_export_name("json")}"'})


@router.get("/api/roofline/export.zip")
async def roofline_export_zip(request: Request, all_attempts: bool = False):
    from fastapi.responses import Response

    from ..export_sizing import zip_bytes
    docs = await asyncio.to_thread(_export_docs, request, not all_attempts)
    data = await asyncio.to_thread(zip_bytes, docs)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="{_export_name("zip")}"'})


@router.get("/api/roofline/candidates")
async def roofline_candidates(limit: int = 8, diverse: bool = True,
                              cached_only: bool = False,
                              spectrum: bool = True, large_limit: int = 3,
                              beyond_limit: int = 2, extra: int = 0,
                              allow_cross_domain_tp: bool = False) -> dict:
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
    from ..roofline import max_tp_of, pick_models, score_models

    hw = await asyncio.to_thread(hardware)
    cat = await asyncio.to_thread(load_model_catalog)
    vram = hw.get("vram_per_gpu_gb")
    ram = hw.get("host_ram_gb")
    gpus = int(hw.get("count") or 8)
    tp_cap = None if allow_cross_domain_tp else max_tp_of(hw)
    limit = max(1, limit)
    picked = await asyncio.to_thread(
        pick_models, cat, vram_per_gpu_gb=vram, host_ram_gb=ram,
        gpu_count=gpus, max_tp=tp_cap, limit=limit, diverse=diverse,
        cached_only=cached_only, spectrum=spectrum,
        large_limit=large_limit, beyond_limit=beyond_limit)
    chosen = {c.id for c in picked}
    rest = [c for c in await asyncio.to_thread(
        score_models, cat, vram_per_gpu_gb=vram, host_ram_gb=ram,
        gpu_count=gpus, max_tp=tp_cap) if c.id not in chosen]
    return {"hardware": hw, "diverse": diverse, "cached_only": cached_only,
            "allow_cross_domain_tp": allow_cross_domain_tp,
            "spectrum": spectrum,
            "candidates": [_asdict(c) for c in
                           (picked + rest)[:limit + max(0, extra)]]}
