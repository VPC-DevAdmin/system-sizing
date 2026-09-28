"""External measurements inside the existing roofline lifecycle and run lock."""

from __future__ import annotations

import asyncio
import re
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from ..benchmarks.campaign import create_campaign, execute, list_campaigns
from ..benchmarks.runtime import prepare, status
from ..benchmarks.spec import BenchmarkSpec, digest
from ..benchmarks.storage import campaign_path, read_json
from .state import ActiveRun

router = APIRouter()


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_id: str
    spec: BenchmarkSpec
    fresh: bool = False


def assert_no_preparation(app):
    for name in ("model_downloads", "engine_pulls"):
        for job in getattr(app.state, name, {}).values():
            proc = job.get("proc")
            if proc is not None and proc.poll() is None:
                raise HTTPException(409, "Asset preparation is running; wait before measuring")


def assert_idle(app):
    if app.state.active and not app.state.active.task.done():
        raise HTTPException(409, "A run is already active")
    opt = app.state.optimizer
    if opt and opt["proc"].poll() is None:
        raise HTTPException(409, "The engine optimizer is active")
    assert_no_preparation(app)


def start(app, coroutine, workload):
    from .runs import _supervise

    app.state.active = ActiveRun(
        task=asyncio.create_task(_supervise(app, coroutine)),
        workload=workload,
        config_path="",
        started_at=time.time(),
        engine_summary="Third-party benchmark",
    )
    return {"accepted": True, **workload}


@router.get("/api/roofline/benchmarks")
async def listing(request: Request):
    base = request.app.state.paths.runs_base
    from ..roofline import load_state

    st = await asyncio.to_thread(load_state, request.app.state.paths.roofline_state)
    return {
        "campaigns": await asyncio.to_thread(list_campaigns, base),
        "tools": status(base),
        "search_report_ready": bool(st and st.plan.get("benchmark_report")),
    }


@router.get("/api/roofline/benchmarks/results")
async def results(request: Request):
    from ..roofline import load_state

    st = await asyncio.to_thread(load_state, request.app.state.paths.roofline_state)
    return {"results": [{"id": digest(row), "result": row} for row in (st.results if st else [])]}


@router.post("/api/roofline/benchmarks/prepare", status_code=202)
async def prepare_tools(spec: BenchmarkSpec, request: Request):
    async with request.app.state.start_lock:
        assert_idle(request.app)
        if not spec.harnesses:
            raise HTTPException(422, "Select at least one harness")

        async def work():
            return [await prepare(request.app.state.paths.runs_base, h) for h in spec.harnesses]

        return start(request.app, work, {"kind": "benchmark_prepare", "harnesses": spec.harnesses})


@router.post("/api/roofline/benchmarks", status_code=202)
async def launch(body: BenchmarkRequest, request: Request):
    from ..roofline import load_state

    app = request.app
    async with app.state.start_lock:
        assert_idle(app)
        st = await asyncio.to_thread(load_state, app.state.paths.roofline_state)
        row = next((r for r in (st.results if st else []) if digest(r) == body.result_id), None)
        if row is None:
            raise HTTPException(409, "Recorded selection changed; refresh before starting")
        try:
            path = await asyncio.to_thread(
                create_campaign,
                app.state.paths.runs_base,
                row,
                body.spec,
                source={"kind": "manual_selection", "roofline_started_at": st.started_at},
                force=body.fresh,
            )
        except (ValueError, OSError) as e:
            raise HTTPException(422, str(e)) from e
        return start(
            app,
            lambda: execute(path, app.state.paths.runs_base),
            {"kind": "third_party_benchmark", "benchmark_id": path.name},
        )


def checked_path(request, campaign_id):
    try:
        path = campaign_path(request.app.state.paths.runs_base, campaign_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if not (path / "manifest.json").is_file():
        raise HTTPException(404, "Unknown benchmark")
    return path


@router.post("/api/roofline/benchmarks/{campaign_id}/resume", status_code=202)
async def resume(campaign_id: str, request: Request):
    async with request.app.state.start_lock:
        assert_idle(request.app)
        path = checked_path(request, campaign_id)
        return start(
            request.app,
            lambda: execute(path, request.app.state.paths.runs_base),
            {"kind": "third_party_benchmark", "benchmark_id": campaign_id},
        )


@router.get("/api/roofline/benchmarks/{campaign_id}")
async def detail(campaign_id: str, request: Request):
    path = checked_path(request, campaign_id)
    return {
        "manifest": await asyncio.to_thread(read_json, path / "manifest.json"),
        "state": await asyncio.to_thread(read_json, path / "state.json"),
    }


@router.get("/api/roofline/benchmarks/{campaign_id}/export/{kind}")
async def export(campaign_id: str, kind: str, request: Request):
    path = checked_path(request, campaign_id)
    if kind not in ("docx", "zip"):
        raise HTTPException(422, "Choose docx or zip")
    file = path / ("report.docx" if kind == "docx" else "evidence.zip")
    if not file.is_file() or file.is_symlink():
        raise HTTPException(409, "Report is not ready; inspect benchmark status")
    return FileResponse(file, filename=f"capsim-{campaign_id}.{kind}")


@router.get("/api/roofline/benchmark-report/{kind}")
async def search_export(kind: str, request: Request):
    from ..roofline import load_state

    st = await asyncio.to_thread(load_state, request.app.state.paths.roofline_state)
    report_id = st.plan.get("benchmark_report") if st else None
    if kind not in ("docx", "zip") or not report_id or not re.fullmatch(r"[a-f0-9]{24}", report_id):
        raise HTTPException(404, "A completed search report is not available")
    folder = request.app.state.paths.runs_base / "benchmark-reports" / report_id
    file = folder / ("report.docx" if kind == "docx" else "evidence.zip")
    if not file.is_file() or file.is_symlink():
        raise HTTPException(404, "Search report artifact is unavailable")
    return FileResponse(file, filename=f"capsim-search-{report_id}.{kind}")
