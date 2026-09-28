"""Infrastructure cards and experiments share the existing service start/stop lifecycle."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..benchmarks.storage import read_json, write_json
from ..infrastructure.campaign import campaign_path, create, execute, listing
from ..infrastructure.capabilities import inspect_host
from ..infrastructure.catalog import listing as projects
from ..infrastructure.spec import ExperimentSpec
from .benchmarks import assert_idle, start

router = APIRouter()


class LaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_id: str | None = None
    idempotency_key: str = Field(min_length=16, max_length=64)
    spec: ExperimentSpec


@router.get("/api/infrastructure")
async def index(request: Request):
    base = request.app.state.paths.runs_base
    readiness = base / "infrastructure" / "readiness.json"
    return {
        "projects": projects(),
        "campaigns": await asyncio.to_thread(listing, base),
        "readiness": read_json(readiness) if readiness.exists() else None,
    }


@router.post("/api/infrastructure/preflight")
async def preflight(request: Request):
    evidence = await asyncio.to_thread(inspect_host)
    write_json(request.app.state.paths.runs_base / "infrastructure" / "readiness.json", evidence)
    return evidence


@router.post("/api/infrastructure/campaigns", status_code=202)
async def launch(body: LaunchRequest, request: Request):
    from ..benchmarks.spec import digest
    from ..roofline import load_state

    app = request.app
    async with app.state.start_lock:
        assert_idle(app)
        st = await asyncio.to_thread(load_state, app.state.paths.roofline_state)
        row = next((r for r in (st.results if st else []) if digest(r) == body.result_id), None)
        if body.spec.project_id == "P05":
            row = {"model": "Qwen-Image / Qwen-Image-Flash", "engine": "diffusion"}
        if row is None:
            raise HTTPException(409, "Recorded selection changed; refresh and select again")
        try:
            path = await asyncio.to_thread(
                create, app.state.paths.runs_base, row, body.spec, body.idempotency_key
            )
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result = start(
            app,
            lambda: execute(path),
            {
                "kind": "infrastructure",
                "project_id": body.spec.project_id,
                "campaign_id": path.name,
            },
        )
        app.state.active.engine_summary = f"Infrastructure experiment {body.spec.project_id}"
        return result


def checked(request, identity):
    try:
        path = campaign_path(request.app.state.paths.runs_base, identity)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not (path / "manifest.json").is_file():
        raise HTTPException(404, "Unknown infrastructure campaign")
    return path


@router.get("/api/infrastructure/campaigns/{identity}")
async def detail(identity: str, request: Request):
    path = checked(request, identity)
    return {
        "manifest": read_json(path / "manifest.json"),
        "state": read_json(path / "state.json"),
        "comparison": read_json(path / "comparison.json")
        if (path / "comparison.json").exists()
        else None,
        "outputs": [read_json(p) for p in sorted((path / "outputs").glob("*/claim.json"))],
    }


@router.get("/api/infrastructure/campaigns/{identity}/export/{kind}")
async def export(identity: str, kind: str, request: Request, output_id: str | None = None):
    path = checked(request, identity)
    manifest = read_json(path / "manifest.json")
    if kind != "zip" and read_json(path / "state.json").get("outcome") == "invalidated":
        raise HTTPException(
            409, "Experiment invalidated by review; original evidence remains available for audit"
        )
    if kind == "zip":
        file = path / "evidence.zip"
    elif kind == "yaml" and manifest["spec"]["project_id"] == "P04":
        file = path / "roofline-candidate.yaml"
    elif kind in ("docx", "md", "json") and output_id in {
        o["id"] for o in manifest["project"]["outputs"]
    }:
        filename = {"docx": "white-paper.docx", "md": "claim-brief.md", "json": "claim.json"}[kind]
        file = path / "outputs" / output_id / filename
    else:
        raise HTTPException(422, "Choose an existing output and supported format")
    if not file.is_file() or file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
        raise HTTPException(409, "Artifact not ready")
    return FileResponse(file, filename=f"capsim-{identity}-{output_id or 'evidence'}.{kind}")


@router.post("/api/infrastructure/campaigns/{identity}/recover")
async def recover_campaign(identity: str, request: Request):
    from ..infrastructure.recovery import recover

    async with request.app.state.start_lock:
        assert_idle(request.app)
        path = checked(request, identity)
        try:
            return await asyncio.to_thread(recover, path)
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc


@router.get("/api/infrastructure/campaigns/{identity}/image-review")
async def image_review(identity: str, request: Request):
    from ..infrastructure.diffusion_report import RUBRIC, inventory

    path = checked(request, identity)
    if read_json(path / "manifest.json")["spec"]["project_id"] != "P05":
        raise HTTPException(422, "Image review requires P05")
    rows = await asyncio.to_thread(inventory, path)
    return {
        "rubric": RUBRIC,
        "images": [
            {
                "blind_id": r["blind_id"],
                "prompt": r["prompt"],
                "url": f"/api/infrastructure/campaigns/{identity}/image/{r['blind_id']}",
            }
            for r in rows
        ],
    }


@router.get("/api/infrastructure/campaigns/{identity}/image/{image_id}")
async def review_image(identity: str, image_id: str, request: Request):
    from ..infrastructure.diffusion_report import inventory

    path = checked(request, identity)
    row = next((r for r in inventory(path) if r["blind_id"] == image_id), None)
    if row is None:
        raise HTTPException(404, "Unknown image")
    file = path / row["path"]
    if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
        raise HTTPException(422, "Invalid image path")
    return FileResponse(file, media_type="image/png")


class ImageReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=200)
    reviewer_kind: str = "human"
    items: list[dict] = Field(min_length=1, max_length=512)


@router.post("/api/infrastructure/campaigns/{identity}/image-review")
async def save_image_review(identity: str, body: ImageReviewRequest, request: Request):
    from ..infrastructure.image_review import submit

    path = checked(request, identity)
    if read_json(path / "manifest.json")["spec"]["project_id"] != "P05":
        raise HTTPException(422, "Image review requires P05")
    try:
        return await asyncio.to_thread(submit, path, body.reviewer, body.reviewer_kind, body.items)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
