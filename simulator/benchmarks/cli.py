"""CLI mirrors the roofline UI, including exact recorded result selection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from .campaign import create_campaign, execute, list_campaigns
from .report import export_bundle
from .runtime import prepare, status
from .spec import BenchmarkSpec, digest
from .storage import campaign_path

app = typer.Typer(help="Run pinned third-party harnesses on recorded roofline results.")


@app.command("list")
def list_cmd(runs_dir: Path = Path("runs")):
    typer.echo(json.dumps(list_campaigns(runs_dir), indent=2))


@app.command("results")
def results_cmd(runs_dir: Path = Path("runs")):
    from ..roofline import load_state

    state = load_state(runs_dir / "roofline.json")
    typer.echo(
        json.dumps(
            [{"id": digest(r), "result": r} for r in state.results] if state else [], indent=2
        )
    )


@app.command("tools")
def tools_cmd(runs_dir: Path = Path("runs")):
    typer.echo(json.dumps(status(runs_dir), indent=2))


@app.command("prepare")
def prepare_cmd(harness: str, runs_dir: Path = Path("runs")):
    from ..service.state import _acquire_runs_lock, _release_runs_lock

    lock = _acquire_runs_lock(runs_dir)
    try:
        typer.echo(json.dumps(asyncio.run(prepare(runs_dir, harness)), indent=2))
    finally:
        _release_runs_lock(lock)


@app.command("run")
def run_cmd(result_id: str, spec: Path, runs_dir: Path = Path("runs"), fresh: bool = False):
    from ..roofline import load_state
    from ..service.state import _acquire_runs_lock, _release_runs_lock

    lock = _acquire_runs_lock(runs_dir)
    try:
        state = load_state(runs_dir / "roofline.json")
        row = next((r for r in (state.results if state else []) if digest(r) == result_id), None)
        if row is None:
            raise typer.BadParameter("Unknown result ID; run capsim benchmark results")
        settings = BenchmarkSpec.model_validate_json(spec.read_text())
        path = create_campaign(
            runs_dir,
            row,
            settings,
            source={"kind": "manual_selection", "roofline_started_at": state.started_at},
            force=fresh,
        )
        typer.echo(json.dumps(asyncio.run(execute(path, runs_dir)), indent=2))
    finally:
        _release_runs_lock(lock)


@app.command("resume")
def resume_cmd(campaign_id: str, runs_dir: Path = Path("runs")):
    from ..service.state import _acquire_runs_lock, _release_runs_lock

    lock = _acquire_runs_lock(runs_dir)
    try:
        typer.echo(
            json.dumps(
                asyncio.run(execute(campaign_path(runs_dir, campaign_id), runs_dir)), indent=2
            )
        )
    finally:
        _release_runs_lock(lock)


@app.command("export")
def export_cmd(campaign_id: str, runs_dir: Path = Path("runs")):
    from .process import Lease

    path = campaign_path(runs_dir, campaign_id)
    with Lease(path / ".worker.lock"):
        typer.echo(str(export_bundle(path)))
