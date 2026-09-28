"""CLI parity for project discovery, readiness, and recorded-source experiments."""

import asyncio
import json
import uuid
from pathlib import Path

import typer

from ..benchmarks.spec import digest
from ..roofline import load_state
from ..service.state import _acquire_runs_lock, _release_runs_lock
from .campaign import create, execute, listing
from .capabilities import inspect_host
from .catalog import listing as projects
from .spec import ExperimentSpec

app = typer.Typer(help="Repeatable infrastructure A/B experiments with separate claim outputs.")


@app.command("list")
def list_cmd(runs_dir: Path = Path("runs")):
    typer.echo(json.dumps({"projects": projects(), "campaigns": listing(runs_dir)}, indent=2))


@app.command("preflight")
def preflight_cmd():
    typer.echo(json.dumps(inspect_host(), indent=2))


@app.command("run")
def run_cmd(result_id: str, spec_file: Path, runs_dir: Path = Path("runs")):
    lock = _acquire_runs_lock(runs_dir)
    try:
        state = load_state(runs_dir / "roofline.json")
        row = next((r for r in (state.results if state else []) if digest(r) == result_id), None)
        spec = ExperimentSpec.model_validate_json(spec_file.read_text())
        if spec.project_id == "P05":
            row = {"model": "Qwen-Image / Qwen-Image-Flash", "engine": "diffusion"}
        if row is None:
            raise typer.BadParameter("Unknown result; use capsim benchmark results")
        path = create(runs_dir, row, spec, uuid.uuid4().hex)
        typer.echo(f"Campaign {path.name}")
        result = asyncio.run(execute(path))
        typer.echo(json.dumps(result, indent=2))
        if result["phase"] != "complete":
            raise typer.Exit(1)
    finally:
        _release_runs_lock(lock)
