"""Command-line interface."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from .config import apply_cli_overrides, load_config
from .engines import make_engine
from .personas import COHORTS

app = typer.Typer(add_completion=False, help="Persona Capacity Simulator")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def run(
    engine: str = typer.Option("vllm", help="Engine type: vllm | sglang"),
    model: str = typer.Option(..., help="Model ID (HF repo or local path)"),
    cohort: str = typer.Option(..., help="Cohort ID"),
    config: Path = typer.Option(Path("config/default.yaml"), help="Config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a single cohort end-to-end."""
    _setup_logging(verbose)
    cfg = load_config(config)
    apply_cli_overrides(cfg, engine=engine, model=model)
    if cohort not in COHORTS:
        raise typer.BadParameter(f"Unknown cohort '{cohort}'. Known: {sorted(COHORTS)}")

    from .runner import run_cohort
    db_path = asyncio.run(run_cohort(cfg, cohort))
    typer.echo(f"Run complete -> {db_path}")


@app.command()
def sweep(
    engine: str = typer.Option("vllm"),
    model: str = typer.Option(...),
    config: Path = typer.Option(Path("config/default.yaml")),
    cohorts: str = typer.Option(
        ",".join(COHORTS.keys()),
        help="Comma-separated cohort IDs (default: all)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run multiple cohorts back-to-back against the same engine."""
    _setup_logging(verbose)
    cfg = load_config(config)
    apply_cli_overrides(cfg, engine=engine, model=model)
    cohort_ids = [c.strip() for c in cohorts.split(",") if c.strip()]
    for cid in cohort_ids:
        if cid not in COHORTS:
            raise typer.BadParameter(f"Unknown cohort '{cid}'")

    from .runner import run_sweep
    paths = asyncio.run(run_sweep(cfg, cohort_ids))
    typer.echo(f"Sweep complete: {len(paths)} cohort runs")
    for p in paths:
        typer.echo(f"  {p}")


@app.command("launch-engine")
def launch_engine(
    engine: str = typer.Option("vllm"),
    model: str = typer.Option(...),
    config: Path = typer.Option(Path("config/default.yaml")),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Launch the engine without running simulations (manual testing)."""
    _setup_logging(verbose)
    cfg = load_config(config)
    apply_cli_overrides(cfg, engine=engine, model=model)
    from .preflight import preflight_check
    preflight_check(cfg.engine.hardware_requirements)
    eng = make_engine(cfg.engine.type, cfg.engine)
    eng.launch(log_dir=cfg.output.db_directory)
    typer.echo(f"Engine up at {eng.base_url}. Ctrl-C to stop.")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        eng.shutdown()


@app.command()
def dashboard(
    run_dir: Path = typer.Option(Path("runs"), "--run-dir", help="Run directory"),
    refresh: float = typer.Option(2.0, help="Refresh interval (seconds)"),
):
    """Live progress view of the most recent run."""
    from .dashboard import watch
    watch(run_dir=run_dir, refresh_s=refresh)


@app.command("export")
def export_cmd(
    input_dir: Path = typer.Option(Path("runs"), "--input-dir"),
    output: Path = typer.Option(Path("buyer_page_data.json"), "--output"),
):
    """Export simplified JSON for the buyer-facing webpage."""
    from .export import export_dir
    doc = export_dir(input_dir, output)
    typer.echo(f"Exported {doc['meta']['cohort_count']} cohorts -> {output}")


@app.command("list-cohorts")
def list_cohorts():
    """List available cohorts."""
    for cid, cohort in COHORTS.items():
        typer.echo(f"  {cid}: {cohort.name} — {cohort.description}")


@app.command("preflight")
def preflight_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Validate the host satisfies a config's ``hardware_requirements``.

    Run before launching the engine to catch hardware/config mismatches
    without waiting for a long model load to fail. Exits non-zero on
    requirements violation; zero on success or when detection isn't
    available (non-Linux dev hosts soft-skip with a warning).
    """
    import sys
    _setup_logging(verbose)
    cfg = load_config(config)
    from .preflight import detect_hardware, preflight_check, PreflightError
    info = detect_hardware()
    typer.echo(
        f"Detected: vendor={info.vendor} model={info.cpu_model!r} "
        f"physical_cores={info.physical_cores} sockets={info.sockets} "
        f"status={info.detection_status}"
    )
    reqs = cfg.engine.hardware_requirements
    if reqs.is_empty():
        typer.echo("No hardware_requirements set in config — skipping check.")
        return
    typer.echo(
        f"Required: vendor={reqs.cpu_vendor} features={reqs.cpu_features} "
        f"min_cores={reqs.min_physical_cores} min_sockets={reqs.min_sockets}"
    )
    try:
        preflight_check(reqs)
    except PreflightError as e:
        typer.echo(str(e), err=True)
        sys.exit(2)
    typer.echo("Preflight OK.")


@app.command("analyze-prefix-cache")
def analyze_prefix_cache(
    db: Path = typer.Argument(..., help="Path to a cohort run .db"),
    hit_ratio: float = typer.Option(
        0.5, help="A session is a hit if later-turn TTFT median is "
        "<= this fraction of turn-0 TTFT",
    ),
):
    """Compute prefix-cache hit rate from captured turn events.

    Validates the multi-turn premise: turn-N+1 TTFT should be much
    lower than turn-0 TTFT for the same session. If it isn't, the
    engine isn't reusing the prefix and the multi-turn cohorts are
    being measured as if every turn were a fresh request.
    """
    from .prefix_cache import analyse_db
    report = analyse_db(db, hit_ratio=hit_ratio)
    typer.echo(f"Verdict: {report.verdict}")
    typer.echo(f"Sessions analysed: {report.sessions_analysed}")
    typer.echo(f"Overall hit rate: {report.overall_hit_rate:.1%}")
    if report.median_ttft_ratio is not None:
        typer.echo(f"Median later/turn0 TTFT ratio: {report.median_ttft_ratio:.2f}")
    typer.echo("Per persona:")
    for pid, info in sorted(report.per_persona.items()):
        typer.echo(
            f"  {pid:<16} sessions={info['sessions']:<5} "
            f"hit_rate={info['hit_rate']:.1%} median_ratio={info['median_ratio']:.2f}"
        )
    if report.notes:
        typer.echo("Notes: " + "; ".join(report.notes))


if __name__ == "__main__":
    app()
