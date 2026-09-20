"""Command-line interface."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from .config import apply_cli_overrides, load_config
from .engines import make_engine
from .personas import (
    COHORTS,
    PERSONAS,
    cohort_from_persona,
    resolve_workload_group,
)

app = typer.Typer(add_completion=False, help="Persona Capacity Simulator")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _parse_pool_sizes(arg: str | None) -> list[int] | None:
    """Parse a ``--pool-sizes`` CLI value into a list of positive ints.

    Returns ``None`` when ``arg`` is empty/None (adaptive stepper mode).
    Otherwise returns the parsed list, raising BadParameter for non-int
    or non-positive values. Preserves caller order — the runner walks
    the list in order, so the caller controls measurement sequence.
    """
    if not arg:
        return None
    try:
        sizes = [int(s.strip()) for s in arg.split(",") if s.strip()]
    except ValueError as e:
        raise typer.BadParameter(
            f"--pool-sizes must be a comma-separated list of integers: {e}"
        ) from e
    if not sizes:
        raise typer.BadParameter("--pool-sizes is empty")
    if any(n <= 0 for n in sizes):
        raise typer.BadParameter(
            f"--pool-sizes must be positive integers (got {sizes})"
        )
    return sizes


_PROFILE_HELP = (
    "Named hardware profile (curated config from config/profiles/, "
    "with plain config/ stems accepted too). Mutually exclusive with "
    "--config. ``capsim list-profiles`` shows what's available."
)


def _config_from_options(
    config: Path | None, profile: str | None,
    default: Path | None = None,
) -> Path:
    """Resolve the --config/--profile pair to a config path."""
    if config is not None and profile is not None:
        raise typer.BadParameter(
            "--config and --profile are mutually exclusive"
        )
    if profile is not None:
        from .config import resolve_profile
        try:
            return resolve_profile(profile)
        except FileNotFoundError as e:
            raise typer.BadParameter(str(e)) from e
    if config is not None:
        return config
    if default is not None:
        return default
    raise typer.BadParameter("pass --config <path> or --profile <name>")


_POOL_SIZES_HELP = (
    "Override the default fixed grid (powers of 2 from 4 to 256) "
    "with a comma-separated list of pool sizes (e.g. "
    "``8,16,32,64,128,256``). Each is measured in order; the run "
    "still early-stops one step past the first observed failure. "
    "Has no effect with --adaptive."
)

_ADAPTIVE_HELP = (
    "Use the two-knee adaptive stepper (Wilson-CI-aware bisection) "
    "instead of the default fixed-grid sweep. The adaptive stepper "
    "localizes both knees and infills curve density around them; "
    "the fixed grid gives uniform x-axis density across powers of 2. "
    "Pick adaptive when you care about precise knee placement; "
    "pick fixed grid (the default) when you're plotting the curve "
    "downstream and want consistent x-axis sampling."
)


def _resolve_stepper_args(
    pool_sizes: str | None, adaptive: bool
) -> tuple[bool, list[int] | None]:
    """Validate the (adaptive, pool_sizes) flag combo and return
    ``(adaptive, fixed_grid)`` ready to pass to run_cohort/run_sweep.
    Rejects ``--adaptive --pool-sizes ...`` since the grid is
    meaningless when the adaptive stepper is in charge."""
    parsed = _parse_pool_sizes(pool_sizes)
    if adaptive and parsed is not None:
        raise typer.BadParameter(
            "--pool-sizes and --adaptive are mutually exclusive: "
            "--adaptive picks pool sizes via the two-knee stepper, "
            "--pool-sizes only applies to fixed-grid mode."
        )
    return adaptive, parsed


@app.command()
def run(
    cohort: str = typer.Option(..., help="Cohort (team mix) id"),
    config: Path = typer.Option(None, help="Config file"),
    profile: str = typer.Option(None, "--profile", help=_PROFILE_HELP),
    engine: str = typer.Option(None, help="Override engine.type from CONFIG (vllm | sglang)"),
    model: str = typer.Option(None, help="Override engine.model_id from CONFIG"),
    pool_sizes: str = typer.Option(None, "--pool-sizes", help=_POOL_SIZES_HELP),
    adaptive: bool = typer.Option(False, "--adaptive", help=_ADAPTIVE_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a single cohort (team mix) end-to-end."""
    _setup_logging(verbose)
    cfg = load_config(_config_from_options(config, profile, Path("config/default.yaml")))
    apply_cli_overrides(cfg, engine=engine, model=model)
    if cohort not in COHORTS:
        raise typer.BadParameter(f"Unknown cohort '{cohort}'. Known: {sorted(COHORTS)}")

    use_adaptive, grid = _resolve_stepper_args(pool_sizes, adaptive)
    from .runner import run_cohort
    db_path = asyncio.run(run_cohort(
        cfg, cohort,
        adaptive=use_adaptive,
        fixed_grid_pool_sizes=grid,
    ))
    typer.echo(f"Run complete -> {db_path}")


@app.command("run-persona")
def run_persona_cmd(
    persona: str = typer.Option(..., help="Persona id (single user archetype)"),
    config: Path = typer.Option(None, help="Config file"),
    profile: str = typer.Option(None, "--profile", help=_PROFILE_HELP),
    engine: str = typer.Option(None, help="Override engine.type from CONFIG"),
    model: str = typer.Option(None, help="Override engine.model_id from CONFIG"),
    pool_sizes: str = typer.Option(None, "--pool-sizes", help=_POOL_SIZES_HELP),
    adaptive: bool = typer.Option(False, "--adaptive", help=_ADAPTIVE_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a single persona end-to-end (one user archetype, no team mix).

    Useful for measuring each archetype's individual capacity so multi-
    persona cohort results can be decomposed: if software_engineering
    underperforms, you can compare against the code_assist persona run
    to see whether the mix itself is producing interference."""
    _setup_logging(verbose)
    cfg = load_config(_config_from_options(config, profile, Path("config/default.yaml")))
    apply_cli_overrides(cfg, engine=engine, model=model)
    if persona not in PERSONAS:
        raise typer.BadParameter(
            f"Unknown persona '{persona}'. Known: {sorted(PERSONAS)}"
        )

    use_adaptive, grid = _resolve_stepper_args(pool_sizes, adaptive)
    from .runner import run_cohort
    db_path = asyncio.run(run_cohort(
        cfg, cohort_from_persona(persona),
        adaptive=use_adaptive,
        fixed_grid_pool_sizes=grid,
    ))
    typer.echo(f"Run complete -> {db_path}")


@app.command()
def sweep(
    config: Path = typer.Option(None),
    profile: str = typer.Option(None, "--profile", help=_PROFILE_HELP),
    engine: str = typer.Option(None, help="Override engine.type from CONFIG"),
    model: str = typer.Option(None, help="Override engine.model_id from CONFIG"),
    type: str = typer.Option(
        "all",
        "--type",
        help=(
            "What to sweep: 'all' (every persona + every cohort), "
            "'personas' (each persona alone), 'cohorts' (every team mix), "
            "or a comma-separated list of persona/cohort ids "
            "(``quick_lookup,chat_heavy``)."
        ),
    ),
    new_run: bool = typer.Option(
        False, "--new-run",
        help=(
            "Start a fresh run_NN+1 directory. Default behavior is to "
            "reuse the latest run_NN and resume — skipping personas/"
            "cohorts that already have final_status='ok' inside it. "
            "Use --new-run when you've changed config/hardware and the "
            "previous run's data should NOT be merged with the new one."
        ),
    ),
    pool_sizes: str = typer.Option(None, "--pool-sizes", help=_POOL_SIZES_HELP),
    adaptive: bool = typer.Option(False, "--adaptive", help=_ADAPTIVE_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run multiple personas + cohorts back-to-back against one engine.

    Personas run first as ephemeral one-persona Cohorts, then cohorts.
    The engine launches once and stays up for the whole sweep.

    Resumes the latest ``runs/run_NN/`` by default — interrupted sweeps
    pick up where they left off automatically. Pass ``--new-run`` to
    cut a fresh ``run_NN+1`` directory."""
    _setup_logging(verbose)
    cfg = load_config(_config_from_options(config, profile, Path("config/default.yaml")))
    apply_cli_overrides(cfg, engine=engine, model=model)
    try:
        persona_ids, cohort_ids = resolve_workload_group(type)
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e
    if not persona_ids and not cohort_ids:
        raise typer.BadParameter(f"Nothing resolved from --type={type!r}")

    use_adaptive, grid = _resolve_stepper_args(pool_sizes, adaptive)
    from .runner import run_sweep
    paths = asyncio.run(run_sweep(
        cfg, persona_ids=persona_ids, cohort_ids=cohort_ids, new_run=new_run,
        adaptive=use_adaptive,
        fixed_grid_pool_sizes=grid,
    ))
    typer.echo(
        f"Sweep complete: {len(paths)} runs "
        f"({len(persona_ids)} personas + {len(cohort_ids)} cohorts)"
    )
    for p in paths:
        typer.echo(f"  {p}")


@app.command("spot-check")
def spot_check(
    plan: Path = typer.Option(
        ...,
        "--plan",
        help=(
            "JSON plan produced by ``scripts/audit_run.py``. "
            "Contains ``rerun_points = [{cohort_id, cohort_run_id, "
            "pool_size, ...}, ...]``."
        ),
    ),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help=(
            "The run directory whose run.db holds the cohort_run rows "
            "to enrich (e.g. runs/run_09). The engine config is loaded "
            "from the existing cohort_run rows so the spot-check uses "
            "the SAME engine + model + KV size as the original run — "
            "any other config would produce non-comparable measurements."
        ),
    ),
    config: Path = typer.Option(
        None,
        help=(
            "Optional: override the engine config (DANGEROUS — emits "
            "a warning, since the spot-check measurements would no "
            "longer be comparable to the original cohort_run's curve)."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Re-measure specific (cohort, pool_size) points from an audit
    plan, appending them to the existing cohort_run rows.

    Workflow:

        python scripts/audit_run.py runs/run_09
        # → writes runs/run_09/audit_report.json

        python -m simulator.cli spot-check \\
            --run-dir runs/run_09 \\
            --plan runs/run_09/audit_report.json

        make export       # picks up the enriched curves
    """
    import json
    _setup_logging(verbose)

    if not plan.exists():
        raise typer.BadParameter(f"Plan file not found: {plan}")
    if not run_dir.exists():
        raise typer.BadParameter(f"Run dir not found: {run_dir}")
    plan_doc = json.loads(plan.read_text())
    if not plan_doc.get("rerun_points"):
        typer.echo("No rerun points in plan — nothing to do.")
        return

    # Default: reconstruct the engine config from the run.db so the
    # spot-check uses the EXACT engine / model / KV pool / quantization
    # the original sweep used. Otherwise data isn't comparable.
    if config is None:
        from .runner import load_config_from_run_dir
        cfg = load_config_from_run_dir(run_dir)
        typer.echo(
            f"spot-check: loaded engine config from {run_dir}/run.db "
            f"(engine.type={cfg.engine.type}, model={cfg.engine.model_id})"
        )
    else:
        typer.echo(
            f"WARNING: spot-check is using --config={config!s} instead "
            f"of the config stored on the original cohort_run. "
            f"Measurements will only be comparable if this matches the "
            f"original engine launch parameters."
        )
        cfg = load_config(config)

    from .runner import run_spot_check
    paths = asyncio.run(run_spot_check(cfg, plan_doc, run_dir=run_dir))
    typer.echo(
        f"Spot-check complete: enriched {len(paths)} cohort_run row(s) "
        f"with {len(plan_doc['rerun_points'])} extra measurement(s)"
    )


@app.command("launch-engine")
def launch_engine(
    config: Path = typer.Option(Path("config/default.yaml")),
    engine: str = typer.Option(None, help="Override engine.type from CONFIG"),
    model: str = typer.Option(None, help="Override engine.model_id from CONFIG"),
    new_run: bool = typer.Option(
        False, "--new-run",
        help="Write engine log into a fresh run_NN+1 directory.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Launch the engine without running simulations (manual testing)."""
    _setup_logging(verbose)
    cfg = load_config(config)
    apply_cli_overrides(cfg, engine=engine, model=model)
    from .preflight import preflight_check
    from .runs import resolve_run_dir
    preflight_check(cfg.engine.hardware_requirements)
    run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    eng = make_engine(cfg.engine.type, cfg.engine)
    eng.launch(log_dir=run_dir)
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
    output: Path = typer.Option(
        None, "--output",
        help=(
            "Override output path. Default: <input_dir>/run_NN/"
            "buyer_page_data.json (or buyer_page_data_slim.json with "
            "--slim) — lands the JSON alongside the run.db it was "
            "built from so per-run artifacts stay grouped."
        ),
    ),
    slim: bool = typer.Option(
        False, "--slim",
        help=(
            "Summary-only export (~99% smaller). Drops per-step "
            "telemetry samples + turn events and the cohort-level "
            "1 Hz heartbeat. Headline numbers, per-step rollup, "
            "landing zones, bottleneck attribution, and prefix-cache "
            "verdict all stay. Use --slim for buyer-page summary "
            "distribution; use the full export when drilling into "
            "per-step diagnostics."
        ),
    ),
):
    """Export simplified JSON for the buyer-facing webpage."""
    from .export import export_dir
    doc, out_path = export_dir(input_dir, output, slim=slim)
    size_kb = out_path.stat().st_size // 1024
    typer.echo(
        f"Exported {doc['meta']['cohort_count']} cohorts -> {out_path} "
        f"({size_kb} KB{'  [slim]' if slim else ''})"
    )


@app.command("list-profiles")
def list_profiles_cmd():
    """List available hardware profiles (curated config/profiles/ plus
    plain config/ stems). Pass a name to --profile on ready / smoke /
    run / run-persona / sweep."""
    from .config import list_profiles
    profiles = list_profiles()
    if not profiles:
        typer.echo(
            "No profiles found — run from the repo root (profiles live "
            "in config/profiles/ and config/)."
        )
        raise typer.Exit(1)
    typer.echo("Profiles (--profile <name>):\n")
    for name, path in profiles.items():
        typer.echo(f"  {name:<40} {path}")


@app.command("list-cohorts")
def list_cohorts():
    """List available team-mix cohorts."""
    typer.echo("Cohorts (team mixes — pass to --cohort or sweep --type cohorts):\n")
    for cid, cohort in COHORTS.items():
        weights_str = ", ".join(
            f"{p}:{w:.0%}" for p, w in cohort.persona_weights.items()
        )
        typer.echo(f"  {cid}")
        typer.echo(f"    {cohort.name} — {cohort.description}")
        typer.echo(f"    weights: {weights_str}")


@app.command("list-personas")
def list_personas():
    """List available user-archetype personas."""
    import math
    typer.echo(
        "Personas (single user archetypes — pass to --persona or "
        "sweep --type personas):\n"
    )
    for pid, persona in PERSONAS.items():
        typer.echo(f"  {pid}")
        typer.echo(f"    {persona.description}")
        typer.echo(
            f"    SLA TTFT:  target ≤{persona.ttft_target_seconds:.0f}s, "
            f"failure ≤{persona.ttft_failure_seconds:.0f}s"
        )
        typer.echo(
            f"    SLA TPOT:  target ≤{persona.tpot_target_ms:.0f}ms, "
            f"failure ≤{persona.tpot_failure_ms:.0f}ms"
        )
        # Median post-response delay = read residual + active think.
        read_med = math.exp(persona.read_time_seconds.mu)
        think_med = math.exp(persona.active_think_seconds.mu)
        typer.echo(
            f"    Post-response: ~{read_med:.0f}s reading + "
            f"~{think_med:.0f}s deliberation = ~{read_med+think_med:.0f}s"
        )


@app.command("ready")
def ready_cmd(
    config: Path = typer.Option(None, "--config", "-c", help="Config file"),
    profile: str = typer.Option(None, "--profile", help=_PROFILE_HELP),
    skip_build: bool = typer.Option(False, "--skip-build", help="Skip docker image build"),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip model download"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Prepare engine + model + host for a config.

    Idempotent — safe to re-run. Skips work that's already done:
      * SGLang Docker images: built only if missing
      * model dir: downloaded only if absent or empty
      * preflight: always validated

    Heavy lifting (docker build / hf download) prints progress to stdout
    and aborts with a non-zero exit code on failure.
    """
    import sys

    _setup_logging(verbose)
    config = _config_from_options(config, profile)
    cfg = load_config(config)

    typer.echo(f"==> Preparing for {config}")
    typer.echo(f"    engine={cfg.engine.type} model={cfg.engine.model_id}")

    # 1. Engine-specific image prep.
    if cfg.engine.type == "sglang" and not skip_build:
        _ensure_sglang_image(cfg.engine.docker_image)
    elif cfg.engine.type == "vllm_dual_socket" and not skip_build:
        _ensure_pulled_image(cfg.engine.vllm_image)
    elif cfg.engine.type == "vllm_cuda" and not skip_build:
        _ensure_pulled_image(cfg.engine.gpu_image)

    # 2. Model staging (only when the config points at a local mount).
    if not skip_download:
        _ensure_model(cfg)

    # 3. Preflight — last so we fail with the freshest hardware view.
    from .preflight import PreflightError, preflight_check
    try:
        preflight_check(cfg.engine.hardware_requirements)
    except PreflightError as e:
        typer.echo(str(e), err=True)
        sys.exit(2)

    typer.echo(f"==> Ready: {config}")


def _ensure_pulled_image(image: str) -> None:
    """For images that exist on a public registry (no local build needed),
    just ``docker pull`` if absent."""
    import subprocess
    inspect = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True,
    )
    if inspect.returncode == 0:
        typer.echo(f"==> Docker image {image} present, skipping pull")
        return
    typer.echo(f"==> Pulling {image}")
    subprocess.run(["docker", "pull", image], check=True)


def _ensure_sglang_image(image: str) -> None:
    """Build the SGLang CPU image only if it's not already present."""
    import os
    import subprocess
    from pathlib import Path

    # Already built?
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if inspect.returncode == 0:
        typer.echo(f"==> Docker image {image} present, skipping build")
        return

    typer.echo(f"==> Docker image {image} missing — building from source")
    sglang_src = Path(os.environ.get("SGLANG_SRC", "/tmp/sglang"))
    sglang_repo = os.environ.get(
        "SGLANG_REPO", "https://github.com/sgl-project/sglang.git"
    )
    base_image = os.environ.get("SGLANG_BASE_IMAGE", "sglang-cpu:xeon")

    if (sglang_src / ".git").exists():
        typer.echo(f"==> Updating {sglang_src}")
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin"],
            cwd=sglang_src, check=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", "origin/HEAD"],
            cwd=sglang_src, check=True,
        )
    else:
        typer.echo(f"==> Cloning {sglang_repo} -> {sglang_src}")
        subprocess.run(
            ["git", "clone", "--depth", "1", sglang_repo, str(sglang_src)],
            check=True,
        )

    dockerfile = sglang_src / "docker" / "xeon.Dockerfile"
    if not dockerfile.exists():
        raise typer.BadParameter(
            f"{dockerfile} not found — inspect {sglang_src}/docker/ for the "
            f"current CPU dockerfile name"
        )

    typer.echo(f"==> Building {base_image} (~15-20 min first run)")
    subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", base_image, str(sglang_src)],
        check=True,
    )
    typer.echo(f"==> Building {image}")
    subprocess.run(
        ["docker", "build", "-f", "Dockerfile.xeon-fixed", "-t", image, "."],
        check=True,
    )


def _ensure_model(cfg) -> None:
    """Download model weights if the host-side dir is missing or empty."""
    import os
    import subprocess
    from pathlib import Path

    container_path = cfg.engine.model_local_path
    if not container_path:
        # vLLM path or HF-id-only config — no local mount to validate.
        return

    # Resolve host_dir from the docker_volumes mapping.
    container_root = "/" + container_path.lstrip("/").split("/")[0]
    host_root = next(
        (h for h, c in (cfg.engine.docker_volumes or {}).items()
         if c == container_root),
        None,
    )
    if host_root is None:
        typer.echo(
            f"WARN: no docker_volume mounts {container_root}; "
            f"can't validate model dir on host"
        )
        return

    rel = container_path[len(container_root):].lstrip("/")
    host_dir = Path(host_root) / rel

    if host_dir.exists() and any(host_dir.iterdir()):
        typer.echo(f"==> Model present at {host_dir}")
        return

    typer.echo(f"==> Model missing at {host_dir} — downloading {cfg.engine.model_id}")
    host_dir.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # Use the venv's ``hf`` binary, not whatever's on PATH. The user
    # may have a ``~/.local/bin/hf`` left over from a different Python
    # which now ImportErrors on huggingface_hub.
    import sys
    hf_bin = Path(sys.executable).parent / "hf"
    if not hf_bin.exists():
        # Fall back to module invocation — same Python, no PATH ambiguity.
        cmd = [sys.executable, "-m", "huggingface_hub.cli.hf", "download",
               cfg.engine.model_id, "--local-dir", str(host_dir)]
    else:
        cmd = [str(hf_bin), "download", cfg.engine.model_id,
               "--local-dir", str(host_dir)]
    subprocess.run(cmd, env=env, check=True)
    typer.echo(f"==> Downloaded to {host_dir}")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Bind address. Localhost by default — the service has no "
             "auth; open it up only on a network you trust.",
    ),
    port: int = typer.Option(8321, "--port"),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir",
        help="Base directory the service reads runs from and starts "
             "runs into.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the control-plane service: run lifecycle over HTTP + live
    telemetry over WebSocket.

    Endpoints under /api (status, profiles, personas, cohorts, runs,
    doctor, export) plus /ws/telemetry streaming the event bus
    (run / snapshot / telemetry / turn / step events) to any client.
    Runs execute in-process — one at a time — via the same run_cohort /
    run_sweep the CLI uses; run.db stays the source of truth.

    Try it:  capsim serve  &  then
             curl -X POST localhost:8321/api/runs -H 'content-type: application/json' \\
                  -d '{"profile": "mock", "workload": {"kind": "cohort", "id": "chat_heavy"}}'
    """
    _setup_logging(verbose)
    from .service import serve
    typer.echo(f"capsim service on http://{host}:{port}  (runs dir: {runs_dir})")
    serve(host=host, port=port, runs_base=runs_dir)


@app.command("doctor")
def doctor_cmd(
    output: Path = typer.Option(
        Path("runs/doctor.json"), "--output", "-o",
        help="Where to write the machine-readable report (default: runs/doctor.json, "
             "next to the run artifacts and outside version control).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Full host validation for a fresh benchmark box.

    Checks CPU capabilities, NUMA, Docker, GPU stack (nvidia-smi +
    container toolkit), disk space, telemetry permissions (PMU, RAPL —
    warn-only, collectors degrade gracefully), and Hugging Face
    reachability. Prints a pass/warn/fail table, recommends candidate
    configs for the detected hardware, writes ``runs/doctor.json``, and
    exits non-zero when any check fails.

    Landing flow on a new host: install → ``capsim doctor`` →
    ``capsim smoke`` → ``capsim ready --config <recommended>``.
    """
    import sys

    from rich.console import Console
    from rich.table import Table

    from .doctor import run_doctor

    _setup_logging(verbose)
    report = run_doctor()

    styles = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}
    table = Table(title="capsim doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for c in report.checks:
        table.add_row(c.name, f"[{styles[c.status]}]{c.status}[/]", c.detail)
    Console().print(table)

    output.write_text(json.dumps(report.to_dict(), indent=2))
    typer.echo(f"Report written to {output}")
    if report.failed:
        typer.echo("doctor: FAIL — fix the failing checks above before running benchmarks", err=True)
        sys.exit(1)
    typer.echo("doctor: host looks usable" + (
        f" — try: capsim smoke --profile {report.recommended_configs[0]}"
        if report.recommended_configs else ""
    ))


@app.command("smoke")
def smoke_cmd(
    config: Path = typer.Option(
        None, "--config", "-c",
        help="Config whose engine/hardware shape to smoke-test.",
    ),
    profile: str = typer.Option(None, "--profile", help=_PROFILE_HELP),
    model: str = typer.Option(
        "Qwen/Qwen3-0.6B", "--model",
        help="Tiny stand-in model (~1.5 GB) — proves the pipeline "
             "before committing to a 30B download.",
    ),
    run_dir: Path = typer.Option(
        Path("runs/smoke"), "--run-dir",
        help="Isolated base directory for smoke artifacts (kept out of "
             "the real runs/ history).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """End-to-end micro-benchmark: launch the real engine with a tiny
    model, run 2 virtual users through a short measured window, export,
    and validate the export against the schema contract.

    Passing smoke means the whole pipeline works on this host — engine
    launch, streaming, telemetry capture, DB write, export contract.
    Takes ~10 minutes on a fresh box (dominated by the model download
    and engine start), a couple of minutes when re-run.
    """
    import sys

    _setup_logging(verbose)
    config = _config_from_options(config, profile)
    cfg = load_config(config)
    apply_cli_overrides(cfg, model=model)

    # Point any pre-staged-weights path at a smoke-specific dir under
    # the same container mount so the big model's dir is untouched and
    # _ensure_model downloads the tiny model to the right host path.
    if cfg.engine.model_local_path:
        container_root = "/" + cfg.engine.model_local_path.lstrip("/").split("/")[0]
        cfg.engine.model_local_path = (
            f"{container_root}/smoke/{model.split('/')[-1]}"
        )

    # Shrink the run: one pool size, few samples, short windows.
    sim = cfg.simulation
    sim.target_samples_per_step = 8
    sim.warmup_min_duration_s = 5
    sim.warmup_max_duration_s = 30
    sim.convergence_window_s = 10
    sim.measurement_timeout_s = 240
    sim.max_total_duration_minutes = 15
    sim.request_timeout_s = 120
    cfg.output.db_directory = str(run_dir)

    gates: list[tuple[str, bool, str]] = []

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append((name, passed, detail))
        typer.echo(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    typer.echo(f"==> smoke: engine={cfg.engine.type} model={model}")
    _ensure_model(cfg)

    from .preflight import PreflightError, preflight_check
    try:
        preflight_check(cfg.engine.hardware_requirements)
    except PreflightError as e:
        typer.echo(str(e), err=True)
        sys.exit(2)

    from .runner import run_cohort
    db_path = None
    try:
        db_path = asyncio.run(run_cohort(
            cfg, cohort_from_persona("quick_lookup"),
            fixed_grid_pool_sizes=[2],
            new_run=True,
        ))
    except Exception as e:  # noqa: BLE001
        gate("engine + run", False, f"{type(e).__name__}: {e}")

    typer.echo("==> smoke gates")
    if db_path is not None:
        from .database import Database
        db = Database(db_path)
        row = db.fetchone(
            "SELECT final_status, collectors_json FROM cohort_run "
            "ORDER BY started_at DESC LIMIT 1"
        )
        samples = db.fetchone(
            "SELECT COALESCE(SUM(sample_size), 0) AS n FROM cohort_measurements"
        )
        db.close()
        status = row["final_status"] if row else None
        gate("engine + run", status == "ok", f"final_status={status}")
        gate(
            "streaming samples", (samples["n"] or 0) > 0,
            f"{samples['n']} turn(s) measured",
        )
        collectors = {}
        if row and row["collectors_json"]:
            collectors = json.loads(row["collectors_json"])
        ok_collectors = sorted(k for k, v in collectors.items() if v == "ok")
        # Warn-only: a box without perf/RAPL access still passes smoke.
        typer.echo(
            f"  [INFO] telemetry collectors ok: "
            f"{', '.join(ok_collectors) or 'none'} "
            f"(statuses: {collectors or 'n/a'})"
        )

        from .export import export_dir, validate_export
        doc, out_path = export_dir(run_dir)
        errors = validate_export(doc)
        gate(
            "export contract", not errors,
            f"{out_path}" if not errors else "; ".join(errors[:3]),
        )

    failed = [name for name, passed, _ in gates if not passed]
    if failed:
        typer.echo(f"smoke: FAIL ({', '.join(failed)})", err=True)
        sys.exit(1)
    typer.echo(
        "smoke: PASS — pipeline verified end-to-end. Next: "
        f"capsim ready --config {config} (full model), then run a sweep."
    )


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
    from .preflight import PreflightError, detect_hardware, preflight_check
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


@app.command("current-run-dir")
def current_run_dir_cmd(
    base: Path = typer.Option(Path("runs"), "--base", help="Runs base directory"),
    new_run: bool = typer.Option(
        False, "--new-run", help="Create the next run_NN+1 instead of reusing the latest",
    ),
):
    """Print the run_NN directory the next invocation would use.

    Creates the directory as a side-effect (so the make-side bg targets
    can ``mkdir -p`` then drop a log into it). With ``--new-run`` it
    advances to ``run_NN+1``; otherwise it returns the latest existing
    one (or creates ``run_01``).
    """
    from .runs import resolve_run_dir
    typer.echo(str(resolve_run_dir(base, new=new_run)))


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
