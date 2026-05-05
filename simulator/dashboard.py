"""Live progress dashboard.

Watches the most-recently-modified .db in the run directory and renders a
periodically refreshing view of: current phase, pool size, in-flight,
completed measurements, and the violation curve so far.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _latest_db(run_dir: Path) -> Path | None:
    """Locate the most recently modified .db.

    With the run_NN/ layout, ``run_dir`` is the base ``runs/`` dir; we
    look in the latest ``run_NN``. Fall back to a flat-dir glob so old
    layouts and ad-hoc paths still work.
    """
    from .runs import latest_run_dir
    if not run_dir.exists():
        return None
    target = latest_run_dir(run_dir) or run_dir
    dbs = sorted(target.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dbs[0] if dbs else None


def _read_state(path: Path) -> dict:
    out: dict = {"path": str(path), "ok": False}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        out["error"] = str(e)
        return out

    try:
        run = conn.execute(
            "SELECT * FROM cohort_run ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if run is None:
            return out
        out["run"] = dict(run)

        latest_snap = conn.execute(
            """SELECT * FROM simulation_snapshots
               WHERE cohort_run_id = ?
               ORDER BY snapshot_at_ms DESC LIMIT 1""",
            (run["cohort_run_id"],),
        ).fetchone()
        out["snapshot"] = dict(latest_snap) if latest_snap else None

        measurements = conn.execute(
            """SELECT step_index, target_pool_size, sample_size,
                      combined_violation_rate, ttft_p95_ms, tpot_p95_ms,
                      capacity_status
               FROM cohort_measurements
               WHERE cohort_run_id = ?
               ORDER BY step_index ASC""",
            (run["cohort_run_id"],),
        ).fetchall()
        out["measurements"] = [dict(m) for m in measurements]

        # Sweep-level tally so the dashboard can say "11/11 complete"
        # when the whole run is done, not just the latest cohort.
        sweep = conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN final_status = 'ok' THEN 1 ELSE 0 END) AS ok,
                 SUM(CASE WHEN final_status IS NOT NULL
                          AND final_status != 'ok' THEN 1 ELSE 0 END) AS other_terminal,
                 SUM(CASE WHEN final_status IS NULL THEN 1 ELSE 0 END) AS in_progress
               FROM cohort_run"""
        ).fetchone()
        out["sweep"] = dict(sweep) if sweep else None
        out["ok"] = True
    finally:
        conn.close()
    return out


# Mapping from cohort_run.final_status values to a single-word
# phase label the dashboard renders when a cohort is terminal.
_FINAL_STATUS_DISPLAY = {
    "ok": "completed",
    "interrupted": "interrupted",
    "time_limit": "time-limit",
    "no_samples": "no samples",
    "unstable": "unstable",
}


def _status_style(status: str | None) -> str:
    if status in (None, "ok"):
        return "green"
    if status in ("interrupted", "time_limit"):
        return "yellow"
    return "red"


def _render(state: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    if not state.get("ok"):
        layout["header"].update(Panel(
            Text("Persona Capacity Simulator — Dashboard", style="bold"), border_style="cyan"))
        layout["body"].update(Panel(
            Text(state.get("error", "No active run database found in run dir."), style="yellow")))
        layout["footer"].update(Panel(Text(time.strftime("%H:%M:%S"))))
        return layout

    run = state["run"]
    snap = state.get("snapshot") or {}
    measurements = state.get("measurements") or []
    sweep = state.get("sweep") or {}
    final_status = run.get("final_status")
    is_terminal = final_status is not None

    header_text = Text()
    header_text.append("Cohort: ", style="bold")
    header_text.append(f"{run['cohort_id']}    ")
    header_text.append("Engine: ", style="bold")
    header_text.append(f"{run['engine_type']}    ")
    header_text.append("Model: ", style="bold")
    header_text.append(f"{run['model_id']}    ")
    header_text.append("Status: ", style="bold")
    header_text.append(str(final_status or "running"), style=_status_style(final_status))
    layout["header"].update(Panel(header_text, border_style="cyan"))

    # Live table of state
    state_tbl = Table(title="Live state", show_header=False, expand=True)
    state_tbl.add_column("k", style="bold")
    state_tbl.add_column("v")
    # Phase: when the cohort is terminal, ``final_status`` is the
    # source of truth — the latest snapshot's ``phase`` column may
    # be a stale "measuring" from before the SnapshotRecorder shut
    # down. Show the final status mapped to a phase-style label.
    if is_terminal:
        phase_label = _FINAL_STATUS_DISPLAY.get(final_status, final_status)
        state_tbl.add_row(
            "Phase",
            Text(str(phase_label), style=_status_style(final_status)),
        )
    else:
        state_tbl.add_row("Phase", str(snap.get("phase", "—")))
    state_tbl.add_row("Pool size (target)", str(snap.get("pool_size", "—")))
    state_tbl.add_row("In-flight", "—" if is_terminal else str(snap.get("in_flight", "—")))
    # Step samples shows progress into the current measurement window
    # (the per-step sample buffer). Target=0 means we're not in a
    # measuring phase right now (warmup / ramp / idle), so render "—".
    if is_terminal:
        state_tbl.add_row("Step samples", "—")
    else:
        step_n = snap.get("step_samples")
        step_target = snap.get("step_target_samples")
        if step_target:
            state_tbl.add_row("Step samples", f"{step_n or 0} / {step_target}")
        else:
            state_tbl.add_row("Step samples", "—")
    state_tbl.add_row("Completed (all-time)", str(snap.get("requests_completed", "—")))
    state_tbl.add_row("Errors", str(snap.get("errors", "—")))
    # Sweep progress across all cohort_runs in this run.db.
    if sweep and sweep.get("total"):
        ok = sweep.get("ok") or 0
        other = sweep.get("other_terminal") or 0
        in_progress = sweep.get("in_progress") or 0
        total = sweep["total"]
        if in_progress == 0:
            sweep_label = f"{ok}/{total} complete"
            if other:
                sweep_label += f" ({other} non-ok)"
            sweep_style = _status_style(None if other == 0 else "interrupted")
        else:
            sweep_label = f"{ok}/{total} complete  ·  {in_progress} running"
            sweep_style = ""
        state_tbl.add_row("Sweep progress", Text(sweep_label, style=sweep_style))

    # Measurements table
    m_tbl = Table(title="Measurement points", expand=True)
    m_tbl.add_column("step", justify="right")
    m_tbl.add_column("pool", justify="right")
    m_tbl.add_column("n", justify="right")
    m_tbl.add_column("viol %", justify="right")
    m_tbl.add_column("ttft p95", justify="right")
    m_tbl.add_column("tpot p95", justify="right")
    m_tbl.add_column("status")
    for m in measurements[-15:]:
        viol = m["combined_violation_rate"] * 100 if m["combined_violation_rate"] is not None else 0
        status = m["capacity_status"] or "—"
        style = {
            "pass": "green",
            "marginal": "yellow",
            "fail": "red",
            "unstable": "magenta",
        }.get(status, "")
        m_tbl.add_row(
            str(m["step_index"]),
            str(m["target_pool_size"]),
            str(m["sample_size"]),
            f"{viol:.1f}",
            f"{m['ttft_p95_ms']:.0f}" if m["ttft_p95_ms"] else "—",
            f"{m['tpot_p95_ms']:.0f}" if m["tpot_p95_ms"] else "—",
            Text(status, style=style),
        )

    body = Layout()
    body.split_row(Layout(state_tbl, name="left", ratio=1), Layout(m_tbl, name="right", ratio=2))
    layout["body"].update(body)

    footer = Text()
    footer.append("DB: ", style="bold")
    footer.append(state["path"])
    footer.append(f"   Refreshed: {time.strftime('%H:%M:%S')}", style="dim")
    layout["footer"].update(Panel(footer))
    return layout


def _render_waiting(run_dir: Path, elapsed_s: float) -> Layout:
    """Pre-DB render: a friendly waiting screen while we poll for the
    run.db to appear. Used during engine boot (loads can take 1-3 min
    on a cold cache) and between cohorts in a sweep."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["header"].update(Panel(
        Text("Persona Capacity Simulator — Dashboard", style="bold cyan"),
        border_style="cyan",
    ))
    body = Text()
    body.append("\n  Waiting for ", style="yellow")
    body.append(f"{run_dir}/run_NN/run.db", style="bold yellow")
    body.append(" to appear...\n\n", style="yellow")
    body.append(
        f"  This is normal during engine boot (~1-3 min cold load) "
        f"or between cohort transitions.\n", style="dim",
    )
    body.append(f"  Polling every 2s. Ctrl-C to exit.\n\n", style="dim")
    body.append(f"  Elapsed: {int(elapsed_s)}s\n", style="bold")
    layout["body"].update(Panel(body, title="Waiting for engine"))
    layout["footer"].update(Panel(
        Text(f"  {time.strftime('%H:%M:%S')}", style="dim"),
    ))
    return layout


def watch(run_dir: str | Path = "runs", refresh_s: float = 2.0) -> None:
    """Live dashboard with auto-wait: polls until a run.db materialises,
    then renders the running cohort's progress. Survives the entire
    "make run-sweep" lifecycle — boot wait → measurement → between-
    cohort transitions → sweep complete — without the user needing
    to restart.
    """
    run_dir = Path(run_dir)
    console = Console()
    waiting_since = time.monotonic()
    db_path: Path | None = _latest_db(run_dir)

    # Initial render: waiting screen if no DB yet, else the dashboard.
    initial = (
        _render(_read_state(db_path)) if db_path is not None
        else _render_waiting(run_dir, 0.0)
    )

    with Live(
        initial, refresh_per_second=1, console=console, screen=True,
    ) as live:
        while True:
            try:
                if db_path is None:
                    # Pre-DB phase: keep polling, update elapsed time.
                    db_path = _latest_db(run_dir)
                    if db_path is None:
                        elapsed = time.monotonic() - waiting_since
                        live.update(_render_waiting(run_dir, elapsed))
                        time.sleep(refresh_s)
                        continue
                # Have a DB — render dashboard. Re-detect each tick
                # in case a new sweep starts a new run_NN/run.db.
                latest = _latest_db(run_dir) or db_path
                state = _read_state(latest)
                live.update(_render(state))
                time.sleep(refresh_s)
            except KeyboardInterrupt:
                break
