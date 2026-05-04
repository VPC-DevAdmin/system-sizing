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
        out["ok"] = True
    finally:
        conn.close()
    return out


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

    header_text = Text()
    header_text.append("Cohort: ", style="bold")
    header_text.append(f"{run['cohort_id']}    ")
    header_text.append("Engine: ", style="bold")
    header_text.append(f"{run['engine_type']}    ")
    header_text.append("Model: ", style="bold")
    header_text.append(f"{run['model_id']}    ")
    header_text.append("Status: ", style="bold")
    header_text.append(str(run.get("final_status") or "running"),
                       style="green" if run.get("final_status") in (None, "ok") else "yellow")
    layout["header"].update(Panel(header_text, border_style="cyan"))

    # Live table of state
    state_tbl = Table(title="Live state", show_header=False, expand=True)
    state_tbl.add_column("k", style="bold")
    state_tbl.add_column("v")
    state_tbl.add_row("Phase", str(snap.get("phase", "—")))
    state_tbl.add_row("Pool size (target)", str(snap.get("pool_size", "—")))
    state_tbl.add_row("In-flight", str(snap.get("in_flight", "—")))
    state_tbl.add_row("Completed", str(snap.get("requests_completed", "—")))
    state_tbl.add_row("Errors", str(snap.get("errors", "—")))

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


def watch(run_dir: str | Path = "runs", refresh_s: float = 2.0) -> None:
    run_dir = Path(run_dir)
    console = Console()
    db_path = _latest_db(run_dir)
    if db_path is None:
        console.print(f"[yellow]No .db files in {run_dir}. Start a run first.[/yellow]")
        return

    with Live(_render(_read_state(db_path)), refresh_per_second=1, console=console, screen=True) as live:
        while True:
            try:
                # Re-detect latest db each refresh in case a sweep starts a new file
                latest = _latest_db(run_dir) or db_path
                state = _read_state(latest)
                live.update(_render(state))
                time.sleep(refresh_s)
            except KeyboardInterrupt:
                break
