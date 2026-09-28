"""Read-only evidence bridge for the parallel workspace UI.

Resolve a result from persisted Roofline state, never an arbitrary client path.
Nested candidate runs do not appear in the top-level run picker.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


def _read_evidence(base: Path, model: str, engine: str, attempt: int | None) -> dict:
    from ..roofline import load_state

    state = load_state(base / "roofline.json")
    if state is None:
        raise HTTPException(404, "No Roofline investigation is recorded")
    matches = [r for r in state.results if r.get("model") == model
               and r.get("engine") == engine]
    if attempt is not None:
        if attempt >= len(matches):
            raise HTTPException(404, "Attempt no longer exists; refresh the investigation")
        selected = matches[attempt]
    else:
        selected = (state.to_dict()["summary"].get("best_per_pair", {})
                    .get(model, {}).get(engine))
    if not selected or not selected.get("run_dir"):
        raise HTTPException(404, "No recorded run directory for this result")

    root = base.resolve()
    stored = Path(selected["run_dir"])
    candidates = ([stored] if stored.is_absolute() else
                  [Path.cwd() / stored, root.parent / stored, root / stored])
    directory = next((p.resolve() for p in candidates
                      if p.resolve().is_relative_to(root)
                      and p.resolve().is_dir()), None)
    if directory is None:
        raise HTTPException(404, "Recorded run is unavailable under this service's runs directory")
    summary_path = directory / "headline_sweep.json"
    if not summary_path.resolve().is_relative_to(root):
        raise HTTPException(404, "Recorded summary is outside the runs directory")
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "Recorded sweep summary is unavailable") from exc
    result = {"selected": selected, "summary": summary,
              "source": str(directory.relative_to(root)),
              "measurement": None, "telemetry": [], "telemetry_total": 0,
              "telemetry_note": "No telemetry database was retained"}
    db = directory / "run.db"
    if not db.is_file() or not db.resolve().is_relative_to(root):
        return result
    try:
        conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Explicitly align to the selected rung in this cohort, rather than
            # showing the latest telemetry or averaging the whole search.
            row = conn.execute(
                "SELECT * FROM cohort_measurements WHERE cohort_run_id = ? "
                "AND target_pool_size = ? ORDER BY step_index DESC LIMIT 1",
                (summary.get("cohort_run_id"), selected.get("concurrency")),
            ).fetchone()
            if row is None:
                result["telemetry_note"] = "No measurement window matches the selected concurrency"
                return result
            result["measurement"] = dict(row)
            mid = row["measurement_id"]
            total = conn.execute(
                "SELECT COUNT(*) FROM measurement_telemetry WHERE measurement_id = ?",
                (mid,),
            ).fetchone()[0]
            result["telemetry_total"] = total
            # Even sampling bounds the response and spans the entire window;
            # aggregates remain the original recorded measurement values.
            stride = max(1, (total + 599) // 600)
            rows = conn.execute(
                "SELECT * FROM measurement_telemetry WHERE measurement_id = ? "
                "ORDER BY sampled_at_ms", (mid,),
            )
            result["telemetry"] = [dict(r) for i, r in enumerate(rows)
                                   if i % stride == 0 or i == total - 1]
            result["telemetry_note"] = (
                f"{total} recorded samples; every {stride} sample(s) shown"
                if total else "No telemetry samples recorded for this window")
        finally:
            conn.close()
    except sqlite3.Error:
        result["telemetry_note"] = "Telemetry is unavailable in this legacy or unreadable database"
    return result


@router.get("/api/workspace/roofline-evidence")
async def roofline_evidence(request: Request, model: str, engine: str,
                           attempt: int | None = Query(default=None, ge=0)) -> dict:
    return await asyncio.to_thread(
        _read_evidence, request.app.state.paths.runs_base, model, engine, attempt)
