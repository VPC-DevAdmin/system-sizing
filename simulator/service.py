"""Control-plane HTTP service (roadmap 2.1).

``capsim serve`` — a single-operator, localhost-by-default FastAPI app
that owns run lifecycle and streams live telemetry. Everything the
Makefile/CLI can do is callable over HTTP; ``run.db`` stays the source
of truth, the service is a thin lifecycle layer over the same
``run_cohort`` / ``run_sweep`` the CLI drives, executed in-process as
asyncio tasks so the event bus reaches WebSocket clients directly.

Security model: none, deliberately — bind stays on 127.0.0.1 unless
the operator explicitly opens it (Phase 4 revisits if the tool ever
grows multi-user ambitions). CORS is likewise not enabled; the Phase 3
UI is served same-origin by this app.

API sketch:

    GET  /api/status            service + active-run state
    GET  /api/profiles          hardware profiles (name -> path)
    GET  /api/personas          persona catalog with SLA floors
    GET  /api/cohorts           cohort catalog with weights
    GET  /api/runs              run_NN dirs with per-cohort summaries
    POST /api/runs              start a run  {profile|config, workload,
                                new_run, pool_sizes?, adaptive?}
    POST /api/runs/stop         cancel the active run
    GET  /api/doctor            host validation report (slow: probes)
    POST /api/export            build export {slim} -> {path, ...}
    GET  /api/export/latest     the built export JSON (404 if absent)
    WS   /ws/telemetry          live bus events {topic, ts, data}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .bus import BUS

log = logging.getLogger(__name__)


# ── Run lifecycle state ───────────────────────────────────────────────


@dataclass
class ActiveRun:
    task: asyncio.Task
    workload: dict
    config_path: str
    started_at: float
    error: Optional[str] = None
    result: Optional[str] = None          # str(db_path) on success

    def describe(self) -> dict:
        return {
            "workload": self.workload,
            "config": self.config_path,
            "started_at": self.started_at,
            "running": not self.task.done(),
            "error": self.error,
            "result": self.result,
        }


class StartRunRequest(BaseModel):
    profile: Optional[str] = None
    config: Optional[str] = None
    # {"kind": "cohort"|"persona", "id": "..."} or {"kind": "sweep",
    # "type": "all"|"personas"|"cohorts"|"a,b,c"}
    workload: dict
    new_run: bool = False
    pool_sizes: Optional[list[int]] = None
    adaptive: bool = False


class ExportRequest(BaseModel):
    slim: bool = False


def _resolve_config_path(req: StartRunRequest) -> Path:
    from .config import resolve_profile
    if req.profile and req.config:
        raise HTTPException(422, "pass either profile or config, not both")
    if req.profile:
        try:
            return resolve_profile(req.profile)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
    if req.config:
        p = Path(req.config)
        if not p.exists():
            raise HTTPException(404, f"config not found: {p}")
        return p
    raise HTTPException(422, "pass profile or config")


def _list_runs(base: Path) -> list[dict]:
    """run_NN dirs, newest first, with per-cohort summaries from each
    run.db (read-only; missing/corrupt DBs degrade to an empty list)."""
    out: list[dict] = []
    if not base.exists():
        return out
    for d in sorted(base.glob("run_[0-9]*"), reverse=True):
        db_path = d / "run.db"
        entry: dict = {"dir": str(d), "name": d.name, "cohorts": []}
        if db_path.exists():
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT cohort_run_id, cohort_id, engine_type, model_id, "
                    "started_at, completed_at, final_status, "
                    "(SELECT COUNT(*) FROM cohort_measurements m "
                    " WHERE m.cohort_run_id = cohort_run.cohort_run_id) AS steps "
                    "FROM cohort_run ORDER BY started_at ASC"
                ).fetchall()
                entry["cohorts"] = [dict(r) for r in rows]
                conn.close()
            except sqlite3.Error as e:
                entry["error"] = str(e)
        exports = sorted(p.name for p in d.glob("buyer_page_data*.json"))
        entry["exports"] = exports
        out.append(entry)
    return out


def create_app(runs_base: Path | str = Path("runs")) -> FastAPI:
    runs_base = Path(runs_base)
    app = FastAPI(title="capsim", version="0.2.0")
    app.state.active: Optional[ActiveRun] = None

    # ── introspection ─────────────────────────────────────────────

    @app.get("/api/status")
    async def status() -> dict:
        active = app.state.active
        return {
            "service": "capsim",
            "bus_subscribers": BUS.subscriber_count,
            "active_run": active.describe() if active else None,
        }

    @app.get("/api/profiles")
    async def profiles() -> dict:
        from .config import list_profiles
        return {name: str(path) for name, path in list_profiles().items()}

    @app.get("/api/personas")
    async def personas() -> list[dict]:
        from .personas import PERSONAS
        return [
            {
                "id": p.id,
                "description": p.description,
                "ttft_target_s": p.ttft_target_seconds,
                "ttft_failure_s": p.ttft_failure_seconds,
                "tpot_target_ms": p.tpot_target_ms,
                "tpot_failure_ms": p.tpot_failure_ms,
            }
            for p in PERSONAS.values()
        ]

    @app.get("/api/cohorts")
    async def cohorts() -> list[dict]:
        from .personas import COHORTS
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "persona_weights": c.persona_weights,
            }
            for c in COHORTS.values()
        ]

    @app.get("/api/runs")
    async def runs() -> list[dict]:
        return _list_runs(runs_base)

    @app.get("/api/doctor")
    async def doctor() -> dict:
        from .doctor import run_doctor
        report = await asyncio.to_thread(run_doctor)
        return report.to_dict()

    # ── run lifecycle ─────────────────────────────────────────────

    @app.post("/api/runs", status_code=202)
    async def start_run(req: StartRunRequest) -> dict:
        active = app.state.active
        if active is not None and not active.task.done():
            raise HTTPException(
                409, "a run is already active — stop it first "
                     "(POST /api/runs/stop)",
            )
        config_path = _resolve_config_path(req)

        from .config import load_config
        from .personas import COHORTS, PERSONAS, cohort_from_persona
        cfg = load_config(config_path)
        cfg.output.db_directory = str(runs_base)

        kind = req.workload.get("kind")
        if kind == "cohort":
            wid = req.workload.get("id")
            if wid not in COHORTS:
                raise HTTPException(404, f"unknown cohort '{wid}'")
            coro_factory = _cohort_coro(cfg, wid, req)
        elif kind == "persona":
            wid = req.workload.get("id")
            if wid not in PERSONAS:
                raise HTTPException(404, f"unknown persona '{wid}'")
            coro_factory = _cohort_coro(cfg, cohort_from_persona(wid), req)
        elif kind == "sweep":
            from .personas import resolve_workload_group
            try:
                persona_ids, cohort_ids = resolve_workload_group(
                    req.workload.get("type", "all")
                )
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            from .runner import run_sweep
            coro_factory = lambda: run_sweep(  # noqa: E731
                cfg, persona_ids=persona_ids, cohort_ids=cohort_ids,
                new_run=req.new_run, adaptive=req.adaptive,
                fixed_grid_pool_sizes=req.pool_sizes,
            )
        else:
            raise HTTPException(
                422, "workload.kind must be cohort | persona | sweep",
            )

        active = ActiveRun(
            task=asyncio.create_task(_supervise(app, coro_factory)),
            workload=req.workload,
            config_path=str(config_path),
            started_at=time.time(),
        )
        app.state.active = active
        return {"accepted": True, "workload": req.workload,
                "config": str(config_path)}

    def _cohort_coro(cfg, cohort, req: StartRunRequest):
        from .runner import run_cohort
        return lambda: run_cohort(
            cfg, cohort,
            new_run=req.new_run,
            adaptive=req.adaptive,
            fixed_grid_pool_sizes=req.pool_sizes,
        )

    async def _supervise(app_ref, coro_factory) -> None:
        """Run the workload; record outcome on the ActiveRun so
        /api/status can report it after completion."""
        active = None
        try:
            result = await coro_factory()
            active = app_ref.state.active
            if active is not None:
                active.result = str(result)
        except asyncio.CancelledError:
            active = app_ref.state.active
            if active is not None:
                active.error = "cancelled"
            raise
        except Exception as e:  # noqa: BLE001
            active = app_ref.state.active
            if active is not None:
                active.error = f"{type(e).__name__}: {e}"
            log.exception("run failed")

    @app.post("/api/runs/stop")
    async def stop_run() -> dict:
        active = app.state.active
        if active is None or active.task.done():
            raise HTTPException(409, "no active run")
        active.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await active.task
        return {"stopped": True}

    # ── export ────────────────────────────────────────────────────

    @app.post("/api/export")
    async def export(req: ExportRequest) -> dict:
        from .export import export_dir
        try:
            doc, out_path = await asyncio.to_thread(
                export_dir, runs_base, None, slim=req.slim,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return {
            "path": str(out_path),
            "schema_version": doc["schema_version"],
            "cohort_count": doc["meta"]["cohort_count"],
            "slim": req.slim,
        }

    @app.get("/api/export/latest")
    async def export_latest(slim: bool = False) -> Any:
        from .runs import latest_run_dir
        d = latest_run_dir(runs_base)
        if d is None:
            raise HTTPException(404, "no runs yet")
        fname = "buyer_page_data_slim.json" if slim else "buyer_page_data.json"
        p = d / fname
        if not p.exists():
            raise HTTPException(
                404, f"{p} not built — POST /api/export first",
            )
        return json.loads(p.read_text())

    # ── live telemetry ────────────────────────────────────────────

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        await ws.accept()
        q = BUS.subscribe()
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            BUS.unsubscribe(q)

    return app


def serve(host: str = "127.0.0.1", port: int = 8321,
          runs_base: Path | str = Path("runs")) -> None:
    """Blocking uvicorn entrypoint used by ``capsim serve``."""
    import uvicorn
    uvicorn.run(create_app(runs_base), host=host, port=port, log_level="info")
