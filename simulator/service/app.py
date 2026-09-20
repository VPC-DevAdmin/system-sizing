"""Assemble the FastAPI app: settings and run state on ``app.state``,
the runs-dir lock held for the app's lifetime, one router per
concern, and the static UI mount."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from .. import __version__
from . import catalog, optimizer, prepare, roofline, runs, telemetry
from .runs import _finalise_orphan_runs
from .state import ActiveRun, Paths, _acquire_runs_lock, _release_runs_lock


def create_app(
    runs_base: Path | str = Path("runs"),
    catalog_dir: Path | str | None = None,
    # The engine optimizer is a repo script (like config/, resolved
    # against the working directory); injectable for tests.
    optimizer_script: Path | str = Path("scripts/engine_optimizer.py"),
) -> FastAPI:
    runs_base = Path(runs_base)
    # Lock first, THEN finalise: a second service on the same dir
    # must refuse to start rather than stamp the live run interrupted.
    runs_lock = _acquire_runs_lock(runs_base)
    _finalise_orphan_runs(runs_base)
    catalog_dir = Path(catalog_dir) if catalog_dir is not None else None
    optimizer_script = Path(optimizer_script)
    paths = Paths(runs_base=runs_base, catalog_dir=catalog_dir,
                  optimizer_script=optimizer_script)

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            _release_runs_lock(runs_lock)

    app = FastAPI(title="capsim", version=__version__, lifespan=_lifespan)
    app.state.paths = paths
    app.state.runs_lock = runs_lock
    app.state.export_locks: dict[str, asyncio.Lock] = {}  # run_name -> lock
    app.state.active: Optional[ActiveRun] = None
    app.state.optimizer: Optional[dict] = None       # {proc, profile, started_at, log}
    app.state.optimizer_catalog: Optional[dict] = None
    app.state.model_downloads: dict = {}             # model -> {proc, log, started_at}
    app.state.engine_pulls: dict = {}                # engine -> {proc, log, started_at}
    # Serializes start_run's check-then-create (see below).
    app.state.start_lock = asyncio.Lock()

    # A previous release saved the shape-search winner as its own
    # "headline_best" persona; the winner now lands IN the Headline:
    # Generation workload, so retire any stale overlay.
    if catalog_dir is None:
        from ..persona_loader import USER_CATALOG_DIR as _ucd
        _stale = _ucd / "headline_best.yaml"
    else:
        _stale = catalog_dir / "headline_best.yaml"
    if _stale.exists():
        _stale.unlink()
        from ..personas import reload_personas as _rp
        _rp(user_dir=catalog_dir)

    for r in (runs.router, catalog.router, prepare.router, roofline.router,
              optimizer.router, telemetry.router):
        app.include_router(r)

    # ── UI (Phase 3) ──────────────────────────────────────────────
    # No-build static frontend shipped as package data; same origin as
    # the API and WebSocket, so no CORS. API routes above win — the
    # mount only catches what they don't.
    ui_dir = Path(__file__).parent.parent / "ui"
    if ui_dir.exists():
        from fastapi.staticfiles import StaticFiles

        class _RevalidatingUI(StaticFiles):
            """Serve the UI with must-revalidate.

            Without this the browser may serve app.js/index.html from
            its own cache for a heuristic period and never ask, so a
            deploy silently shows the OLD interface — which looks
            exactly like the deploy having failed. ETags still do the
            real work: revalidation returns 304 and no body, so the
            cost is one conditional request per file per load.
            """

            def file_response(self, *args, **kwargs):
                resp = super().file_response(*args, **kwargs)
                resp.headers["Cache-Control"] = "no-cache, must-revalidate"
                return resp

        app.mount("/", _RevalidatingUI(directory=ui_dir, html=True),
                  name="ui")

    return app


def serve(host: str = "127.0.0.1", port: int = 8321,
          runs_base: Path | str = Path("runs")) -> None:
    """Blocking uvicorn entrypoint used by ``capsim serve``. The
    loopback-only guard lives in the CLI (``--insecure``); this
    function binds wherever it is told."""
    import uvicorn
    uvicorn.run(create_app(runs_base), host=host, port=port, log_level="info")
