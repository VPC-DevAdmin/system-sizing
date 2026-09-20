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

# Tests patch ``service.subprocess.Popen``; keep the module reachable here.
import subprocess  # noqa: F401

from .app import create_app, serve
from .optimizer import _log_heartbeat
from .prepare import _tail_text
from .runs import (
    _CUSTOM_INT_FIELDS,
    _build_custom_config,
    _finalise_orphan_runs,
    _list_runs,
    _resolve_config_path,
    _validate_custom_ints,
)
from .schemas import (
    EnginePullRequest,
    ExportRequest,
    ModelAddRequest,
    ModelDownloadRequest,
    OptimizerStartRequest,
    PromoteRequest,
    RooflineRequest,
    SaveSpecRequest,
    StartRunRequest,
    StorageRequest,
)
from .state import (
    _HELD_RUNS_LOCKS,
    SERVE_LOCK_NAME,
    STOP_TIMEOUT_S,
    ActiveRun,
    Paths,
    RunsDirBusy,
    _acquire_runs_lock,
    _read_json,
    _release_runs_lock,
)

__all__ = [
    "ActiveRun",
    "EnginePullRequest",
    "ExportRequest",
    "ModelAddRequest",
    "ModelDownloadRequest",
    "OptimizerStartRequest",
    "Paths",
    "PromoteRequest",
    "RooflineRequest",
    "RunsDirBusy",
    "SERVE_LOCK_NAME",
    "STOP_TIMEOUT_S",
    "SaveSpecRequest",
    "StartRunRequest",
    "StorageRequest",
    "_CUSTOM_INT_FIELDS",
    "_HELD_RUNS_LOCKS",
    "_acquire_runs_lock",
    "_build_custom_config",
    "_finalise_orphan_runs",
    "_list_runs",
    "_log_heartbeat",
    "_read_json",
    "_release_runs_lock",
    "_resolve_config_path",
    "_tail_text",
    "_validate_custom_ints",
    "create_app",
    "serve",
]
