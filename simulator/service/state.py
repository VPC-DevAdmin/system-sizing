"""Service state: the active run, the paths every router resolves
against, and the runs-dir lock."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# How long POST /api/runs/stop waits for engine teardown before
# answering 202 "stopping" instead of holding the request open.
STOP_TIMEOUT_S = 20.0


# ── Run lifecycle state ───────────────────────────────────────────────


@dataclass
class ActiveRun:
    task: asyncio.Task
    workload: dict
    config_path: str
    started_at: float
    error: Optional[str] = None
    result: Optional[str] = None          # str(db_path) on success
    # Human summary of the engine actually launched ("custom: 8×tp1
    # pack · KV fp8" / "optimized profile x") — the banner's answer to
    # "what exactly is running?", surviving page reloads.
    engine_summary: Optional[str] = None
    # Mutable progress holder for multi-cell runs (the headline shape
    # search updates it in place: {cell, budget, shape, best, done}).
    progress: Optional[dict] = None

    def describe(self) -> dict:
        return {
            "workload": self.workload,
            "config": self.config_path,
            "started_at": self.started_at,
            "running": not self.task.done(),
            "error": self.error,
            "result": self.result,
            "engine_summary": self.engine_summary,
            "progress": self.progress,
        }


@dataclass(frozen=True)
class Paths:
    """Where this service reads and writes — fixed in create_app and
    carried on ``app.state.paths`` so routers and helpers resolve
    against it instead of capturing lexical variables."""
    runs_base: Path
    # None = the user overlay dir (persona_loader.USER_CATALOG_DIR).
    catalog_dir: Optional[Path]
    # The engine optimizer is a repo script (like config/, resolved
    # against the working directory); injectable for tests.
    optimizer_script: Path

    @property
    def opt_out(self) -> Path:
        return self.runs_base / "engine_optimizer" / "run.json"

    @property
    def search_out(self) -> Path:
        return self.runs_base / "engine_optimizer" / "search.json"

    @property
    def history_dir(self) -> Path:
        return self.opt_out.parent / "history"

    @property
    def roofline_state(self) -> Path:
        return self.runs_base / "roofline.json"


SERVE_LOCK_NAME = ".capsim-serve.lock"


class RunsDirBusy(RuntimeError):
    """Another ``capsim serve`` holds this runs dir."""


# Locks this process holds, by resolved runs dir: [file, refcount].
# Re-entrant within the process (tests build several apps on one
# dir; flock on a second descriptor would otherwise self-deadlock),
# exclusive across processes, which is the case that matters.
_HELD_RUNS_LOCKS: dict[str, list] = {}


def _acquire_runs_lock(base: Path):
    """Exclusive advisory lock on ``<runs>/.capsim-serve.lock``, held
    for the service's lifetime. A second ``capsim serve`` on the same
    dir would otherwise run ``_finalise_orphan_runs`` and stamp the
    FIRST service's live run 'interrupted' (D3). Returns a lock record
    for _release_runs_lock, or None when the platform has no flock."""
    try:
        import fcntl
    except ImportError:            # not a supported host anyway
        return None
    key = str(base.resolve())
    held = _HELD_RUNS_LOCKS.get(key)
    if held is not None:
        held[1] += 1
        return held
    base.mkdir(parents=True, exist_ok=True)
    fh = open(base / SERVE_LOCK_NAME, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        fh.close()
        raise RunsDirBusy(
            f"another capsim serve is already using {base} (lock "
            f"{SERVE_LOCK_NAME} is held) — stop it, or pass a different "
            f"--runs-dir") from e
    fh.seek(0)
    fh.truncate()
    import os as _os
    fh.write(f"{_os.getpid()}\n")
    fh.flush()
    rec = [fh, 1, key]
    _HELD_RUNS_LOCKS[key] = rec
    return rec


def _release_runs_lock(rec) -> None:
    if rec is None:
        return
    rec[1] -= 1
    if rec[1] > 0:
        return
    fh, _, key = rec
    _HELD_RUNS_LOCKS.pop(key, None)
    with contextlib.suppress(OSError):
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        fh.close()


def _read_json(p: Path) -> Any:
    """json.loads off the event loop — exports and search docs run to
    tens of MB, and parsing them inline stalled every other client."""
    return json.loads(p.read_text())
