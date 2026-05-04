"""Run-directory layout helpers.

Each invocation of ``make run-cohort`` / ``run-persona`` / ``run-sweep``
writes its artifacts into a numbered subdirectory:

    runs/
      run_01/
        20260504T231642Z_sglang_..._chat_heavy.db
        engine_sglang_1714851402.log
        perf_m0_8.csv
        sweep_20260504T231642.log
      run_02/
        ...

Default behaviour is **resume**: a new invocation reuses the highest-
numbered ``run_NN`` so artifacts that belong together (sweep log,
engine log, per-cohort DBs, perf telemetry CSVs) stay grouped, and
``--resume`` on a sweep finds the work already done in this run dir.

Pass ``--new-run`` (or ``RUN_NEW=true`` from make) to start a fresh
``run_NN+1``. There is no implicit "rotate after N hours"; the user
decides when a run is over.
"""

from __future__ import annotations

import re
from pathlib import Path

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def list_run_dirs(base: str | Path) -> list[Path]:
    """Return existing ``run_NN`` subdirectories of ``base``, sorted ascending."""
    base = Path(base)
    if not base.exists():
        return []
    out = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = _RUN_DIR_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return [p for _, p in out]


def latest_run_dir(base: str | Path) -> Path | None:
    dirs = list_run_dirs(base)
    return dirs[-1] if dirs else None


def next_run_dir(base: str | Path) -> Path:
    """Create and return ``base/run_(N+1)`` (or ``run_01`` if none exist)."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    dirs = list_run_dirs(base)
    if dirs:
        last = int(_RUN_DIR_RE.match(dirs[-1].name).group(1))
        n = last + 1
    else:
        n = 1
    p = base / f"run_{n:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_run_dir(base: str | Path, *, new: bool = False) -> Path:
    """Pick the directory the next run should write into.

    * ``new=True``: always create the next ``run_NN``.
    * ``new=False`` (default): reuse the latest ``run_NN``; create
      ``run_01`` if none exist yet.
    """
    base = Path(base)
    if new:
        return next_run_dir(base)
    latest = latest_run_dir(base)
    if latest is not None:
        return latest
    return next_run_dir(base)
