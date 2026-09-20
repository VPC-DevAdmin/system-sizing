#!/usr/bin/env python3
"""Generate docs/database_schema.md from the live DDL in simulator/database.py.

The doc used to be hand-written and fell six schema versions behind.
Now the column tables come straight from ``database.SCHEMA`` (the
``--`` comments in the DDL become the notes column), the migration
list from ``database.MIGRATIONS``, and the engine-type list from the
engine registry. A test asserts the checked-in doc matches this
output, so the doc cannot drift again.

    python scripts/gen_schema_doc.py            # rewrite docs/database_schema.md
    python scripts/gen_schema_doc.py --check    # exit 1 if the doc is stale
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import database  # noqa: E402


def _engine_types() -> list[str]:
    """The registry is a chain of ``engine_type == "x"`` tests in
    simulator/engines/__init__.py; read them from the source so this
    stays in sync without importing every engine module."""
    src = (ROOT / "simulator" / "engines" / "__init__.py").read_text()
    return sorted(set(re.findall(r'engine_type == "(\w+)"', src)))

DOC_PATH = ROOT / "docs" / "database_schema.md"

# Hand-written notes for columns whose DDL comment says less than a
# reader needs. Keyed "table.column".
NOTES: dict[str, str] = {
    "cohort_run.cohort_run_id": "`uuid.uuid4().hex`, minted when the run starts.",
    "cohort_run.started_at": "ISO-8601 UTC.",
    "cohort_run.completed_at": "Set by `finalise_run`; NULL while in flight.",
    "cohort_run.engine_type": "One of the registry keys: "
                              + ", ".join(f"`{k}`" for k in _engine_types()) + ".",
    "cohort_run.model_id": "HF repo id (e.g. `Qwen/Qwen3-30B-A3B-Instruct-2507`).",
    "cohort_run.cohort_id": "Cohort id, or a persona id when the run is a single persona.",
    "cohort_run.cohort_definition_json": "JSON of `{id, name, description, category, persona_weights}`.",
    "cohort_run.config_json": "The full `Config` dataclass as JSON — every knob that produced "
                              "this measurement.",
    "cohort_run.final_status": "`ok`, `interrupted` (SIGINT/closed-loop), `cancelled` (user stop, "
                               "open-loop), `error`, `time_limit`, `no_samples`, or NULL while in "
                               "progress. Resume only skips `ok`.",
    "cohort_measurements.measurement_id": "Foreign-key target for events and telemetry.",
    "cohort_measurements.cohort_run_id": "→ `cohort_run.cohort_run_id`.",
    "cohort_measurements.step_index": "0-based ordinal within the ramp / rate search.",
    "cohort_measurements.target_pool_size": "Closed loop: commanded pool size. Open loop: "
                                            "`round(active_sessions_mean)` (see the v7 note).",
    "cohort_measurements.capacity_status": "`pending` (row inserted up front), `pass`, "
                                           "`marginal`, `fail` — Wilson-CI bands on the "
                                           "combined violation rate.",
    "cohort_measurements.stability": "`stable`, `divergent`, `client_limited`, `superseded` "
                                     "(re-measured after the load generator scaled out), "
                                     "or NULL on closed-loop rows.",
    "turn_events.measurement_id": "→ `cohort_measurements.measurement_id`.",
    "turn_events.session_id": "Fresh UUID per multi-turn session — turn 0 is cold, turns 1..N "
                              "replay history (the prefix-cache hit candidates).",
    "turn_events.ttft_ms": "First-token latency. On a failed turn this carries the end-to-end "
                           "time (a failure is a violation in both axes).",
    "turn_events.tpot_ms": "Per-output-token time. NULL/excluded from TPOT percentiles for "
                           "turns that failed before any token.",
    "simulation_snapshots.cohort_run_id": "→ `cohort_run.cohort_run_id`.",
    "simulation_snapshots.phase": "`ramp`, `warmup`, `measure`, `drain`, `idle`, …",
    "measurement_telemetry.measurement_id": "→ `cohort_measurements.measurement_id`.",
    "measurement_telemetry.queue_depth": "Engine-reported waiting requests (`num_waiting`).",
    "virtual_users.cohort_run_id": "→ `cohort_run.cohort_run_id`. Closed-loop only: open-loop "
                                   "sessions live in worker subprocesses and are not persisted "
                                   "here.",
}

TABLE_TITLES: dict[str, str] = {
    "cohort_run": "one row per cohort/persona run",
    "cohort_measurements": "one row per measurement window (ramp step or rate window)",
    "turn_events": "one row per LLM turn captured in a window",
    "simulation_snapshots": "one row per second of the whole run",
    "measurement_telemetry": "one row per second within a window",
    "virtual_users": "one row per simulated user lifecycle (closed loop)",
}

PREAMBLE = """# `run.db` schema

*Generated by `scripts/gen_schema_doc.py` from `simulator/database.py` —
edit the DDL comments there, not this file. `tests/test_schema_doc.py`
fails when this file is stale.*

One SQLite file per `runs/run_NN/` directory. Every cohort/persona
invocation against that run dir appends rows to the same file;
isolation is by `cohort_run_id`. The DDL lives in
[simulator/database.py](../simulator/database.py) and is applied on
`Database` open with `CREATE TABLE IF NOT EXISTS`.

Connection settings: `journal_mode=WAL`, `synchronous=NORMAL`,
`isolation_level=None` (autocommit), `check_same_thread=False`. A
single process-wide `threading.Lock` wraps every cursor. Readers
(export, dashboard, service) open the file with `mode=ro`.

## Schema versioning

The current shape is **schema version {version}**, stamped into SQLite's
`PRAGMA user_version`. A fresh file gets the full DDL and the current
stamp. An existing file is lifted by the ordered, idempotent
migrations below (pre-versioning files read as 0 and receive every
one); a file stamped *newer* than the running code is refused rather
than written into.

| version | change |
|---|---|
{migrations}

## Entity diagram

```
cohort_run                          (one row per run)
   │  cohort_run_id (PK)
   │
   ├──< cohort_measurements          (one row per window; PMU/BW/power/
   │     │  measurement_id (PK)       GPU aggregates inlined as nullable
   │     │                            columns)
   │     ├──< turn_events              (one row per LLM turn in the window)
   │     └──< measurement_telemetry    (per-second samples in the window)
   │
   ├──< simulation_snapshots         (per-second run-wide tick)
   └──< virtual_users                (closed loop: one row per user)
```

Open-loop runs (`cohort_run.mode = 'open_loop'`) use the same tables:
the control variable is `cohort_measurements.arrival_rate_per_min`,
the verdict is `stability`, and `target_pool_size` holds the mean
concurrent sessions observed so legacy consumers still read a
concurrency.
"""

POSTAMBLE = """
## What is *not* persisted

- Per-request streaming chunks, unless tier-3 capture is on
  (`turn_events.token_timestamps_json`).
- Raw `perf stat` / IMC / RAPL samples — only the per-second rows in
  `measurement_telemetry` and the window rollups in
  `cohort_measurements`.
- Engine container logs (`runs/run_NN/engine_*.log`) and the
  optimizer/roofline state (`runs/engine_optimizer/*.json`,
  `runs/roofline.json`) — files beside the DB, not rows in it.
- Open-loop session objects: they live in the load-generator worker
  subprocesses; only their turns (`turn_events`) and the per-second
  population (`simulation_snapshots.active_sessions`) land in the DB.
"""

_CREATE_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", re.S)
_INDEX_RE = re.compile(r"CREATE INDEX IF NOT EXISTS (\w+) ON (\w+)\(([^)]*)\);")


def _parse_columns(body: str) -> list[tuple[str, str, str]]:
    """Return (name, type, note) triples from a CREATE TABLE body,
    attaching the run of ``--`` comment lines above a column (and any
    trailing ``-- …`` on the same line) as its note."""
    out: list[tuple[str, str, str]] = []
    pending: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            pending.append(line[2:].strip())
            continue
        trailing = ""
        if "--" in line:
            line, trailing = line.split("--", 1)
            trailing = trailing.strip()
            line = line.strip()
        # A line may declare several columns: "a REAL, b REAL, c REAL,"
        for decl in [d.strip() for d in line.rstrip(",").split(",") if d.strip()]:
            parts = decl.split()
            name, ctype = parts[0], " ".join(parts[1:])
            note = " ".join(pending) if pending else ""
            if trailing:
                note = (note + " " + trailing).strip() if note else trailing
            out.append((name, ctype, note))
            # The comment run applies to the first column of a group
            # (that is how the DDL is written); later columns in the
            # same block share it only via their own trailing text.
            pending = []
    return out


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("``", "`")


def render() -> str:
    migrations = "\n".join(
        f"| {v} | {desc} |" for v, desc, _ in database.MIGRATIONS)
    doc = [PREAMBLE.format(version=database.SCHEMA_VERSION, migrations=migrations)]
    indexes: dict[str, list[str]] = {}
    for name, table, cols in _INDEX_RE.findall(database.SCHEMA):
        indexes.setdefault(table, []).append(f"`{name}` on (`{cols}`)")
    for table, body in _CREATE_RE.findall(database.SCHEMA):
        doc.append(f"\n## `{table}` — {TABLE_TITLES.get(table, '')}\n")
        doc.append("| column | type | notes |\n|---|---|---|")
        for col, ctype, note in _parse_columns(body):
            override = NOTES.get(f"{table}.{col}")
            text = override if override else note
            doc.append(f"| `{col}` | {ctype or '—'} | {_md_escape(text)} |")
        if table in indexes:
            doc.append("\nIndexes: " + "; ".join(indexes[table]) + ".")
    doc.append(POSTAMBLE)
    return "\n".join(doc).rstrip() + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = DOC_PATH.read_text() if DOC_PATH.exists() else ""
        if current != text:
            print(f"{DOC_PATH} is stale — run: python scripts/gen_schema_doc.py",
                  file=sys.stderr)
            return 1
        return 0
    DOC_PATH.write_text(text)
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
