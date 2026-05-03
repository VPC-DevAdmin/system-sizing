"""SQLite schema and capture helpers.

One database per cohort run. Designed for moderate write rates (a few hundred
events per measurement window) — direct sqlite3 with WAL mode is sufficient.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS cohort_run (
    cohort_run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    engine_type TEXT NOT NULL,
    model_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    cohort_definition_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    final_status TEXT
);

CREATE TABLE IF NOT EXISTS cohort_measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort_run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    target_pool_size INTEGER NOT NULL,
    measured_avg_pool_size REAL NOT NULL,
    measured_avg_in_flight REAL NOT NULL,
    measurement_started_at TEXT NOT NULL,
    measurement_duration_s INTEGER NOT NULL,
    sample_size INTEGER NOT NULL,
    ttft_violation_rate REAL NOT NULL,
    tpot_violation_rate REAL NOT NULL,
    combined_violation_rate REAL NOT NULL,
    violation_rate_ci_lower REAL NOT NULL,
    violation_rate_ci_upper REAL NOT NULL,
    ttft_p50_ms REAL, ttft_p75_ms REAL, ttft_p95_ms REAL,
    tpot_p50_ms REAL, tpot_p75_ms REAL, tpot_p95_ms REAL,
    avg_kv_cache_pct REAL,
    estimated_prefix_hit_rate REAL,
    capacity_status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_measurements_run ON cohort_measurements(cohort_run_id);

CREATE TABLE IF NOT EXISTS turn_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL,
    persona_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    ttft_ms REAL NOT NULL,
    completed_at_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    history_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    tpot_ms REAL NOT NULL,
    end_to_end_ms REAL NOT NULL,
    in_flight_at_submit INTEGER NOT NULL,
    in_flight_avg_during REAL,
    in_flight_peak_during INTEGER,
    sla_ttft_violation INTEGER NOT NULL,
    sla_tpot_violation INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_measurement ON turn_events(measurement_id);

CREATE TABLE IF NOT EXISTS simulation_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort_run_id TEXT NOT NULL,
    snapshot_at_ms INTEGER NOT NULL,
    phase TEXT NOT NULL,
    pool_size INTEGER NOT NULL,
    in_flight INTEGER NOT NULL,
    requests_completed INTEGER NOT NULL,
    errors INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_run_time ON simulation_snapshots(cohort_run_id, snapshot_at_ms);

CREATE TABLE IF NOT EXISTS measurement_telemetry (
    telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL,
    sampled_at_ms INTEGER NOT NULL,
    kv_cache_used_pct REAL,
    queue_depth INTEGER,
    prefix_cache_hits INTEGER,
    prefix_cache_misses INTEGER,
    cpu_util_avg REAL,
    memory_used_gb REAL,
    engine_rss_gb REAL,
    freq_mhz_mean REAL,
    freq_mhz_stddev REAL,
    freq_mhz_min REAL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_measurement ON measurement_telemetry(measurement_id);

-- Window-level telemetry aggregates: one row per measurement.
-- Stores PMU totals, BW summaries, AMX, power, and effective frequency.
CREATE TABLE IF NOT EXISTS measurement_aggregate (
    measurement_id INTEGER PRIMARY KEY,
    pmu_cycles REAL,
    pmu_instructions REAL,
    pmu_ipc REAL,
    pmu_stalls_mem_any REAL,
    pmu_stalls_l3_miss REAL,
    pmu_stall_mem_ratio REAL,
    pmu_amx_ops REAL,
    amx_perf_event_name TEXT,
    pmu_llc_reference REAL,
    pmu_llc_miss REAL,
    mem_local_fraction REAL,
    mem_remote_fraction REAL,
    memory_bw_read_gb_s_avg REAL,
    memory_bw_read_gb_s_peak REAL,
    memory_bw_write_gb_s_avg REAL,
    memory_bw_write_gb_s_peak REAL,
    bandwidth_status TEXT,
    power_w_avg REAL,
    power_w_peak REAL,
    power_status TEXT,
    effective_freq_ghz_mean REAL,
    effective_freq_ghz_stddev REAL,
    effective_freq_ghz_min REAL,
    onednn_amx_time_fraction REAL,
    onednn_matmul_dispatches_amx INTEGER,
    onednn_matmul_dispatches_non_amx INTEGER
);

CREATE TABLE IF NOT EXISTS virtual_users (
    user_id TEXT PRIMARY KEY,
    cohort_run_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    spawned_at_ms INTEGER NOT NULL,
    terminated_at_ms INTEGER,
    sessions_target INTEGER NOT NULL,
    sessions_completed INTEGER NOT NULL,
    turns_total INTEGER NOT NULL,
    pool_size_at_spawn INTEGER NOT NULL,
    replaced_user_id TEXT
);
"""


class Database:
    """Thread-safe-enough wrapper around a single SQLite file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # -- High-level inserts ---------------------------------------------------

    def insert_run(
        self,
        cohort_run_id: str,
        started_at: str,
        engine_type: str,
        model_id: str,
        cohort_id: str,
        cohort_definition: dict,
        config: dict,
    ) -> None:
        with self.cursor() as c:
            c.execute(
                """INSERT INTO cohort_run
                (cohort_run_id, started_at, engine_type, model_id, cohort_id,
                 cohort_definition_json, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    cohort_run_id, started_at, engine_type, model_id, cohort_id,
                    json.dumps(cohort_definition), json.dumps(config),
                ),
            )

    def finalise_run(self, cohort_run_id: str, completed_at: str, status: str) -> None:
        with self.cursor() as c:
            c.execute(
                "UPDATE cohort_run SET completed_at = ?, final_status = ? WHERE cohort_run_id = ?",
                (completed_at, status, cohort_run_id),
            )

    def insert_measurement(self, row: dict) -> int:
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        with self.cursor() as c:
            c.execute(
                f"INSERT INTO cohort_measurements ({','.join(cols)}) VALUES ({placeholders})",
                [row[k] for k in cols],
            )
            return c.lastrowid

    def insert_events(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ",".join("?" for _ in cols)
        with self.cursor() as c:
            c.executemany(
                f"INSERT INTO turn_events ({','.join(cols)}) VALUES ({placeholders})",
                [[r[k] for k in cols] for r in rows],
            )

    def insert_snapshot(self, row: dict) -> None:
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        with self.cursor() as c:
            c.execute(
                f"INSERT INTO simulation_snapshots ({','.join(cols)}) VALUES ({placeholders})",
                [row[k] for k in cols],
            )

    def upsert_aggregate(self, row: dict) -> None:
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "measurement_id")
        with self.cursor() as c:
            c.execute(
                f"""INSERT INTO measurement_aggregate ({','.join(cols)}) VALUES ({placeholders})
                ON CONFLICT(measurement_id) DO UPDATE SET {updates}""",
                [row[k] for k in cols],
            )

    def insert_telemetry(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ",".join("?" for _ in cols)
        with self.cursor() as c:
            c.executemany(
                f"INSERT INTO measurement_telemetry ({','.join(cols)}) VALUES ({placeholders})",
                [[r[k] for k in cols] for r in rows],
            )

    def upsert_user(self, row: dict) -> None:
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "user_id")
        with self.cursor() as c:
            c.execute(
                f"""INSERT INTO virtual_users ({','.join(cols)}) VALUES ({placeholders})
                ON CONFLICT(user_id) DO UPDATE SET {updates}""",
                [row[k] for k in cols],
            )

    # -- Read helpers ---------------------------------------------------------

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.cursor() as c:
            c.execute(sql, params)
            return c.fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self.cursor() as c:
            c.execute(sql, params)
            return c.fetchone()
