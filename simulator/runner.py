"""Top-level cohort-run orchestration."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from .adaptive import StepResult, choose_next_pool_size
from .config import Config
from .database import Database
from .engines import Engine, make_engine
from .measurement import (
    PHASE_IDLE,
    PhaseTracker,
    run_measurement_step,
)
from .personas import Cohort, get_cohort
from .pool_manager import PoolManager
from .telemetry import MeasurementTelemetry, SnapshotRecorder
from .tokenizer_corpus import TokenCorpus
from .virtual_user import SharedState, _now_ms

log = logging.getLogger(__name__)


def _run_db_path(cfg: Config, cohort_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_model = cfg.engine.model_id.replace("/", "_")
    db_dir = Path(cfg.output.db_directory)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / f"{ts}_{cfg.engine.type}_{short_model}_{cohort_id}.db"


def _config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def _cohort_to_dict(cohort: Cohort) -> dict:
    return {
        "id": cohort.id,
        "name": cohort.name,
        "description": cohort.description,
        "persona_weights": cohort.persona_weights,
    }


async def run_cohort(
    cfg: Config,
    cohort_id: str,
    *,
    engine: Engine | None = None,
    db_path: Path | None = None,
) -> Path:
    """Run a single cohort end-to-end. Returns the resulting db path."""
    cohort = get_cohort(cohort_id)
    own_engine = engine is None
    if own_engine:
        engine = make_engine(cfg.engine.type, cfg.engine)
        engine.launch(log_dir=cfg.output.db_directory)

    if db_path is None:
        db_path = _run_db_path(cfg, cohort_id)

    cohort_run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    db = Database(db_path)
    db.insert_run(
        cohort_run_id=cohort_run_id,
        started_at=started_at,
        engine_type=cfg.engine.type,
        model_id=cfg.engine.model_id,
        cohort_id=cohort_id,
        cohort_definition=_cohort_to_dict(cohort),
        config=_config_to_dict(cfg),
    )
    log.info("Cohort run %s -> %s", cohort_run_id, db_path)

    state = SharedState()
    phase_tracker = PhaseTracker()
    corpus = TokenCorpus(cfg.engine.model_id)
    client = AsyncOpenAI(base_url=engine.base_url, api_key="EMPTY")

    user_termination_buffer: list = []
    pool = PoolManager(
        cohort=cohort,
        client=client,
        model_id=engine.model_id,
        corpus=corpus,
        state=state,
        request_timeout_s=cfg.simulation.request_timeout_s,
        on_user_terminated=lambda s: user_termination_buffer.append(s),
    )
    pool.start()

    snap = SnapshotRecorder(
        db=db,
        cohort_run_id=cohort_run_id,
        state=state,
        pool=pool,
        get_phase=phase_tracker.get,
        interval_s=cfg.simulation.snapshot_interval_s,
    )
    snap.start()

    telemetry = MeasurementTelemetry(
        cfg.telemetry, engine, perf_events=cfg.telemetry.perf_events
    )

    history: list[StepResult] = []
    final_status = "ok"
    run_started = time.monotonic()
    max_total_s = cfg.simulation.max_total_duration_minutes * 60
    step_index = 0

    try:
        next_size = choose_next_pool_size(
            history,
            initial_pool_size=cfg.simulation.initial_pool_size,
            max_pool_size=cfg.simulation.max_pool_size,
            knee_slope_threshold=cfg.simulation.knee_slope_threshold,
            stop_violation_threshold=cfg.simulation.stop_violation_threshold,
        )
        while next_size is not None:
            if time.monotonic() - run_started > max_total_s:
                log.warning("Max run duration reached; stopping ramp")
                final_status = "time_limit"
                break

            result = await run_measurement_step(
                pool=pool,
                state=state,
                phase_tracker=phase_tracker,
                telemetry=telemetry,
                db=db,
                cohort_run_id=cohort_run_id,
                step_index=step_index,
                target_pool_size=next_size,
                target_samples=cfg.simulation.target_samples_per_step,
                measurement_timeout_s=cfg.simulation.measurement_timeout_s,
                cv_threshold=cfg.simulation.stabilization_cv_threshold,
                stab_min_s=cfg.simulation.stabilization_min_duration_s,
                stab_max_s=cfg.simulation.stabilization_max_duration_s,
            )

            # Persist any user terminations seen
            _flush_users(db, cohort_run_id, user_termination_buffer)

            if result.status == "unstable":
                final_status = "unstable"
                break
            if result.status == "no_samples":
                final_status = "no_samples"
                break

            history.append(StepResult(
                pool_size=result.target_pool_size,
                violation_rate=result.combined_violation_rate,
            ))
            step_index += 1

            next_size = choose_next_pool_size(
                history,
                initial_pool_size=cfg.simulation.initial_pool_size,
                max_pool_size=cfg.simulation.max_pool_size,
                knee_slope_threshold=cfg.simulation.knee_slope_threshold,
                stop_violation_threshold=cfg.simulation.stop_violation_threshold,
            )
    except KeyboardInterrupt:
        final_status = "interrupted"
        raise
    finally:
        phase_tracker.set(PHASE_IDLE)
        await snap.stop()
        await pool.stop()
        _flush_users(db, cohort_run_id, user_termination_buffer)
        db.finalise_run(
            cohort_run_id=cohort_run_id,
            completed_at=datetime.now(timezone.utc).isoformat(),
            status=final_status,
        )
        db.close()
        if own_engine:
            engine.shutdown()

    return db_path


def _flush_users(db: Database, cohort_run_id: str, buffer: list) -> None:
    while buffer:
        s = buffer.pop()
        db.upsert_user({
            "user_id": s.user_id,
            "cohort_run_id": cohort_run_id,
            "persona_id": s.persona_id,
            "spawned_at_ms": s.spawned_at_ms,
            "terminated_at_ms": s.terminated_at_ms or _now_ms(),
            "sessions_target": s.sessions_target,
            "sessions_completed": s.sessions_completed,
            "turns_total": s.turns_total,
            "pool_size_at_spawn": s.pool_size_at_spawn,
            "replaced_user_id": s.replaced_user_id,
        })


async def run_sweep(cfg: Config, cohort_ids: list[str]) -> list[Path]:
    """Run multiple cohorts back-to-back against the same engine."""
    engine = make_engine(cfg.engine.type, cfg.engine)
    engine.launch(log_dir=cfg.output.db_directory)
    paths: list[Path] = []
    try:
        for cid in cohort_ids:
            log.info("=== Sweep: cohort %s ===", cid)
            path = await run_cohort(cfg, cid, engine=engine)
            paths.append(path)
    finally:
        engine.shutdown()
    return paths
