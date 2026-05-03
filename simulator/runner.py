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
from .amx_utilization import parse_amx_utilization
from .config import Config
from .cpu_binding import expand_thread_binding
from .database import Database
from .engines import Engine, make_engine
from .preflight import preflight_check
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
        # Validate the host before any subprocess / docker / model load —
        # SGLang FP8 on AMD wastes 10-20 minutes before the assertion.
        preflight_check(cfg.engine.hardware_requirements)
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
        capture_token_timestamps=cfg.simulation.enable_token_timestamps,
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

    # Engine bound-CPU set drives frequency aggregation. Without this filter,
    # idle cores on the unused socket pull the host-wide average toward
    # half-nominal — the number lies. See cpu_binding.py.
    #
    # Prefer the engine-agnostic cpu_bind field; fall back to vLLM's
    # specific env var so existing configs keep working.
    bind_str = (
        cfg.engine.cpu_bind
        or cfg.engine.vllm_extra_env.get("VLLM_CPU_OMP_THREADS_BIND")
        or ""
    )
    bound_cpus = expand_thread_binding(bind_str) or None

    telemetry = MeasurementTelemetry(
        cfg.telemetry,
        engine,
        bound_cpus=bound_cpus,
        engine_pid=getattr(engine, "pid", None),
        artifacts_dir=cfg.output.db_directory,
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

        # Parse oneDNN verbose output (only meaningful after engine has run).
        # AMX dispatch fraction is a run-level signal — store it against the
        # last measurement so the export consumer can read it as the
        # "near-knee" AMX picture for that cohort.
        try:
            log_path = getattr(engine, "log_path", None)
            if log_path is not None and Path(log_path).exists():
                amx = parse_amx_utilization(log_path)
                if not amx.is_empty():
                    last_mid = db.fetchone(
                        "SELECT measurement_id FROM cohort_measurements "
                        "WHERE cohort_run_id = ? ORDER BY step_index DESC LIMIT 1",
                        (cohort_run_id,),
                    )
                    if last_mid is not None:
                        db.upsert_aggregate({
                            "measurement_id": last_mid["measurement_id"],
                            "onednn_amx_time_fraction": amx.onednn_amx_time_fraction,
                            "onednn_matmul_dispatches_amx": amx.onednn_matmul_dispatches_amx,
                            "onednn_matmul_dispatches_non_amx": amx.onednn_matmul_dispatches_non_amx,
                        })
        except Exception as e:  # noqa: BLE001
            log.debug("AMX util parse failed: %s", e)

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
    preflight_check(cfg.engine.hardware_requirements)
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
