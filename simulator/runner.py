"""Top-level cohort-run orchestration."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from .adaptive import FixedGridStepper, StepResult, TwoKneeStepper
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
from .runs import resolve_run_dir
from .telemetry import MeasurementTelemetry, SnapshotRecorder
from .tokenizer_corpus import TokenCorpus
from .virtual_user import SharedState, _now_ms

log = logging.getLogger(__name__)


def _run_db_path(run_dir: Path) -> Path:
    """The canonical DB path for a run dir.

    One DB per ``run_NN/`` — every cohort/persona invocation against
    the same run dir appends a new ``cohort_run`` row (and its
    measurements / events / telemetry) to the same file. The schema
    keys all rows by ``cohort_run_id`` so cohorts do not interfere.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "run.db"


def _config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def load_config_from_run_dir(run_dir: Path) -> Config:
    """Reconstruct the Config that was used to produce the run.db in
    ``run_dir``. Reads any cohort_run row's ``config_json`` and applies
    it onto a default ``Config()`` via the same merger ``load_config``
    uses for YAML files.

    Used by spot-check / audit follow-on runs to guarantee the
    follow-on uses the SAME engine / model / KV pool / quantization
    as the original sweep — otherwise the new measurements aren't
    comparable to the cohort's existing curve.
    """
    from .config import _merge_dataclass, ReplicaConfig
    db_path = _run_db_path(run_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"No run.db in {run_dir}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT config_json FROM cohort_run "
            "ORDER BY started_at ASC LIMIT 1"
        ).fetchone()
    if row is None or not row[0]:
        raise ValueError(
            f"No cohort_run rows in {db_path} — cannot reconstruct "
            f"engine config from this run dir."
        )
    import json as _json
    raw = _json.loads(row[0])
    cfg = Config()
    _merge_dataclass(cfg, raw)
    # Same special-case as load_config(): list[ReplicaConfig] needs
    # explicit dataclass reconstruction since the generic merger only
    # handles dataclass-typed scalars.
    if cfg.engine.replicas and isinstance(cfg.engine.replicas[0], dict):
        cfg.engine.replicas = [ReplicaConfig(**r) for r in cfg.engine.replicas]
    return cfg


def _cohort_to_dict(cohort: Cohort) -> dict:
    return {
        "id": cohort.id,
        "name": cohort.name,
        "description": cohort.description,
        "category": cohort.category,
        "persona_weights": cohort.persona_weights,
    }


async def run_cohort(
    cfg: Config,
    cohort: Cohort | str,
    *,
    engine: Engine | None = None,
    db_path: Path | None = None,
    run_dir: Path | None = None,
    new_run: bool = False,
    # Spot-check / audit-only escape hatch: measure exactly these pool
    # sizes in order, NO early-stop, regardless of violation rate. Used
    # by ``run_spot_check`` to re-measure flagged points. Production
    # sweeps should use ``adaptive`` + ``fixed_grid_pool_sizes`` below
    # — those go through the FixedGridStepper / TwoKneeStepper
    # interface and get early-stop-on-failure semantics.
    pool_size_override: list[int] | None = None,
    # Stepper-mode controls (mutually exclusive with pool_size_override):
    #   * adaptive=True → TwoKneeStepper (knee-localizing bisection).
    #   * adaptive=False (default) → FixedGridStepper at the powers-of-2
    #     grid in ``adaptive.DEFAULT_FIXED_GRID``, OR at
    #     ``fixed_grid_pool_sizes`` when supplied. Both fixed-grid
    #     paths early-stop one step past the first observed failure.
    adaptive: bool = False,
    fixed_grid_pool_sizes: list[int] | None = None,
    # Append mode: when set, attach the new measurements to an
    # existing ``cohort_run`` row instead of creating a new one.
    # Used together with ``pool_size_override`` to enrich a
    # previously-completed cohort with extra points (e.g. an
    # audit-driven spot-check). The pre-existing ``final_status``
    # is preserved — we don't re-finalise.
    existing_cohort_run_id: str | None = None,
) -> Path:
    """Run a single cohort end-to-end. Returns the resulting db path.

    ``cohort`` accepts either a Cohort object (e.g. one built via
    ``cohort_from_persona`` for persona runs) or a cohort-id string
    looked up in COHORTS.

    ``run_dir`` selects the ``run_NN/`` artifact directory; if omitted
    the latest existing one is reused (or ``run_01`` is created).
    Pass ``new_run=True`` to force a fresh ``run_NN+1``.
    """
    if isinstance(cohort, str):
        cohort = get_cohort(cohort)
    cohort_id = cohort.id
    if run_dir is None:
        run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    own_engine = engine is None
    if own_engine:
        # Validate the host before any subprocess / docker / model load —
        # SGLang FP8 on AMD wastes 10-20 minutes before the assertion.
        preflight_check(cfg.engine.hardware_requirements)
        engine = make_engine(cfg.engine.type, cfg.engine)
        engine.launch(log_dir=run_dir)

    if db_path is None:
        db_path = _run_db_path(run_dir)

    db = Database(db_path)
    append_mode = existing_cohort_run_id is not None
    if append_mode:
        cohort_run_id = existing_cohort_run_id
        # Continue step_index after the existing rows.
        last_step = db.fetchone(
            "SELECT MAX(step_index) AS s FROM cohort_measurements "
            "WHERE cohort_run_id = ?",
            (cohort_run_id,),
        )
        starting_step_index = (
            (last_step["s"] + 1) if last_step and last_step["s"] is not None
            else 0
        )
        log.info(
            "Append-mode cohort run %s (existing) — continuing step_index "
            "from %d", cohort_run_id, starting_step_index,
        )
    else:
        cohort_run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        db.insert_run(
            cohort_run_id=cohort_run_id,
            started_at=started_at,
            engine_type=cfg.engine.type,
            model_id=cfg.engine.model_id,
            cohort_id=cohort_id,
            cohort_definition=_cohort_to_dict(cohort),
            config=_config_to_dict(cfg),
        )
        starting_step_index = 0
        log.info("Cohort run %s -> %s", cohort_run_id, db_path)

    state = SharedState()
    phase_tracker = PhaseTracker()
    corpus = TokenCorpus(cfg.engine.model_id)
    # One AsyncOpenAI client per replica. Single-backend engines
    # return ``[base_url]`` from replica_urls, so the typical case
    # produces a one-element list. Multi-replica engines (dual-socket
    # NUMA-pinned vLLM) produce N elements; the pool manager
    # hash-routes each virtual user to a stable replica so multi-turn
    # conversations preserve prefix-cache locality on one backend.
    clients = [
        AsyncOpenAI(base_url=url, api_key=engine.api_key)
        for url in engine.replica_urls
    ]
    log.info(
        "Built %d client(s) for replica URLs: %s",
        len(clients), engine.replica_urls,
    )
    log.info("API model name (sent in chat/completions body): %s",
             engine.api_model_name)

    user_termination_buffer: list = []
    pool = PoolManager(
        cohort=cohort,
        clients=clients,
        model_id=engine.api_model_name,
        corpus=corpus,
        state=state,
        request_timeout_s=cfg.simulation.request_timeout_s,
        on_user_terminated=lambda s: user_termination_buffer.append(s),
        capture_token_timestamps=cfg.simulation.enable_token_timestamps,
        ramp_spawn_interval_s=cfg.simulation.ramp_spawn_interval_s,
        initial_phase_offset_enabled=cfg.simulation.initial_phase_offset_enabled,
        # Only pass reasoning_effort when the engine is declared as a
        # reasoning model — keeps non-reasoning request bodies clean.
        reasoning_effort=(
            cfg.engine.reasoning_effort if cfg.engine.reasoning else None
        ),
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
        artifacts_dir=run_dir,
    )

    # Stepper selection (precedence: spot-check override → adaptive →
    # fixed grid). All three drive the same measurement loop below;
    # only the next-pool-size decision differs:
    #   * pool_size_override  — explicit list, no early-stop. Used only
    #                           by run_spot_check / audit re-runs.
    #   * adaptive            — TwoKneeStepper (locates both knees +
    #                           infill via Wilson-CI-aware bisection).
    #   * default             — FixedGridStepper at powers-of-2
    #                           [4..256] (or ``fixed_grid_pool_sizes``
    #                           when supplied) with early-stop one
    #                           step past the first observed failure.
    if pool_size_override is not None:
        stepper = None
        override_iter = iter(pool_size_override)
    elif adaptive:
        stepper = TwoKneeStepper(
            initial_pool_size=cfg.simulation.initial_pool_size,
            max_pool_size=cfg.simulation.max_pool_size,
            stop_violation_threshold=cfg.simulation.stop_violation_threshold,
        )
        override_iter = None
    else:
        stepper = FixedGridStepper(
            grid=fixed_grid_pool_sizes,
            failure_threshold=cfg.simulation.stop_violation_threshold,
        )
        override_iter = None
    final_status = "ok"
    run_started = time.monotonic()
    max_total_s = cfg.simulation.max_total_duration_minutes * 60
    step_index = starting_step_index

    try:
        next_size = (
            stepper.next_pool_size() if stepper is not None
            else next(override_iter, None)
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
                warmup_min_duration_s=cfg.simulation.warmup_min_duration_s,
                warmup_max_duration_s=cfg.simulation.warmup_max_duration_s,
                convergence_window_s=cfg.simulation.convergence_window_s,
                convergence_threshold=cfg.simulation.convergence_threshold,
                convergence_min_completions_per_window=cfg.simulation.convergence_min_completions_per_window,
            )

            # Persist any user terminations seen
            _flush_users(db, cohort_run_id, user_termination_buffer)

            if result.status == "no_samples":
                final_status = "no_samples"
                break

            if stepper is not None:
                stepper.record(StepResult(
                    pool_size=result.target_pool_size,
                    violation_rate=result.combined_violation_rate,
                    target_miss_rate=result.combined_target_miss_rate,
                    # Wilson-CI-aware stepper: gate phase decisions on
                    # the actual sample size rather than the implicit
                    # n=100 default. Critical when measurement windows
                    # capture few samples (e.g. long-context cohorts at
                    # high concurrency where only n=5-9 turns finish).
                    sample_size=result.sample_size,
                ))
            step_index += 1

            next_size = (
                stepper.next_pool_size() if stepper is not None
                else next(override_iter, None)
            )
    except KeyboardInterrupt:
        final_status = "interrupted"
        raise
    finally:
        phase_tracker.set(PHASE_IDLE)
        await snap.stop()
        # Sticky-routing sanity check before tearing the pool down.
        # Multi-replica engines should show ±1 user-assignment balance;
        # heavy skew indicates a bug in _client_for_user.
        if len(clients) > 1:
            log.info("Replica routing balance: %s", pool.routing_summary())
        # End-of-run engine metrics scrape — captures the cumulative
        # prefix-cache counters so the export can show "engine-reported
        # X% hit rate" alongside the per-session TTFT analysis. Runs
        # for ALL engine types (vllm, sglang, vllm_dual_socket); the
        # dual_socket case sums across replicas.
        try:
            eng_metrics = engine.get_metrics()
            hits = eng_metrics.get("prefix_cache_hits")
            queries = eng_metrics.get("prefix_cache_queries")
            hit_rate = eng_metrics.get("prefix_cache_hit_rate")
            if hits is not None and queries is not None:
                log.info(
                    "Prefix-cache (engine, aggregated): hits=%s queries=%s "
                    "hit_rate=%s",
                    int(hits), int(queries),
                    f"{hit_rate:.1%}" if hit_rate is not None else "—",
                )
            elif hit_rate is not None:
                log.info(
                    "Prefix-cache (engine, aggregated): hit_rate=%.1f%%",
                    hit_rate * 100.0,
                )
            if hits is not None or hit_rate is not None:
                db.update_cohort_run(cohort_run_id, {
                    "prefix_cache_engine_hits": (
                        int(hits) if hits is not None else None
                    ),
                    "prefix_cache_engine_queries": (
                        int(queries) if queries is not None else None
                    ),
                    "prefix_cache_engine_hit_rate": (
                        float(hit_rate) if hit_rate is not None else None
                    ),
                })
        except Exception as e:  # noqa: BLE001
            log.debug("end-of-run metrics scrape failed: %s", e)
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
                        db.update_measurement(last_mid["measurement_id"], {
                            "onednn_amx_time_fraction": amx.onednn_amx_time_fraction,
                            "onednn_matmul_dispatches_amx": amx.onednn_matmul_dispatches_amx,
                            "onednn_matmul_dispatches_non_amx": amx.onednn_matmul_dispatches_non_amx,
                        })
        except Exception as e:  # noqa: BLE001
            log.debug("AMX util parse failed: %s", e)

        if not append_mode:
            db.finalise_run(
                cohort_run_id=cohort_run_id,
                completed_at=datetime.now(timezone.utc).isoformat(),
                status=final_status,
            )
        # Append-mode: leave the existing cohort_run row alone — its
        # final_status was set when the original sweep finished. The
        # spot-check just enriched the cohort_measurements rows.
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


def find_completed_runs(
    runs_dir: Path,
    engine_type: str,
    model_id: str,
) -> set[str]:
    """Return the set of cohort_ids that already have a completed
    ``cohort_run`` row for the given (engine_type, model_id) inside
    ``runs_dir/run.db``.

    A run counts as completed iff ``cohort_run.final_status == 'ok'``.
    Other statuses — ``interrupted`` (Ctrl-C / SSH disconnect),
    ``time_limit`` (max duration hit), ``no_samples`` / ``unstable``
    (didn't capture useful data) — are deliberately NOT skipped: those
    are exactly the runs the user probably wants to retry. Their
    leftover rows aren't deleted; they're available for inspection
    inside the same DB, just not counted as "done."

    For backwards compatibility with the pre-consolidation flat layout,
    this also scans any other ``*.db`` files in the directory.
    """
    completed: set[str] = set()
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return completed
    for db_path in runs_dir.glob("*.db"):
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                rows = conn.execute(
                    "SELECT cohort_id FROM cohort_run "
                    "WHERE engine_type = ? AND model_id = ? "
                    "AND final_status = 'ok'",
                    (engine_type, model_id),
                ).fetchall()
                for r in rows:
                    completed.add(r[0])
        except sqlite3.Error:
            continue
    return completed


async def run_sweep(
    cfg: Config,
    persona_ids: list[str] | None = None,
    cohort_ids: list[str] | None = None,
    *,
    new_run: bool = False,
    # Stepper mode for every cohort in the sweep. Defaults match
    # run_cohort: adaptive=False uses FixedGridStepper (the new
    # default), adaptive=True opts into the TwoKneeStepper.
    # ``fixed_grid_pool_sizes`` overrides the default
    # powers-of-2 grid when adaptive=False.
    adaptive: bool = False,
    fixed_grid_pool_sizes: list[int] | None = None,
) -> list[Path]:
    """Run multiple personas + cohorts back-to-back against the same engine.

    Personas run first (each as an ephemeral one-persona Cohort), then
    cohorts. The engine is launched once and stays up across the sweep.

    Resume is the default: artifacts land in the latest ``run_NN/``
    subdirectory of ``cfg.output.db_directory``, and personas/cohorts
    that already have ``final_status='ok'`` in that directory are
    skipped. To start a fresh ``run_NN+1`` (no resume), pass
    ``new_run=True``.
    """
    from .personas import cohort_from_persona

    persona_ids = list(persona_ids or [])
    cohort_ids = list(cohort_ids or [])
    if not persona_ids and not cohort_ids:
        raise ValueError("run_sweep requires at least one persona_id or cohort_id")

    run_dir = resolve_run_dir(cfg.output.db_directory, new=new_run)
    log.info("Sweep run dir: %s (new_run=%s)", run_dir, new_run)

    if not new_run:
        completed = find_completed_runs(
            run_dir,
            cfg.engine.type,
            cfg.engine.model_id,
        )
        skipped_p = [p for p in persona_ids if p in completed]
        skipped_c = [c for c in cohort_ids if c in completed]
        persona_ids = [p for p in persona_ids if p not in completed]
        cohort_ids = [c for c in cohort_ids if c not in completed]
        if skipped_p or skipped_c:
            log.info(
                "Resume: skipping %d already-completed (%s)",
                len(skipped_p) + len(skipped_c),
                ", ".join(skipped_p + skipped_c),
            )
        if not persona_ids and not cohort_ids:
            log.info("Resume: nothing to do — all workloads already completed")
            return []

    preflight_check(cfg.engine.hardware_requirements)
    engine = make_engine(cfg.engine.type, cfg.engine)
    engine.launch(log_dir=run_dir)
    if adaptive:
        log.info("Sweep mode: adaptive (TwoKneeStepper)")
    else:
        from .adaptive import DEFAULT_FIXED_GRID
        grid = fixed_grid_pool_sizes or DEFAULT_FIXED_GRID
        log.info(
            "Sweep mode: fixed grid %s (early-stop one step past first failure)",
            grid,
        )
    paths: list[Path] = []
    try:
        for pid in persona_ids:
            log.info("=== Sweep: persona %s ===", pid)
            path = await run_cohort(
                cfg, cohort_from_persona(pid), engine=engine, run_dir=run_dir,
                adaptive=adaptive,
                fixed_grid_pool_sizes=fixed_grid_pool_sizes,
            )
            paths.append(path)
        for cid in cohort_ids:
            log.info("=== Sweep: cohort %s ===", cid)
            path = await run_cohort(
                cfg, cid, engine=engine, run_dir=run_dir,
                adaptive=adaptive,
                fixed_grid_pool_sizes=fixed_grid_pool_sizes,
            )
            paths.append(path)
    finally:
        engine.shutdown()
    return paths


async def run_spot_check(
    cfg: Config,
    plan: dict,
    *,
    run_dir: Path,
) -> list[Path]:
    """Re-measure specific (cohort, pool_size) points from an audit
    plan. Each point is appended to its existing cohort_run row,
    enriching the curve with extra measurements without re-running
    the full ramp.

    ``plan`` is the JSON produced by ``scripts/audit_run.py``: a dict
    with key ``rerun_points = [{"cohort_id", "cohort_run_id", "pool_size", ...}, ...]``.
    Points are grouped by cohort_run_id; one engine launch covers all
    of them.
    """
    from collections import defaultdict
    from .personas import cohort_from_persona

    rerun = plan.get("rerun_points") or []
    if not rerun:
        log.info("Spot-check: no rerun points in plan — nothing to do")
        return []

    by_crid: dict[str, dict] = defaultdict(
        lambda: {"cohort_id": None, "pools": []}
    )
    for entry in rerun:
        crid = entry["cohort_run_id"]
        by_crid[crid]["cohort_id"] = entry["cohort_id"]
        by_crid[crid]["pools"].append(int(entry["pool_size"]))

    log.info(
        "Spot-check: %d existing cohort_run row(s), %d total point(s)",
        len(by_crid), sum(len(v["pools"]) for v in by_crid.values()),
    )

    preflight_check(cfg.engine.hardware_requirements)
    engine = make_engine(cfg.engine.type, cfg.engine)
    engine.launch(log_dir=run_dir)
    paths: list[Path] = []
    try:
        for crid, info in by_crid.items():
            cohort_id = info["cohort_id"]
            pools = sorted(set(info["pools"]))
            log.info(
                "=== Spot-check: cohort %s (run_id=%s) at pools %s ===",
                cohort_id, crid, pools,
            )
            # Build the cohort object — try the cohorts dict first,
            # fall back to ephemeral persona-cohort for persona-only
            # entries (e.g. forklifted personas).
            try:
                cohort: Cohort | str = get_cohort(cohort_id)
            except KeyError:
                cohort = cohort_from_persona(cohort_id)
            path = await run_cohort(
                cfg, cohort, engine=engine, run_dir=run_dir,
                pool_size_override=pools,
                existing_cohort_run_id=crid,
            )
            paths.append(path)
    finally:
        engine.shutdown()
    return paths
