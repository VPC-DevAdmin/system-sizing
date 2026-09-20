"""Run lifecycle: start, stop and delete runs, list them, build and
serve exports, and replay history for a page opened mid-run — the
thin layer over the same ``run_cohort`` / ``run_sweep`` (and friends)
the CLI drives."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from ..bus import BUS
from .schemas import ExportRequest, StartRunRequest
from .state import ActiveRun, _read_json

log = logging.getLogger(__name__)

router = APIRouter()


def _pkg():
    """The ``simulator.service`` package, resolved at call time. Tests
    monkeypatch names on it (``service._build_custom_config``,
    ``service.STOP_TIMEOUT_S``), so the routes read those through the
    package rather than binding them at import."""
    import simulator.service as pkg
    return pkg


def _build_custom_config(custom: dict, runs_base: Path) -> Path:
    """Advanced path: a downloaded model + hand-set engine shape →
    a generated config file (same schema as promoted profiles).
    Devices come from the detected topology; infeasible shapes are
    refused with the reason."""
    # The shape -> engine translation lives in engines/custom.py so the
    # arena driver builds candidates through the SAME code (levers,
    # the one-memory-knob translation, refused knobs) rather than a
    # private copy that drifts.
    from ..engines.custom import ShapeError, config_doc, custom_engine
    try:
        engine = custom_engine(custom)
    except ShapeError as e:
        raise HTTPException(422, str(e)) from e
    doc = config_doc(engine, runs_base)
    import yaml as _yaml
    out = runs_base / "custom_benchmark.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_yaml.safe_dump(doc, sort_keys=False))
    return out


def _resolve_config_path(req: StartRunRequest) -> Path:
    from ..config import resolve_profile
    if req.profile and req.config:
        raise HTTPException(422, "pass either profile or config, not both")
    if req.profile:
        try:
            return resolve_profile(req.profile)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
    if req.config:
        p = Path(req.config)
        if p.suffix.lower() not in (".yaml", ".yml"):
            raise HTTPException(
                422, f"config must be a .yaml file, got {p.name!r}")
        if not p.exists():
            raise HTTPException(404, f"config not found: {p}")
        return p
    raise HTTPException(422, "pass profile or config")


def _list_runs(base: Path) -> list[dict]:
    """run_NN dirs, newest first, with per-cohort summaries from each
    run.db (read-only; missing/corrupt DBs degrade to an empty list).

    Each cohort row carries enough for the Results run list to stand
    alone: display name, methodology, and a cheap headline verdict
    (max stable arrival rate for open-loop runs, max passing pool for
    closed-loop) read straight from the measurements — no export
    build needed to render the list."""
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
                m_cols = {
                    r[1] for r in
                    conn.execute("PRAGMA table_info(cohort_measurements)")
                }
                r_cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(cohort_run)")
                }
                mode_sel = ("mode" if "mode" in r_cols else "NULL AS mode")
                rows = conn.execute(
                    "SELECT cohort_run_id, cohort_id, engine_type, model_id, "
                    "started_at, completed_at, final_status, "
                    f"cohort_definition_json, {mode_sel}, "
                    "(SELECT COUNT(*) FROM cohort_measurements m "
                    " WHERE m.cohort_run_id = cohort_run.cohort_run_id) AS steps "
                    "FROM cohort_run ORDER BY started_at ASC"
                ).fetchall()
                cohorts = []
                for r in rows:
                    c = dict(r)
                    try:
                        cdef = json.loads(c.pop("cohort_definition_json") or "{}")
                        c["cohort_name"] = cdef.get("name") or c["cohort_id"]
                    except (TypeError, ValueError):
                        c["cohort_name"] = c["cohort_id"]
                    crid = c["cohort_run_id"]
                    # Headline: open-loop rate verdict when present…
                    if "arrival_rate_per_min" in m_cols:
                        h = conn.execute(
                            "SELECT MAX(arrival_rate_per_min) AS rate_max, "
                            "MAX(CASE WHEN capacity_status='pass' THEN "
                            "arrival_rate_per_min END) AS rate_sla "
                            "FROM cohort_measurements WHERE cohort_run_id=? "
                            "AND stability='stable'", (crid,),
                        ).fetchone()
                        c["rate_max_per_min"] = h["rate_max"]
                        c["rate_sla_per_min"] = h["rate_sla"]
                        if h["rate_max"] is not None:
                            s = conn.execute(
                                "SELECT active_sessions_mean FROM "
                                "cohort_measurements WHERE cohort_run_id=? "
                                "AND stability='stable' "
                                "ORDER BY arrival_rate_per_min DESC LIMIT 1",
                                (crid,),
                            ).fetchone()
                            c["sessions_at_max"] = (
                                s["active_sessions_mean"] if s else None)
                    # …and the closed-loop pool verdict as fallback.
                    p = conn.execute(
                        "SELECT MAX(CASE WHEN capacity_status='pass' THEN "
                        "target_pool_size END) AS cap "
                        "FROM cohort_measurements WHERE cohort_run_id=?",
                        (crid,),
                    ).fetchone()
                    c["capacity_pool"] = p["cap"] if p else None
                    # A saturation sweep's headline is peak output
                    # throughput, which lives in its own summary file
                    # rather than the SLA-shaped measurement columns.
                    if c.get("mode") == "headline_sweep":
                        try:
                            sw = json.loads(
                                (d / "headline_sweep.json").read_text())
                            pk = sw.get("peak") or {}
                            c["peak_out_tok_s"] = pk.get("out_tok_s")
                            c["peak_streams"] = pk.get("in_flight")
                        except (OSError, json.JSONDecodeError):
                            pass
                    cohorts.append(c)
                entry["cohorts"] = cohorts
                conn.close()
            except sqlite3.Error as e:
                entry["error"] = str(e)
        exports = sorted(p.name for p in d.glob("buyer_page_data*.json"))
        entry["exports"] = exports
        out.append(entry)
    return out


_CUSTOM_INT_FIELDS = ("replicas", "tp", "max_num_seqs",
                      "max_num_batched_tokens", "max_model_len")


def _validate_custom_ints(custom: dict) -> None:
    """The Advanced form's integer levers arrive as JSON from any
    client; a non-integer used to surface as a 500 from int()."""
    for key in _CUSTOM_INT_FIELDS:
        v = custom.get(key)
        if v is None or v == "" or v == "default":
            continue
        ok = (isinstance(v, int) and not isinstance(v, bool)) or (
            isinstance(v, str) and v.strip().isdigit())
        if not ok or int(v) < 1:
            raise HTTPException(
                422, f"custom.{key} must be a positive integer, got {v!r}")


def _finalise_orphan_runs(base: Path) -> None:
    """Stamp 'interrupted' on cohort_run rows left unfinalised by a
    hard serve kill. Runs at service startup, when nothing can be
    executing in-process — without this, a killed run reads as
    'running' in the UI forever. Only safe under the runs-dir lock:
    see _acquire_runs_lock."""
    if not base.exists():
        return
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    for d in base.glob("run_[0-9]*"):
        p = d / "run.db"
        if not p.exists():
            continue
        try:
            conn = sqlite3.connect(p)
            conn.execute(
                "UPDATE cohort_run SET final_status = 'interrupted', "
                "completed_at = COALESCE(completed_at, ?) "
                "WHERE final_status IS NULL", (now,))
            conn.commit()
            conn.close()
        except sqlite3.Error:
            continue


# ── introspection ─────────────────────────────────────────────

@router.get("/api/status")
async def status(request: Request) -> dict:
    active = request.app.state.active
    return {
        "service": "capsim",
        "bus_subscribers": BUS.subscriber_count,
        "active_run": active.describe() if active else None,
    }


@router.get("/api/runs")
async def runs(request: Request) -> list[dict]:
    # One sqlite open per run dir — dozens of them on a box that
    # has been benchmarking for a month. Off the loop.
    return await asyncio.to_thread(_list_runs, request.app.state.paths.runs_base)


@router.get("/api/runs/{run}/headline")
async def headline_sweep_doc(run: str, request: Request) -> dict:
    """The saturation-sweep summary for a headline run. Separate
    from the capacity export because it answers a different
    question and shares none of its shape."""
    if "/" in run or run.startswith("."):
        raise HTTPException(422, "bad run name")
    path = request.app.state.paths.runs_base / run / "headline_sweep.json"
    if not path.exists():
        raise HTTPException(404, f"{run} has no headline sweep summary")
    try:
        return await asyncio.to_thread(_read_json, path)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"unreadable sweep summary: {e}") from e


@router.get("/api/doctor")
async def doctor() -> dict:
    from ..doctor import run_doctor
    report = await asyncio.to_thread(run_doctor)
    return report.to_dict()


# ── run lifecycle ─────────────────────────────────────────────

@router.post("/api/runs", status_code=202)
async def start_run(req: StartRunRequest, request: Request) -> dict:
    # Serialize the whole check-then-create sequence. There is an
    # await between "is a run already active?" and setting
    # app.state.active, so two near-simultaneous POSTs both
    # cleared the guard and both spawned runs. They then destroyed
    # each other: every engine launch sweeps stale vllm-*
    # containers, so concurrent runs removed one another's
    # replicas and every candidate failed while the GPUs sat idle.
    async with request.app.state.start_lock:
        return await _start_run_locked(request.app, req)


async def _plan_roofline(spec: dict) -> tuple[list[str], list[str]]:
    """(engines, models) a roofline will search: the spec's lists,
    else every staged engine and the ranker's top picks."""
    from ..engine_runtimes import available_engines

    engines = list(spec.get("engines") or available_engines())
    if not engines:
        raise HTTPException(
            422, "no engine runtime is staged — pull one in "
                 "Prepare before starting a roofline")
    models = list(spec.get("models") or [])
    if not models:
        from ..arena import hardware as _hw
        from ..model_catalog import load_model_catalog
        from ..roofline import pick_models
        hw = await asyncio.to_thread(_hw)
        cat = await asyncio.to_thread(load_model_catalog)
        picked = await asyncio.to_thread(
            pick_models, cat,
            vram_per_gpu_gb=hw.get("vram_per_gpu_gb"),
            limit=int(spec.get("model_limit") or 3),
            cached_only=bool(spec.get("cached_only")))
        models = [c.id for c in picked]
    if not models:
        raise HTTPException(422, "no model fits this host")
    return engines, models


async def _start_run_locked(app, req: StartRunRequest) -> dict:
    runs_base = app.state.paths.runs_base
    active = app.state.active
    if active is not None and not active.task.done():
        raise HTTPException(
            409, "a run is already active — stop it first "
                 "(POST /api/runs/stop)",
        )
    opt = app.state.optimizer
    if opt and opt["proc"].poll() is None:
        raise HTTPException(
            409, "the engine optimizer is running — it owns the "
                 "engines/GPUs; stop it first (POST /api/optimizer/stop)",
        )
    roofline_plan: Optional[tuple[list[str], list[str]]] = None
    if (req.workload or {}).get("kind") == "roofline":
        # A roofline varies the model AND the engine per cell, so
        # its `custom` is a TEMPLATE. The UI sends none; default it
        # to the first planned engine, and borrow the first planned
        # (or auto-picked) model just to satisfy the pre-flight
        # build -- the run rebuilds a config for every cell anyway,
        # and validating the template here still catches a shape
        # that does not fit the box.
        roofline_plan = await _plan_roofline(
            dict((req.workload or {}).get("spec") or {}))
        engines, models = roofline_plan
        if req.custom is None:
            req.custom = {"engine": engines[0]}
        if not req.custom.get("model_id"):
            req.custom = {**req.custom, "model_id": models[0]}
    if req.custom is not None:
        _validate_custom_ints(req.custom)
        config_path = await asyncio.to_thread(
            _pkg()._build_custom_config, dict(req.custom), runs_base)
    else:
        config_path = _resolve_config_path(req)

    import yaml as _yaml

    from ..config import load_config
    from ..headline_shapes import is_headline_persona
    from ..personas import COHORTS, PERSONAS, cohort_from_persona
    try:
        cfg = await asyncio.to_thread(load_config, config_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except (_yaml.YAMLError, TypeError, ValueError, AttributeError,
            KeyError) as e:
        # A file that exists but is not a capsim config (wrong
        # schema, not YAML at all) is the caller's error, not a
        # crash.
        raise HTTPException(
            422, f"{config_path} is not a valid capsim config: "
                 f"{type(e).__name__}: {e}") from e
    cfg.output.db_directory = str(runs_base)

    kind = req.workload.get("kind")
    # A headline workload asks a different question than capacity,
    # so it gets a different instrument: a saturation sweep over
    # concurrency rather than an arrival-rate stability search.
    # Selecting one in the UI swaps the mechanism automatically —
    # the user picks a workload, not a methodology.
    sweep_progress: dict = {}
    is_sweep = False
    if kind == "persona" and is_headline_persona(req.workload.get("id")) \
            and req.mode != "closed":
        wid = req.workload.get("id")
        if wid not in PERSONAS:
            raise HTTPException(404, f"unknown persona '{wid}'")
        from ..headline_sweep import run_headline_sweep
        is_sweep = True
        coro_factory = lambda: run_headline_sweep(  # noqa: E731
            cfg, cohort_from_persona(wid), new_run=req.new_run,
            max_concurrency=req.max_concurrency,
            progress=sweep_progress,
        )
    elif kind == "cohort":
        wid = req.workload.get("id")
        if wid not in COHORTS:
            raise HTTPException(404, f"unknown cohort '{wid}'")
        coro_factory = _cohort_coro(cfg, wid, req)
    elif kind == "persona":
        wid = req.workload.get("id")
        if wid not in PERSONAS:
            raise HTTPException(404, f"unknown persona '{wid}'")
        coro_factory = _cohort_coro(cfg, cohort_from_persona(wid), req)
    elif kind == "roofline":
        # Autopilot: stage models, search models x engines x shapes,
        # confirm each model's winner. Hours long by design, and
        # every step is persisted -- see simulator/roofline.py.
        from ..roofline import run_roofline

        spec = dict(req.workload.get("spec") or {})
        assert roofline_plan is not None
        engines, models = roofline_plan

        # GPU-engine defaults. Engines that are not GPU-resident
        # servers (KTransformers: one replica, no KV precision
        # knob) override these per cell -- roofline.engine_defaults
        # -- through the same builder, so the cell records what
        # actually launched.
        base_custom = dict(req.custom or {})
        base_custom.setdefault("device", "gpu")
        base_custom.setdefault("replicas", 8)
        base_custom.setdefault("tp", 1)
        base_custom.setdefault("gpu_memory_utilization", 0.95)
        base_custom.setdefault("kv_cache_dtype", "fp8")
        base_custom.setdefault("max_model_len", 2048)

        def _build_rf(overrides: dict) -> Path:
            return _pkg()._build_custom_config(
                {**base_custom, **overrides}, runs_base)

        from ..roofline import shape_of
        shapes = {"max_num_seqs": spec.get("max_num_seqs"),
                  "output_tokens": spec.get("output_tokens")}
        # Fresh run when either the top-level new_run or the spec's
        # resume:false says so; the UI sends new_run.
        resume = bool(spec.get("resume", True)) and not req.new_run
        coro_factory = lambda: run_roofline(  # noqa: E731
            models=models, engines=engines,
            shapes={k: v for k, v in shapes.items() if v},
            input_tokens=int(spec.get("input_tokens") or 128),
            build_config=_build_rf, runs_base=runs_base,
            resume=resume,
            confirm_winners=bool(spec.get("confirm_winners", True)),
            engine_shape=shape_of(base_custom),
        )
    elif kind == "headline_optimize":
        # Engine shape and request shape are coupled, so they are
        # searched together; see simulator/headline_optimize.py.
        from ..headline_optimize import run_headline_optimize
        wid = req.workload.get("id") or "headline_generation"
        if wid not in PERSONAS:
            raise HTTPException(404, f"unknown persona '{wid}'")
        if req.custom is None:
            raise HTTPException(
                422, "the joint search needs a custom engine (pick a "
                     "model and engine settings) — it varies "
                     "max_num_seqs against that baseline")
        base_custom = dict(req.custom)

        def _build(overrides: dict) -> Path:
            return _pkg()._build_custom_config(
                {**base_custom, **overrides}, runs_base)

        is_sweep = True          # reuse the sweep progress holder
        coro_factory = lambda: run_headline_optimize(  # noqa: E731
            cfg, cohort_from_persona(wid),
            preset=req.preset or "standard",
            max_num_seqs=req.search_max_num_seqs,
            output_tokens=req.search_output_tokens,
            engines=req.search_engines,
            input_tokens=req.input_tokens or 128,
            build_config=_build, runs_base=runs_base,
            progress=sweep_progress,
        )
    elif kind == "headline_search":
        from ..headline_search import run_headline_search
        shape_progress: dict = {}
        coro_factory = lambda: run_headline_search(  # noqa: E731
            cfg, new_run=req.new_run, progress=shape_progress,
            input_tokens=req.input_tokens,
        )
    elif kind == "sweep":
        from ..personas import resolve_workload_group
        try:
            persona_ids, cohort_ids = resolve_workload_group(
                req.workload.get("type", "all")
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        from ..runner import run_sweep
        coro_factory = lambda: run_sweep(  # noqa: E731
            cfg, persona_ids=persona_ids, cohort_ids=cohort_ids,
            new_run=req.new_run, adaptive=req.adaptive,
            fixed_grid_pool_sizes=req.pool_sizes,
        )
    else:
        raise HTTPException(
            422, "workload.kind must be cohort | persona | "
                 "headline_search | headline_optimize | "
                 "roofline | sweep",
        )

    if req.custom is not None:
        c = req.custom
        summary = (
            "custom CPU engine" if c.get("device") == "cpu"
            else "custom: "
            + f"{c.get('replicas') or 1}×tp{c.get('tp') or 1}"
            + f" {c.get('placement') or 'pack'}"
            + f" · gmu {c.get('gpu_memory_utilization') or 0.92}"
            + (f" · mns {c['max_num_seqs']}"
               if c.get("max_num_seqs") else "")
            + (f" · mbt {c['max_num_batched_tokens']}"
               if c.get("max_num_batched_tokens") else "")
            + f" · KV {c.get('kv_cache_dtype') or 'auto'}"
            + (" · EP on" if c.get("expert_parallel") else "")
            + (" · trust-remote-code"
               if c.get("trust_remote_code") else "")
        )
    elif req.profile:
        summary = f"profile {req.profile}"
    else:
        summary = f"config {config_path.name}"
    active = ActiveRun(
        task=asyncio.create_task(_supervise(app, coro_factory)),
        workload=req.workload,
        config_path=str(config_path),
        started_at=time.time(),
        engine_summary=summary,
        progress=(shape_progress if kind == "headline_search"
                  else sweep_progress if is_sweep else None),
    )
    app.state.active = active
    return {"accepted": True, "workload": req.workload,
            "config": str(config_path), "engine_summary": summary}


def _cohort_coro(cfg, cohort, req: StartRunRequest):
    # Explicit closed-loop knobs (pool grid / adaptive stepper)
    # imply the legacy methodology even if mode wasn't set.
    mode = req.mode or getattr(cfg.simulation, "mode", "open")
    closed = (
        mode == "closed" or req.adaptive or bool(req.pool_sizes)
    )
    if closed:
        from ..runner import run_cohort
        return lambda: run_cohort(
            cfg, cohort,
            new_run=req.new_run,
            adaptive=req.adaptive,
            fixed_grid_pool_sizes=req.pool_sizes,
        )
    from ..open_loop import run_cohort_open_loop
    return lambda: run_cohort_open_loop(
        cfg, cohort, new_run=req.new_run,
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


@router.delete("/api/runs/{run_name}/cohorts/{cohort_run_id}")
async def delete_cohort_run(
        run_name: str, cohort_run_id: str, request: Request) -> dict:
    """Fully delete one cohort run's data — measurements, turns,
    telemetry, snapshots, users, and the run row. When it was the
    last cohort in its run_NN dir, the whole dir goes (engine
    logs included). Refused while any run is active: the runner
    writes to these tables in-process."""
    if "/" in run_name or run_name.startswith("."):
        raise HTTPException(422, "bad run name")
    active = request.app.state.active
    if active is not None and not active.task.done():
        raise HTTPException(
            409, "a run is active — deleting run data while the "
                 "runner writes to it is unsafe; stop it first")
    d = request.app.state.paths.runs_base / run_name
    db_path = d / "run.db"
    if not db_path.exists():
        raise HTTPException(404, f"no run.db in {d}")

    def _delete() -> dict:
        import shutil
        conn = sqlite3.connect(db_path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM cohort_run WHERE cohort_run_id = ?",
                (cohort_run_id,),
            ).fetchone()
            if not exists:
                raise HTTPException(
                    404, f"unknown cohort run {cohort_run_id}")
            mids = [r[0] for r in conn.execute(
                "SELECT measurement_id FROM cohort_measurements "
                "WHERE cohort_run_id = ?", (cohort_run_id,),
            ).fetchall()]
            if mids:
                ph = ",".join("?" for _ in mids)
                conn.execute(
                    f"DELETE FROM turn_events WHERE measurement_id IN ({ph})",
                    mids)
                conn.execute(
                    f"DELETE FROM measurement_telemetry "
                    f"WHERE measurement_id IN ({ph})", mids)
            for table in ("cohort_measurements", "simulation_snapshots",
                          "virtual_users", "cohort_run"):
                conn.execute(
                    f"DELETE FROM {table} WHERE cohort_run_id = ?",
                    (cohort_run_id,))
            conn.commit()
            remaining = conn.execute(
                "SELECT COUNT(*) FROM cohort_run").fetchone()[0]
        finally:
            conn.close()
        # Cached exports are stale either way.
        for p in d.glob("buyer_page_data*.json"):
            p.unlink(missing_ok=True)
        run_removed = remaining == 0
        if run_removed:
            shutil.rmtree(d, ignore_errors=True)
        return {"deleted": cohort_run_id, "run_removed": run_removed}

    return await asyncio.to_thread(_delete)


@router.post("/api/runs/stop")
async def stop_run(request: Request) -> dict:
    active = request.app.state.active
    if active is None or active.task.done():
        raise HTTPException(409, "no active run")
    active.task.cancel()
    STOP_TIMEOUT_S = _pkg().STOP_TIMEOUT_S
    # Engine teardown (container stop, replica drain) can take
    # longer than a browser is willing to wait on one request.
    # After STOP_TIMEOUT_S answer 202 "stopping": the cancel is
    # delivered and /api/status will flip when teardown ends.
    done, _ = await asyncio.wait({active.task}, timeout=STOP_TIMEOUT_S)
    if not done:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=202, content={
            "stopped": False, "stopping": True,
            "detail": f"stop requested; engine teardown still running "
                      f"after {STOP_TIMEOUT_S:.0f}s — poll /api/status",
        })
    return {"stopped": True}


# ── export ────────────────────────────────────────────────────

@router.post("/api/export")
async def export(req: ExportRequest, request: Request) -> dict:
    from ..export import export_dir
    try:
        doc, out_path = await asyncio.to_thread(
            export_dir, request.app.state.paths.runs_base, None, slim=req.slim,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return {
        "path": str(out_path),
        "schema_version": doc["schema_version"],
        "cohort_count": doc["meta"]["cohort_count"],
        "slim": req.slim,
    }


@router.get("/api/export/latest")
async def export_latest(request: Request, slim: bool = False) -> Any:
    from ..runs import latest_run_dir
    d = latest_run_dir(request.app.state.paths.runs_base)
    if d is None:
        raise HTTPException(404, "no runs yet")
    fname = "buyer_page_data_slim.json" if slim else "buyer_page_data.json"
    p = d / fname
    if not p.exists():
        raise HTTPException(
            404, f"{p} not built — POST /api/export first",
        )
    return await asyncio.to_thread(_read_json, p)


@router.get("/api/runs/{run_name}/export")
async def run_export(
        run_name: str, request: Request, slim: bool = False) -> Any:
    """Export JSON for one run_NN dir — reads the built file, or
    builds it on first request (a few seconds on a big run.db).
    Powers the UI's results + run-comparison views."""
    if "/" in run_name or run_name.startswith("."):
        raise HTTPException(422, "bad run name")
    d = request.app.state.paths.runs_base / run_name
    if not (d / "run.db").exists():
        raise HTTPException(404, f"no run.db in {d}")
    fname = "buyer_page_data_slim.json" if slim else "buyer_page_data.json"
    p = d / fname
    # One build at a time per run: the Results list and a
    # comparison can ask for the same run together, and two
    # concurrent export_dir calls wrote the same file over each
    # other. The second waiter re-checks staleness under the lock
    # and finds the first one's fresh file.
    lock = request.app.state.export_locks.setdefault(
        f"{run_name}|{fname}", asyncio.Lock())
    async with lock:
        # Rebuild when the DB has newer data than the cached export
        # — a Results tab opened mid-run builds a partial export,
        # and serving that snapshot forever would freeze the run at
        # whatever step it happened to be on.
        stale = (
            p.exists()
            and p.stat().st_mtime < (d / "run.db").stat().st_mtime
        )
        if not p.exists() or stale:
            from ..export import export_dir
            try:
                await asyncio.to_thread(export_dir, d, p, slim=slim)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, f"export failed: {e}") from e
        return await asyncio.to_thread(_read_json, p)


@router.get("/api/live/backfill")
async def live_backfill(request: Request, window_s: int = 600) -> dict:
    """History for the newest cohort run, shaped like the live
    WS events — a page opened mid-run (or after) replays this
    into the same chart handlers, so the view shows where the
    run IS and what it has done, not just what happens next."""
    from ..runs import latest_run_dir
    runs_base = request.app.state.paths.runs_base

    def _read() -> dict:
        d = latest_run_dir(runs_base)
        if d is None or not (d / "run.db").exists():
            return {"run": None, "snapshots": [], "telemetry": [],
                    "turns": [], "steps": []}
        conn = sqlite3.connect(f"file:{d / 'run.db'}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            run = conn.execute(
                "SELECT cohort_run_id, cohort_id, engine_type, "
                "model_id, started_at, completed_at, final_status "
                "FROM cohort_run ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if run is None:
                return {"run": None, "snapshots": [], "telemetry": [],
                        "turns": [], "steps": []}
            crid = run["cohort_run_id"]
            cutoff = int((time.time() - window_s) * 1000)
            snapshots = [dict(r) for r in conn.execute(
                "SELECT * FROM simulation_snapshots WHERE "
                "cohort_run_id = ? AND snapshot_at_ms > ? "
                "ORDER BY snapshot_at_ms DESC LIMIT 600",
                (crid, cutoff)).fetchall()][::-1]
            # SELECT * + shape in Python: the DB is opened
            # read-only (no migrations), so naming late-added
            # columns (e.g. v7's arrival_rate_per_min) would 500
            # on any pre-migration run.db.
            steps = []
            for r in conn.execute(
                "SELECT * FROM cohort_measurements WHERE "
                "cohort_run_id = ? ORDER BY step_index", (crid,),
            ).fetchall():
                row = dict(r)
                steps.append({
                    "step_index": row.get("step_index"),
                    "pool_size": row.get("target_pool_size"),
                    "sample_size": row.get("sample_size"),
                    "combined_violation_rate":
                        row.get("combined_violation_rate"),
                    "combined_target_miss_rate":
                        row.get("combined_target_miss_rate"),
                    "ttft_p95_ms": row.get("ttft_p95_ms"),
                    "tpot_p95_ms": row.get("tpot_p95_ms"),
                    "capacity_status": row.get("capacity_status"),
                    "arrival_rate_per_min":
                        row.get("arrival_rate_per_min"),
                    "stability": row.get("stability"),
                })
            mids = [r[0] for r in conn.execute(
                "SELECT measurement_id FROM cohort_measurements "
                "WHERE cohort_run_id = ?", (crid,)).fetchall()]
            telemetry, turns = [], []
            if mids:
                ph = ",".join("?" for _ in mids)
                telemetry = [dict(r) for r in conn.execute(
                    f"SELECT * FROM measurement_telemetry WHERE "
                    f"measurement_id IN ({ph}) AND sampled_at_ms > ? "
                    f"ORDER BY sampled_at_ms DESC LIMIT 300",
                    (*mids, cutoff)).fetchall()][::-1]
                turns = [dict(r) for r in conn.execute(
                    f"SELECT completed_at_ms, ttft_ms, tpot_ms, "
                    f"error FROM turn_events WHERE "
                    f"measurement_id IN ({ph}) "
                    f"ORDER BY completed_at_ms DESC LIMIT 40",
                    (*mids,)).fetchall()][::-1]
            return {"run": dict(run), "snapshots": snapshots,
                    "telemetry": telemetry, "turns": turns,
                    "steps": steps}
        finally:
            conn.close()
    return await asyncio.to_thread(_read)
