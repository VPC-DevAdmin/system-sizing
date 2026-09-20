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

import asyncio
import contextlib
import json
import logging
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from . import __version__
from .bus import BUS

log = logging.getLogger(__name__)

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


class StartRunRequest(BaseModel):
    profile: Optional[str] = None
    config: Optional[str] = None
    # Advanced: benchmark a downloaded model with hand-set engine
    # shape instead of a saved profile. {"model_id", "replicas",
    # "tp", "max_num_seqs"?, "max_num_batched_tokens"?,
    # "kv_cache_dtype"?} — devices are assigned from this machine's
    # detected topology.
    custom: Optional[dict] = None
    # {"kind": "cohort"|"persona", "id": "..."} or {"kind": "sweep",
    # "type": "all"|"personas"|"cohorts"|"a,b,c"}
    workload: dict
    new_run: bool = False
    pool_sizes: Optional[list[int]] = None
    adaptive: bool = False
    # Methodology. "open" (default) — open-loop Poisson session
    # arrivals; capacity is the arrival rate where the engine's queue
    # turns divergent. "closed" — the legacy fixed-pool ramp (kept for
    # comparison runs and for the pool_sizes / adaptive knobs, which
    # only apply there). Sweeps always run closed-loop.
    mode: str = "open"
    # Headline sweeps only: cap the concurrency ladder (the UI's
    # "max concurrent streams" control). None = the full ladder.
    max_concurrency: Optional[int] = None
    # Shape search only: the pinned prompt length. None = config
    # default (128, the vendor convention).
    input_tokens: Optional[int] = None
    # Joint engine+shape search: which grid to walk. The explicit
    # lists override the preset — a model's KV cost per token decides
    # which shapes are even reachable (Llama-70B is 16x Qwen3.6's, so
    # its grid belongs at short outputs), and that is not something a
    # fixed preset can know.
    preset: Optional[str] = None
    search_max_num_seqs: Optional[list[int]] = None
    search_output_tokens: Optional[list[int]] = None
    # Which servers the joint search covers. None = vLLM only, so an
    # unqualified search costs what it always did; naming both makes
    # the engine a measured dimension rather than an assumption.
    search_engines: Optional[list[str]] = None


class ExportRequest(BaseModel):
    slim: bool = False


class SaveSpecRequest(BaseModel):
    # Either form saves the same catalog entry. The graphical designer
    # sends ``spec`` (structured JSON — the UI has no YAML anywhere);
    # ``yaml`` remains for API users and older clients.
    yaml: Optional[str] = None
    spec: Optional[dict] = None


class ModelDownloadRequest(BaseModel):
    model: str


class ModelAddRequest(BaseModel):
    model: str                          # HF repo id, org/name
    family: Optional[str] = None        # default: inferred from the id
    quant: Optional[str] = None         # default: inferred from the id
    notes: str = ""
    # Verify the repo exists on the Hub before adding (best-effort:
    # an offline box adds unverified rather than being blocked).
    check_hub: bool = True
    # Rich metadata — the discovery flow fills these from the Hub's
    # own safetensors metadata so an added model arrives fully sized.
    series: Optional[str] = None
    params_b: Optional[float] = None
    moe: Optional[bool] = None
    approx_size_gb: Optional[float] = None
    min_vram_gb: Optional[float] = None
    specialty: Optional[str] = None


class RooflineRequest(BaseModel):
    """Autopilot: stage models, search engines x shapes, confirm, report."""
    # Explicit model list, or None to let the ranker choose.
    models: Optional[list[str]] = None
    model_limit: int = 3
    cached_only: bool = False
    engines: Optional[list[str]] = None      # None = every staged engine
    max_num_seqs: Optional[list[int]] = None
    output_tokens: Optional[list[int]] = None
    input_tokens: int = 128
    resume: bool = True
    confirm_winners: bool = True


class EnginePullRequest(BaseModel):
    engine: str                         # key in engine_runtimes.RUNTIMES


class StorageRequest(BaseModel):
    hf_cache: str      # absolute directory for model weights


class PromoteRequest(BaseModel):
    # "search" promotes the guided search's best candidate; "registry"
    # promotes the named sweep config (the UI passes its ranked #1).
    source: str
    config_name: Optional[str] = None
    # Promote from an ARCHIVED search (a runs/engine_optimizer/history
    # file name) instead of the live search.json.
    file: Optional[str] = None


class OptimizerStartRequest(BaseModel):
    # mode "arena": guided search over the full feasible space for
    #   this host (optionally narrowed by ``arena``).
    # mode "search": guided coarse-to-fine over a config/search/ YAML.
    # mode "registry": sweep a fixed set of hand-curated configs.
    # The default stays "registry" for wire compatibility (a bare
    # {"profile": ...} body must keep meaning a registry sweep); the
    # UI always sends mode explicitly and defaults to "arena" there.
    mode: str = "registry"
    profile: Optional[str] = None      # registry mode
    space: Optional[str] = None        # search mode: space name or path
    only: Optional[list[str]] = None   # registry mode: config subset
    # arena mode: {"models": [hf ids], "dims": {dim: [values]}} —
    # empty/absent means "everything feasible".
    arena: Optional[dict] = None
    budget: Optional[int] = None       # arena mode: evaluation budget
    new_run: bool = False


def _build_custom_config(custom: dict, runs_base: Path) -> Path:
    """Advanced path: a downloaded model + hand-set engine shape →
    a generated config file (same schema as promoted profiles).
    Devices come from the detected topology; infeasible shapes are
    refused with the reason."""
    # The shape -> engine translation lives in engines/custom.py so the
    # arena driver builds candidates through the SAME code (levers,
    # the one-memory-knob translation, refused knobs) rather than a
    # private copy that drifts.
    from .engines.custom import ShapeError, config_doc, custom_engine
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
    from .config import resolve_profile
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


def _log_heartbeat(log_path) -> Optional[dict]:
    """Live progress derived from the optimizer's own log tail: which
    config of how many, which phase and for how long, and the last
    measured cell — so the UI shows a heartbeat through the
    minutes-long engine launches instead of dead air."""
    import re
    try:
        p = Path(log_path)
        with p.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 32768))
            tail = f.read().decode("utf-8", "replace")
        mtime = p.stat().st_mtime
    except OSError:
        return None
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None
    hb: dict = {
        "last_line": lines[-1][-220:],
        "last_activity_s": max(0, round(time.time() - mtime)),
    }
    cfg_re = re.compile(r"=== Config (\d+)/(\d+): (\S+) ===")
    ph_re = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] phase: (.+)$")
    meas_re = re.compile(r"\] (ladder_\S+: .*tok/s=[\d.]+.*)$")
    for ln in reversed(lines):
        m = cfg_re.search(ln)
        if m and "config" not in hb:
            hb["config"] = int(m.group(1))
            hb["configs_total"] = int(m.group(2))
            hb["config_name"] = m.group(3)
        m = ph_re.match(ln)
        if m and "phase" not in hb:
            hb["phase"] = m.group(4)
            # Log stamps are host-local HH:MM:SS; the service runs on
            # the same host, so a wall-clock diff is exact (with a
            # midnight wrap guard).
            now = time.localtime()
            dt = (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
                  - (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                     + int(m.group(3))))
            hb["phase_elapsed_s"] = dt + 86400 if dt < 0 else dt
        m = meas_re.search(ln)
        if m and "last_measure" not in hb:
            hb["last_measure"] = m.group(1)[:160]
        if all(k in hb for k in ("config", "phase", "last_measure")):
            break
    return hb


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


def _tail_text(path: str | Path, n: int = 4096) -> str:
    """Last ``n`` bytes of a log, decoded leniently. Download logs
    grow to MB over a 60 GB pull and /api/models is polled every
    2.5 s; reading the whole file each time was the stall."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


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

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            _release_runs_lock(runs_lock)

    app = FastAPI(title="capsim", version=__version__, lifespan=_lifespan)
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
        from .persona_loader import USER_CATALOG_DIR as _ucd
        _stale = _ucd / "headline_best.yaml"
    else:
        _stale = catalog_dir / "headline_best.yaml"
    if _stale.exists():
        _stale.unlink()
        from .personas import reload_personas as _rp
        _rp(user_dir=catalog_dir)

    # ── introspection ─────────────────────────────────────────────

    @app.get("/api/status")
    async def status() -> dict:
        active = app.state.active
        return {
            "service": "capsim",
            "bus_subscribers": BUS.subscriber_count,
            "active_run": active.describe() if active else None,
        }

    @app.get("/api/profiles")
    async def profiles() -> dict:
        """Profiles with operator-facing metadata: what model/shape
        each one runs, whether it matches THIS machine's hardware,
        and whether it came from the optimizer — so the Benchmark
        picker can lead with relevant, plainly-labeled choices
        instead of a flat list of filenames."""
        import yaml as _yaml

        from .arena import hardware
        from .config import list_profiles

        hw = await asyncio.to_thread(hardware)
        has_gpu = bool(hw["count"])
        _CPU_ENGINES = ("vllm", "sglang", "vllm_dual_socket")

        def _describe(name: str, path) -> Optional[dict]:
            engine_type, model, label, detail = "?", "", name, ""
            try:
                raw = _yaml.safe_load(Path(path).read_text()) or {}
                eng = raw.get("engine") or {}
                if not eng:
                    # Not a benchmark profile (e.g. config/arena.yaml,
                    # the hardware-hints file) — keep it out of the
                    # picker entirely.
                    return None
                engine_type = str(eng.get("type", "?"))
                model = str(eng.get("model_id") or eng.get("model") or "")
                short = model.split("/")[-1] if model else ""
                if engine_type == "vllm_cuda_multi":
                    reps = eng.get("replica_devices") or []
                    tp = max((len(g) for g in reps), default=1)
                    label = f"{short} — whole box, {len(reps)} replicas"
                    detail = (f"tp{tp} per replica"
                              if tp > 1 else "one GPU per replica")
                elif engine_type in ("trtllm", "sglang_cuda",
                                     "ktransformers"):
                    from .engines.knobs import ENGINE_LABELS
                    reps = eng.get("replica_devices") or []
                    tp = max((len(g) for g in reps), default=1)
                    label = (f"{short} — "
                             f"{ENGINE_LABELS.get(engine_type, engine_type)}"
                             f", {len(reps)} replicas")
                    detail = (f"tp{tp} per replica"
                              if tp > 1 else "one GPU per replica")
                elif engine_type == "vllm_cuda":
                    tp = eng.get("tensor_parallel_size", 1)
                    label = f"{short} — single engine"
                    detail = f"tp{tp}" if tp and tp > 1 else "one GPU"
                elif engine_type in _CPU_ENGINES:
                    label = f"{short} — CPU engine"
                    detail = engine_type
                elif engine_type == "mock":
                    label = "Self-test (mock engine)"
                    detail = "no hardware needed — verifies the pipeline"
                elif engine_type == "remote":
                    label = f"Remote endpoint{' — ' + short if short else ''}"
            except Exception:  # noqa: BLE001
                pass
            from .engines.knobs import GPU_ENGINES
            gpu_engine = engine_type in GPU_ENGINES or engine_type == "vllm_cuda"
            # The searched engine dimensions, extracted so the
            # benchmark form can prefill its Advanced settings with
            # exactly what the optimization landed on.
            params: dict = {}
            try:
                flags = [str(f) for f in (eng.get("vllm_extra_flags") or [])]
                def _flag(key):
                    return (flags[flags.index(key) + 1]
                            if key in flags
                            and flags.index(key) + 1 < len(flags) else None)
                reps = eng.get("replica_devices") or []
                params = {
                    # Which server the profile was measured on, so the
                    # benchmark form prefills the engine too and does
                    # not silently re-measure on a different one.
                    "engine": (engine_type
                               if engine_type in GPU_ENGINES
                               else "vllm_cuda_multi"),
                    "replicas": len(reps) if reps else 1,
                    "tp": (max((len(g) for g in reps), default=1) if reps
                           else int(eng.get("tensor_parallel_size") or 1)),
                    "gpu_memory_utilization":
                        eng.get("gpu_memory_utilization"),
                    "max_model_len": eng.get("max_model_len"),
                    "max_num_seqs": _flag("--max-num-seqs"),
                    "max_num_batched_tokens":
                        _flag("--max-num-batched-tokens"),
                    "kv_cache_dtype": _flag("--kv-cache-dtype") or "auto",
                    "expert_parallel": "--enable-expert-parallel" in flags,
                }
            except Exception:  # noqa: BLE001
                params = {}
            return {
                "path": str(path),
                "engine_type": engine_type,
                "model_id": model,
                "label": label,
                "detail": detail,
                "optimized": name.startswith("optimized-"),
                "engine": params,
                # Does this profile match THIS machine?
                "fits_hardware": (
                    has_gpu if gpu_engine
                    else (not has_gpu) if engine_type in _CPU_ENGINES
                    else True   # mock / remote / unknown: always usable
                ),
            }

        out = {}
        for name, path in list_profiles().items():
            entry = await asyncio.to_thread(_describe, name, path)
            if entry is not None:
                out[name] = entry
        return out

    def _persona_summary(p) -> dict:
        """The numbers a human needs to know what a persona MEANS:
        question/answer sizes, session length, and the read/think gap
        between turns — analytic, not sampled."""
        from .distributions import summarize
        inp = summarize(p.input_tokens)
        out = summarize(p.output_tokens)
        turns = summarize(p.turns_per_session)
        read = summarize(p.read_time_seconds)
        think = summarize(p.active_think_seconds)
        gap = {
            k: ((read[k] or 0) + (think[k] or 0))
            if read[k] is not None and think[k] is not None else None
            for k in ("median", "mean", "p90")
        }
        return {"input_tokens": inp, "output_tokens": out,
                "turns_per_session": turns, "think_gap_s": gap}

    # Working personas of the shape search — real registry entries
    # (workers must resolve them) but not workloads a user picks.
    _INTERNAL_PERSONAS = {"headline_cell", "headline_best"}

    @app.get("/api/personas")
    async def personas() -> list[dict]:
        from .personas import PERSONAS
        return [
            {
                "id": p.id,
                "name": getattr(p, "name", "") or p.id,
                "description": p.description,
                "ttft_target_s": p.ttft_target_seconds,
                "ttft_failure_s": p.ttft_failure_seconds,
                "tpot_target_ms": p.tpot_target_ms,
                "tpot_failure_ms": p.tpot_failure_ms,
                "summary": _persona_summary(p),
            }
            for p in PERSONAS.values()
            if p.id not in _INTERNAL_PERSONAS
        ]

    @app.get("/api/cohorts")
    async def cohorts() -> list[dict]:
        from .personas import COHORTS, PERSONAS

        def _blended(c) -> Optional[dict]:
            """Weight-averaged medians across the mix — 'what a
            typical turn of this team looks like'."""
            total = sum(c.persona_weights.values()) or 1.0
            acc = {"input_tokens": 0.0, "output_tokens": 0.0,
                   "turns_per_session": 0.0, "think_gap_s": 0.0}
            for pid, w in c.persona_weights.items():
                p = PERSONAS.get(pid)
                if p is None:
                    return None
                s = _persona_summary(p)
                acc["input_tokens"] += (s["input_tokens"]["median"] or 0) * w
                acc["output_tokens"] += (s["output_tokens"]["median"] or 0) * w
                acc["turns_per_session"] += (
                    s["turns_per_session"]["mean"] or 0) * w
                acc["think_gap_s"] += (s["think_gap_s"]["median"] or 0) * w
            return {k: round(v / total, 1) for k, v in acc.items()}

        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "persona_weights": c.persona_weights,
                "blended": _blended(c),
            }
            for c in COHORTS.values()
        ]

    # ── persona/cohort editor (Phase 4) ───────────────────────────
    # The catalog is data: packaged defaults + config/personas/*.yaml
    # overlays. The editor GETs a spec as YAML, PUTs it back; the
    # server validates against the full merged catalog before the
    # overlay file lands, so a bad save can't wedge the registry.

    def _catalog_dir() -> Path:
        from .persona_loader import USER_CATALOG_DIR
        return catalog_dir if catalog_dir is not None else USER_CATALOG_DIR

    def _save_catalog_entry(
        kind: str, entry_id: str,
        spec_yaml: Optional[str] = None, spec: Optional[dict] = None,
    ) -> None:
        import yaml as _yaml

        from .persona_loader import PersonaSpecError, load_catalog
        from .personas import reload_personas

        if not entry_id.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(422, "id must be alphanumeric/_/-")
        if spec is None:
            if spec_yaml is None:
                raise HTTPException(422, "pass spec or yaml")
            try:
                spec = _yaml.safe_load(spec_yaml)
            except _yaml.YAMLError as e:
                raise HTTPException(422, f"invalid YAML: {e}") from e
        if not isinstance(spec, dict):
            raise HTTPException(422, "spec must be a mapping")

        target = _catalog_dir() / f"{entry_id}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = target.read_text() if target.exists() else None
        target.write_text(_yaml.safe_dump(
            {kind: {entry_id: spec}}, sort_keys=False, allow_unicode=True,
        ))
        try:
            load_catalog(user_dir=_catalog_dir())   # validate with the new file
        except PersonaSpecError as e:
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(previous)
            raise HTTPException(422, str(e)) from e
        reload_personas(user_dir=_catalog_dir())

    @app.get("/api/personas/{persona_id}")
    async def persona_detail(persona_id: str) -> dict:
        import yaml as _yaml

        from .persona_loader import serialize_persona
        from .personas import PERSONAS
        if persona_id not in PERSONAS:
            raise HTTPException(404, f"unknown persona '{persona_id}'")
        spec = serialize_persona(PERSONAS[persona_id])
        return {
            "id": persona_id,
            "spec": spec,
            "yaml": _yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
            "editable_file": str(_catalog_dir() / f"{persona_id}.yaml"),
        }

    @app.put("/api/personas/{persona_id}")
    async def persona_save(persona_id: str, req: SaveSpecRequest) -> dict:
        _save_catalog_entry("personas", persona_id, req.yaml, req.spec)
        return {"saved": persona_id}

    @app.get("/api/cohorts/{cohort_id}")
    async def cohort_detail(cohort_id: str) -> dict:
        import yaml as _yaml

        from .persona_loader import serialize_cohort
        from .personas import COHORTS
        if cohort_id not in COHORTS:
            raise HTTPException(404, f"unknown cohort '{cohort_id}'")
        spec = serialize_cohort(COHORTS[cohort_id])
        return {
            "id": cohort_id,
            "spec": spec,
            "yaml": _yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
            "editable_file": str(_catalog_dir() / f"{cohort_id}.yaml"),
        }

    @app.put("/api/cohorts/{cohort_id}")
    async def cohort_save(cohort_id: str, req: SaveSpecRequest) -> dict:
        _save_catalog_entry("cohorts", cohort_id, req.yaml, req.spec)
        return {"saved": cohort_id}

    # ── headline shape store ──────────────────────────────────────
    # The shape search's winner is stored per MODEL FAMILY; the UI
    # asks whether the selected model's family has one, and can apply
    # it to the Headline: Generation workload with one click.

    @app.get("/api/headline-shape")
    async def headline_shape(model: str) -> dict:
        from .headline_shapes import (
            generation_shape,
            model_family,
            shape_for,
            shapes_path,
        )
        stored = shape_for(shapes_path(_catalog_dir()), model)
        current = generation_shape()
        return {
            "family": model_family(model),
            "shape": stored,
            "active": bool(
                stored and current
                and current == (stored.get("input_tokens"),
                                stored.get("output_tokens"))),
        }

    @app.post("/api/headline-shape/apply")
    async def headline_shape_apply(req: dict) -> dict:
        from .headline_shapes import (
            apply_shape_to_generation,
            shape_for,
            shapes_path,
        )
        model = str(req.get("model") or "")
        stored = shape_for(shapes_path(_catalog_dir()), model)
        if not stored:
            raise HTTPException(
                404, f"no optimized shape stored for '{model}' — run "
                     "the shape search first")
        await asyncio.to_thread(
            apply_shape_to_generation, _catalog_dir(),
            stored["input_tokens"], stored["output_tokens"])
        return {"applied": True, "shape": stored}

    @app.get("/api/hardware")
    async def hardware_summary() -> dict:
        """Tiny hardware probe for the UI — GPU count/names so the
        benchmark form can gray out the CPU/GPU toggle honestly."""
        from .arena import hardware as _hw
        try:
            hw = await asyncio.to_thread(_hw)
        except Exception:  # noqa: BLE001
            hw = {}
        # `gpus` keeps the arena's view (config may pin the shape);
        # `detected_gpus` is what nvidia-smi saw, which is what decides
        # whether a GPU engine can actually launch here.
        return {
            "gpus": hw.get("count", 0) or 0,
            "detected_gpus": hw.get("detected_count", hw.get("count", 0)) or 0,
            "source": hw.get("source", "detected"),
            "vram_per_gpu_gb": hw.get("vram_per_gpu_gb"),
        }

    @app.get("/api/runs")
    async def runs() -> list[dict]:
        # One sqlite open per run dir — dozens of them on a box that
        # has been benchmarking for a month. Off the loop.
        return await asyncio.to_thread(_list_runs, runs_base)

    @app.get("/api/runs/{run}/headline")
    async def headline_sweep_doc(run: str) -> dict:
        """The saturation-sweep summary for a headline run. Separate
        from the capacity export because it answers a different
        question and shares none of its shape."""
        if "/" in run or run.startswith("."):
            raise HTTPException(422, "bad run name")
        path = runs_base / run / "headline_sweep.json"
        if not path.exists():
            raise HTTPException(404, f"{run} has no headline sweep summary")
        try:
            return await asyncio.to_thread(_read_json, path)
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, f"unreadable sweep summary: {e}") from e

    @app.get("/api/doctor")
    async def doctor() -> dict:
        from .doctor import run_doctor
        report = await asyncio.to_thread(run_doctor)
        return report.to_dict()

    # ── run lifecycle ─────────────────────────────────────────────

    @app.post("/api/runs", status_code=202)
    async def start_run(req: StartRunRequest) -> dict:
        # Serialize the whole check-then-create sequence. There is an
        # await between "is a run already active?" and setting
        # app.state.active, so two near-simultaneous POSTs both
        # cleared the guard and both spawned runs. They then destroyed
        # each other: every engine launch sweeps stale vllm-*
        # containers, so concurrent runs removed one another's
        # replicas and every candidate failed while the GPUs sat idle.
        async with app.state.start_lock:
            return await _start_run_locked(req)

    async def _start_run_locked(req: StartRunRequest) -> dict:
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
        if req.custom is not None:
            # A roofline varies the model per cell, so its `custom` is
            # a TEMPLATE with no model_id. Borrow the first planned
            # model just to satisfy this pre-flight build -- the run
            # rebuilds a config for every cell anyway, and validating
            # the template here still catches a bad engine or a shape
            # that does not fit the box.
            _validate_custom_ints(req.custom)
            pre = dict(req.custom)
            if not pre.get("model_id"):
                planned = ((req.workload or {}).get("spec") or {}).get("models")
                if planned:
                    pre["model_id"] = planned[0]
            config_path = await asyncio.to_thread(
                _build_custom_config, pre, runs_base)
        else:
            config_path = _resolve_config_path(req)

        import yaml as _yaml

        from .config import load_config
        from .headline_shapes import is_headline_persona
        from .personas import COHORTS, PERSONAS, cohort_from_persona
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
            from .headline_sweep import run_headline_sweep
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
            from .engine_runtimes import available_engines
            from .roofline import run_roofline

            spec = dict(req.workload.get("spec") or {})
            engines = spec.get("engines") or available_engines()
            if not engines:
                raise HTTPException(
                    422, "no engine runtime is staged — pull one in "
                         "Prepare before starting a roofline")
            models = spec.get("models")
            if not models:
                from .arena import hardware as _hw
                from .model_catalog import load_model_catalog
                from .roofline import pick_models
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

            base_custom = dict(req.custom or {})
            base_custom.setdefault("device", "gpu")
            base_custom.setdefault("replicas", 8)
            base_custom.setdefault("tp", 1)
            base_custom.setdefault("gpu_memory_utilization", 0.95)
            base_custom.setdefault("kv_cache_dtype", "fp8")
            base_custom.setdefault("max_model_len", 2048)

            def _build_rf(overrides: dict) -> Path:
                return _build_custom_config(
                    {**base_custom, **overrides}, runs_base)

            shapes = {"max_num_seqs": spec.get("max_num_seqs"),
                      "output_tokens": spec.get("output_tokens")}
            coro_factory = lambda: run_roofline(  # noqa: E731
                models=models, engines=engines,
                shapes={k: v for k, v in shapes.items() if v},
                input_tokens=int(spec.get("input_tokens") or 128),
                build_config=_build_rf, runs_base=runs_base,
                resume=bool(spec.get("resume", True)),
                confirm_winners=bool(spec.get("confirm_winners", True)),
            )
        elif kind == "headline_optimize":
            # Engine shape and request shape are coupled, so they are
            # searched together; see simulator/headline_optimize.py.
            from .headline_optimize import run_headline_optimize
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
                return _build_custom_config(
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
            from .headline_search import run_headline_search
            shape_progress: dict = {}
            coro_factory = lambda: run_headline_search(  # noqa: E731
                cfg, new_run=req.new_run, progress=shape_progress,
                input_tokens=req.input_tokens,
            )
        elif kind == "sweep":
            from .personas import resolve_workload_group
            try:
                persona_ids, cohort_ids = resolve_workload_group(
                    req.workload.get("type", "all")
                )
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            from .runner import run_sweep
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
        closed = (
            req.mode == "closed" or req.adaptive or bool(req.pool_sizes)
        )
        if closed:
            from .runner import run_cohort
            return lambda: run_cohort(
                cfg, cohort,
                new_run=req.new_run,
                adaptive=req.adaptive,
                fixed_grid_pool_sizes=req.pool_sizes,
            )
        from .open_loop import run_cohort_open_loop
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

    @app.delete("/api/runs/{run_name}/cohorts/{cohort_run_id}")
    async def delete_cohort_run(run_name: str, cohort_run_id: str) -> dict:
        """Fully delete one cohort run's data — measurements, turns,
        telemetry, snapshots, users, and the run row. When it was the
        last cohort in its run_NN dir, the whole dir goes (engine
        logs included). Refused while any run is active: the runner
        writes to these tables in-process."""
        if "/" in run_name or run_name.startswith("."):
            raise HTTPException(422, "bad run name")
        active = app.state.active
        if active is not None and not active.task.done():
            raise HTTPException(
                409, "a run is active — deleting run data while the "
                     "runner writes to it is unsafe; stop it first")
        d = runs_base / run_name
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

    @app.post("/api/runs/stop")
    async def stop_run() -> dict:
        active = app.state.active
        if active is None or active.task.done():
            raise HTTPException(409, "no active run")
        active.task.cancel()
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




    # ── storage (choose the disk + location for model weights) ────
    # Lists every mounted filesystem and every large unmounted disk;
    # lets the user pick where weights live. The choice persists to
    # ~/.config/capsim/storage.json and everything downstream (doctor,
    # downloads, engine mounts, optimizer) resolves through it. The
    # service NEVER formats or mounts disks — that is root-privileged
    # and destructive, so unmounted disks come with the exact commands
    # for a human instead.

    @app.get("/api/storage")
    async def storage_status() -> dict:
        import psutil

        from .doctor import parse_lsblk_unmounted
        from .models import hf_cache_dir, hf_cache_source

        def _fs_list() -> list[dict]:
            out, seen = [], set()
            for part in psutil.disk_partitions(all=False):
                mp = part.mountpoint
                if part.fstype in ("tmpfs", "devtmpfs", "squashfs",
                                   "overlay", "vfat", "autofs", "nullfs") \
                        or mp.startswith(("/boot", "/snap", "/System",
                                          "/private/var", "/dev")):
                    continue
                try:
                    usage = shutil.disk_usage(mp)
                    dev = Path(mp).stat().st_dev
                except OSError:
                    continue
                if dev in seen:
                    continue
                seen.add(dev)
                out.append({
                    "mountpoint": mp,
                    "device": part.device,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / 1e9, 1),
                    "free_gb": round(usage.free / 1e9, 1),
                })
            return sorted(out, key=lambda f: -f["free_gb"])

        unmounted: list[dict] = []
        try:
            res = await asyncio.to_thread(
                subprocess.run,
                ["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT"],
                capture_output=True, text=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            res = None                     # non-Linux host — no lsblk
        if res and res.returncode == 0 and res.stdout.strip():
            try:
                for name, size_gb, has_partitions in parse_lsblk_unmounted(
                    json.loads(res.stdout), min_gb=200.0,
                ):
                    # Commands for a HUMAN with sudo — shown, never
                    # run. A partitioned disk was used by SOMETHING
                    # before (leftover ZFS pools look exactly like
                    # this): its recipe starts with an explicit
                    # check-then-wipe, never a bare mkfs.
                    if has_partitions:
                        commands = [
                            f"# {name} has existing partitions — check what's on it first:",
                            f"sudo blkid /dev/{name}* ; lsblk -f /dev/{name}",
                            "# ONLY if the contents are disposable:",
                            f"sudo wipefs -a /dev/{name}",
                            f"sudo mkfs.ext4 -L capsim-data /dev/{name}",
                        ]
                    else:
                        commands = [
                            f"sudo mkfs.ext4 -L capsim-data /dev/{name}",
                        ]
                    commands += [
                        "sudo mkdir -p /data",
                        f"sudo mount /dev/{name} /data",
                        "grep -q capsim-data /etc/fstab || echo 'LABEL=capsim-data /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab",
                        "sudo chown $USER /data",
                    ]
                    unmounted.append({
                        "name": name,
                        "size_gb": round(size_gb, 1),
                        "has_partitions": has_partitions,
                        "commands": commands,
                    })
            except (ValueError, KeyError):
                pass

        cache = hf_cache_dir()
        probe = cache
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            cache_free = round(shutil.disk_usage(probe).free / 1e9, 1)
        except OSError:
            cache_free = None
        return {
            "hf_cache": str(cache),
            "hf_cache_source": hf_cache_source(),
            "hf_cache_free_gb": cache_free,
            "filesystems": await asyncio.to_thread(_fs_list),
            "unmounted": unmounted,
        }

    @app.post("/api/storage")
    async def storage_set(req: StorageRequest) -> dict:
        from .models import hf_cache_dir, set_hf_cache_dir
        try:
            chosen = await asyncio.to_thread(set_hf_cache_dir, req.hf_cache)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"hf_cache": str(chosen), "resolved": str(hf_cache_dir())}

    # ── model staging (weights into the shared HF cache) ──────────
    # The UI's Models panel: which models do profiles/search spaces
    # reference, are the weights cached where the engine containers
    # mount, and one-click downloads with tailed progress. Downloads
    # are HF_HOME-pinned to that same cache.

    @app.get("/api/models")
    async def models_list() -> dict:
        from .models import hf_cache_dir, referenced_models
        entries = await asyncio.to_thread(referenced_models)
        downloads = {}
        for model, dl in list(app.state.model_downloads.items()):
            exit_code = dl["proc"].poll()
            tail = (await asyncio.to_thread(_tail_text, dl["log"]))[-400:]
            downloads[model] = {
                "running": exit_code is None,
                "exit_code": exit_code,
                "started_at": dl["started_at"],
                "log": dl["log"],
                "log_tail": tail,
            }
        return {
            "cache_dir": str(hf_cache_dir()),
            "models": entries,
            "downloads": downloads,
        }

    @app.post("/api/models/add")
    async def models_add(req: ModelAddRequest) -> dict:
        """Add a model to the local catalog (config/models/local.yaml)
        and suggest its quantized siblings. Adding is what makes a
        model downloadable and searchable — the catalog is the explicit
        operator-curated list, so the download gate stays meaningful."""
        from .model_catalog import (
            CatalogError,
            add_catalog_model,
            hub_model_exists,
            suggest_quant_siblings,
        )
        if req.check_hub:
            exists = await asyncio.to_thread(hub_model_exists, req.model)
            if exists is False:
                raise HTTPException(
                    404, f"'{req.model}' does not exist on the Hugging "
                         f"Face Hub — check the org/name spelling",
                )
        else:
            exists = None
        try:
            entry, created = await asyncio.to_thread(
                lambda: add_catalog_model(
                    req.model, family=req.family, quant=req.quant,
                    notes=req.notes, series=req.series,
                    params_b=req.params_b, moe=req.moe,
                    approx_size_gb=req.approx_size_gb,
                    min_vram_gb=req.min_vram_gb, specialty=req.specialty,
                ),
            )
        except CatalogError as e:
            raise HTTPException(422, str(e)) from e
        siblings = await asyncio.to_thread(
            lambda: suggest_quant_siblings(req.model, verify=req.check_hub),
        )
        return {"entry": entry, "created": created,
                "hub_verified": exists, "siblings": siblings}

    @app.get("/api/models/discover")
    async def models_discover(orgs: Optional[str] = None) -> dict:
        """Live Hub discovery, validated against THIS box: recent
        text-generation models from the leading orgs with real
        parameter counts, native-FP8 detection, capability tags, and
        the feasible TP set for the detected GPUs. Slow (one Hub call
        per candidate) — the UI calls it on demand, not on load."""
        from .arena import hardware
        from .discovery import DEFAULT_ORGS, discover_models
        from .model_catalog import load_model_catalog
        hw = await asyncio.to_thread(hardware)
        known = {e["id"] for e in await asyncio.to_thread(load_model_catalog)}
        org_list = (tuple(o.strip() for o in orgs.split(",") if o.strip())
                    if orgs else DEFAULT_ORGS)
        max_tp = max((len(g) for g in hw["device_groups"]), default=1)
        entries = await asyncio.to_thread(
            discover_models, org_list, 6, hw["vram_per_gpu_gb"], max_tp, known,
        )
        if not entries:
            raise HTTPException(
                502, "the Hub returned nothing — is this box offline? "
                     "Discovery needs huggingface.co reachable.",
            )
        return {"hardware": hw, "models": entries}

    @app.post("/api/models/download", status_code=202)
    async def models_download(req: ModelDownloadRequest) -> dict:
        from .models import download_command, referenced_models
        known = {m["model"] for m in await asyncio.to_thread(referenced_models)}
        if req.model not in known:
            # Only models the configs actually reference — the service
            # is not a general download proxy.
            raise HTTPException(
                404, f"'{req.model}' is not referenced by any profile "
                     f"or search space",
            )
        existing = app.state.model_downloads.get(req.model)
        if existing and existing["proc"].poll() is None:
            raise HTTPException(409, "download already running for this model")
        argv, extra_env = download_command(req.model)
        log_dir = runs_base / "model_downloads"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            req.model.replace("/", "--")
            + f"_{time.strftime('%Y%m%dT%H%M%S')}.log"
        )
        import os as _os
        # The child inherits the descriptor; our copy closes either way
        # (it leaked one handle per download otherwise).
        with open(log_path, "w") as log_file:
            try:
                proc = subprocess.Popen(
                    argv, stdout=log_file, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    env={**_os.environ, **extra_env},
                )
            except FileNotFoundError as e:
                raise HTTPException(
                    500, f"hf CLI not found ({e}) — is huggingface_hub "
                         f"installed in the service environment?",
                ) from e
        app.state.model_downloads[req.model] = {
            "proc": proc, "log": str(log_path), "started_at": time.time(),
        }
        return {"accepted": True, "model": req.model, "log": str(log_path)}

    # ── roofline autopilot ───────────────────────────────────────
    # A roofline run takes hours and the operator will not be watching.
    # Everything it knows lives in runs/roofline.json, written after
    # every step, so reconnecting is a GET rather than a replay of
    # events nobody was listening for.

    _roofline_state = runs_base / "roofline.json"

    @app.get("/api/roofline")
    async def roofline_state() -> dict:
        from .roofline import load_state
        st = await asyncio.to_thread(load_state, _roofline_state)
        if st is None:
            return {"kind": "roofline", "status": "none", "done": True,
                    "results": [], "summary": {"best": None,
                                               "best_per_model": {},
                                               "best_per_engine": {}}}
        doc = st.to_dict()
        # Whether THIS service is still driving it. A state file left
        # at "searching" by a killed process must not read as running,
        # or the page will sit forever waiting for a dead run.
        active = app.state.active
        desc = active.describe() if active else None
        doc["live"] = bool(desc and desc.get("running")
                           and (desc.get("workload") or {}).get("kind")
                           == "roofline")
        return doc

    @app.get("/api/roofline/candidates")
    async def roofline_candidates(limit: int = 12) -> dict:
        """The ranked model shortlist, with the reasoning shown."""
        from dataclasses import asdict as _asdict

        from .arena import hardware
        from .model_catalog import load_model_catalog
        from .roofline import score_models

        hw = await asyncio.to_thread(hardware)
        cat = await asyncio.to_thread(load_model_catalog)
        ranked = await asyncio.to_thread(
            score_models, cat, vram_per_gpu_gb=hw.get("vram_per_gpu_gb"))
        return {"hardware": hw,
                "candidates": [_asdict(c) for c in ranked[:max(1, limit)]]}

    # ── engine runtimes (Prepare: stage the servers) ─────────────
    # An engine is only offered downstream once its image is on the
    # box (see engine_runtimes). These two endpoints are what make
    # that gate openable from the UI.

    def _pull_progress(log_path: str) -> str:
        """Last meaningful line of a docker pull log — the layer
        progress, so a 59 GB pull is visibly alive."""
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 8192))
                lines = f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return ""
        for ln in reversed(lines):
            ln = ln.strip()
            if ln and not ln.startswith("\r"):
                return ln[:200]
        return ""

    @app.get("/api/engines")
    async def engines_status() -> dict:
        from .engine_runtimes import image_store_root, runtime_status
        rows = await asyncio.to_thread(runtime_status)
        store = await asyncio.to_thread(image_store_root)
        for r in rows:
            pull = app.state.engine_pulls.get(r["engine"])
            if not pull:
                continue
            rc = pull["proc"].poll()
            r["pull"] = {
                "running": rc is None,
                "returncode": rc,
                "started_at": pull["started_at"],
                "log": pull["log"],
                "progress": await asyncio.to_thread(
                    _pull_progress, pull["log"]),
            }
        return {"runtimes": rows, "image_store": store,
                "available": [r["engine"] for r in rows if r["staged"]]}

    @app.post("/api/engines/pull", status_code=202)
    async def engines_pull(req: EnginePullRequest) -> dict:
        from .engine_runtimes import RUNTIMES, image_store_root
        meta = RUNTIMES.get(req.engine)
        if meta is None:
            raise HTTPException(
                404, f"unknown engine '{req.engine}' — one of "
                     f"{', '.join(RUNTIMES)}")
        existing = app.state.engine_pulls.get(req.engine)
        if existing and existing["proc"].poll() is None:
            raise HTTPException(409, "a pull is already running for this "
                                     "engine")
        # Refuse rather than wedge the disk: these images are tens of
        # GB and the store is frequently not on the roomiest volume.
        store = await asyncio.to_thread(image_store_root)
        free = store.get("free_gb")
        if free is not None and free < meta["approx_gb"] * 1.1:
            raise HTTPException(
                507, f"{meta['label']} needs about {meta['approx_gb']} GB "
                     f"but only {free:.0f} GB is free on "
                     f"{store.get('path')}"
                     + (f" ({store['note']})" if store.get("note") else "")
                     + " — free space or move the image store first")
        log_dir = runs_base / "engine_pulls"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            req.engine + f"_{time.strftime('%Y%m%dT%H%M%S')}.log")
        with open(log_path, "w") as log_file:
            try:
                proc = subprocess.Popen(
                    ["docker", "pull", meta["image"]],
                    stdout=log_file, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            except FileNotFoundError as e:
                raise HTTPException(
                    500, f"docker not found ({e}) — the service host must "
                         f"be the one that launches engines") from e
        app.state.engine_pulls[req.engine] = {
            "proc": proc, "log": str(log_path), "started_at": time.time(),
        }
        return {"accepted": True, "engine": req.engine,
                "image": meta["image"], "log": str(log_path)}

    # ── engine optimizer (find the best launch shape first) ──────
    # Runs scripts/engine_optimizer.py as a supervised subprocess —
    # same artifact contract as `make optimize-engine` (incremental
    # runs/engine_optimizer/run.json, resumable), so the UI, the CLI,
    # and a second SSH session all watch the same file.

    _opt_out = runs_base / "engine_optimizer" / "run.json"
    _search_out = runs_base / "engine_optimizer" / "search.json"

    def _lock_state() -> Optional[dict]:
        """Attach point for optimizers this service did not start
        (serve restarted mid-run, or a CLI launch): the runner holds
        an exclusive flock on .optimizer.lock with {pid, started_at,
        argv} inside. If the flock is TAKEN, someone is running —
        return their state; if we can acquire it, nobody is (the
        kernel releases flocks when the holder dies, so a stale file
        never reads as running)."""
        import fcntl
        p = _opt_out.parent / ".optimizer.lock"
        try:
            fh = open(p)
        except OSError:
            return None
        try:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                try:
                    doc = json.loads(fh.read() or "{}")
                except json.JSONDecodeError:
                    return {}
                if isinstance(doc, dict):
                    return doc
                if isinstance(doc, int):
                    return {"pid": doc}    # pre-JSON lock: bare PID
                return {}
            fcntl.flock(fh, fcntl.LOCK_UN)
            return None
        finally:
            fh.close()

    def _optimizer_running() -> bool:
        opt = app.state.optimizer
        if opt and opt["proc"].poll() is None:
            return True
        return _lock_state() is not None

    _history_dir = _opt_out.parent / "history"

    def _archive_search_results() -> None:
        """A fresh run overwrites search.json — archive the previous
        run first so optimizer results have history like benchmark
        runs do. The space file is archived alongside and the doc's
        space_file re-pointed at the copy, so a historical winner can
        still be promoted (the space hash still verifies)."""
        if not _search_out.exists():
            return
        try:
            doc = json.loads(_search_out.read_text())
        except (OSError, json.JSONDecodeError):
            return
        ts = (doc.get("generated_at") or "")[:19].replace(":", "").replace(
            "-", "") or time.strftime("%Y%m%dT%H%M%S")
        _history_dir.mkdir(parents=True, exist_ok=True)
        base = f"search_{ts}_{doc.get('space', 'space')}"
        space_src = Path(doc.get("space_file") or "")
        if space_src.exists():
            space_copy = _history_dir / f"{base}_space.yaml"
            space_copy.write_text(space_src.read_text())
            doc["space_file"] = str(space_copy)
        (_history_dir / f"{base}.json").write_text(json.dumps(doc, indent=2))

    def _space_models(space_file) -> list[str]:
        import yaml as _yaml
        try:
            raw = _yaml.safe_load(Path(space_file).read_text()) or {}
            return sorted(str(v.get("model"))
                          for v in (raw.get("model_variants") or {}).values()
                          if isinstance(v, dict) and v.get("model"))
        except Exception:  # noqa: BLE001
            return []

    def _group_of(models: list[str]) -> Optional[dict]:
        """Runs group by (series, size-range) — "Qwen3 16–45B": a
        follow-up over the same family/size bracket is an addendum to
        the same investigation even if the exact model subset differs;
        Qwen3.6 is a different series and never mixes. See
        arena.group_for_models."""
        from .arena import group_for_models
        return group_for_models(models)

    def _doc_summary_entry(doc: dict, file_name: str) -> dict:
        s = doc.get("summary") or {}
        best = s.get("best") or {}
        group = _group_of(_space_models(doc.get("space_file") or ""))
        return {
            "file": file_name,
            "space": doc.get("space"),
            "generated_at": doc.get("generated_at"),
            "evaluated": s.get("evaluated"),
            "done_reason": s.get("done_reason"),
            "best_score": best.get("score"),
            "best_key": best.get("key"),
            "group_key": group and group["key"],
            "group_label": group and group["label"],
        }

    def _history_entries() -> list[dict]:
        if not _history_dir.exists():
            return []
        out = []
        for p in sorted(_history_dir.glob("search_*.json"), reverse=True):
            try:
                doc = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append(_doc_summary_entry(doc, p.name))
        return out

    def _combined_group(group_key: str) -> dict:
        """Merge every run of one model-set group — history plus the
        current run — into a single ranking. Evaluations dedupe by
        canonical candidate key (best score wins); runs whose
        objective or measurement fingerprint differs from the group's
        newest run are EXCLUDED and named, never silently mixed."""
        from .search import Objective, best_rung

        docs: list[tuple[Optional[str], dict]] = []   # (file|None=current, doc)
        if _search_out.exists():
            try:
                docs.append((None, json.loads(_search_out.read_text())))
            except (OSError, json.JSONDecodeError):
                pass
        if _history_dir.exists():
            for p in sorted(_history_dir.glob("search_*.json"), reverse=True):
                try:
                    docs.append((p.name, json.loads(p.read_text())))
                except (OSError, json.JSONDecodeError):
                    continue
        group_docs = []
        for fname, doc in docs:
            g = _group_of(_space_models(doc.get("space_file") or ""))
            if g and g["key"] == group_key:
                group_docs.append((fname, doc))
        if not group_docs:
            raise HTTPException(404, f"no runs in group '{group_key}'")

        # Comparable scores only: group by (objective, measurement)
        # fingerprint and keep the MAJORITY cohort (ties go to the
        # cohort containing the newest run) — one oddball run must
        # not evict the rest of the evidence.
        def _fp(doc: dict) -> str:
            return json.dumps(
                [doc.get("objective"), doc.get("measurement")],
                sort_keys=True)
        counts: dict[str, int] = {}
        for _f, doc in group_docs:
            counts[_fp(doc)] = counts.get(_fp(doc), 0) + 1
        newest_fp = _fp(max(
            group_docs, key=lambda fd: fd[1].get("generated_at") or "")[1])
        fingerprint = max(
            counts, key=lambda f: (counts[f], f == newest_fp))
        included, excluded = [], []
        for fname, doc in group_docs:
            if _fp(doc) == fingerprint:
                included.append((fname, doc))
            else:
                excluded.append(fname or "current")

        ref = included[0][1]
        objective = Objective(**(ref.get("objective") or {}))
        merged: dict[str, tuple[dict, Optional[str]]] = {}
        for fname, doc in included:
            evaluated = (doc.get("state") or {}).get("evaluated") or {}
            for k, e in evaluated.items():
                if e.get("status") != "ok" or e.get("score") is None:
                    continue
                if k not in merged or e["score"] > merged[k][0]["score"]:
                    merged[k] = (e, fname)
        ranked = sorted(merged.items(), key=lambda kv: kv[1][0]["score"],
                        reverse=True)
        top = [{
            "key": k, "score": e["score"], "params": e.get("params"),
            "iteration": e.get("iteration"),
            "config_name": e.get("config_name"),
            "best_rung": best_rung(e.get("cells") or [], objective),
            "source": src or "current",
        } for k, (e, src) in ranked[:10]]
        best = top[0] if top else None
        group = _group_of(_space_models(ref.get("space_file") or ""))
        return {
            "kind": "combined",
            "space": group["label"] if group else group_key,
            "generated_at": max(
                (d.get("generated_at") or "" for _f, d in included),
                default=""),
            "runs": [f or "current" for f, _d in included],
            "excluded": excluded,
            # The winner still promotes: from its source run's file
            # (None = the current live results).
            "promote_file": (ranked[0][1][1] if ranked else None),
            "summary": {
                "evaluated": len(merged),
                "ok": len(merged),
                "failed": 0,
                "iterations": [],
                "done_reason": (
                    f"combined view of {len(included)} run(s)"
                    + (f"; {len(excluded)} excluded "
                       f"(different objective/measurement)"
                       if excluded else "")),
                "best": best,
                "top": top,
            },
        }

    def _build_seed_file(space_doc: dict) -> tuple[Path, int]:
        """Collect ok evaluations from the new run's investigation
        group (history archives with the same (series, size-range)
        identity AND the same objective+measurement fingerprint) into
        a seed file for the driver. Best score wins on duplicates."""
        import dataclasses as _dc

        from .search import Measurement, Objective

        def _fp_of(objective, measurement) -> str:
            try:
                return json.dumps([
                    _dc.asdict(Objective(**(objective or {}))),
                    _dc.asdict(Measurement(**(measurement or {}))),
                ], sort_keys=True)
            except TypeError:
                return "?"

        models = sorted({str(v.get("model"))
                         for v in (space_doc.get("model_variants") or {}).values()
                         if isinstance(v, dict) and v.get("model")})
        group = _group_of(models)
        if group is None:
            return _opt_out.parent / "seed.json", 0
        want_fp = _fp_of(space_doc.get("objective"),
                         space_doc.get("measurement"))
        merged: dict[str, dict] = {}
        if _history_dir.exists():
            for p in sorted(_history_dir.glob("search_*.json")):
                try:
                    hdoc = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                g = _group_of(_space_models(hdoc.get("space_file") or ""))
                if not g or g["key"] != group["key"]:
                    continue
                if _fp_of(hdoc.get("objective"),
                          hdoc.get("measurement")) != want_fp:
                    continue
                for k, e in ((hdoc.get("state") or {}).get("evaluated")
                             or {}).items():
                    if e.get("status") != "ok" or e.get("score") is None:
                        continue
                    if k not in merged or e["score"] > merged[k]["score"]:
                        merged[k] = e
        seed_path = _opt_out.parent / "seed.json"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(json.dumps({"evaluated": merged}))
        return seed_path, len(merged)

    @app.get("/api/optimizer/combined/{group_key}")
    async def optimizer_combined(group_key: str) -> dict:
        if not group_key.isalnum():
            raise HTTPException(422, "bad group key")
        return await asyncio.to_thread(_combined_group, group_key)

    @app.get("/api/optimizer/history")
    async def optimizer_history() -> list[dict]:
        return await asyncio.to_thread(_history_entries)

    @app.get("/api/optimizer/history/{name}")
    async def optimizer_history_doc(name: str) -> Any:
        if "/" in name or not name.startswith("search_") \
                or not name.endswith(".json"):
            raise HTTPException(422, "bad history name")
        p = _history_dir / name
        if not p.exists():
            raise HTTPException(404, f"no archived search '{name}'")
        return await asyncio.to_thread(_read_json, p)

    async def _optimizer_catalog() -> dict:
        if app.state.optimizer_catalog is None:
            if not optimizer_script.exists():
                raise HTTPException(
                    404, f"{optimizer_script} not found — run the service "
                         f"from the repo root",
                )
            import sys
            res = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(optimizer_script), "--list-json"],
                capture_output=True, text=True, timeout=60,
            )
            if res.returncode != 0:
                raise HTTPException(
                    500, f"optimizer --list-json failed: {res.stderr[-500:]}",
                )
            app.state.optimizer_catalog = json.loads(res.stdout)
        return app.state.optimizer_catalog

    @app.get("/api/arena")
    async def arena() -> dict:
        """The full test arena for this host: every catalog model with
        its feasible TP set, every dimension with all values. The UI
        renders this with everything selected; the operator subtracts."""
        from .arena import full_arena
        return await asyncio.to_thread(full_arena)

    @app.post("/api/arena/preview")
    async def arena_preview(req: OptimizerStartRequest) -> dict:
        """Shape count + restart-cost estimate for a selection, before
        committing a night to it."""
        from .arena import build_space_doc, summarize_space_doc
        try:
            doc = await asyncio.to_thread(
                build_space_doc, req.arena or {}, None, req.budget,
            )
            return await asyncio.to_thread(summarize_space_doc, doc)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/optimizer")
    async def optimizer_status() -> dict:
        catalog = await _optimizer_catalog()
        opt = app.state.optimizer
        results = None
        if _opt_out.exists():
            try:
                results = json.loads(_opt_out.read_text())
            except (OSError, json.JSONDecodeError):
                results = None
        search_results = None
        if _search_out.exists():
            try:
                search_results = json.loads(_search_out.read_text())
            except (OSError, json.JSONDecodeError):
                search_results = None
        # The arena selection the current/last search was built from —
        # the UI restores its cards from this and defaults to resume.
        arena_space = None
        arena_space_path = _opt_out.parent / "arena_space.yaml"
        if arena_space_path.exists():
            import yaml as _yaml
            try:
                arena_space = _yaml.safe_load(arena_space_path.read_text())
            except (OSError, _yaml.YAMLError):
                arena_space = None
        from .search import SearchSpaceError, list_spaces, load_space
        spaces = list_spaces()

        def _details() -> dict:
            from .promote import space_overview
            out = {}
            for name, path in spaces.items():
                try:
                    out[name] = space_overview(load_space(path))
                except SearchSpaceError as e:
                    out[name] = {"error": str(e)}
            return out
        out: dict = {
            "running": _optimizer_running(),
            "catalog": catalog,
            "results": results,
            "out_path": str(_opt_out),
            "spaces": spaces,
            "space_details": await asyncio.to_thread(_details),
            "search_results": search_results,
            "search_out_path": str(_search_out),
            "arena_space": arena_space,
            # Model-set group of the current run, for history grouping.
            "search_group": (
                _group_of(_space_models(search_results.get("space_file")))
                if search_results else None),
        }
        if opt:
            out["active"] = {
                "mode": opt.get("mode", "registry"),
                "profile": opt["profile"],
                "started_at": opt["started_at"],
                "log": opt["log"],
                "exit_code": opt["proc"].poll(),
            }
        elif out["running"]:
            # Not our child — attach to the lock holder so a freshly
            # loaded UI still shows the in-flight run and can stop it.
            ext = _lock_state() or {}
            argv = ext.get("argv") or []
            if "--search" in argv:
                mode = ("arena" if any("arena_space" in str(a) for a in argv)
                        else "search")
            elif argv:
                mode = "registry"
            else:
                # Pre-JSON lock: nothing to infer from — say so rather
                # than guess, and let the UI route by file freshness.
                mode = "unknown"
            logs = sorted(_opt_out.parent.glob("optimizer_*.log"))
            out["active"] = {
                "mode": mode,
                "profile": mode,
                "started_at": ext.get("started_at"),
                "log": str(logs[-1]) if logs else None,
                "exit_code": None,
                "external": True,
                "pid": ext.get("pid"),
            }
        if out["running"] and out.get("active", {}).get("log"):
            out["active"]["heartbeat"] = await asyncio.to_thread(
                _log_heartbeat, out["active"]["log"])
        return out

    @app.post("/api/optimizer/start", status_code=202)
    async def optimizer_start(req: OptimizerStartRequest) -> dict:
        active = app.state.active
        if active is not None and not active.task.done():
            raise HTTPException(
                409, "a capacity run is active — the optimizer needs the "
                     "engines/GPUs to itself; stop the run first",
            )
        if _optimizer_running():
            raise HTTPException(409, "optimizer already running")
        import sys
        _opt_out.parent.mkdir(parents=True, exist_ok=True)
        log_path = _opt_out.parent / (
            f"optimizer_{time.strftime('%Y%m%dT%H%M%S')}.log"
        )
        seeded = 0
        if req.new_run:
            _archive_search_results()
        if req.mode == "arena":
            import yaml as _yaml

            from .arena import build_space_doc
            try:
                doc = await asyncio.to_thread(
                    build_space_doc, req.arena or {}, None, req.budget,
                )
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            space_path = _opt_out.parent / "arena_space.yaml"
            space_path.parent.mkdir(parents=True, exist_ok=True)
            space_path.write_text(_yaml.safe_dump(doc, sort_keys=False))
            cmd = [sys.executable, str(optimizer_script),
                   "--search", str(space_path),
                   "--search-out", str(_search_out)]
            if req.new_run:
                cmd.append("--new-run")
                # Reopening an investigation: seed everything the
                # group's earlier runs already measured, so this run
                # ADDS to "Qwen3 16-45B" instead of re-measuring it.
                seed_path, seeded = await asyncio.to_thread(
                    _build_seed_file, doc,
                )
                if seeded:
                    cmd.extend(["--seed-results", str(seed_path)])
        elif req.mode == "search":
            from .search import list_spaces
            spaces = list_spaces()
            space_path = spaces.get(req.space or "") or req.space
            if not space_path or not Path(space_path).exists():
                raise HTTPException(
                    404, f"unknown search space '{req.space}' — "
                         f"known: {sorted(spaces)}",
                )
            cmd = [sys.executable, str(optimizer_script),
                   "--search", str(space_path),
                   "--search-out", str(_search_out)]
            if req.new_run:
                cmd.append("--new-run")
        elif req.mode == "registry":
            catalog = await _optimizer_catalog()
            if req.profile not in catalog["profiles"]:
                raise HTTPException(
                    404, f"unknown optimizer profile '{req.profile}' — "
                         f"known: {sorted(catalog['profiles'])}",
                )
            cmd = [sys.executable, str(optimizer_script),
                   "--out", str(_opt_out), "--profile", req.profile]
            if req.new_run:
                cmd.append("--new-run")
            if req.only:
                cmd.extend(["--only", *req.only])
        else:
            raise HTTPException(422, "mode must be arena | search | registry")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        app.state.optimizer = {
            "proc": proc,
            "profile": req.profile or req.space or "arena",
            "mode": req.mode,
            "started_at": time.time(), "log": str(log_path),
        }
        return {"accepted": True, "mode": req.mode,
                "profile": req.profile, "space": req.space,
                "seeded": seeded, "log": str(log_path)}

    @app.post("/api/optimizer/promote")
    async def optimizer_promote(req: PromoteRequest) -> dict:
        """Winner → benchmark profile (config/profiles/optimized-*.yaml).
        Writes repo config the same way the persona editor does — the
        generated file is plain YAML the operator can read and rename."""
        from .promote import (
            PromoteError,
            promote_registry_winner,
            promote_search_winner,
        )
        try:
            if req.source == "search":
                if req.file:
                    if "/" in req.file or not req.file.startswith("search_"):
                        raise HTTPException(422, "bad history file name")
                    src = _opt_out.parent / "history" / req.file
                    if not src.exists():
                        raise HTTPException(404, f"no archived search "
                                                 f"'{req.file}'")
                else:
                    src = _search_out
                    if not src.exists():
                        raise HTTPException(404, "no guided-search results yet")
                doc = await asyncio.to_thread(_read_json, src)
                result = await asyncio.to_thread(promote_search_winner, doc)
            elif req.source == "registry":
                if not req.config_name:
                    raise HTTPException(422, "registry promote needs config_name")
                if not _opt_out.exists():
                    raise HTTPException(404, "no registry-sweep results yet")
                doc = await asyncio.to_thread(_read_json, _opt_out)
                catalog = await _optimizer_catalog()
                result = await asyncio.to_thread(
                    promote_registry_winner, catalog, doc, req.config_name,
                )
            else:
                raise HTTPException(422, "source must be search | registry")
        except PromoteError as e:
            raise HTTPException(409, str(e)) from e
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(500, f"could not read results: {e}") from e
        return result

    @app.post("/api/optimizer/stop")
    async def optimizer_stop() -> dict:
        if not _optimizer_running():
            raise HTTPException(409, "no optimizer running")
        import os as _os
        import signal as _signal
        opt = app.state.optimizer
        if opt and opt["proc"].poll() is None:
            proc = opt["proc"]
            with contextlib.suppress(ProcessLookupError):
                _os.killpg(_os.getpgid(proc.pid), _signal.SIGTERM)
            await asyncio.to_thread(proc.wait, 20)
        else:
            # External runner (started before a serve restart, or from
            # the CLI): kill the lock holder's process group and wait
            # for the kernel to release the flock.
            ext = _lock_state() or {}
            pid = ext.get("pid")
            if not pid:
                raise HTTPException(
                    409, "an optimizer holds the lock but its PID is "
                         "unreadable (started by an older build) — kill "
                         "the engine_optimizer process on the host, or "
                         "let it finish",
                )
            with contextlib.suppress(ProcessLookupError, PermissionError):
                _os.killpg(_os.getpgid(int(pid)), _signal.SIGTERM)

            def _wait_released() -> None:
                deadline = time.time() + 20
                while time.time() < deadline and _lock_state() is not None:
                    time.sleep(0.5)
            await asyncio.to_thread(_wait_released)
        # The optimizer cleans containers between configs, not on
        # SIGTERM — sweep up any vllm-* container it left running.
        def _cleanup() -> None:
            with contextlib.suppress(Exception):
                res = subprocess.run(
                    ["docker", "ps", "-aq", "--filter", "name=vllm-"],
                    capture_output=True, text=True, timeout=20,
                )
                cids = res.stdout.split()
                if cids:
                    subprocess.run(["docker", "rm", "-f", *cids],
                                   capture_output=True, timeout=60)
        await asyncio.to_thread(_cleanup)
        return {"stopped": True}

    # ── export ────────────────────────────────────────────────────

    @app.post("/api/export")
    async def export(req: ExportRequest) -> dict:
        from .export import export_dir
        try:
            doc, out_path = await asyncio.to_thread(
                export_dir, runs_base, None, slim=req.slim,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return {
            "path": str(out_path),
            "schema_version": doc["schema_version"],
            "cohort_count": doc["meta"]["cohort_count"],
            "slim": req.slim,
        }

    @app.get("/api/export/latest")
    async def export_latest(slim: bool = False) -> Any:
        from .runs import latest_run_dir
        d = latest_run_dir(runs_base)
        if d is None:
            raise HTTPException(404, "no runs yet")
        fname = "buyer_page_data_slim.json" if slim else "buyer_page_data.json"
        p = d / fname
        if not p.exists():
            raise HTTPException(
                404, f"{p} not built — POST /api/export first",
            )
        return await asyncio.to_thread(_read_json, p)

    @app.get("/api/runs/{run_name}/export")
    async def run_export(run_name: str, slim: bool = False) -> Any:
        """Export JSON for one run_NN dir — reads the built file, or
        builds it on first request (a few seconds on a big run.db).
        Powers the UI's results + run-comparison views."""
        if "/" in run_name or run_name.startswith("."):
            raise HTTPException(422, "bad run name")
        d = runs_base / run_name
        if not (d / "run.db").exists():
            raise HTTPException(404, f"no run.db in {d}")
        fname = "buyer_page_data_slim.json" if slim else "buyer_page_data.json"
        p = d / fname
        # One build at a time per run: the Results list and a
        # comparison can ask for the same run together, and two
        # concurrent export_dir calls wrote the same file over each
        # other. The second waiter re-checks staleness under the lock
        # and finds the first one's fresh file.
        lock = app.state.export_locks.setdefault(
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
                from .export import export_dir
                try:
                    await asyncio.to_thread(export_dir, d, p, slim=slim)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(500, f"export failed: {e}") from e
            return await asyncio.to_thread(_read_json, p)

    @app.get("/api/live/backfill")
    async def live_backfill(window_s: int = 600) -> dict:
        """History for the newest cohort run, shaped like the live
        WS events — a page opened mid-run (or after) replays this
        into the same chart handlers, so the view shows where the
        run IS and what it has done, not just what happens next."""
        from .runs import latest_run_dir

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

    # ── live telemetry ────────────────────────────────────────────

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        await ws.accept()
        q = BUS.subscribe()

        # Two halves: the sender pushes bus events; the receiver only
        # exists to notice the client going away. Without it a closed
        # tab left the handler parked on q.get() forever, and its
        # subscriber queue kept filling (the bus drops oldest, so it
        # was a leak of one queue per stale tab, not a crash).
        async def _send() -> None:
            while True:
                event = await q.get()
                await ws.send_json(event)

        async def _recv() -> None:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return

        tasks = [asyncio.create_task(_send()), asyncio.create_task(_recv())]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                with contextlib.suppress(WebSocketDisconnect, RuntimeError,
                                         asyncio.CancelledError):
                    t.result()
        finally:
            for t in tasks:
                t.cancel()
            BUS.unsubscribe(q)

    # ── UI (Phase 3) ──────────────────────────────────────────────
    # No-build static frontend shipped as package data; same origin as
    # the API and WebSocket, so no CORS. API routes above win — the
    # mount only catches what they don't.
    ui_dir = Path(__file__).parent / "ui"
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
