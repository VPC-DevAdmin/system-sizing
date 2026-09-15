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

from .bus import BUS

log = logging.getLogger(__name__)


# ── Run lifecycle state ───────────────────────────────────────────────


@dataclass
class ActiveRun:
    task: asyncio.Task
    workload: dict
    config_path: str
    started_at: float
    error: Optional[str] = None
    result: Optional[str] = None          # str(db_path) on success

    def describe(self) -> dict:
        return {
            "workload": self.workload,
            "config": self.config_path,
            "started_at": self.started_at,
            "running": not self.task.done(),
            "error": self.error,
            "result": self.result,
        }


class StartRunRequest(BaseModel):
    profile: Optional[str] = None
    config: Optional[str] = None
    # {"kind": "cohort"|"persona", "id": "..."} or {"kind": "sweep",
    # "type": "all"|"personas"|"cohorts"|"a,b,c"}
    workload: dict
    new_run: bool = False
    pool_sizes: Optional[list[int]] = None
    adaptive: bool = False


class ExportRequest(BaseModel):
    slim: bool = False


class SaveSpecRequest(BaseModel):
    yaml: str


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


class StorageRequest(BaseModel):
    hf_cache: str      # absolute directory for model weights


class PromoteRequest(BaseModel):
    # "search" promotes the guided search's best candidate; "registry"
    # promotes the named sweep config (the UI passes its ranked #1).
    source: str
    config_name: Optional[str] = None


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
        if not p.exists():
            raise HTTPException(404, f"config not found: {p}")
        return p
    raise HTTPException(422, "pass profile or config")


def _list_runs(base: Path) -> list[dict]:
    """run_NN dirs, newest first, with per-cohort summaries from each
    run.db (read-only; missing/corrupt DBs degrade to an empty list)."""
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
                rows = conn.execute(
                    "SELECT cohort_run_id, cohort_id, engine_type, model_id, "
                    "started_at, completed_at, final_status, "
                    "(SELECT COUNT(*) FROM cohort_measurements m "
                    " WHERE m.cohort_run_id = cohort_run.cohort_run_id) AS steps "
                    "FROM cohort_run ORDER BY started_at ASC"
                ).fetchall()
                entry["cohorts"] = [dict(r) for r in rows]
                conn.close()
            except sqlite3.Error as e:
                entry["error"] = str(e)
        exports = sorted(p.name for p in d.glob("buyer_page_data*.json"))
        entry["exports"] = exports
        out.append(entry)
    return out


def create_app(
    runs_base: Path | str = Path("runs"),
    catalog_dir: Path | str | None = None,
    # The engine optimizer is a repo script (like config/, resolved
    # against the working directory); injectable for tests.
    optimizer_script: Path | str = Path("scripts/engine_optimizer.py"),
) -> FastAPI:
    runs_base = Path(runs_base)
    catalog_dir = Path(catalog_dir) if catalog_dir is not None else None
    optimizer_script = Path(optimizer_script)
    app = FastAPI(title="capsim", version="0.2.0")
    app.state.active: Optional[ActiveRun] = None
    app.state.optimizer: Optional[dict] = None       # {proc, profile, started_at, log}
    app.state.optimizer_catalog: Optional[dict] = None
    app.state.model_downloads: dict = {}             # model -> {proc, log, started_at}

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
        from .config import list_profiles
        return {name: str(path) for name, path in list_profiles().items()}

    @app.get("/api/personas")
    async def personas() -> list[dict]:
        from .personas import PERSONAS
        return [
            {
                "id": p.id,
                "description": p.description,
                "ttft_target_s": p.ttft_target_seconds,
                "ttft_failure_s": p.ttft_failure_seconds,
                "tpot_target_ms": p.tpot_target_ms,
                "tpot_failure_ms": p.tpot_failure_ms,
            }
            for p in PERSONAS.values()
        ]

    @app.get("/api/cohorts")
    async def cohorts() -> list[dict]:
        from .personas import COHORTS
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "persona_weights": c.persona_weights,
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

    def _save_catalog_entry(kind: str, entry_id: str, spec_yaml: str) -> None:
        import yaml as _yaml

        from .persona_loader import PersonaSpecError, load_catalog
        from .personas import reload_personas

        if not entry_id.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(422, "id must be alphanumeric/_/-")
        try:
            spec = _yaml.safe_load(spec_yaml)
        except _yaml.YAMLError as e:
            raise HTTPException(422, f"invalid YAML: {e}") from e
        if not isinstance(spec, dict):
            raise HTTPException(422, "spec must be a YAML mapping")

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
        _save_catalog_entry("personas", persona_id, req.yaml)
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
        _save_catalog_entry("cohorts", cohort_id, req.yaml)
        return {"saved": cohort_id}

    @app.get("/api/runs")
    async def runs() -> list[dict]:
        return _list_runs(runs_base)

    @app.get("/api/doctor")
    async def doctor() -> dict:
        from .doctor import run_doctor
        report = await asyncio.to_thread(run_doctor)
        return report.to_dict()

    # ── run lifecycle ─────────────────────────────────────────────

    @app.post("/api/runs", status_code=202)
    async def start_run(req: StartRunRequest) -> dict:
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
        config_path = _resolve_config_path(req)

        from .config import load_config
        from .personas import COHORTS, PERSONAS, cohort_from_persona
        cfg = load_config(config_path)
        cfg.output.db_directory = str(runs_base)

        kind = req.workload.get("kind")
        if kind == "cohort":
            wid = req.workload.get("id")
            if wid not in COHORTS:
                raise HTTPException(404, f"unknown cohort '{wid}'")
            coro_factory = _cohort_coro(cfg, wid, req)
        elif kind == "persona":
            wid = req.workload.get("id")
            if wid not in PERSONAS:
                raise HTTPException(404, f"unknown persona '{wid}'")
            coro_factory = _cohort_coro(cfg, cohort_from_persona(wid), req)
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
                422, "workload.kind must be cohort | persona | sweep",
            )

        active = ActiveRun(
            task=asyncio.create_task(_supervise(app, coro_factory)),
            workload=req.workload,
            config_path=str(config_path),
            started_at=time.time(),
        )
        app.state.active = active
        return {"accepted": True, "workload": req.workload,
                "config": str(config_path)}

    def _cohort_coro(cfg, cohort, req: StartRunRequest):
        from .runner import run_cohort
        return lambda: run_cohort(
            cfg, cohort,
            new_run=req.new_run,
            adaptive=req.adaptive,
            fixed_grid_pool_sizes=req.pool_sizes,
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

    @app.post("/api/runs/stop")
    async def stop_run() -> dict:
        active = app.state.active
        if active is None or active.task.done():
            raise HTTPException(409, "no active run")
        active.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await active.task
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
            tail = ""
            try:
                text = Path(dl["log"]).read_text(errors="replace")
                tail = text[-400:]
            except OSError:
                pass
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
                    notes=req.notes,
                ),
            )
        except CatalogError as e:
            raise HTTPException(422, str(e)) from e
        siblings = await asyncio.to_thread(
            lambda: suggest_quant_siblings(req.model, verify=req.check_hub),
        )
        return {"entry": entry, "created": created,
                "hub_verified": exists, "siblings": siblings}

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
        log_file = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                argv, stdout=log_file, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env={**_os.environ, **extra_env},
            )
        except FileNotFoundError as e:
            log_file.close()
            raise HTTPException(
                500, f"hf CLI not found ({e}) — is huggingface_hub "
                     f"installed in the service environment?",
            ) from e
        app.state.model_downloads[req.model] = {
            "proc": proc, "log": str(log_path), "started_at": time.time(),
        }
        return {"accepted": True, "model": req.model, "log": str(log_path)}

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
                    return json.loads(fh.read() or "{}") or {}
                except json.JSONDecodeError:
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
            else:
                mode = "registry"
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
                "log": str(log_path)}

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
                if not _search_out.exists():
                    raise HTTPException(404, "no guided-search results yet")
                doc = json.loads(_search_out.read_text())
                result = await asyncio.to_thread(promote_search_winner, doc)
            elif req.source == "registry":
                if not req.config_name:
                    raise HTTPException(422, "registry promote needs config_name")
                if not _opt_out.exists():
                    raise HTTPException(404, "no registry-sweep results yet")
                doc = json.loads(_opt_out.read_text())
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
            if pid:
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
        return json.loads(p.read_text())

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
        if not p.exists():
            from .export import export_dir
            try:
                await asyncio.to_thread(export_dir, d, p, slim=slim)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(500, f"export failed: {e}") from e
        return json.loads(p.read_text())

    # ── live telemetry ────────────────────────────────────────────

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        await ws.accept()
        q = BUS.subscribe()
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            BUS.unsubscribe(q)

    # ── UI (Phase 3) ──────────────────────────────────────────────
    # No-build static frontend shipped as package data; same origin as
    # the API and WebSocket, so no CORS. API routes above win — the
    # mount only catches what they don't.
    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app


def serve(host: str = "127.0.0.1", port: int = 8321,
          runs_base: Path | str = Path("runs")) -> None:
    """Blocking uvicorn entrypoint used by ``capsim serve``."""
    import uvicorn
    uvicorn.run(create_app(runs_base), host=host, port=port, log_level="info")
