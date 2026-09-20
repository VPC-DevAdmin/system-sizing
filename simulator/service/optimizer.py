"""Engine optimizer and arena: the supervised
``scripts/engine_optimizer.py`` subprocess, its results and history,
and promotion of a winner to a benchmark profile."""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from .schemas import OptimizerStartRequest, PromoteRequest
from .state import Paths, _read_json

router = APIRouter()


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


# ── engine optimizer (find the best launch shape first) ──────
# Runs scripts/engine_optimizer.py as a supervised subprocess —
# same artifact contract as `make optimize-engine` (incremental
# runs/engine_optimizer/run.json, resumable), so the UI, the CLI,
# and a second SSH session all watch the same file.

def _lock_state(paths: Paths) -> Optional[dict]:
    """Attach point for optimizers this service did not start
    (serve restarted mid-run, or a CLI launch): the runner holds
    an exclusive flock on .optimizer.lock with {pid, started_at,
    argv} inside. If the flock is TAKEN, someone is running —
    return their state; if we can acquire it, nobody is (the
    kernel releases flocks when the holder dies, so a stale file
    never reads as running)."""
    import fcntl
    p = paths.opt_out.parent / ".optimizer.lock"
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


def _optimizer_running(app) -> bool:
    opt = app.state.optimizer
    if opt and opt["proc"].poll() is None:
        return True
    return _lock_state(app.state.paths) is not None


def _archive_search_results(paths: Paths) -> None:
    """A fresh run overwrites search.json — archive the previous
    run first so optimizer results have history like benchmark
    runs do. The space file is archived alongside and the doc's
    space_file re-pointed at the copy, so a historical winner can
    still be promoted (the space hash still verifies)."""
    if not paths.search_out.exists():
        return
    try:
        doc = json.loads(paths.search_out.read_text())
    except (OSError, json.JSONDecodeError):
        return
    ts = (doc.get("generated_at") or "")[:19].replace(":", "").replace(
        "-", "") or time.strftime("%Y%m%dT%H%M%S")
    paths.history_dir.mkdir(parents=True, exist_ok=True)
    base = f"search_{ts}_{doc.get('space', 'space')}"
    space_src = Path(doc.get("space_file") or "")
    if space_src.exists():
        space_copy = paths.history_dir / f"{base}_space.yaml"
        space_copy.write_text(space_src.read_text())
        doc["space_file"] = str(space_copy)
    (paths.history_dir / f"{base}.json").write_text(json.dumps(doc, indent=2))


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
    from ..arena import group_for_models
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


def _history_entries(paths: Paths) -> list[dict]:
    if not paths.history_dir.exists():
        return []
    out = []
    for p in sorted(paths.history_dir.glob("search_*.json"), reverse=True):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(_doc_summary_entry(doc, p.name))
    return out


def _combined_group(paths: Paths, group_key: str) -> dict:
    """Merge every run of one model-set group — history plus the
    current run — into a single ranking. Evaluations dedupe by
    canonical candidate key (best score wins); runs whose
    objective or measurement fingerprint differs from the group's
    newest run are EXCLUDED and named, never silently mixed."""
    from ..search import Objective, best_rung

    docs: list[tuple[Optional[str], dict]] = []   # (file|None=current, doc)
    if paths.search_out.exists():
        try:
            docs.append((None, json.loads(paths.search_out.read_text())))
        except (OSError, json.JSONDecodeError):
            pass
    if paths.history_dir.exists():
        for p in sorted(paths.history_dir.glob("search_*.json"), reverse=True):
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


def _build_seed_file(paths: Paths, space_doc: dict) -> tuple[Path, int]:
    """Collect ok evaluations from the new run's investigation
    group (history archives with the same (series, size-range)
    identity AND the same objective+measurement fingerprint) into
    a seed file for the driver. Best score wins on duplicates."""
    import dataclasses as _dc

    from ..search import Measurement, Objective

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
        return paths.opt_out.parent / "seed.json", 0
    want_fp = _fp_of(space_doc.get("objective"),
                     space_doc.get("measurement"))
    merged: dict[str, dict] = {}
    if paths.history_dir.exists():
        for p in sorted(paths.history_dir.glob("search_*.json")):
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
    seed_path = paths.opt_out.parent / "seed.json"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps({"evaluated": merged}))
    return seed_path, len(merged)


@router.get("/api/optimizer/combined/{group_key}")
async def optimizer_combined(group_key: str, request: Request) -> dict:
    if not group_key.isalnum():
        raise HTTPException(422, "bad group key")
    return await asyncio.to_thread(
        _combined_group, request.app.state.paths, group_key)


@router.get("/api/optimizer/history")
async def optimizer_history(request: Request) -> list[dict]:
    return await asyncio.to_thread(_history_entries, request.app.state.paths)


@router.get("/api/optimizer/history/{name}")
async def optimizer_history_doc(name: str, request: Request) -> Any:
    if "/" in name or not name.startswith("search_") \
            or not name.endswith(".json"):
        raise HTTPException(422, "bad history name")
    p = request.app.state.paths.history_dir / name
    if not p.exists():
        raise HTTPException(404, f"no archived search '{name}'")
    return await asyncio.to_thread(_read_json, p)


async def _optimizer_catalog(app) -> dict:
    optimizer_script = app.state.paths.optimizer_script
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


@router.get("/api/arena")
async def arena() -> dict:
    """The full test arena for this host: every catalog model with
    its feasible TP set, every dimension with all values. The UI
    renders this with everything selected; the operator subtracts."""
    from ..arena import full_arena
    return await asyncio.to_thread(full_arena)


@router.post("/api/arena/preview")
async def arena_preview(req: OptimizerStartRequest) -> dict:
    """Shape count + restart-cost estimate for a selection, before
    committing a night to it."""
    from ..arena import build_space_doc, summarize_space_doc
    try:
        doc = await asyncio.to_thread(
            build_space_doc, req.arena or {}, None, req.budget,
        )
        return await asyncio.to_thread(summarize_space_doc, doc)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/api/optimizer")
async def optimizer_status(request: Request) -> dict:
    app = request.app
    paths = app.state.paths
    catalog = await _optimizer_catalog(app)
    opt = app.state.optimizer
    results = None
    if paths.opt_out.exists():
        try:
            results = json.loads(paths.opt_out.read_text())
        except (OSError, json.JSONDecodeError):
            results = None
    search_results = None
    if paths.search_out.exists():
        try:
            search_results = json.loads(paths.search_out.read_text())
        except (OSError, json.JSONDecodeError):
            search_results = None
    # The arena selection the current/last search was built from —
    # the UI restores its cards from this and defaults to resume.
    arena_space = None
    arena_space_path = paths.opt_out.parent / "arena_space.yaml"
    if arena_space_path.exists():
        import yaml as _yaml
        try:
            arena_space = _yaml.safe_load(arena_space_path.read_text())
        except (OSError, _yaml.YAMLError):
            arena_space = None
    from ..search import SearchSpaceError, list_spaces, load_space
    spaces = list_spaces()

    def _details() -> dict:
        from ..promote import space_overview
        out = {}
        for name, path in spaces.items():
            try:
                out[name] = space_overview(load_space(path))
            except SearchSpaceError as e:
                out[name] = {"error": str(e)}
        return out
    out: dict = {
        "running": _optimizer_running(app),
        "catalog": catalog,
        "results": results,
        "out_path": str(paths.opt_out),
        "spaces": spaces,
        "space_details": await asyncio.to_thread(_details),
        "search_results": search_results,
        "search_out_path": str(paths.search_out),
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
        ext = _lock_state(paths) or {}
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
        logs = sorted(paths.opt_out.parent.glob("optimizer_*.log"))
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


@router.post("/api/optimizer/start", status_code=202)
async def optimizer_start(req: OptimizerStartRequest, request: Request) -> dict:
    # Same lock as run start: without it a run and an optimizer
    # could both clear their guards and start together, and each
    # engine launch sweeps the other's containers.
    async with request.app.state.start_lock:
        return await _optimizer_start_locked(request.app, req)


def _refuse_if_busy(app) -> None:
    active = app.state.active
    if active is not None and not active.task.done():
        raise HTTPException(
            409, "a capacity run is active — the optimizer needs the "
                 "engines/GPUs to itself; stop the run first",
        )
    if _optimizer_running(app):
        raise HTTPException(409, "optimizer already running")


async def _optimizer_start_locked(app, req: OptimizerStartRequest) -> dict:
    paths = app.state.paths
    optimizer_script = paths.optimizer_script
    _refuse_if_busy(app)
    import sys
    paths.opt_out.parent.mkdir(parents=True, exist_ok=True)
    log_path = paths.opt_out.parent / (
        f"optimizer_{time.strftime('%Y%m%dT%H%M%S')}.log"
    )
    seeded = 0
    if req.new_run:
        _archive_search_results(paths)
    if req.mode == "arena":
        import yaml as _yaml

        from ..arena import build_space_doc
        try:
            doc = await asyncio.to_thread(
                build_space_doc, req.arena or {}, None, req.budget,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        space_path = paths.opt_out.parent / "arena_space.yaml"
        space_path.parent.mkdir(parents=True, exist_ok=True)
        space_path.write_text(_yaml.safe_dump(doc, sort_keys=False))
        cmd = [sys.executable, str(optimizer_script),
               "--search", str(space_path),
               "--search-out", str(paths.search_out)]
        if req.new_run:
            cmd.append("--new-run")
            # Reopening an investigation: seed everything the
            # group's earlier runs already measured, so this run
            # ADDS to "Qwen3 16-45B" instead of re-measuring it.
            seed_path, seeded = await asyncio.to_thread(
                _build_seed_file, paths, doc,
            )
            if seeded:
                cmd.extend(["--seed-results", str(seed_path)])
    elif req.mode == "search":
        from ..search import list_spaces
        spaces = list_spaces()
        space_path = spaces.get(req.space or "") or req.space
        if not space_path or not Path(space_path).exists():
            raise HTTPException(
                404, f"unknown search space '{req.space}' — "
                     f"known: {sorted(spaces)}",
            )
        cmd = [sys.executable, str(optimizer_script),
               "--search", str(space_path),
               "--search-out", str(paths.search_out)]
        if req.new_run:
            cmd.append("--new-run")
    elif req.mode == "registry":
        catalog = await _optimizer_catalog(app)
        if req.profile not in catalog["profiles"]:
            raise HTTPException(
                404, f"unknown optimizer profile '{req.profile}' — "
                     f"known: {sorted(catalog['profiles'])}",
            )
        cmd = [sys.executable, str(optimizer_script),
               "--out", str(paths.opt_out), "--profile", req.profile]
        if req.new_run:
            cmd.append("--new-run")
        if req.only:
            cmd.extend(["--only", *req.only])
    else:
        raise HTTPException(422, "mode must be arena | search | registry")
    # Re-checked after every await above: the lock keeps runs out,
    # but an optimizer started from the CLI takes the flock without
    # asking this process.
    _refuse_if_busy(app)
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


@router.post("/api/optimizer/promote")
async def optimizer_promote(req: PromoteRequest, request: Request) -> dict:
    """Winner → benchmark profile (config/profiles/optimized-*.yaml).
    Writes repo config the same way the persona editor does — the
    generated file is plain YAML the operator can read and rename."""
    from ..promote import (
        PromoteError,
        promote_registry_winner,
        promote_search_winner,
    )
    app = request.app
    paths = app.state.paths
    try:
        if req.source == "search":
            if req.file:
                if "/" in req.file or not req.file.startswith("search_"):
                    raise HTTPException(422, "bad history file name")
                src = paths.opt_out.parent / "history" / req.file
                if not src.exists():
                    raise HTTPException(404, f"no archived search "
                                             f"'{req.file}'")
            else:
                src = paths.search_out
                if not src.exists():
                    raise HTTPException(404, "no guided-search results yet")
            doc = await asyncio.to_thread(_read_json, src)
            result = await asyncio.to_thread(promote_search_winner, doc)
        elif req.source == "registry":
            if not req.config_name:
                raise HTTPException(422, "registry promote needs config_name")
            if not paths.opt_out.exists():
                raise HTTPException(404, "no registry-sweep results yet")
            doc = await asyncio.to_thread(_read_json, paths.opt_out)
            catalog = await _optimizer_catalog(app)
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


@router.post("/api/optimizer/stop")
async def optimizer_stop(request: Request) -> dict:
    app = request.app
    paths = app.state.paths
    if not _optimizer_running(app):
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
        ext = _lock_state(paths) or {}
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
            while time.time() < deadline and _lock_state(paths) is not None:
                time.sleep(0.5)
        await asyncio.to_thread(_wait_released)
    # The optimizer cleans containers between configs, not on
    # SIGTERM — sweep up any capsim engine container it left
    # running (a search may launch any engine, not just vLLM).
    # Anchored to capsim's prefixes: ``name=vllm-`` also matched
    # a user's my-vllm-dev.
    def _cleanup() -> None:
        from ..engines.docker_replica import container_name_filter
        for flt in container_name_filter():
            with contextlib.suppress(Exception):
                res = subprocess.run(
                    ["docker", "ps", "-aq", "--filter", flt],
                    capture_output=True, text=True, timeout=20,
                )
                cids = res.stdout.split()
                if cids:
                    subprocess.run(["docker", "rm", "-f", *cids],
                                   capture_output=True, timeout=60)
    await asyncio.to_thread(_cleanup)
    return {"stopped": True}
