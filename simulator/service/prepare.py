"""Prepare: where model weights live, staging models into the shared
HF cache, and pulling engine runtime images — the gates a run must
pass before an engine can launch."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    EnginePullRequest,
    ModelAddRequest,
    ModelDownloadRequest,
    StorageRequest,
)

router = APIRouter()


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


# ── storage (choose the disk + location for model weights) ────
# Lists every mounted filesystem and every large unmounted disk;
# lets the user pick where weights live. The choice persists to
# ~/.config/capsim/storage.json and everything downstream (doctor,
# downloads, engine mounts, optimizer) resolves through it. The
# service NEVER formats or mounts disks — that is root-privileged
# and destructive, so unmounted disks come with the exact commands
# for a human instead.

@router.get("/api/storage")
async def storage_status() -> dict:
    import psutil

    from ..doctor import parse_lsblk_unmounted
    from ..models import hf_cache_dir, hf_cache_source

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


@router.post("/api/storage")
async def storage_set(req: StorageRequest) -> dict:
    from ..models import hf_cache_dir, set_hf_cache_dir
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

@router.get("/api/models")
async def models_list(request: Request) -> dict:
    from ..models import hf_cache_dir, referenced_models
    entries = await asyncio.to_thread(referenced_models)
    downloads = {}
    for model, dl in list(request.app.state.model_downloads.items()):
        exit_code = dl["proc"].poll()
        tail = (await asyncio.to_thread(_tail_text, dl["log"]))[-400:]
        downloads[model] = {
            "running": exit_code is None,
            "exit_code": exit_code,
            "started_at": dl["started_at"],
            "log": dl["log"],
            "log_tail": tail,
        }
    # A companion download is keyed "<model>#gguf"; the row's gguf
    # block says whether one is in flight so the UI needs no join.
    for row in entries:
        if row.get("gguf"):
            dl = downloads.get(f"{row['model']}#gguf")
            row["gguf"]["downloading"] = bool(dl and dl["running"])
    return {
        "cache_dir": str(hf_cache_dir()),
        "models": entries,
        "downloads": downloads,
    }


@router.post("/api/models/add")
async def models_add(req: ModelAddRequest) -> dict:
    """Add a model to the local catalog (config/models/local.yaml)
    and suggest its quantized siblings. Adding is what makes a
    model downloadable and searchable — the catalog is the explicit
    operator-curated list, so the download gate stays meaningful."""
    from ..model_catalog import (
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


@router.get("/api/models/discover")
async def models_discover(orgs: Optional[str] = None) -> dict:
    """Live Hub discovery, validated against THIS box: recent
    text-generation models from the leading orgs with real
    parameter counts, native-FP8 detection, capability tags, and
    the feasible TP set for the detected GPUs. Slow (one Hub call
    per candidate) — the UI calls it on demand, not on load."""
    from ..arena import hardware
    from ..discovery import DEFAULT_ORGS, discover_models
    from ..model_catalog import load_model_catalog
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


@router.post("/api/models/download", status_code=202)
async def models_download(req: ModelDownloadRequest, request: Request) -> dict:
    from ..models import download_command, referenced_models
    known = {m["model"]: m for m in await asyncio.to_thread(referenced_models)}
    if req.model not in known:
        # Only models the configs actually reference — the service
        # is not a general download proxy.
        raise HTTPException(
            404, f"'{req.model}' is not referenced by any profile "
                 f"or search space",
        )
    # The GGUF companion (KTransformers' weights) is staged under its
    # own key so it can run beside, and be reported apart from, the
    # safetensors download of the same model.
    key = req.model if req.companion is None else f"{req.model}#{req.companion}"
    if req.companion == "gguf" and not known[req.model].get("gguf"):
        raise HTTPException(
            422, f"'{req.model}' has no GGUF companion in the catalog — "
                 f"add a gguf: {{repo, file}} block to its entry",
        )
    existing = request.app.state.model_downloads.get(key)
    if existing and existing["proc"].poll() is None:
        raise HTTPException(409, "download already running for this model")
    try:
        argv, extra_env = download_command(req.model, companion=req.companion)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    log_dir = request.app.state.paths.runs_base / "model_downloads"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        key.replace("/", "--").replace("#", "--")
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
    request.app.state.model_downloads[key] = {
        "proc": proc, "log": str(log_path), "started_at": time.time(),
    }
    return {"accepted": True, "model": req.model, "companion": req.companion,
            "key": key, "log": str(log_path)}


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


@router.get("/api/engines")
async def engines_status(request: Request) -> dict:
    from ..engine_runtimes import image_store_root, runtime_status
    rows = await asyncio.to_thread(runtime_status)
    store = await asyncio.to_thread(image_store_root)
    for r in rows:
        pull = request.app.state.engine_pulls.get(r["engine"])
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


@router.post("/api/engines/pull", status_code=202)
async def engines_pull(req: EnginePullRequest, request: Request) -> dict:
    from ..engine_runtimes import RUNTIMES, image_store_root
    meta = RUNTIMES.get(req.engine)
    if meta is None:
        raise HTTPException(
            404, f"unknown engine '{req.engine}' — one of "
                 f"{', '.join(RUNTIMES)}")
    existing = request.app.state.engine_pulls.get(req.engine)
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
    log_dir = request.app.state.paths.runs_base / "engine_pulls"
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
    request.app.state.engine_pulls[req.engine] = {
        "proc": proc, "log": str(log_path), "started_at": time.time(),
    }
    return {"accepted": True, "engine": req.engine,
            "image": meta["image"], "log": str(log_path)}
