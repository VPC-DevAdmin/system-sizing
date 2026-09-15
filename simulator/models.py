"""Model weight staging: HF cache inspection + download commands.

Backs the UI's Models panel: which models do the profiles and search
spaces reference, are their weights already in the HF cache the engine
containers mount, and how do we fetch the missing ones?

The cache directory resolution matches the engine/optimizer mounts —
``OPTIMIZER_HF_CACHE`` env override, else ``/data/ml/huggingface``
when that layout exists, else ``~/.cache/huggingface`` — and every
download is pinned to it via ``HF_HOME`` so weights land where the
containers will look, not wherever the CLI's default happens to be.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml


def hf_cache_dir() -> Path:
    override = os.environ.get("OPTIMIZER_HF_CACHE")
    if override:
        return Path(override)
    data = Path("/data/ml/huggingface")
    if data.exists():
        return data
    return Path.home() / ".cache" / "huggingface"


def _model_dir(model_id: str, cache: Path) -> Path:
    return cache / "hub" / ("models--" + model_id.replace("/", "--"))


def _dir_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / 1e9


def model_status(model_id: str, cache: Path | None = None) -> dict:
    """Cache status for one HF model id. ``cached`` means a snapshot
    revision with files exists and no blob is mid-download."""
    cache = cache or hf_cache_dir()
    d = _model_dir(model_id, cache)
    snapshots = d / "snapshots"
    has_snapshot = snapshots.exists() and any(
        rev.is_dir() and any(rev.iterdir()) for rev in snapshots.iterdir()
    )
    incomplete = any((d / "blobs").glob("*.incomplete")) if (d / "blobs").exists() else False
    return {
        "model": model_id,
        "cached": bool(has_snapshot and not incomplete),
        "partial": bool(incomplete or (d.exists() and not has_snapshot)),
        "size_gb": round(_dir_size_gb(d), 1) if d.exists() else 0.0,
        "path": str(d),
    }


def referenced_models(
    profiles_dirs: tuple[str, ...] = ("config/profiles", "config"),
    search_dir: str = "config/search",
) -> list[dict]:
    """Every HF model id referenced by GPU profiles and search spaces,
    with where it came from. Local ``/models/...`` paths (the CPU
    pre-staged flow) are skipped — they aren't HF downloads."""
    out: dict[str, set[str]] = {}

    def _add(model, source):
        if not model or str(model).startswith("/"):
            return
        out.setdefault(str(model), set()).add(source)

    for d in profiles_dirs:
        for p in sorted(Path(d).glob("*.yaml")) if Path(d).exists() else []:
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            engine = (raw.get("engine") or {})
            # Only engines that pull from HF at launch (the GPU flow);
            # the CPU flow stages into /models via `capsim ready`.
            if engine.get("type") == "vllm_cuda":
                _add(engine.get("model_id"), f"profile:{p.stem}")
    if Path(search_dir).exists():
        for p in sorted(Path(search_dir).glob("*.yaml")):
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            for vname, v in (raw.get("model_variants") or {}).items():
                if isinstance(v, dict):
                    _add(v.get("model"), f"space:{p.stem}/{vname}")

    cache = hf_cache_dir()
    return [
        {**model_status(m, cache), "referenced_by": sorted(srcs)}
        for m, srcs in sorted(out.items())
    ]


def download_command(model_id: str) -> tuple[list[str], dict]:
    """(argv, extra_env) for a cache-pinned weight download. Prefers
    the ``hf`` CLI sitting next to this interpreter (the tool venv),
    falling back to PATH."""
    hf_bin = Path(sys.executable).parent / "hf"
    argv = [str(hf_bin) if hf_bin.exists() else (shutil.which("hf") or "hf"),
            "download", model_id]
    env = {"HF_HOME": str(hf_cache_dir())}
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            env[var] = os.environ[var]
    return argv, env
