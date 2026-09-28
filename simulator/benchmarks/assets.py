"""Resolve local assets without downloading or trusting a mutable image tag."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .storage import sha256

IMAGE_FIELDS = {
    "vllm_cuda": "gpu_image",
    "vllm_cuda_multi": "gpu_image",
    "sglang_cuda": "sglang_image",
    "trtllm": "trtllm_image",
    "sglang": "docker_image",
}


def local_snapshot(model: str) -> str:
    if Path(model).is_dir():
        return str(Path(model).resolve())
    from huggingface_hub import snapshot_download

    from ..models import hf_cache_dir

    return snapshot_download(model, cache_dir=str(hf_cache_dir() / "hub"), local_files_only=True)


def fingerprint(cfg) -> dict:
    """Small metadata hashes plus immutable HF blob names; weights are not rehashed."""
    result = {"model_id": cfg.model_id, "engine_type": cfg.type}
    try:
        snapshot = Path(local_snapshot(cfg.model_id))
        result["snapshot"] = str(snapshot)
        result["files"] = {
            str(p.relative_to(snapshot)): {
                "bytes": p.stat().st_size,
                "blob": p.resolve().name,
                "sha256": sha256(p) if p.stat().st_size < 16 * 1024**2 else None,
            }
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
    except (OSError, ValueError):
        result["snapshot"] = None
    field = IMAGE_FIELDS.get(cfg.type)
    if field and getattr(cfg, field, None):
        try:
            p = subprocess.run(
                ["docker", "image", "inspect", getattr(cfg, field), "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if p.returncode == 0:
                result["image"] = {"field": field, "id": p.stdout.strip()}
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def pin_model(cfg, snapshot: str):
    """Mount the complete HF repo cache so snapshot symlinks still reach blobs."""
    if cfg.type not in IMAGE_FIELDS or cfg.model_local_path:
        return
    path = Path(snapshot)
    # Recent HF caches can chain snapshot -> repo blobs -> shared hub blobs.
    # Preserve the entire hub at its original absolute path, so relative and
    # absolute symlinks both remain valid inside the container.
    cache = path.parent.parent.parent if path.parent.name == "snapshots" else path
    for item in path.rglob("*"):
        if item.is_symlink() and not item.exists():
            raise ValueError(f"Broken cached model link: {item.name}")
        if item.is_file() and not item.resolve().is_relative_to(cache.resolve()):
            raise ValueError(
                "Model files resolve outside the mounted cache; stage a self-contained model cache"
            )
    cfg.docker_volumes[str(cache)] = str(cache) + ":ro"
    cfg.model_local_path = str(path)
    cfg.served_model_name = cfg.served_model_name or cfg.model_id
