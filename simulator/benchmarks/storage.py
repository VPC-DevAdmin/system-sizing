"""Atomic experiment records and portable evidence, independent of UI lifetime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

SECRET = re.compile(r"token$|(?:api|master|access)[_-]?key|password|secret|authorization", re.I)


def redact(value):
    if isinstance(value, dict):
        return {
            k: ("[redacted]" if SECRET.search(k) and v else redact(v)) for k, v in value.items()
        }
    if isinstance(value, list):
        out = []
        hide_next = False
        for item in value:
            if hide_next:
                out.append("[redacted]")
                hide_next = False
            elif isinstance(item, str) and "=" in item and SECRET.search(item.split("=", 1)[0]):
                out.append(item.split("=", 1)[0] + "=[redacted]")
            elif isinstance(item, str) and item.startswith("--") and SECRET.search(item):
                out.append(item)
                hide_next = True
            else:
                out.append(redact(item))
        return out
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def campaign_path(base: Path, campaign_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{24}", campaign_id):
        raise ValueError("Invalid benchmark ID")
    root = (base / "benchmarks").resolve()
    path = (root / campaign_id).resolve()
    if path.parent != root:
        raise ValueError("Benchmark directory must not be a symlink outside the run directory")
    return path
