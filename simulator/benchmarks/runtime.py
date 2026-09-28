"""Pinned harnesses in separate environments, never in the engine environment."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .process import Lease, run_process
from .spec import VERSIONS
from .storage import read_json, write_json


def runtime_dir(base: Path, name: str) -> Path:
    if name not in VERSIONS:
        raise ValueError("Unknown harness")
    return base / "benchmark-tools" / f"{name}-{VERSIONS[name]}"


def status(base: Path) -> list[dict]:
    rows = []
    for name, version in VERSIONS.items():
        root = runtime_dir(base, name)
        ready = root / "ready.json"
        rows.append(
            {
                "id": name,
                "version": version,
                "ready": ready.is_file() and (root / "bin" / name).is_file(),
                "environment": str(root),
            }
        )
    return rows


def python_for_tools() -> str:
    choices = [
        os.getenv("CAPSIM_BENCHMARK_PYTHON"),
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.13"),
        sys.executable,
    ]
    for binary in dict.fromkeys(p for p in choices if p):
        try:
            p = subprocess.run(
                [binary, "-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            major, minor = map(int, p.stdout.strip().split("."))
            if major == 3 and 11 <= minor <= 13:
                return binary
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    raise RuntimeError("Harness preparation needs Python 3.11–3.13; set CAPSIM_BENCHMARK_PYTHON")


def validate_runtime(root: Path, name: str) -> dict:
    ready = read_json(root / "ready.json")
    python = root / "bin" / "python"
    if not (root / "bin" / name).is_file():
        raise RuntimeError("Harness entry point missing")
    version = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m;print(m.version(" + json.dumps(name) + "))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze"], capture_output=True, text=True, timeout=30, check=True
    ).stdout
    if version != VERSIONS[name] or set(freeze.splitlines()) != set(
        ready["requirements"].splitlines()
    ):
        raise RuntimeError("Harness dependency drift")
    return ready


async def prepare(base: Path, name: str) -> dict:
    root = runtime_dir(base.resolve(), name)
    root.parent.mkdir(parents=True, exist_ok=True)
    with Lease(root.parent / f".{name}.lock"):
        ready = root / "ready.json"
        python = root / "bin" / "python"
        if ready.exists():
            try:
                data = await asyncio.to_thread(validate_runtime, root, name)
                return {**data, "binary": str(root / "bin" / name)}
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                # Preserve the broken environment; rebuild exactly once from its saved lock.
                if not (root.parent / f"{name}-{VERSIONS[name]}.requirements.txt").is_file():
                    raise RuntimeError(
                        "Saved dependency lock missing; cannot repair the existing environment faithfully"
                    ) from None
                root.rename(root.with_name(root.name + f".broken-{time.time_ns()}"))
        if not ready.exists():
            creator = await asyncio.to_thread(python_for_tools)
            if await run_process(
                [creator, "-m", "venv", str(root)], root.parent / f"{name}-venv.log", timeout=120
            ):
                raise RuntimeError(f"Unable to create {name} environment")
            lock = root.parent / f"{name}-{VERSIONS[name]}.requirements.txt"
            install = ["-r", str(lock)] if lock.exists() else [f"{name}=={VERSIONS[name]}"]
            if await run_process(
                [str(python), "-m", "pip", "install", *install],
                root.parent / f"{name}-install.log",
                timeout=3600,
            ):
                raise RuntimeError(
                    f"{name} install failed; inspect benchmark-tools/{name}-install.log"
                )
            proc = await asyncio.create_subprocess_exec(
                str(python), "-m", "pip", "freeze", stdout=asyncio.subprocess.PIPE
            )
            data, _ = await proc.communicate()
            if proc.returncode:
                raise RuntimeError("Unable to record harness dependency lock")
            lock.write_bytes(data)
            write_json(
                ready,
                {
                    "harness": name,
                    "version": VERSIONS[name],
                    "requirements": data.decode(),
                    "python": str(python),
                },
            )
        data = await asyncio.to_thread(validate_runtime, root, name)
        return {**data, "binary": str(root / "bin" / name)}
