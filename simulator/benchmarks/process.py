"""Cancellable process groups with time limits and parent-death cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path


def module_argv(package_root: str, module: str, *args: str) -> list[str]:
    """Keep the worker on the caller's source even when cwd contains an older checkout."""
    return [
        sys.executable,
        "-c",
        "import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); "
        "sys.path.insert(0,root); runpy.run_module(module,run_name='__main__')",
        package_root,
        module,
        *args,
    ]


async def run_process(
    argv: list[str],
    log: Path,
    *,
    timeout: float,
    env: dict | None = None,
    cwd: Path | None = None,
    termination_grace: float = 10,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            str(os.getpid()),
            str(termination_grace),
            *argv,
            stdout=stream,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            env=env,
            cwd=cwd,
        )
        try:
            return await asyncio.wait_for(proc.wait(), timeout)
        finally:
            # Also kill surviving grandchildren after a leader exits normally.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), termination_grace)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)


def hardware_lease_path() -> Path:
    """One lock file per user for the physical accelerators.

    Deliberately not under ``tempfile.gettempdir()``: a service started by
    launchd/systemd and a CLI shell often see different ``TMPDIR`` values and
    would otherwise never contend, and tmp cleaners unlink held lock files.
    ``CAPSIM_HARDWARE_LOCK`` overrides the location for shared hosts.
    """
    override = os.environ.get("CAPSIM_HARDWARE_LOCK")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "capsim" / "hardware.lock"


class Lease:
    """Cross-process lease; the kernel releases it even after an unclean exit."""

    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise RuntimeError(
                "Benchmark resource is busy; another process holds its lease"
            ) from None
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()
            self.file = None


def guard(parent: int, grace: float, argv: list[str]) -> int:
    """Keep the tool group bounded even if its Python supervisor is SIGKILLed."""
    import subprocess
    import time

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    proc = subprocess.Popen(argv)
    while proc.poll() is None and not stopping and os.getppid() == parent:
        time.sleep(0.2)
    if stopping or os.getppid() != parent:
        os.killpg(os.getpgrp(), signal.SIGTERM)
        deadline = time.monotonic() + grace
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        # All descendants inherit this group. Include the guard itself so no
        # extra process can survive a lost owner; the cancelled caller ignores rc.
        os.killpg(os.getpgrp(), signal.SIGKILL)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(guard(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]))
