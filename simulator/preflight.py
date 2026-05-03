"""Hardware-compatibility preflight check.

Runs before any engine launch to fail fast on host/config mismatches —
e.g. SGLang's CPU FP8 path requires Intel AMX, and asking it to load a
30B model on AMD wastes 10-20 minutes before the assertion fires deep in
weight processing. The check inspects ``/proc/cpuinfo`` for vendor and
ISA-extension flags and validates them against the per-config
``hardware_requirements`` block.

Soft-skip on non-Linux hosts (developer Macs, etc.): we log a warning
and don't fail. The actual production host would fail on its own
constraints; we just can't validate from a Mac.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# /proc/cpuinfo vendor strings → canonical short names we use in configs.
_VENDOR_MAP = {
    "genuineintel": "intel",
    "authenticamd": "amd",
    "arm": "arm",
}


@dataclass
class HardwareInfo:
    vendor: Optional[str]            # "intel" | "amd" | "arm" | None
    cpu_model: Optional[str]
    flags: set[str]
    physical_cores: Optional[int]
    sockets: Optional[int]
    detection_status: str            # "ok" | "no_proc_cpuinfo" | "parse_error"


def detect_hardware(proc_cpuinfo: str = "/proc/cpuinfo") -> HardwareInfo:
    """Best-effort hardware detection from ``/proc/cpuinfo``.

    Returns a populated ``HardwareInfo`` on Linux; on hosts without
    /proc/cpuinfo the ``detection_status`` field is ``no_proc_cpuinfo``
    and the caller treats requirements checks as skipped.
    """
    p = Path(proc_cpuinfo)
    if not p.exists():
        return HardwareInfo(
            vendor=None, cpu_model=None, flags=set(),
            physical_cores=None, sockets=None,
            detection_status="no_proc_cpuinfo",
        )

    try:
        text = p.read_text()
    except OSError:
        return HardwareInfo(
            vendor=None, cpu_model=None, flags=set(),
            physical_cores=None, sockets=None,
            detection_status="parse_error",
        )

    vendor: Optional[str] = None
    cpu_model: Optional[str] = None
    flags: set[str] = set()
    cores_per_socket: Optional[int] = None
    socket_ids: set[int] = set()

    # /proc/cpuinfo is one block per logical CPU; each block has
    # "key : value" lines. The same physical-id appears once per
    # logical CPU on that socket; "cpu cores" is per-socket physical
    # core count.
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key == "vendor_id" and vendor is None:
            v = value.lower().strip()
            vendor = _VENDOR_MAP.get(v)
            if vendor is None:
                # Catch ARM-style "implementer" / "ARM" fallthrough.
                vendor = "arm" if "arm" in v else v
        elif key == "model name" and cpu_model is None:
            cpu_model = value
        elif key == "flags":
            flags.update(value.split())
        elif key == "cpu cores" and cores_per_socket is None:
            try:
                cores_per_socket = int(value)
            except ValueError:
                pass
        elif key == "physical id":
            try:
                socket_ids.add(int(value))
            except ValueError:
                pass

    sockets = len(socket_ids) if socket_ids else None
    physical_cores = (
        cores_per_socket * sockets if cores_per_socket and sockets else None
    )
    return HardwareInfo(
        vendor=vendor, cpu_model=cpu_model, flags=flags,
        physical_cores=physical_cores, sockets=sockets,
        detection_status="ok",
    )


@dataclass
class HardwareRequirements:
    """Per-config hardware constraints.

    All fields default to None / empty, meaning 'no constraint'. Only
    populated fields are checked.
    """
    cpu_vendor: Optional[str] = None        # "intel" | "amd" | "arm"
    cpu_features: list[str] = field(default_factory=list)  # /proc/cpuinfo flags
    min_physical_cores: Optional[int] = None
    min_sockets: Optional[int] = None
    notes: str = ""                          # free-form note shown on fail

    def is_empty(self) -> bool:
        return (
            self.cpu_vendor is None
            and not self.cpu_features
            and self.min_physical_cores is None
            and self.min_sockets is None
        )


class PreflightError(RuntimeError):
    """Raised when hardware doesn't satisfy a config's requirements."""


def check_requirements(
    info: HardwareInfo, reqs: HardwareRequirements,
) -> list[str]:
    """Return a list of human-readable failure messages.

    Empty list = all requirements satisfied (or skipped due to absent
    detection signal).
    """
    failures: list[str] = []
    if reqs.is_empty():
        return failures
    if info.detection_status != "ok":
        return failures   # soft-skip on non-Linux

    if reqs.cpu_vendor and info.vendor and info.vendor != reqs.cpu_vendor:
        failures.append(
            f"cpu_vendor mismatch: required '{reqs.cpu_vendor}', "
            f"detected '{info.vendor}' ({info.cpu_model or 'unknown'})"
        )

    missing = [f for f in reqs.cpu_features if f not in info.flags]
    if missing:
        failures.append(
            f"missing CPU feature flags: {sorted(missing)} "
            f"(detected on {info.cpu_model or 'unknown'}: "
            f"{sorted(f for f in reqs.cpu_features if f in info.flags)})"
        )

    if reqs.min_physical_cores and info.physical_cores is not None \
            and info.physical_cores < reqs.min_physical_cores:
        failures.append(
            f"min_physical_cores={reqs.min_physical_cores}, "
            f"detected {info.physical_cores}"
        )
    if reqs.min_sockets and info.sockets is not None \
            and info.sockets < reqs.min_sockets:
        failures.append(
            f"min_sockets={reqs.min_sockets}, detected {info.sockets}"
        )
    return failures


def preflight_check(reqs: HardwareRequirements, *, raise_on_fail: bool = True) -> bool:
    """Run hardware detection and validate requirements.

    Returns True if everything is satisfied (or skipped). When
    ``raise_on_fail`` is True (the default) any failure raises
    :class:`PreflightError` with a multi-line explanatory message.
    Otherwise returns False on failure.
    """
    info = detect_hardware()
    if info.detection_status == "no_proc_cpuinfo":
        log.warning(
            "preflight: /proc/cpuinfo unavailable (non-Linux host?); "
            "skipping hardware requirements validation"
        )
        return True

    log.info(
        "preflight: vendor=%s model=%r physical_cores=%s sockets=%s",
        info.vendor, info.cpu_model, info.physical_cores, info.sockets,
    )

    failures = check_requirements(info, reqs)
    if not failures:
        return True

    msg_lines = [
        "Hardware preflight FAILED — config requires:",
    ]
    if reqs.cpu_vendor:
        msg_lines.append(f"  cpu_vendor: {reqs.cpu_vendor}")
    if reqs.cpu_features:
        msg_lines.append(f"  cpu_features: {reqs.cpu_features}")
    if reqs.min_physical_cores:
        msg_lines.append(f"  min_physical_cores: {reqs.min_physical_cores}")
    if reqs.min_sockets:
        msg_lines.append(f"  min_sockets: {reqs.min_sockets}")
    if reqs.notes:
        msg_lines.append(f"  notes: {reqs.notes}")
    msg_lines.append("Detected:")
    msg_lines.append(f"  vendor: {info.vendor}")
    msg_lines.append(f"  cpu_model: {info.cpu_model}")
    msg_lines.append(f"  physical_cores: {info.physical_cores}")
    msg_lines.append(f"  sockets: {info.sockets}")
    msg_lines.append("Failures:")
    for f in failures:
        msg_lines.append(f"  * {f}")
    msg = "\n".join(msg_lines)

    if raise_on_fail:
        raise PreflightError(msg)
    log.error(msg)
    return False
