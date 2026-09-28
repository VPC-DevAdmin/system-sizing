"""Reversible GPU power controls and explicit energy measurement boundaries.

Callers must hold the capsim host lease. This module never labels GPU board
telemetry as whole-system energy, and refuses to overwrite a pending restore.
"""

import csv
import io
import math
import subprocess
from pathlib import Path

from ..benchmarks.storage import read_json, write_json


def _run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("GPU power command failed: " + result.stderr.strip())
    return result.stdout


def limits():
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,power.limit,power.min_limit,power.max_limit",
            "--format=csv,noheader,nounits",
        ]
    )
    result = {}
    for row in csv.reader(io.StringIO(raw)):
        identity, current, low, high = (s.strip() for s in row)
        values = [float(x) for x in (current, low, high)]
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("GPU power limits are not finite positive values")
        result[identity] = dict(zip(("current_w", "minimum_w", "maximum_w"), values, strict=True))
    if not result:
        raise ValueError("No readable GPU power limits")
    return result


class PowerTransaction:
    def __init__(self, journal, requested):
        self.journal = Path(journal)
        self.requested = requested

    def apply(self):
        if self.journal.exists() and read_json(self.journal).get("phase") != "restored":
            raise RuntimeError("A pending power restore must be completed first")
        before = limits()
        if not self.requested:
            raise ValueError("Specify at least one GPU power limit")
        for identity, watts in self.requested.items():
            bounds = before.get(identity)
            if (
                not bounds
                or not math.isfinite(watts)
                or not bounds["minimum_w"] <= watts <= bounds["maximum_w"]
            ):
                raise ValueError("Power request outside discovered device limits")
        state = {
            "phase": "applying",
            "before": {k: before[k]["current_w"] for k in self.requested},
            "requested": self.requested,
        }
        # Persist ALL original settings before the first host change.
        write_json(self.journal, state)
        try:
            for identity, watts in self.requested.items():
                _run(["sudo", "-n", "/usr/bin/nvidia-smi", "-i", identity, "-pl", str(watts)])
            observed = limits()
            if any(abs(observed[k]["current_w"] - v) > 0.5 for k, v in self.requested.items()):
                raise RuntimeError("GPU did not apply requested power limit")
            state["phase"] = "applied"
            write_json(self.journal, state)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        state = read_json(self.journal)
        if state["phase"] == "restored":
            return
        failures = []
        for identity, watts in state["before"].items():
            try:
                _run(["sudo", "-n", "/usr/bin/nvidia-smi", "-i", identity, "-pl", str(watts)])
            except Exception as exc:
                failures.append(str(exc))
        observed = limits()
        for identity, watts in state["before"].items():
            if identity not in observed or abs(observed[identity]["current_w"] - watts) > 0.5:
                failures.append("Original limit not restored for " + identity)
        state["phase"] = "restore_failed" if failures else "restored"
        state["restore_errors"] = failures
        write_json(self.journal, state)
        if failures:
            raise RuntimeError("Power restore needs attention: " + "; ".join(failures))


def integrate_energy(samples, start, end, *, boundary, max_gap_s):
    """Trapezoidal integration with bracketing and gap rejection; no extrapolation."""
    if boundary not in ("gpu_board", "server_psu_input", "pdu_outlet"):
        raise ValueError("Declare a supported sensor boundary")
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid measurement window")
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Specify a positive maximum telemetry gap")
    ordered = sorted(samples, key=lambda s: s["timestamp"])
    if len(ordered) < 2 or ordered[0]["timestamp"] > start or ordered[-1]["timestamp"] < end:
        raise ValueError("Telemetry does not bracket the complete measurement")
    energy = 0.0
    covered = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        t0, t1 = left["timestamp"], right["timestamp"]
        w0, w1 = left["watts"], right["watts"]
        if not all(math.isfinite(v) for v in (t0, t1, w0, w1)) or min(w0, w1) < 0 or t1 <= t0:
            raise ValueError("Invalid or duplicate power sample")
        lo, hi = max(start, t0), min(end, t1)
        if hi <= lo:
            continue
        if t1 - t0 > max_gap_s:
            raise ValueError("Telemetry gap exceeds the declared maximum")
        wl = w0 + (w1 - w0) * (lo - t0) / (t1 - t0)
        wh = w0 + (w1 - w0) * (hi - t0) / (t1 - t0)
        energy += (wl + wh) / 2 * (hi - lo)
        covered += hi - lo
    if abs(covered - (end - start)) > 1e-6:
        raise ValueError("Incomplete telemetry coverage")
    return {
        "joules": energy,
        "duration_s": end - start,
        "boundary": boundary,
        "whole_server": boundary in ("server_psu_input", "pdu_outlet"),
        "method": "Trapezoidal interpolation at measurement boundaries",
    }
