"""Memory bandwidth via Intel IMC uncore events through ``perf stat``.

GNR (Granite Rapids / Xeon 6) splits the legacy ``uncore_imc`` PMU into
per-controller PMUs (``uncore_imc_0`` ... ``uncore_imc_N``). The legacy
alias resolves to ZERO PMUs on GNR — perf returns rc=0 with no events,
silently producing empty samples. We discover the per-controller PMUs
via sysfs and SUM across them at each timestamp; the legacy alias is
the SPR/EMR fallback.

Other lessons baked in:
  * Modern perf scales ``cas_count_*`` events via the event's built-in
    ``.scale=64``, emitting ``"238.47 MiB"`` instead of a raw CAS count.
    The ``.isdigit()``-style filter dropped all such lines on Linux 6.8.
    We parse as float and scale by the unit column.
  * Pre-flight ``perf stat -a -e <events> -- sleep 0`` so silent-zero
    uncore_imc resolution surfaces as a real failure status.
"""

from __future__ import annotations

import logging
import re
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

UTC = timezone.utc

_IMC_PMU_ROOT = "/sys/bus/event_source/devices"
_IMC_READ_EVENT_LEGACY = "uncore_imc/cas_count_read/"
_IMC_WRITE_EVENT_LEGACY = "uncore_imc/cas_count_write/"
_BYTES_PER_CAS = 64  # cache-line size; only used for raw-count fallback.


def _discover_imc_events(root: str | Path = _IMC_PMU_ROOT) -> tuple[list[str], list[str]] | None:
    """Return (read_events, write_events) for every per-controller IMC PMU,
    or None if the host only has the legacy aggregate (SPR/EMR) or none.

    ``uncore_imc_free_running_*`` entries are excluded — they're free-running
    counters with a different event set.
    """
    p = Path(root)
    if not p.is_dir():
        return None
    reads: list[str] = []
    writes: list[str] = []
    for entry in sorted(p.iterdir()):
        name = entry.name
        if not name.startswith("uncore_imc"):
            continue
        if name == "uncore_imc":
            continue  # legacy alias handled as fallback
        if "free_running" in name:
            continue
        reads.append(f"{name}/cas_count_read/")
        writes.append(f"{name}/cas_count_write/")
    if not reads:
        return None
    return reads, writes


@dataclass
class BandwidthSample:
    sampled_at: datetime
    read_gb_s: float
    write_gb_s: float


@dataclass
class BandwidthCollector:
    interval_ms: int = 1000
    perf_path: str | None = None
    _status: str = field(default="not_started", init=False)
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _samples: list[BandwidthSample] = field(default_factory=list, init=False, repr=False)
    _raw_lines: list[str] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _reader_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> None:
        resolved = self.perf_path or shutil.which("perf")
        if resolved is None:
            self._status = "perf_not_found"
            return

        # Per-controller events on GNR; legacy alias on SPR/EMR.
        discovered = _discover_imc_events()
        if discovered is not None:
            reads, writes = discovered
            events_csv = ",".join(reads + writes)
        else:
            events_csv = f"{_IMC_READ_EVENT_LEGACY},{_IMC_WRITE_EVENT_LEGACY}"

        # Pre-flight: silent zero-output is a real failure mode on GNR
        # when only the legacy alias is present. An honest exit code
        # comes from per-controller event names if discovered.
        try:
            probe = subprocess.run(
                [resolved, "stat", "-a", "-e", events_csv, "--", "sleep", "0"],
                capture_output=True, text=True, timeout=3,
            )
            combined = (probe.stdout + probe.stderr).lower()
            if probe.returncode != 0 or "not supported" in combined or "no such" in combined:
                self._status = "perf_uncore_unavailable"
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            log.debug("uncore probe failed: %s", e)
            self._status = "perf_uncore_unavailable"
            return

        try:
            self._process = subprocess.Popen(
                [resolved, "stat", "-a", "-I", str(self.interval_ms),
                 "-x,", "-e", events_csv],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as e:
            log.warning("Failed to launch bandwidth perf: %s", e)
            self._status = "perf_launch_failed"
            return

        self._status = "running"
        self._reader_thread = threading.Thread(
            target=self._read_output, name="bandwidth-reader", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> list[BandwidthSample]:
        if self._status != "running" or self._process is None:
            self._parse_raw_output()  # might have been started but never produced
            return list(self._samples)

        process = self._process
        self._process = None
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None
        self._status = "stopped"
        self._parse_raw_output()
        return list(self._samples)

    # ---- internals ----------------------------------------------------------

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                with self._lock:
                    self._raw_lines.append(line)
        except Exception:
            pass

    def _parse_raw_output(self) -> None:
        with self._lock:
            lines = list(self._raw_lines)
        # buckets[ts_str] -> {"read_bytes": ..., "write_bytes": ...}
        buckets: dict[str, dict[str, float]] = {}
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            ts_str = parts[0].strip()
            value_str = parts[1].strip()
            unit = parts[2].strip().lower() if len(parts) > 2 else ""
            event = parts[3].strip().lower()
            if not ts_str or not value_str:
                continue
            if "cas_count_read" not in event and "cas_count_write" not in event:
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            try:
                ts_bucket = f"{float(ts_str):.1f}"
            except ValueError:
                continue
            unit_scale = {
                "": _BYTES_PER_CAS,        # raw CAS count
                "count": _BYTES_PER_CAS,
                "bytes": 1.0,
                "kib": 1024.0,
                "mib": 1024.0 * 1024.0,
                "gib": 1024.0 * 1024.0 * 1024.0,
            }.get(unit)
            if unit_scale is None:
                continue
            bytes_delta = value * unit_scale

            bucket = buckets.setdefault(ts_bucket, {"read_bytes": 0.0, "write_bytes": 0.0})
            # SUM across per-controller PMUs at the same timestamp; do not overwrite.
            if "cas_count_read" in event:
                bucket["read_bytes"] += bytes_delta
            else:
                bucket["write_bytes"] += bytes_delta

        interval_s = self.interval_ms / 1000.0
        now = datetime.now(UTC)
        last_ts = max((float(k) for k in buckets.keys()), default=0.0)
        for ts_str, b in sorted(buckets.items()):
            try:
                offset_s = float(ts_str)
            except ValueError:
                offset_s = 0.0
            sampled_at = datetime.fromtimestamp(
                now.timestamp() - (last_ts - offset_s), tz=UTC
            )
            self._samples.append(BandwidthSample(
                sampled_at=sampled_at,
                read_gb_s=b["read_bytes"] / interval_s / 1e9,
                write_gb_s=b["write_bytes"] / interval_s / 1e9,
            ))


def bandwidth_summary(samples: list[BandwidthSample]) -> dict[str, float | None]:
    if not samples:
        return {
            "memory_bw_read_gb_s_avg": None,
            "memory_bw_read_gb_s_peak": None,
            "memory_bw_write_gb_s_avg": None,
            "memory_bw_write_gb_s_peak": None,
        }
    reads = [s.read_gb_s for s in samples]
    writes = [s.write_gb_s for s in samples]
    return {
        "memory_bw_read_gb_s_avg": sum(reads) / len(reads),
        "memory_bw_read_gb_s_peak": max(reads),
        "memory_bw_write_gb_s_avg": sum(writes) / len(writes),
        "memory_bw_write_gb_s_peak": max(writes),
    }
