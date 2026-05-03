"""Telemetry capture: per-second snapshots, PMU/perf, memory bandwidth, engine metrics.

Two coordinator classes:
  * `SnapshotRecorder` — always-on Tier-1 per-second snapshots of pool state.
  * `MeasurementTelemetry` — Tier-2 capture during measurement windows.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


# -----------------------------------------------------------------------------
# Tier 1: per-second simulation snapshots
# -----------------------------------------------------------------------------

class SnapshotRecorder:
    """Per-second simulation snapshots, written directly to the database."""

    def __init__(self, db, cohort_run_id: str, state, pool, get_phase, interval_s: int = 1):
        self.db = db
        self.cohort_run_id = cohort_run_id
        self.state = state
        self.pool = pool
        self.get_phase = get_phase
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._stopped = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self) -> None:
        try:
            while not self._stopped:
                self.db.insert_snapshot({
                    "cohort_run_id": self.cohort_run_id,
                    "snapshot_at_ms": _now_ms(),
                    "phase": self.get_phase(),
                    "pool_size": self.pool.target_size,
                    "in_flight": self.state.in_flight,
                    "requests_completed": self.state.completed,
                    "errors": self.state.errors,
                })
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            pass


# -----------------------------------------------------------------------------
# Tier 2: PMU + memory bandwidth + engine metrics during measurement
# -----------------------------------------------------------------------------


@dataclass
class TelemetrySample:
    sampled_at_ms: int
    kv_cache_used_pct: Optional[float] = None
    queue_depth: Optional[int] = None
    prefix_cache_hits: Optional[int] = None
    prefix_cache_misses: Optional[int] = None
    pmu_cycles: Optional[int] = None
    pmu_instructions: Optional[int] = None
    pmu_stalls_mem_any: Optional[int] = None
    pmu_stalls_l3_miss: Optional[int] = None
    pmu_amx_ops: Optional[int] = None
    memory_bw_read_gb_s: Optional[float] = None
    memory_bw_write_gb_s: Optional[float] = None
    cpu_util_avg: Optional[float] = None


class _PerfStatRunner:
    """Background `perf stat -x , -I 1000` subprocess parsing CSV output.

    Each output row is one event/interval. We bucket rows by interval timestamp
    and emit a per-second dict of {event_name: value}. If perf isn't available
    or fails to start, the runner becomes a no-op.
    """

    def __init__(self, events: list[str]):
        self.events = events
        self._proc: subprocess.Popen | None = None
        self._reader_task: asyncio.Task | None = None
        self._buckets: dict[float, dict[str, int]] = {}
        self._latest_completed_ts: float = 0.0
        self._latest_values: dict[str, int] = {}
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        perf = shutil.which("perf")
        if perf is None:
            log.info("perf not on PATH; PMU collection disabled")
            return
        cmd = [perf, "stat", "-a", "-x", ",", "-I", "1000", "-e", ",".join(self.events)]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to start perf: %s", e)
            return
        self._available = True
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._proc is not None
        loop = asyncio.get_event_loop()
        # Read line-by-line from stderr (perf stat writes events to stderr)
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            while True:
                line = await loop.run_in_executor(None, stream.readline)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                # Format: <ts>,<value>,<unit>,<event>,<run>,<pcnt>,...
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    ts = float(parts[0])
                    value_str = parts[1].strip()
                    if value_str in ("<not counted>", "<not supported>"):
                        continue
                    value = int(value_str.replace(",", ""))
                    event = parts[3].strip()
                except ValueError:
                    continue
                self._buckets.setdefault(ts, {})[event] = value
                self._latest_completed_ts = ts
                self._latest_values = dict(self._buckets[ts])
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("perf reader exited: %s", e)

    def latest(self) -> dict[str, int]:
        return dict(self._latest_values)

    async def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass


class MeasurementTelemetry:
    """Capture per-second telemetry rows during a measurement window."""

    def __init__(self, telemetry_config, engine, perf_events: Optional[list[str]] = None):
        self.cfg = telemetry_config
        self.engine = engine
        self.perf_events = perf_events or telemetry_config.perf_events
        self._perf: _PerfStatRunner | None = None
        self._task: asyncio.Task | None = None
        self._samples: list[TelemetrySample] = []
        self._stopped = False
        self._measurement_id: int | None = None

    def start(self, measurement_id: int) -> None:
        self._measurement_id = measurement_id
        self._samples = []
        self._stopped = False
        if self.cfg.enable_pmu:
            self._perf = _PerfStatRunner(self.perf_events)
            self._perf.start()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> tuple[int, list[dict]]:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._perf is not None:
            await self._perf.stop()
        rows = [self._sample_to_row(s) for s in self._samples]
        return (self._measurement_id or -1), rows

    async def _loop(self) -> None:
        try:
            while not self._stopped:
                sample = TelemetrySample(sampled_at_ms=_now_ms())

                # Engine metrics
                if self.cfg.enable_engine_metrics and self.engine is not None:
                    try:
                        m = await asyncio.to_thread(self.engine.get_metrics)
                    except Exception:
                        m = {}
                    sample.kv_cache_used_pct = m.get("kv_cache_used_pct")
                    qd = m.get("queue_depth")
                    if qd is not None:
                        sample.queue_depth = int(qd)
                    if "prefix_cache_hits" in m:
                        sample.prefix_cache_hits = int(m["prefix_cache_hits"])

                # PMU
                if self._perf is not None and self._perf.available:
                    raw = self._perf.latest()
                    sample.pmu_cycles = raw.get("cycles")
                    sample.pmu_instructions = raw.get("instructions")
                    sample.pmu_stalls_mem_any = raw.get("cycle_activity.stalls_mem_any")
                    sample.pmu_stalls_l3_miss = raw.get("cycle_activity.stalls_l3_miss")

                # CPU util
                sample.cpu_util_avg = _read_cpu_util()

                self._samples.append(sample)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    def _sample_to_row(self, s: TelemetrySample) -> dict:
        return {
            "measurement_id": self._measurement_id,
            "sampled_at_ms": s.sampled_at_ms,
            "kv_cache_used_pct": s.kv_cache_used_pct,
            "queue_depth": s.queue_depth,
            "prefix_cache_hits": s.prefix_cache_hits,
            "prefix_cache_misses": s.prefix_cache_misses,
            "pmu_cycles": s.pmu_cycles,
            "pmu_instructions": s.pmu_instructions,
            "pmu_stalls_mem_any": s.pmu_stalls_mem_any,
            "pmu_stalls_l3_miss": s.pmu_stalls_l3_miss,
            "pmu_amx_ops": s.pmu_amx_ops,
            "memory_bw_read_gb_s": s.memory_bw_read_gb_s,
            "memory_bw_write_gb_s": s.memory_bw_write_gb_s,
            "cpu_util_avg": s.cpu_util_avg,
        }


def _read_cpu_util() -> Optional[float]:
    """Best-effort CPU utilisation (Linux /proc/stat delta)."""
    if not os.path.exists("/proc/stat"):
        return None
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu":
            return None
        nums = [int(x) for x in parts[1:]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        global _LAST_CPU
        try:
            last_idle, last_total = _LAST_CPU
        except NameError:
            _LAST_CPU = (idle, total)
            return None
        d_idle = idle - last_idle
        d_total = total - last_total
        _LAST_CPU = (idle, total)
        if d_total <= 0:
            return None
        return 100.0 * (1.0 - d_idle / d_total)
    except Exception:
        return None


_LAST_CPU: tuple[int, int] | None = None
