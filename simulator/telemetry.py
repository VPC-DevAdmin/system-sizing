"""Telemetry orchestration.

Three layers:

* :class:`SnapshotRecorder` — Tier-1 always-on per-second snapshots of
  the simulator's own pool/in-flight state, written straight to the DB.
* :class:`MeasurementTelemetry` — Tier-2 per-measurement-window
  capture: engine metrics, frequency (CPU-filtered), bandwidth (per-IMC),
  power (RAPL), and per-second CPU util / memory.
* :class:`PerfStatCollector` (in ``perf_collector``) and the AMX
  utilisation parser are owned and run by this orchestrator.

Lessons baked in (from the GNR / R470 work):
  * frequency aggregation is filtered to the engine's bound CPU set
  * bandwidth uses per-IMC PMU discovery on GNR (legacy alias on SPR)
  * RSS for the engine walks ``children(recursive=True)``
  * perf is system-wide (``-a``); never per-PID for TP>1 engines

If a collector lacks the host capability it self-reports ``not_supported``
(or similar) without failing the run — the run completes with NULLs in
the corresponding telemetry columns.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .bandwidth import BandwidthCollector, bandwidth_summary
from .frequency import FrequencyCollector
from .perf_collector import PerfStatCollector
from .power_probe import PowerProbe

log = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── Optional psutil for engine RSS ────────────────────────────────────

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def engine_rss_gb(engine_pid: int | None) -> float | None:
    """Return summed RSS (GB) of the engine process and all descendants.

    Walks ``psutil.Process(pid).children(recursive=True)``. vLLM with
    TP>1 runs each tensor-parallel worker as a separate subprocess —
    reading only the parent's RSS gives a wildly low number (~5% of
    actual on a 7B model).
    """
    if not _HAS_PSUTIL or engine_pid is None:
        return None
    try:
        parent = psutil.Process(engine_pid)
        rss = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return rss / (1024 ** 3)


# ── Tier 1: always-on simulation snapshots ────────────────────────────


class SnapshotRecorder:
    def __init__(self, db, cohort_run_id, state, pool, get_phase, interval_s=1):
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
        # Snapshot the terminal state. Without this the dashboard's
        # "latest snapshot" lookup returns whatever phase the loop was
        # in last (typically ``measuring`` since cancellation usually
        # interrupts the per-second sleep), so the dashboard reads as
        # "still measuring" forever after a cohort finishes.
        try:
            self.db.insert_snapshot({
                "cohort_run_id": self.cohort_run_id,
                "snapshot_at_ms": _now_ms(),
                "phase": self.get_phase(),
                "pool_size": self.pool.target_size,
                "in_flight": self.state.in_flight,
                "requests_completed": self.state.completed,
                "errors": self.state.errors,
                "step_samples": self.state.step_samples,
                "step_target_samples": self.state.step_target_samples,
            })
        except Exception:
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
                    "step_samples": self.state.step_samples,
                    "step_target_samples": self.state.step_target_samples,
                })
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            pass


# ── Tier 2: per-measurement telemetry ─────────────────────────────────


@dataclass
class _IntervalSample:
    sampled_at_ms: int
    kv_cache_used_pct: Optional[float] = None
    queue_depth: Optional[int] = None
    prefix_cache_hits: Optional[int] = None
    prefix_cache_misses: Optional[int] = None
    # Host-wide CPU utilization (mean across every logical CPU the
    # kernel sees). Useful for "did anything else compete with the
    # engine on this box" forensics but misleading as a workload
    # utilization number — on a 128-logical-CPU host with the engine
    # pinned to 64 cores, this caps at ~50% even when the engine is
    # pegged because the unused HT siblings / unused socket are
    # averaged in.
    cpu_util_avg: Optional[float] = None
    # Bound-set CPU utilization (mean across only the cores in the
    # engine's cpu_bind cpuset). This is the actual workload
    # utilization number — at 100% the engine has saturated its
    # allocated CPU budget. Falls back to host-wide when no
    # bound_cpus is configured.
    cpu_util_bound_avg: Optional[float] = None
    memory_used_gb: Optional[float] = None
    engine_rss_gb: Optional[float] = None
    freq_mhz_mean: Optional[float] = None
    freq_mhz_stddev: Optional[float] = None
    freq_mhz_min: Optional[float] = None


class MeasurementTelemetry:
    """Collects rich telemetry during a single measurement window.

    Construction is cheap; ``start()`` spins up the heavyweight perf /
    bandwidth / power collectors. ``stop()`` returns:

      * a list of per-second interval rows (DB-ready)
      * a window-aggregate dict (PMU totals, BW summary, power, AMX event
        name, frequency aggregates)
    """

    def __init__(
        self,
        telemetry_config,
        engine,
        *,
        bound_cpus: set[int] | None = None,
        engine_pid: int | None = None,
        artifacts_dir: str | Path = "runs",
    ):
        self.cfg = telemetry_config
        self.engine = engine
        self.bound_cpus = bound_cpus or None
        self.engine_pid = engine_pid
        self.artifacts_dir = Path(artifacts_dir)

        self._freq = FrequencyCollector(cpu_filter=self.bound_cpus)
        self._perf: PerfStatCollector | None = None
        self._bandwidth: BandwidthCollector | None = None
        self._power: PowerProbe | None = None
        self._task: asyncio.Task | None = None
        self._samples: list[_IntervalSample] = []
        self._stopped = False
        self._measurement_id: int | None = None

    def start(self, measurement_id: int) -> None:
        self._measurement_id = measurement_id
        self._samples = []
        self._stopped = False

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.enable_pmu:
            perf_log = self.artifacts_dir / f"perf_m{measurement_id}.csv"
            self._perf = PerfStatCollector(raw_output_path=perf_log)
            self._perf.start()
        if self.cfg.enable_memory_bandwidth:
            self._bandwidth = BandwidthCollector(interval_ms=1000)
            self._bandwidth.start()
        if getattr(self.cfg, "enable_power", True):
            self._power = PowerProbe()
            self._power.start()

        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> tuple[int, list[dict], dict]:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

        # Stop the heavy collectors and gather aggregates.
        agg: dict = {}
        if self._perf is not None:
            try:
                pmu = await asyncio.to_thread(self._perf.stop)
                agg.update({k: v for k, v in pmu.items() if v is not None})
            except Exception as e:
                log.warning("perf.stop() failed: %s", e)

        if self._bandwidth is not None:
            try:
                bw_samples = await asyncio.to_thread(self._bandwidth.stop)
                agg.update({k: v for k, v in bandwidth_summary(bw_samples).items()
                            if v is not None})
                agg["bandwidth_status"] = self._bandwidth.status
            except Exception as e:
                log.warning("bandwidth.stop() failed: %s", e)

        if self._power is not None:
            try:
                p = await asyncio.to_thread(self._power.stop)
                if p.get("power_w_avg") is not None:
                    agg["power_w_avg"] = p["power_w_avg"]
                if p.get("power_w_peak") is not None:
                    agg["power_w_peak"] = p["power_w_peak"]
                agg["power_status"] = p.get("status")
            except Exception as e:
                log.warning("power.stop() failed: %s", e)

        # Frequency aggregates from our own per-second samples.
        freq_means = [s.freq_mhz_mean for s in self._samples if s.freq_mhz_mean is not None]
        freq_stds = [s.freq_mhz_stddev for s in self._samples if s.freq_mhz_stddev is not None]
        freq_mins = [s.freq_mhz_min for s in self._samples if s.freq_mhz_min is not None]
        if freq_means:
            agg["effective_freq_ghz_mean"] = (sum(freq_means) / len(freq_means)) / 1000.0
        if freq_stds:
            agg["effective_freq_ghz_stddev"] = max(freq_stds) / 1000.0
        if freq_mins:
            agg["effective_freq_ghz_min"] = min(freq_mins) / 1000.0

        rows = [self._sample_to_row(s) for s in self._samples]
        return (self._measurement_id or -1), rows, agg

    async def _loop(self) -> None:
        last_cpu: dict[int, tuple[int, int]] | None = None
        try:
            while not self._stopped:
                sample = _IntervalSample(sampled_at_ms=_now_ms())

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

                # CPU util via /proc/stat — both views from a single
                # per-CPU read so the two metrics share an interval.
                host_util, bound_util, last_cpu = _read_cpu_util(
                    last_cpu, bound_cpus=self.bound_cpus,
                )
                sample.cpu_util_avg = host_util
                sample.cpu_util_bound_avg = bound_util

                # Memory used (system + engine RSS)
                sample.memory_used_gb = _read_memory_used_gb()
                sample.engine_rss_gb = engine_rss_gb(self.engine_pid)

                # Frequency
                mean_mhz, std_mhz, min_mhz = self._freq.sample()
                sample.freq_mhz_mean = mean_mhz
                sample.freq_mhz_stddev = std_mhz
                sample.freq_mhz_min = min_mhz

                self._samples.append(sample)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    def _sample_to_row(self, s: _IntervalSample) -> dict:
        return {
            "measurement_id": self._measurement_id,
            "sampled_at_ms": s.sampled_at_ms,
            "kv_cache_used_pct": s.kv_cache_used_pct,
            "queue_depth": s.queue_depth,
            "prefix_cache_hits": s.prefix_cache_hits,
            "prefix_cache_misses": s.prefix_cache_misses,
            "cpu_util_avg": s.cpu_util_avg,
            "cpu_util_bound_avg": s.cpu_util_bound_avg,
            "memory_used_gb": s.memory_used_gb,
            "engine_rss_gb": s.engine_rss_gb,
            "freq_mhz_mean": s.freq_mhz_mean,
            "freq_mhz_stddev": s.freq_mhz_stddev,
            "freq_mhz_min": s.freq_mhz_min,
        }


# ── /proc helpers ─────────────────────────────────────────────────────


def _read_cpu_util(
    last: dict[int, tuple[int, int]] | None,
    *,
    bound_cpus: set[int] | None = None,
) -> tuple[Optional[float], Optional[float], dict[int, tuple[int, int]] | None]:
    """Read /proc/stat per-CPU and return BOTH host-wide and bound-set
    utilization for the interval since ``last``.

    Returns ``(host_util, bound_util, new_last)``:
      * ``host_util`` — mean utilization across every logical CPU
        the kernel reports. Useful for forensic "did anything else
        compete with the engine" diagnosis. On a multi-socket or
        HT-enabled host with a single-socket cpuset, this is diluted
        by idle cores outside the binding and dramatically
        understates the workload's actual CPU pressure.
      * ``bound_util`` — mean utilization across only the CPUs in
        ``bound_cpus``. This is the workload-relevant number — at
        100% the engine has saturated its allocated CPU budget.
        Falls back to ``host_util`` when ``bound_cpus`` is None
        (e.g. an engine without an explicit cpu_bind).

    Both metrics are computed from a single /proc/stat read so the
    two views share the same interval and any sampling skew is
    identical. ``new_last`` is a dict of ``{cpu_id: (idle_ticks,
    total_ticks)}`` for the next delta computation.

    On the first call ``last`` is None — returns ``(None, None,
    initial_snapshot)`` since you need two samples to compute a
    delta. Returns ``(None, None, last)`` on any read failure;
    callers should tolerate Nones and skip the sample.
    """
    if not os.path.exists("/proc/stat"):
        return None, None, last
    try:
        per_cpu_now: dict[int, tuple[int, int]] = {}
        with open("/proc/stat") as f:
            for line in f:
                # Per-CPU rows look like ``cpu0 ...``, ``cpu127 ...``;
                # the aggregate row is ``cpu ...`` (note trailing space)
                # which we skip — we synthesize it from the per-CPU
                # numbers so host and bound aggregates share a base.
                if not line.startswith("cpu"):
                    break  # /proc/stat puts all cpu lines first; bail
                if line.startswith("cpu "):
                    continue
                parts = line.split()
                if not parts or len(parts) < 5:
                    continue
                try:
                    cpu_id = int(parts[0][3:])
                except ValueError:
                    continue
                nums = [int(x) for x in parts[1:]]
                # idle + iowait (col 3 + 4 in /proc/stat) — same
                # definition as the prior single-line reader.
                idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
                total = sum(nums)
                per_cpu_now[cpu_id] = (idle, total)
        if not per_cpu_now:
            return None, None, last
        if last is None:
            return None, None, per_cpu_now

        def _mean_util(cpu_ids) -> Optional[float]:
            utils: list[float] = []
            for cpu_id in cpu_ids:
                if cpu_id not in last or cpu_id not in per_cpu_now:
                    continue
                d_idle = per_cpu_now[cpu_id][0] - last[cpu_id][0]
                d_total = per_cpu_now[cpu_id][1] - last[cpu_id][1]
                if d_total <= 0:
                    continue
                utils.append(100.0 * (1.0 - d_idle / d_total))
            if not utils:
                return None
            return sum(utils) / len(utils)

        host_util = _mean_util(per_cpu_now.keys())
        if bound_cpus:
            bound_util = _mean_util(bound_cpus)
        else:
            # No explicit binding → bound view equals host view, so
            # consumers can read either field safely without
            # special-casing the "no cpu_bind set" case.
            bound_util = host_util
        return host_util, bound_util, per_cpu_now
    except Exception:
        return None, None, last


def _read_memory_used_gb() -> Optional[float]:
    """Total - MemAvailable from /proc/meminfo, in GB."""
    if not os.path.exists("/proc/meminfo"):
        return None
    try:
        kv: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                rest = rest.strip()
                if rest.endswith("kB"):
                    rest = rest[:-2].strip()
                try:
                    kv[key.strip()] = int(rest)
                except ValueError:
                    pass
        total = kv.get("MemTotal")
        avail = kv.get("MemAvailable")
        if total is None:
            return None
        if avail is None:
            avail = kv.get("MemFree", 0)
        used_kb = total - avail
        return used_kb / (1024 * 1024)
    except Exception:
        return None
