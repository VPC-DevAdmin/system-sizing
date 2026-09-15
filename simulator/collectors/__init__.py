"""Telemetry collector plugins (roadmap 1.2).

A collector is anything that can be probed for availability on the
current host and sampled at 1 Hz during a measurement window. The
informal protocol every collector follows:

* ``name`` — stable identifier, used in the export's ``collectors``
  status block.
* ``is_available() -> bool`` — cheap host-capability probe; a collector
  that isn't available is skipped and reported (never an error).
* ``sample()`` — one reading; returns None on transient failure.
* ``status`` — "ok" / "not_available" / collector-specific error.

The pre-existing CPU/host collectors (``perf_collector``,
``bandwidth``, ``power_probe``, ``frequency``) predate this package
and are orchestrated directly by ``telemetry.MeasurementTelemetry``;
they already behave like plugins (self-reporting availability,
degrading to NULLs). New collectors land here.
"""

from .gpu import GpuCollector, GpuSample, aggregate_gpu_samples

__all__ = ["GpuCollector", "GpuSample", "aggregate_gpu_samples"]
