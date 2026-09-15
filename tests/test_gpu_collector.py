"""GPU collector plugin (roadmap 1.2): CSV parsing, multi-device
aggregation, window aggregates, and the schema-v3 migration."""

from __future__ import annotations

from simulator.collectors.gpu import (
    GpuSample,
    aggregate_gpu_samples,
    parse_smi_csv,
    _combine_devices,
)


def test_parse_smi_csv_two_devices() -> None:
    text = (
        "87, 71680, 81920, 285.31, 1410, 0x0000000000000000\n"
        "93, 70656, 81920, 301.77, 1395, 0x0000000000000004\n"
    )
    devices = parse_smi_csv(text)
    assert len(devices) == 2
    assert devices[0]["sm_util_pct"] == 87.0
    assert devices[0]["vram_used_gb"] == 71680 / 1024
    assert devices[0]["vram_total_gb"] == 80.0
    assert devices[0]["throttled"] is False
    # 0x4 = SwPowerCap → throttled
    assert devices[1]["throttled"] is True


def test_parse_smi_csv_na_fields() -> None:
    # Some SKUs report power as [N/A]; parse must degrade per-field.
    devices = parse_smi_csv("42, 1024, 8192, [N/A], 900, 0x0000000000000001\n")
    assert devices[0]["power_w"] is None
    assert devices[0]["sm_util_pct"] == 42.0
    # 0x1 = GpuIdle — NOT a throttle reason.
    assert devices[0]["throttled"] is False


def test_combine_devices_aggregation() -> None:
    sample = _combine_devices(parse_smi_csv(
        "80, 10240, 20480, 100, 1400, 0x0\n"
        "60, 5120, 20480, 200, 1200, 0x0\n"
    ))
    assert sample.sm_util_pct == 70.0            # mean across devices
    assert sample.vram_used_gb == 15.0           # summed
    assert sample.vram_total_gb == 40.0          # summed
    assert sample.power_w == 300.0               # summed
    assert sample.sm_clock_mhz == 1300.0         # mean
    assert sample.throttled is False


def test_combine_devices_empty() -> None:
    assert _combine_devices([]) is None


def test_aggregate_gpu_samples() -> None:
    samples = [
        GpuSample(sm_util_pct=80, vram_used_gb=60, vram_total_gb=80,
                  power_w=300, sm_clock_mhz=1400, throttled=False),
        GpuSample(sm_util_pct=90, vram_used_gb=70, vram_total_gb=80,
                  power_w=350, sm_clock_mhz=1300, throttled=True),
        GpuSample(sm_util_pct=100, vram_used_gb=75, vram_total_gb=80,
                  power_w=360, sm_clock_mhz=1200, throttled=True),
    ]
    agg = aggregate_gpu_samples(samples)
    assert agg["gpu_sm_util_pct_avg"] == 90.0
    assert agg["gpu_sm_util_pct_peak"] == 100.0
    assert agg["gpu_vram_used_gb_peak"] == 75.0
    assert agg["gpu_vram_total_gb"] == 80.0
    assert agg["gpu_power_w_peak"] == 360.0
    assert agg["gpu_sm_clock_mhz_min"] == 1200.0
    assert abs(agg["gpu_throttle_fraction"] - 2 / 3) < 1e-9


def test_aggregate_gpu_samples_empty() -> None:
    assert aggregate_gpu_samples([]) == {}


def test_schema_v3_migration_adds_gpu_columns(tmp_path) -> None:
    """A v2-era DB opened by current code gains the GPU columns and
    lands at SCHEMA_VERSION."""
    import sqlite3

    from simulator.database import SCHEMA_VERSION, Database

    db_path = tmp_path / "run.db"
    # Build a fresh DB with current code, then wind the stamp back to
    # 2 and strip a GPU column to simulate a v2 DB.
    db = Database(db_path)
    db.close()
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE measurement_telemetry DROP COLUMN gpu_sm_util_pct")
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    db = Database(db_path)
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(measurement_telemetry)")}
    assert "gpu_sm_util_pct" in cols
    agg_cols = {r["name"] for r in db.fetchall("PRAGMA table_info(cohort_measurements)")}
    assert "gpu_throttle_fraction" in agg_cols
    version = db.fetchone("PRAGMA user_version")[0]
    assert version == SCHEMA_VERSION
    db.close()


def test_gpu_collector_unavailable_on_gpuless_host(monkeypatch) -> None:
    """On a host without an NVIDIA stack the collector reports
    not_available and samples None — never raises."""
    import simulator.collectors.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: None)
    monkeypatch.setitem(__import__("sys").modules, "pynvml", None)
    c = gpu_mod.GpuCollector()
    assert c.is_available() is False
    assert c.status == "not_available"
    assert c.sample() is None
