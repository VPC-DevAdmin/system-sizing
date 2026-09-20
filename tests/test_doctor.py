"""Doctor host-validation checks (roadmap 0.4).

These run on whatever host executes the suite (often a dev Mac), so
they assert structure and status vocabulary rather than specific
hardware outcomes — every check must land in ok/warn/fail/skip and the
report must serialize.
"""

from __future__ import annotations

import json
from pathlib import Path

from simulator.doctor import FAIL, OK, SKIP, WARN, DoctorReport, run_doctor

VALID = {OK, WARN, FAIL, SKIP}


def test_run_doctor_report_structure() -> None:
    report = run_doctor()
    names = [c.name for c in report.checks]
    # Core checks always present regardless of host.
    for expected in ("cpu", "docker", "gpu", "disk", "hf_reachability"):
        assert expected in names, f"missing check {expected}"
    assert all(c.status in VALID for c in report.checks)
    assert all(c.detail for c in report.checks)


def test_report_serializes_and_failed_flag() -> None:
    report = DoctorReport()
    report.add("a", OK, "fine")
    report.add("b", WARN, "meh")
    assert report.failed is False
    report.add("c", FAIL, "broken")
    assert report.failed is True
    doc = json.loads(json.dumps(report.to_dict()))
    assert doc["failed"] is True
    assert len(doc["checks"]) == 3
    assert doc["checks"][2] == {"name": "c", "status": "fail", "detail": "broken"}


def test_skip_never_fails_report() -> None:
    report = DoctorReport()
    report.add("gpu", SKIP, "no nvidia-smi — CPU-only host")
    assert report.failed is False


def test_parse_lsblk_unmounted() -> None:
    """Large disks with no mountpoint anywhere in their partition tree
    are flagged; mounted ones and small ones are not."""
    from simulator.doctor import parse_lsblk_unmounted

    doc = {"blockdevices": [
        # Boot disk: mounted via a child partition.
        {"name": "sda", "size": 480_000_000_000, "type": "disk",
         "mountpoint": None,
         "children": [{"name": "sda1", "size": 479_000_000_000,
                       "type": "part", "mountpoint": "/"}]},
        # Data NVMe nobody mounted — the lab-box case.
        {"name": "nvme1n1", "size": 3_840_000_000_000, "type": "disk",
         "mountpoint": None},
        # Small unmounted disk: ignored.
        {"name": "sdb", "size": 100_000_000_000, "type": "disk",
         "mountpoint": None},
        # Newer lsblk uses mountpoints (plural) lists.
        {"name": "nvme2n1", "size": 1_920_000_000_000, "type": "disk",
         "mountpoints": [None],
         "children": [{"name": "nvme2n1p1", "size": 1_900_000_000_000,
                       "type": "part", "mountpoints": ["/data"]}]},
    ]}
    out = parse_lsblk_unmounted(doc)
    assert out == [("nvme1n1", 3840.0, False)]


def test_parse_lsblk_blank_disks_sort_first() -> None:
    """A partitioned-but-unmounted disk (leftover ZFS pool shape) must
    never be the default formatting example — blank disks sort first
    even when smaller."""
    from simulator.doctor import parse_lsblk_unmounted

    doc = {"blockdevices": [
        {"name": "nvme1n1", "size": 3_000_000_000_000, "type": "disk",
         "mountpoint": None,
         "children": [
             {"name": "nvme1n1p1", "size": 2_990_000_000_000,
              "type": "part", "mountpoint": None},
             {"name": "nvme1n1p9", "size": 8_000_000, "type": "part",
              "mountpoint": None}]},
        {"name": "nvme3n1", "size": 1_500_000_000_000, "type": "disk",
         "mountpoint": None},
    ]}
    out = parse_lsblk_unmounted(doc)
    assert out == [("nvme3n1", 1500.0, False), ("nvme1n1", 3000.0, True)]


def test_disk_check_reports_real_consumers(tmp_path, monkeypatch) -> None:
    """The disk row measures where the tool actually writes (HF cache,
    docker root), not just the largest common path."""
    from simulator.doctor import DoctorReport, _check_disk

    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    report = DoctorReport()
    _check_disk(report, docker_ok=False)
    row = next(c for c in report.checks if c.name == "disk")
    assert "hf-cache" in row.detail
    assert str(cache) in row.detail


def test_a_broken_nvidia_smi_is_a_failure_not_a_warning(monkeypatch) -> None:
    """A host that has nvidia-smi and cannot run it has a broken GPU
    stack; every GPU engine will fail the same way, so the CLI must
    exit non-zero rather than warn and carry on."""
    import simulator.doctor as doc

    def fake_run(cmd, timeout=15):
        if cmd[0] == "nvidia-smi":
            return 1, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
        return 127, f"{cmd[0]}: not found"
    monkeypatch.setattr(doc, "_run", fake_run)
    report = DoctorReport()
    assert doc._check_gpu(report, docker_ok=False) is False
    row = next(c for c in report.checks if c.name == "gpu")
    assert row.status == FAIL
    assert "couldn't communicate" in row.detail
    assert report.failed

    # Absent entirely is still a CPU-only host, not a failure.
    monkeypatch.setattr(doc, "_run", lambda cmd, timeout=15: (127, "x"))
    report = DoctorReport()
    doc._check_gpu(report, docker_ok=False)
    assert next(c for c in report.checks if c.name == "gpu").status == SKIP


def test_hf_token_detection_honours_the_hub_env(tmp_path, monkeypatch) -> None:
    """The check looked only at HF_TOKEN and ~/.cache/huggingface, so a
    host with HF_HOME on the data disk reported no token while every
    download used one."""
    from simulator.doctor import hf_token_source

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH",
                "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    assert hf_token_source() is None

    hf_home = tmp_path / "data" / "hf"
    hf_home.mkdir(parents=True)
    (hf_home / "token").write_text("hf_x")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    assert hf_token_source() == str(hf_home / "token")

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_y")
    assert hf_token_source() == "$HUGGING_FACE_HUB_TOKEN"
    monkeypatch.setenv("HF_TOKEN", "hf_z")
    assert hf_token_source() == "$HF_TOKEN"
