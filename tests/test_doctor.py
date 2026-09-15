"""Doctor host-validation checks (roadmap 0.4).

These run on whatever host executes the suite (often a dev Mac), so
they assert structure and status vocabulary rather than specific
hardware outcomes — every check must land in ok/warn/fail/skip and the
report must serialize.
"""

from __future__ import annotations

import json

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
