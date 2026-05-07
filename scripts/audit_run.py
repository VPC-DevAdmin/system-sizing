#!/usr/bin/env python3
"""Audit a finished run.db for curve-quality anomalies and emit a
targeted-rerun plan.

Detects four kinds of anomaly that warrant re-measuring specific
(cohort, pool_size) points rather than redoing entire cohorts:

  1. ``no_marginal_band`` — failure-axis capacity_pool_size equals
     soft_capacity_pool_size, meaning bisection skipped over the
     5–30% violation band entirely. Schedule the midpoint between
     capacity and fail.

  2. ``no_fail_observed`` — fail_pool_size is None even though the
     curve reached or exceeded marginal status. The doubling phase
     stopped early (pre-#2 algorithm) or the curve plateaus in
     marginal. Schedule one or two pools beyond the last measured
     point to push past the 30% fail threshold.

  3. ``single_point_rescue`` — capacity_pool_size is set by a single
     measurement that "rescued" a curve from a marginal-status run.
     Pattern: pool=N is pass while both immediate neighbors (lower
     and upper) are marginal/fail. Schedule a re-measurement of N
     to confirm reproducibility.

  4. ``boundary_status`` — a measurement landed exactly on a status
     threshold (rate ≈ 0.05 or ≈ 0.30) where Wilson-CI-aware
     classification (post-#5 algorithm) would have stayed marginal.
     Schedule a re-measurement to tighten the CI.

Usage:
    python scripts/audit_run.py runs/run_09
    python scripts/audit_run.py runs/run_09 --output /tmp/plan.json
    python scripts/audit_run.py runs/run_09 --quiet           # JSON only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Make ``simulator.*`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.measurement import _classify_status  # noqa: E402


# Status thresholds — must match measurement._classify_status / export.
PASS_UPPER = 0.05
FAIL_LOWER = 0.30


@dataclass
class Anomaly:
    kind: str
    cohort_id: str
    cohort_run_id: str
    pool_sizes_to_remeasure: list[int]
    rationale: str


@dataclass
class AuditResult:
    source_dir: str
    audited_at: str
    anomalies: list[Anomaly] = field(default_factory=list)

    def as_json(self) -> dict:
        return {
            "source_dir": self.source_dir,
            "audited_at": self.audited_at,
            "anomalies": [a.__dict__ for a in self.anomalies],
            "rerun_points": _dedupe_rerun_points(self.anomalies),
        }


def _dedupe_rerun_points(anomalies: list[Anomaly]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for a in anomalies:
        for pool in a.pool_sizes_to_remeasure:
            key = (a.cohort_id, pool)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "cohort_id": a.cohort_id,
                "cohort_run_id": a.cohort_run_id,
                "pool_size": pool,
                "reason": a.kind,
            })
    return out


def _read_cohort_runs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT cohort_run_id, cohort_id, final_status FROM cohort_run "
        "WHERE final_status = 'ok' ORDER BY started_at"
    ).fetchall()
    return [dict(r) for r in rows]


def _read_measurements(conn: sqlite3.Connection, crid: str) -> list[dict]:
    rows = conn.execute(
        """SELECT step_index, target_pool_size, sample_size,
                  combined_violation_rate, combined_target_miss_rate,
                  ttft_violation_rate, tpot_violation_rate,
                  ttft_target_miss_rate, tpot_target_miss_rate,
                  capacity_status, target_status
             FROM cohort_measurements
            WHERE cohort_run_id = ?
            ORDER BY target_pool_size""",
        (crid,),
    ).fetchall()
    return [dict(r) for r in rows]


def _check_no_marginal_band(
    crid: str, cohort_id: str, ms: list[dict],
) -> Anomaly | None:
    """Detect: failure-axis bracket [last_pass, first_violation>=5%]
    contains no measurement with capacity_status='marginal'."""
    pass_pools = [
        m["target_pool_size"] for m in ms
        if m["capacity_status"] == "pass"
    ]
    fail_pools = [
        m["target_pool_size"] for m in ms
        if m["capacity_status"] == "fail"
    ]
    marginal_pools = [
        m["target_pool_size"] for m in ms
        if m["capacity_status"] == "marginal"
    ]
    if not pass_pools or not fail_pools:
        return None
    last_pass = max(pass_pools)
    first_fail = min(p for p in fail_pools if p > last_pass) \
        if any(p > last_pass for p in fail_pools) else None
    if first_fail is None:
        return None
    # Marginal in the bracket?
    if any(last_pass < p < first_fail for p in marginal_pools):
        return None
    gap = first_fail - last_pass
    if gap < 2:
        return None
    midpoint = (last_pass + first_fail) // 2
    return Anomaly(
        kind="no_marginal_band",
        cohort_id=cohort_id,
        cohort_run_id=crid,
        pool_sizes_to_remeasure=[midpoint],
        rationale=(
            f"failure-axis cliff is sharp: pool={last_pass} pass → "
            f"pool={first_fail} fail with no marginal measurement in between. "
            f"Re-measure pool={midpoint} to characterise the marginal band."
        ),
    )


def _check_no_fail_observed(
    crid: str, cohort_id: str, ms: list[dict],
) -> Anomaly | None:
    """Detect: no measurement reached capacity_status='fail'."""
    if any(m["capacity_status"] == "fail" for m in ms):
        return None
    if not ms:
        return None
    # Suggest doubling the last-measured pool until we cross 30%.
    # Cap at 2× the current max so we don't blast past hardware.
    max_pool = max(m["target_pool_size"] for m in ms)
    suggestions = [max_pool * 2, max_pool * 4]
    return Anomaly(
        kind="no_fail_observed",
        cohort_id=cohort_id,
        cohort_run_id=crid,
        pool_sizes_to_remeasure=suggestions,
        rationale=(
            f"sweep ended at pool={max_pool} without observing a "
            f"sustained-fail measurement (capacity_status='fail', ≥30% "
            f"violation). Probe {suggestions} to locate fail_pool_size."
        ),
    )


def _check_single_point_rescue(
    crid: str, cohort_id: str, ms: list[dict],
) -> Anomaly | None:
    """Detect: capacity is set by a single 'pass' measurement whose
    immediate neighbors are both worse-than-pass. Models the
    writer pool=120 case from the May 2026 Intel run."""
    pass_pools = [
        m["target_pool_size"] for m in ms
        if m["capacity_status"] == "pass"
    ]
    if not pass_pools:
        return None
    last_pass = max(pass_pools)
    # Find immediate neighbors in the curve.
    sorted_ms = sorted(ms, key=lambda m: m["target_pool_size"])
    idx = next(
        (i for i, m in enumerate(sorted_ms)
         if m["target_pool_size"] == last_pass),
        None,
    )
    if idx is None:
        return None
    left = sorted_ms[idx - 1] if idx > 0 else None
    right = sorted_ms[idx + 1] if idx + 1 < len(sorted_ms) else None
    if left is None or right is None:
        return None
    # Suspicious if BOTH neighbors are non-pass — the lone pass is
    # an island in a marginal/fail sea.
    if left["capacity_status"] == "pass" or right["capacity_status"] == "pass":
        return None
    # Equally suspicious if only the target_status is the one that
    # "rescued" — pattern from writer:
    #   pool=112 target_status=marginal, target_miss=0.150
    #   pool=120 target_status=pass, target_miss=0.040
    #   pool=128 target_status=fail, target_miss=1.000
    # Surface this as a separate-but-similar concern.
    return Anomaly(
        kind="single_point_rescue",
        cohort_id=cohort_id,
        cohort_run_id=crid,
        pool_sizes_to_remeasure=[last_pass],
        rationale=(
            f"capacity_pool_size={last_pass} is set by a single 'pass' "
            f"measurement with non-pass neighbors "
            f"(pool={left['target_pool_size']} {left['capacity_status']!r}, "
            f"pool={right['target_pool_size']} {right['capacity_status']!r}). "
            f"Re-measure pool={last_pass} to confirm reproducibility before "
            f"relying on the rescue."
        ),
    )


def _check_boundary_status(
    crid: str, cohort_id: str, ms: list[dict],
) -> Anomaly | None:
    """Detect: a status='fail' or 'pass' measurement that Wilson-CI
    classification would have left marginal. Targets old runs that
    pre-date the #5 fix.

    Re-runs the modern classification on (misses, n) and reports any
    measurement whose stored status disagrees."""
    suspect_pools = []
    for m in ms:
        n = m["sample_size"] or 0
        if n < 10:
            continue
        for status_field, rate_field in (
            ("capacity_status", "combined_violation_rate"),
            ("target_status", "combined_target_miss_rate"),
        ):
            stored = m[status_field]
            if stored not in ("pass", "fail"):
                continue
            rate = m[rate_field] or 0.0
            misses = round(rate * n)
            modern = _classify_status(misses, n)
            if modern == "marginal" and stored in ("pass", "fail"):
                suspect_pools.append(m["target_pool_size"])
                break  # once per pool
    if not suspect_pools:
        return None
    return Anomaly(
        kind="boundary_status",
        cohort_id=cohort_id,
        cohort_run_id=crid,
        pool_sizes_to_remeasure=sorted(set(suspect_pools)),
        rationale=(
            "one or more measurements were classified pass/fail under "
            "the legacy point-estimate threshold but Wilson-CI bounds "
            "say the data is consistent with marginal — likely a "
            "single-sample boundary flicker. Re-measure to tighten the CI."
        ),
    )


def audit(run_dir: Path) -> AuditResult:
    db_path = run_dir / "run.db"
    if not db_path.exists():
        sys.exit(f"No run.db in {run_dir}")
    result = AuditResult(
        source_dir=str(run_dir),
        audited_at=datetime.now(timezone.utc).isoformat(),
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for run in _read_cohort_runs(conn):
            ms = _read_measurements(conn, run["cohort_run_id"])
            if not ms:
                continue
            for checker in (
                _check_no_marginal_band,
                _check_no_fail_observed,
                _check_single_point_rescue,
                _check_boundary_status,
            ):
                anomaly = checker(run["cohort_run_id"], run["cohort_id"], ms)
                if anomaly is not None:
                    result.anomalies.append(anomaly)
    finally:
        conn.close()
    return result


def _print_human_summary(result: AuditResult) -> None:
    if not result.anomalies:
        print(f"audit: no anomalies found in {result.source_dir}")
        return
    print(f"audit: {len(result.anomalies)} anomaly/anomalies in {result.source_dir}")
    for a in result.anomalies:
        pools = ", ".join(str(p) for p in a.pool_sizes_to_remeasure)
        print(f"  [{a.kind}] {a.cohort_id} (rerun pools: {pools})")
        print(f"    {a.rationale}")
    rerun = _dedupe_rerun_points(result.anomalies)
    print()
    print(f"rerun plan: {len(rerun)} (cohort, pool) points")
    for p in rerun:
        print(f"  - {p['cohort_id']:25} pool={p['pool_size']:4}  ({p['reason']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Run directory (e.g. runs/run_09)")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the JSON plan. Defaults to "
                        "<source>/audit_report.json.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the human-readable summary "
                        "(JSON file is still written).")
    args = p.parse_args()

    if not args.source.exists():
        sys.exit(f"Source dir not found: {args.source}")

    result = audit(args.source)

    output_path = args.output or (args.source / "audit_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.as_json(), indent=2))

    if not args.quiet:
        _print_human_summary(result)
        print()
        print(f"plan written to: {output_path}")


if __name__ == "__main__":
    main()
