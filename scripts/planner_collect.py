#!/usr/bin/env python3
"""Collect planner campaign results into one CSV.

Reads <runs_dir>/<job_id>/run_*/tiers.json (simulator.rescore) for every
job in a queue file and writes one row per job and workload: the knee
and each service tier's capacity (sessions/min arriving, concurrent
sessions, tokens/s, power), plus the job's model, devices and settings.
Jobs that failed or have no tiers.json are listed with their state, so
outliers and gaps are easy to find for re-runs.

    planner_collect.py queue.yaml out.csv [--rescore]

--rescore re-runs simulator.rescore on each run first (e.g. after the
wall-power file has grown, or with new tier definitions).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

TIERS = ("responsive", "standard", "relaxed", "batch")
FIELDS = ("rate_per_min", "sessions", "turns_per_s", "prompt_tok_per_s",
          "output_tok_per_s", "wall_power_w", "gpu_power_w", "power_w")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    qpath, out = Path(args[0]), Path(args[1])
    doc = yaml.safe_load(qpath.read_text())
    runs_dir = Path(doc["runs_dir"])
    status_path = Path(doc["state_dir"]) / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    defaults = doc.get("defaults") or {}
    rows = []
    for job in doc["jobs"]:
        jid = job["id"]
        st = status.get(jid, {})
        jd = runs_dir / jid
        rds = sorted(jd.glob("run_*"), key=lambda p: int(p.name.split("_")[1])) if jd.exists() else []
        if rds and "--rescore" in sys.argv:
            env = dict(os.environ)
            if doc.get("wall_power_csv"):
                env["CAPSIM_WALL_POWER_CSV"] = doc["wall_power_csv"]
            subprocess.run([sys.executable, "-m", "simulator.rescore", str(rds[-1])],
                           env=env, capture_output=True)
        tiers_path = rds[-1] / "tiers.json" if rds else None
        base = {
            "job": jid, "state": st.get("state", "pending"),
            "model": job.get("model"), "devices": job.get("devices"),
            "tp": job.get("tp", defaults.get("tp")),
            "engine": job.get("engine") or (job.get("cpu_engine") or {}).get("type")
            or defaults.get("engine"),
            "history": (job.get("simulation") or {}).get("history_window_turns"),
            "max_model_len": job.get("max_model_len", defaults.get("max_model_len")),
            "kv_cache_dtype": job.get("kv_cache_dtype"),
            "reasoning_effort": job.get("reasoning_effort"),
        }
        if not tiers_path or not tiers_path.exists():
            rows.append({**base, "workload": ",".join(job.get("workloads", []))})
            continue
        res = json.loads(tiers_path.read_text())
        for wl, e in res["workloads"].items():
            row = {**base, "workload": wl, "windows": e.get("windows"),
                   "knee_per_min": e.get("knee_per_min"),
                   "ceiling_per_min": e.get("ceiling_per_min"),
                   "knee_nonmonotone": e.get("knee_nonmonotone")}
            kp = e.get("knee_point") or {}
            for f in FIELDS:
                row[f"knee_{f}"] = kp.get(f)
            for t in TIERS:
                v = (e.get("tiers") or {}).get(t) or {}
                row[f"{t}_basis"] = v.get("basis")
                for f in FIELDS:
                    row[f"{t}_{f}"] = v.get(f)
            rows.append(row)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    done = sum(1 for r in rows if r.get("knee_per_min") is not None)
    print(f"{len(rows)} rows, {done} with a knee -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
