"""Append the H wave to a live campaign queue.

    python scripts/hwave_append.py QUEUE.yaml xe7740|r770|r770-twave [--smoke-after JOB_ID] [--before JOB_ID]

The live queues on the boxes carry manual jobs that planner_queues.py
does not generate, so the H wave is added to them rather than written
over them. Jobs already in the queue are left alone. --smoke-after puts
the H0 smoke jobs straight after that job, so they start as soon as it
finishes; everything else goes at the end, or straight before JOB_ID
with --before, so new jobs run next. r770-twave appends the T wave
(task-model jobs) instead. The old queue is kept beside
it as QUEUE.pre-hwave-append.yaml.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_queues as pq  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    qpath, box = Path(args[0]), args[1]
    smoke_after = args[args.index("--smoke-after") + 1] if "--smoke-after" in args else None
    before = args[args.index("--before") + 1] if "--before" in args else None
    q = yaml.safe_load(qpath.read_text())
    have = {j["id"] for j in q["jobs"]}
    wave = {"xe7740": pq.hwave_xe7740, "r770": pq.hwave_r770, "r770-twave": pq.twave_r770}[box]()
    new = [j for j in wave if j["id"] not in have]
    smoke = [j for j in new if j["id"].startswith("H0-")]
    rest = [j for j in new if not j["id"].startswith("H0-")]
    jobs = list(q["jobs"])
    if smoke_after and smoke_after in have:
        i = next(k for k, j in enumerate(jobs) if j["id"] == smoke_after) + 1
        jobs[i:i] = smoke
    else:
        rest = smoke + rest
    if before:
        if before not in have:
            raise SystemExit(f"--before {before}: not in the queue")
        i = next(k for k, j in enumerate(jobs) if j["id"] == before)
        jobs[i:i] = rest
    else:
        jobs += rest
    shutil.copy(qpath, qpath.with_name(qpath.stem + ".pre-hwave-append.yaml"))
    q["jobs"] = jobs
    tmp = qpath.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(q, sort_keys=False, width=120))
    tmp.replace(qpath)
    print(f"added {len(new)} jobs; queue now {len(jobs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
