"""One-shot P02 readiness check and owned launch; never stop other workloads.

A heartbeat invokes this module. It does not install a server-side polling job.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from ..benchmarks.process import Lease, hardware_lease_path, module_argv
from ..benchmarks.storage import read_json, write_json
from .campaign import create, execute
from .spec import ExperimentSpec


def idle_reasons(plan):
    with urllib.request.urlopen(plan["api"] + "/api/status", timeout=15) as response:
        status = json.load(response)
    reasons = []
    if (status.get("active_run") or {}).get("running"):
        reasons.append("capsim has active work")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    if result.stdout.strip():
        reasons.append("GPU compute processes are present")
    gpus = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    if any(
        float(parts[1]) > 1024 or float(parts[2]) > 5
        for line in gpus.stdout.splitlines()
        if len(parts := line.split(",")) == 3
    ):
        reasons.append("GPU memory or utilization remains active")
    running = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout.splitlines()
    if set(running) & set(plan["wait_containers"]):
        reasons.append("Existing GPU workload containers have not finished")
    try:
        with Lease(hardware_lease_path()):
            pass
    except RuntimeError:
        reasons.append("Hardware lease is held")
    return reasons


def check(path, launch=False):
    path = Path(path)
    plan = read_json(path)
    root = path.parent
    with Lease(root / ".deferred.lock"):
        statepath = root / "readiness.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = (
            read_json(statepath)
            if statepath.exists()
            else dict(phase="waiting", plan_sha256=digest)
        )
        if state["plan_sha256"] != digest:
            raise ValueError("Deferred plan changed; use a new directory")
        if state["phase"] in ("launching", "running", "complete", "failed", "cancelled"):
            return state
        if (root / "hold").exists():
            state.update(phase="waiting", reason=["Explicit P02 hold marker"], idle_since=None)
        else:
            reasons = idle_reasons(plan)
            if reasons:
                state.update(phase="waiting", reason=reasons, idle_since=None)
            elif state.get("idle_since") is None:
                state.update(
                    phase="waiting",
                    reason=["First idle observation; require another after 120 seconds"],
                    idle_since=time.time(),
                )
            elif time.time() - state["idle_since"] < 120:
                state.update(phase="waiting", reason=["Waiting for second idle observation"])
            elif not launch:
                state.update(phase="ready", reason=[])
            else:
                state.update(phase="launching", reason=[])
                write_json(statepath, state)
                env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
                package = str(Path(__file__).resolve().parents[2])
                with (root / "controller.log").open("ab") as out:
                    proc = subprocess.Popen(
                        module_argv(
                            package, "simulator.infrastructure.session_deferred", str(path), "--run"
                        ),
                        cwd=package,
                        env=env,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                write_json(root / "controller.json", {"pid": proc.pid})
                return read_json(statepath)
        state["checked_at"] = time.time()
        write_json(statepath, state)
        return state


def run(path):
    path = Path(path)
    plan = read_json(path)
    root = path.parent
    dependencies = root / "dependencies"
    if dependencies.is_dir():
        os.environ["PYTHONPATH"] = str(dependencies) + os.pathsep + os.environ.get("PYTHONPATH", "")
    statepath = root / "readiness.json"
    with Lease(root / ".runner.lock"):
        # Check again immediately before creating work. The worker claims the same hardware lease.
        reasons = idle_reasons(plan)
        state = read_json(statepath)
        if reasons or (root / "hold").exists():
            state.update(phase="waiting", reason=reasons or ["P02 hold marker"], idle_since=None)
            write_json(statepath, state)
            return
        try:
            source = read_json(Path(plan["source_manifest"]))["source_result"]
            if source["model"] != plan["model"]:
                raise ValueError("Unexpected source model")
            settings = ExperimentSpec(**plan["spec"])
            campaign = create(Path(plan["runs_root"]), source, settings, state["plan_sha256"])
            state.update(phase="running", campaign_id=campaign.name, started_at=time.time())
            write_json(statepath, state)
            result = asyncio.run(execute(campaign))
            # execute returns no state on some successful paths; read the authoritative artifact.
            result = read_json(campaign / "state.json")
            state.update(
                phase=result["phase"],
                outcome=result.get("outcome"),
                error=result.get("error"),
                finished_at=time.time(),
            )
        except Exception as exc:
            state.update(phase="failed", error=str(exc), finished_at=time.time())
        write_json(statepath, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.plan)
    else:
        print(json.dumps(check(args.plan, args.launch)))
