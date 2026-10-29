#!/usr/bin/env python3
"""Planner campaign queue runner: open-loop persona and mix runs in
parallel lanes on one box.

A queue file lists jobs. Each job asks for a number of devices (GPUs
on the XE7740, sockets on the R770) and runs one ``capsim sweep`` over
its workloads with the SLA gate off. The runner hands out devices in
queue order, so jobs that need one GPU run eight at a time and a job
that needs the whole box waits for every lane to come free. Each lane
gets its own port window, container tag (CAPSIM_LANE), GPU telemetry
filter (CAPSIM_GPU_INDEXES) and run folder, so runs never touch each
other's engines. After a sweep the runner writes the slim export and
``tiers.json`` (simulator.rescore).

    planner_campaign.py run queue.yaml          # run until the queue is done
    planner_campaign.py status queue.yaml       # one line per job

State lives in <state_dir>/status.json; a restart skips finished jobs
and re-runs anything that was running when the runner stopped (the
sweep resumes its run folder, skipping finished workloads).

Queue file:

    box: xe7740                     # xe7740 (GPUs) or r770 (sockets)
    state_dir: /data/capsim-planner/state
    runs_dir: /data/capsim-planner/runs
    defaults:
      engine: vllm_cuda_multi
      max_model_len: 32768
      gpu_memory_utilization: 0.90
      simulation: {...}             # merged into every job
    jobs:
      - id: G1.1-qwen30b-quick_lookup
        model: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
        devices: 1                  # GPUs (xe7740) or sockets (r770)
        tp: 1
        barrier: false              # true: start only when nothing else runs
        initial_rate_per_s: 0.25    # first arrival rate tried
        workloads: [quick_lookup]
        simulation: {history_window_turns: 8}
        engine_extra: {kv_cache_dtype: fp8}
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
UV = os.environ.get("UV", "uv")
GPU_DOMAINS = [[0, 1, 2, 3], [4, 5, 6, 7]]
# R770: socket -> physical cores (even = socket 0, odd = socket 1),
# NUMA node = socket. Hyper-threads stay idle.
R770_SOCKET_CPUS = {
    0: ",".join(str(c) for c in range(0, 172, 2)),
    1: ",".join(str(c) for c in range(1, 172, 2)),
}
BASE_PORT = 9100
LANE_PORTS = 128

PLANNER_SIMULATION = {
    "mode": "open",
    "open_loop_sla_gate": False,
    "open_loop_error_ceiling": 0.05,
    "open_loop_step_factor": 2.0,
    "open_loop_backfill_factor": 1.414,
    "open_loop_backfill_floor_ratio": 8.0,
    "open_loop_resolution_pct": 10.0,
    "open_loop_window_s": 120,
    "open_loop_refine_window_s": 120,
    "open_loop_warmup_s": 30,
    "open_loop_settle_window_s": 30,
    "open_loop_settle_max_s": 90,
    "open_loop_warmup_follows_sessions": False,
    "open_loop_seed_sessions": True,
    "planner_timeouts": True,
    "max_total_duration_minutes": 240,
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Queue:
    def __init__(self, path: Path):
        self.path = path
        self.doc = yaml.safe_load(path.read_text())
        self.box = self.doc.get("box", "xe7740")
        self.state_dir = Path(self.doc["state_dir"])
        self.runs_dir = Path(self.doc["runs_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.state_dir / "status.json"
        self.status = (json.loads(self.status_path.read_text())
                       if self.status_path.exists() else {})
        defaults = self.doc.get("defaults") or {}
        self.jobs = [deep_merge(defaults, j) for j in self.doc.get("jobs") or []]
        ids = [j["id"] for j in self.jobs]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise SystemExit(f"duplicate job ids: {sorted(dup)}")

    def save(self) -> None:
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.status, indent=1))
        tmp.replace(self.status_path)

    def total_devices(self) -> list[int]:
        return list(range(8)) if self.box == "xe7740" else [0, 1]


def pick_devices(free: list[int], n: int, box: str) -> list[int] | None:
    """Contiguous devices inside one PCIe domain when they fit there
    (TP groups must not span the two domains); the whole box for 8."""
    if box != "xe7740":
        if n >= 2:
            return [0, 1] if set(free) >= {0, 1} else None
        return [free[0]] if free else None
    if n >= 8:
        return list(range(8)) if len(free) == 8 else None
    for dom in GPU_DOMAINS:
        avail = [d for d in dom if d in free]
        for start in range(0, len(dom), n):
            want = dom[start:start + n]
            if len(want) == n and all(d in avail for d in want):
                return want
    return None


def gpu_config(job: dict, devices: list[int], lane: str, rundir: Path) -> dict:
    sys.path.insert(0, str(REPO))
    from simulator.engines.custom import config_doc, custom_engine
    tp = int(job.get("tp") or 1)
    replicas = max(1, len(devices) // tp)
    custom = {
        "engine": job.get("engine") or "vllm_cuda_multi",
        "model_id": job["model"], "tp": tp, "replicas": replicas,
        "placement": "pack",
        "max_model_len": job.get("max_model_len"),
        "gpu_memory_utilization": job.get("gpu_memory_utilization", 0.90),
    }
    for k in ("kv_cache_dtype", "max_num_seqs", "trust_remote_code",
              "served_model_name"):
        if job.get(k) is not None:
            custom[k] = job[k]
    hw = {"count": len(devices), "device_groups": [devices],
          "vram_per_gpu_gb": 95.6}
    engine = custom_engine(custom, hw=hw)
    engine["port"] = BASE_PORT + LANE_PORTS * int(devices[0])
    engine.update(job.get("engine_extra") or {})
    doc = config_doc(engine, str(rundir))
    # Host-wide PMU and memory-bandwidth collectors would read every
    # lane at once; GPU telemetry is filtered to the lane's devices.
    doc["telemetry"] = {"enable_pmu": False, "enable_memory_bandwidth": False,
                        "enable_power": False, "enable_engine_metrics": True,
                        "enable_gpu": True}
    return doc


def cpu_config(job: dict, devices: list[int], lane: str, rundir: Path) -> dict:
    """R770: one engine replica per socket in the lane, pinned to that
    socket's physical cores and memory node."""
    engine = copy.deepcopy(job.get("cpu_engine") or {})
    if not engine:
        raise SystemExit(f"job {job['id']}: r770 jobs need cpu_engine")
    if engine.get("type") == "sglang":
        if len(devices) > 1 and int(engine.get("tensor_parallel_size", 1)) > 1:
            engine["cpu_bind"] = "|".join(R770_SOCKET_CPUS[d] for d in devices)
        else:
            engine["cpu_bind"] = R770_SOCKET_CPUS[devices[0]]
        engine["port"] = BASE_PORT + LANE_PORTS * int(devices[0])
        engine.setdefault("docker_extra_env", {})
    else:  # vllm_dual_socket: a replica per socket
        reps = []
        for i, d in enumerate(devices):
            r = copy.deepcopy((engine.get("replicas") or [{}])[0])
            r.update({"name": f"s{d}", "cpuset_cpus": R770_SOCKET_CPUS[d],
                      "cpuset_mems": str(d),
                      "port": BASE_PORT + LANE_PORTS * int(devices[0]) + i})
            # vLLM-CPU's KV pool defaults to 4 GiB; give it room on a
            # 1 TB socket, and one OpenMP thread per physical core.
            r.setdefault("env", {})
            r["env"].setdefault("VLLM_CPU_KVCACHE_SPACE", "100")
            r["env"]["VLLM_CPU_OMP_THREADS_BIND"] = R770_SOCKET_CPUS[d]
            reps.append(r)
        engine["replicas"] = reps
    if job.get("max_model_len"):
        engine["max_model_len"] = job["max_model_len"]
    engine.update(job.get("engine_extra") or {})
    return {
        "engine": engine,
        "telemetry": {"enable_pmu": False, "enable_memory_bandwidth": len(devices) > 1,
                      "enable_power": True, "enable_engine_metrics": True,
                      "enable_amx_utilization": False, "enable_gpu": False},
        "output": {"db_directory": str(rundir)},
    }


def build_config(q: Queue, job: dict, devices: list[int], lane: str,
                 rundir: Path) -> Path:
    if q.box == "xe7740":
        doc = gpu_config(job, devices, lane, rundir)
    else:
        doc = cpu_config(job, devices, lane, rundir)
    sim = deep_merge(PLANNER_SIMULATION, job.get("simulation") or {})
    workers = int(job.get("workers_per_device", 8 if q.box == "xe7740" else 4))
    sim.setdefault("open_loop_max_workers", max(8, workers * len(devices)))
    sim.setdefault("open_loop_min_workers", max(2, workers * len(devices) // 2))
    if job.get("initial_rate_per_s"):
        sim["open_loop_initial_rate_per_s"] = float(job["initial_rate_per_s"])
    doc["simulation"] = sim
    if job.get("reasoning_effort"):
        doc["engine"]["reasoning"] = True
        doc["engine"]["reasoning_effort"] = job["reasoning_effort"]
    rundir.mkdir(parents=True, exist_ok=True)
    cfg = rundir / "config.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
    return cfg


def launch(q: Queue, job: dict, devices: list[int]) -> subprocess.Popen:
    lane = ("g" if q.box == "xe7740" else "s") + "".join(str(d) for d in devices)
    rundir = q.runs_dir / job["id"]
    cfg = build_config(q, job, devices, lane, rundir)
    env = dict(os.environ)
    env["CAPSIM_LANE"] = lane
    if q.doc.get("wall_power_csv"):
        env["CAPSIM_WALL_POWER_CSV"] = q.doc["wall_power_csv"]
    if q.box == "xe7740":
        env["CAPSIM_GPU_INDEXES"] = ",".join(str(d) for d in devices)
    workloads = ",".join(job["workloads"])
    script = (
        f"set -o pipefail; cd {REPO} && "
        f"{UV} run -q python -m simulator.cli sweep --config {cfg} "
        f"--type {workloads} --mode open && "
        f"rd=$(ls -d {rundir}/run_* | sort -V | tail -1) && "
        f"{UV} run -q python -m simulator.cli export --input-dir $rd "
        f"--output $rd/export_slim.json --slim ; "
        f"rd=$(ls -d {rundir}/run_* | sort -V | tail -1) && "
        f"{UV} run -q python -m simulator.rescore $rd"
    )
    log = open(rundir / "job.log", "a")
    log.write(f"\n=== {now()} start lane={lane} devices={devices}\n")
    log.flush()
    return subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def run(q: Queue) -> None:
    free = q.total_devices()
    running: dict[str, tuple[subprocess.Popen, list[int]]] = {}
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    for j in q.jobs:  # jobs left "running" by a dead runner start over
        st = q.status.get(j["id"])
        if st and st.get("state") == "running":
            st["state"] = "pending"
    q.save()

    while True:
        # Reap.
        for jid, (proc, devs) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            st = q.status[jid]
            st.update({"state": "done" if rc == 0 else "failed", "rc": rc,
                       "finished": now()})
            free = sorted(set(free) | set(devs))
            del running[jid]
            q.save()
        if stop["flag"]:
            for proc, _ in running.values():
                os.killpg(proc.pid, signal.SIGINT)
            for proc, _ in running.values():
                try:
                    proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            return
        # Start in queue order; a job that does not fit blocks those
        # behind it (no back-filling), so whole-box jobs are not starved.
        pending = [j for j in q.jobs
                   if q.status.get(j["id"], {}).get("state") in (None, "pending")]
        for job in pending:
            # barrier: wait until everything before it has finished.
            if job.get("barrier") and running:
                break
            need = int(job.get("devices") or 1)
            devs = pick_devices(free, need, q.box)
            if devs is None:
                break
            proc = launch(q, job, devs)
            free = [d for d in free if d not in devs]
            running[job["id"]] = (proc, devs)
            q.status[job["id"]] = {"state": "running", "devices": devs,
                                   "started": now(), "pid": proc.pid}
            q.save()
        if not running and not [j for j in q.jobs
                                if q.status.get(j["id"], {}).get("state") in (None, "pending")]:
            return
        time.sleep(10)


def status(q: Queue) -> None:
    for j in q.jobs:
        st = q.status.get(j["id"], {})
        print(f"{j['id']:<48} {st.get('state', 'pending'):<8} "
              f"{st.get('devices', '')!s:<14} {st.get('started', '')} {st.get('finished', '')}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("run", "status"):
        raise SystemExit(__doc__)
    q = Queue(Path(sys.argv[2]))
    if sys.argv[1] == "run":
        run(q)
    else:
        status(q)


if __name__ == "__main__":
    main()
