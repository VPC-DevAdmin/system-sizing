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
# R770 with SubNumaCluster on (SNC2, from lscpu on 2026-11-04): four
# domains of 43 physical cores, NUMA node = domain. Socket 0 holds
# domains 0 and 1, socket 1 domains 2 and 3. A queue with r770_snc: true
# treats each domain as a device and pins its engine's memory there too.
R770_SNC_CPUS = {
    0: ",".join(str(c) for c in range(0, 86, 2)),
    1: ",".join(str(c) for c in range(86, 172, 2)),
    2: ",".join(str(c) for c in range(1, 87, 2)),
    3: ",".join(str(c) for c in range(87, 172, 2)),
}
R770_HT_CPUS = "172-343"
R770_HT_OFFSET = 172
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
    # The XE7740's 8-GPU quick_lookup sweep was still stable at the
    # default 256/s rail.
    "open_loop_max_rate_per_s": 1024.0,
    "open_loop_window_s": 120,
    "open_loop_refine_window_s": 120,
    "open_loop_warmup_s": 30,
    "open_loop_settle_window_s": 30,
    "open_loop_settle_max_s": 90,
    "open_loop_warmup_follows_sessions": False,
    "open_loop_seed_sessions": True,
    "planner_timeouts": True,
    "max_total_duration_minutes": 240,
    # Shared system prompt per persona (tokens): an application's
    # instructions, policies and tool definitions, identical for every
    # user of it, so the engine serves it from its prefix cache as in
    # production. Everything the user sends stays unique. Sized to the
    # kind of application: short bots ~0.5K, chat ~0.8K, RAG and
    # document work 1-1.5K, IDE and customer assistants ~2K, tool-using
    # agents 4-6K. A job sets simulation.system_prompt_tokens: null to
    # run without.
    "system_prompt_tokens": {
        "quick_lookup": 500, "translation": 400, "meeting_summary": 600,
        "conversational": 800, "writer": 800, "long_form_generator": 800,
        "voice_assistant": 800, "document_qa": 1000, "reasoning_analysis": 1000,
        "inbox_triage": 1200, "knowledge_search": 1500, "document_extraction": 1500,
        "code_assist": 2000, "customer_assistant": 2000,
        "agent_task_runner": 4000, "coding_agent": 6000,
        # H wave (config/personas/hwave.yaml): real agents carry their
        # tool definitions and rules in a ~10-24K prefix.
        "h_assistant": 2000, "h_document_qa": 1500, "h_coding_agent": 24000,
        "h_research_agent": 20000, "h_service_agent": 10000,
        "h_document_pipeline": 2000, "h_deep_analysis": 2000,
        # T wave: task-model jobs; their instructions and label sets.
        "t_ticket_triage": 1500, "t_document_intake": 1200,
        "t_privacy_screening": 800, "t_agent_support": 1500,
    },
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
        if self.box == "xe7740":
            return list(range(8))
        return [0, 1, 2, 3] if self.doc.get("r770_snc") else [0, 1]


def check_numa(q: Queue, nodes: int | None = None) -> None:
    """The R770 queue's device map must match the box: two NUMA nodes
    (one per socket) normally, four with SNC on and r770_snc set. A
    mismatch pins engines to the wrong cores, so the runner stops."""
    if q.box != "r770":
        return
    if nodes is None:
        nodes = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
    want = 4 if q.doc.get("r770_snc") else 2
    if nodes != want:
        raise SystemExit(f"R770 has {nodes} NUMA nodes but the queue expects {want} "
                         f"(r770_snc={bool(q.doc.get('r770_snc'))})")


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
    for k in ("kv_cache_dtype", "max_num_seqs", "max_num_batched_tokens",
              "expert_parallel", "trust_remote_code", "served_model_name"):
        if job.get(k) is not None:
            custom[k] = job[k]
    hw = {"count": len(devices), "device_groups": [devices],
          "vram_per_gpu_gb": 95.6}
    engine = custom_engine(custom, hw=hw)
    engine["port"] = BASE_PORT + LANE_PORTS * int(devices[0])
    # Flag lists add to the ones the job's settings produced (KV dtype,
    # remote code) rather than replacing them.
    for k, v in (job.get("engine_extra") or {}).items():
        if isinstance(v, list) and isinstance(engine.get(k), list):
            engine[k] = engine[k] + list(v)
        else:
            engine[k] = v
    doc = config_doc(engine, str(rundir))
    # Host-wide PMU and memory-bandwidth collectors would read every
    # lane at once; GPU telemetry is filtered to the lane's devices.
    doc["telemetry"] = {"enable_pmu": False, "enable_memory_bandwidth": False,
                        "enable_power": False, "enable_engine_metrics": True,
                        "enable_gpu": True}
    return doc


def cpu_config(job: dict, devices: list[int], lane: str, rundir: Path,
               snc: bool = False) -> dict:
    """R770: one engine replica per socket in the lane, pinned to that
    socket's physical cores and memory node. With ``snc`` each device is
    an SNC domain instead: 43 cores and its own memory node."""
    engine = copy.deepcopy(job.get("cpu_engine") or {})
    if not engine:
        raise SystemExit(f"job {job['id']}: r770 jobs need cpu_engine")
    cpu_map = R770_SNC_CPUS if snc else R770_SOCKET_CPUS
    cores = int(job.get("socket_cores") or (43 if snc else 86))

    def socket_cpus(d: int) -> str:
        cpus = cpu_map[d].split(",")[:cores]
        if int(job.get("socket_threads") or 1) == 2:
            # Hyper-threads too: each core's sibling is CPU n + 172.
            cpus += [str(int(c) + R770_HT_OFFSET) for c in cpus]
        return ",".join(cpus)
    if engine.get("type") == "sglang":
        if len(devices) > 1 and int(engine.get("tensor_parallel_size", 1)) > 1:
            engine["cpu_bind"] = "|".join(socket_cpus(d) for d in devices)
        else:
            engine["cpu_bind"] = socket_cpus(devices[0])
        engine["port"] = BASE_PORT + LANE_PORTS * int(devices[0])
        engine.setdefault("docker_extra_env", {})
        if snc:
            engine["docker_extra_args"] = list(engine.get("docker_extra_args") or []) + [
                "--cpuset-mems", ",".join(str(d) for d in devices)]
    else:  # vllm_dual_socket: a replica per socket
        if snc:
            raise SystemExit(f"job {job['id']}: only SGLang jobs run under r770_snc")
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
        doc = cpu_config(job, devices, lane, rundir, snc=bool(q.doc.get("r770_snc")))
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
    allowance = job.get("reasoning_overhead_tokens")
    if job.get("reasoning_from"):
        # Level-matching rerun (T2): the allowance is the variant's mean
        # reasoning on this workload, measured by its reasoning_probe job,
        # and the answer is timed from that point (simulator.rescore).
        allowance = measured_allowance(q, job)
        doc["engine"]["answer_start_after_tokens"] = int(allowance)
        # Every turn is a fixed length, so splitting reasoning from the
        # answer adds nothing; with a parser, a turn that reasons past the
        # mean streams no answer text and is filed as no_content_tokens,
        # which the search reads as errors and backs off from. Without it
        # every token streams as content and the answer start comes from
        # the allowance above.
        flags = list(doc["engine"].get("vllm_extra_flags") or [])
        if "--reasoning-parser" in flags:
            i = flags.index("--reasoning-parser")
            del flags[i:i + 2]
            doc["engine"]["vllm_extra_flags"] = flags
    if allowance:
        # Applies with or without an effort setting: models with no effort
        # knob still reason.
        doc["engine"]["reasoning_overhead_tokens"] = int(allowance)
    rundir.mkdir(parents=True, exist_ok=True)
    cfg = rundir / "config.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False))
    return cfg


def measured_allowance(q: Queue, job: dict) -> int:
    """Mean reasoning tokens per turn for the job's workload, from the
    summary its ``reasoning_from`` probe job wrote."""
    src = job["reasoning_from"]
    f = q.runs_dir / src / SCORE_JOBS["reasoning_probe"][1]
    summary = json.loads(f.read_text())
    if summary.get("status") != "ok":
        raise ValueError(f"job {job['id']}: probe {src} did not finish ok")
    (wl,) = job["workloads"]
    row = (summary.get("workloads") or {}).get(wl)
    if not row or row.get("reasoning_mean") is None:
        raise ValueError(f"job {job['id']}: probe {src} has no reasoning for {wl}")
    return max(1, round(row["reasoning_mean"]))


def launch(q: Queue, job: dict, devices: list[int]) -> subprocess.Popen:
    lane = ("g" if q.box == "xe7740" else "s") + "".join(str(d) for d in devices)
    rundir = q.runs_dir / job["id"]
    cfg = build_config(q, job, devices, lane, rundir)
    env = dict(os.environ)
    env["CAPSIM_LANE"] = lane
    # The simulator's hardware lease is one lock per host by default;
    # lanes own disjoint devices, so each takes its own.
    (q.state_dir / "locks").mkdir(exist_ok=True)
    env["CAPSIM_HARDWARE_LOCK"] = str(q.state_dir / "locks" / f"{lane}.lock")
    if q.doc.get("wall_power_csv"):
        env["CAPSIM_WALL_POWER_CSV"] = q.doc["wall_power_csv"]
    if q.box == "xe7740":
        env["CAPSIM_GPU_INDEXES"] = ",".join(str(d) for d in devices)
    if job.get("kind") in SCORE_JOBS:
        # A quality score, not a capacity sweep: same engine build, then the
        # kind's own script (scripts/livebench_job.py, scripts/opencompass_job.py)
        # writes a summary the runner checks.
        (rundir / "job.json").write_text(json.dumps(job, indent=1, default=str))
        script = (f"cd {REPO} && {UV} run -q python scripts/{SCORE_JOBS[job['kind']][0]} "
                  f"--config {cfg} --job-file {rundir / 'job.json'}")
        log = open(rundir / "job.log", "a")
        log.write(f"\n=== {now()} start {job['kind']} lane={lane} devices={devices}\n")
        log.flush()
        return subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    workloads = ",".join(job["workloads"])
    # On the R770 the engine's OpenMP threads own every physical core of
    # their socket and stall whenever another process is scheduled on
    # one; the load generator runs on the hyper-thread siblings instead.
    pin = f"taskset -c {R770_HT_CPUS} " if q.box == "r770" else ""
    if q.box == "r770" and job.get("loadgen_other_socket"):
        # Hyper-threading off: no sibling CPUs exist, so the generator
        # takes the second socket the job reserved.
        pin = f"taskset -c {R770_SOCKET_CPUS[devices[1]]} "
    elif q.box == "r770" and int(job.get("socket_threads") or 1) == 2:
        # The engine has every thread of devices[0]; the generator
        # takes the second socket the job reserved.
        other = devices[1]
        pin = (f"taskset -c {R770_SOCKET_CPUS[other]},"
               f"{','.join(str(int(c) + R770_HT_OFFSET) for c in R770_SOCKET_CPUS[other].split(','))} ")
    # A deliberate re-run (status "fresh") starts a new run folder; the
    # default resumes the latest one, which is right after a crash but
    # would skip a workload that already finished.
    new_run = " --new-run" if q.status.get(job["id"], {}).get("fresh") else ""
    # Re-score even after a failed sweep (partial windows are worth
    # keeping), but exit with the sweep's status so a cancelled or
    # broken run is not marked done.
    script = (
        f"set -o pipefail; cd {REPO} && "
        f"{pin}{UV} run -q python -m simulator.cli sweep --config {cfg} "
        f"--type {workloads} --mode open{new_run} && "
        f"rd=$(ls -d {rundir}/run_* | sort -V | tail -1) && "
        f"{UV} run -q python -m simulator.cli export --input-dir $rd "
        f"--output $rd/export_slim.json --slim ; rc=$? ; "
        f"rd=$(ls -d {rundir}/run_* | sort -V | tail -1) && "
        f"{UV} run -q python -m simulator.rescore $rd ; exit $rc"
    )
    log = open(rundir / "job.log", "a")
    log.write(f"\n=== {now()} start lane={lane} devices={devices}\n")
    log.flush()
    return subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def _load_box_setting(q: Queue) -> dict:
    f = q.state_dir / "box_setting.json"
    return json.loads(f.read_text()) if f.exists() else {"name": None}


def _switch_box_setting(q: Queue, current: dict, name: str | None) -> None:
    """Reset the setting in force (if any), apply ``name`` (if any).
    Settings live in the queue file: box_settings: {name: {apply, reset}}."""
    settings = q.doc.get("box_settings") or {}
    if current.get("name") == name:
        return
    if current.get("name"):
        cmd = (settings.get(current["name"]) or {}).get("reset")
        if cmd:
            subprocess.run(["bash", "-c", cmd], check=False)
        print(f"{now()} box setting {current['name']} reset", flush=True)
    if name:
        cmd = (settings.get(name) or {}).get("apply")
        if cmd:
            subprocess.run(["bash", "-c", cmd], check=True)
        print(f"{now()} box setting {name} applied", flush=True)
    current.clear()
    current["name"] = name
    (q.state_dir / "box_setting.json").write_text(json.dumps(current))


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _sweeps_ok(q: Queue, jid: str) -> bool:
    """Every open-loop workload in the job's last run folder finished
    with status ok (a cancelled or broken sweep still gets re-scored,
    so tiers.json alone proves nothing)."""
    import sqlite3
    rds = sorted((q.runs_dir / jid).glob("run_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not rds or not (rds[-1] / "tiers.json").exists():
        return False
    try:
        conn = sqlite3.connect(str(rds[-1] / "run.db"))
        rows = conn.execute("SELECT final_status FROM cohort_run "
                            "WHERE mode = 'open_loop'").fetchall()
        conn.close()
    except sqlite3.Error:
        return False
    return bool(rows) and all(r[0] == "ok" for r in rows)


# Quality-score job kinds: the script the runner starts and the summary file
# whose status must be "ok" for the job to count as done.
SCORE_JOBS = {
    "livebench": ("livebench_job.py", "livebench_summary.json"),
    "opencompass": ("opencompass_job.py", "opencompass_summary.json"),
    # Not a quality score, but the same shape: one engine, a script, a
    # summary. Measures reasoning length per workload (level matching, T1).
    "reasoning_probe": ("reasoning_probe_job.py", "reasoning_probe_summary.json"),
}


def _score_ok(q: Queue, jid: str, kind: str) -> bool:
    f = q.runs_dir / jid / SCORE_JOBS[kind][1]
    try:
        return json.loads(f.read_text()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def _livebench_ok(q: Queue, jid: str) -> bool:
    return _score_ok(q, jid, "livebench")


def _finished_ok(q: Queue, jid: str) -> bool:
    """A quality-score job is done when its summary says ok; a capacity job
    when its sweeps finished."""
    job = next((j for j in q.jobs if j["id"] == jid), {})
    kind = job.get("kind")
    return _score_ok(q, jid, kind) if kind in SCORE_JOBS else _sweeps_ok(q, jid)


def _job_outcome(q: Queue, jid: str) -> str:
    """For a job whose process is not our child (adopted after a runner
    restart): done when it finished cleanly."""
    return "done" if _finished_ok(q, jid) else "failed"


# ── Engine tuning ──────────────────────────────────────────────────────
# Tuning jobs carry tune_group (one per model and box) and tune_name;
# persona jobs carry tune_key. When every job of a group has finished,
# the runner scores them and writes <state_dir>/tuned/<group>.json; a
# job with tune_key starts only once that file exists and runs with
# the winner's settings merged in.

def _tiers(q: Queue, jid: str) -> dict | None:
    rds = sorted((q.runs_dir / jid).glob("run_*"))
    if not rds or not (rds[-1] / "tiers.json").exists():
        return None
    return json.loads((rds[-1] / "tiers.json").read_text())


def pick_tuned(q: Queue, group: str) -> dict:
    """Winner = highest mean knee across the group's workloads,
    relative to the group's baseline (tune_name 'default'). The
    Standard-tier leader is recorded too; when it differs by more than
    10% it is kept as the latency alternative."""
    jobs = [j for j in q.jobs if j.get("tune_group") == group]
    rows = []
    for j in jobs:
        t = _tiers(q, j["id"])
        if not t:
            rows.append({"name": j.get("tune_name"), "job": j["id"], "ok": False})
            continue
        knees, std = {}, {}
        for wl, e in t["workloads"].items():
            knees[wl] = e.get("knee_per_min")
            st = (e.get("tiers") or {}).get("standard") or {}
            std[wl] = st.get("rate_per_min")
        rows.append({"name": j.get("tune_name"), "job": j["id"], "ok": True,
                     "knees": knees, "standard": std,
                     "overrides": j.get("tune_overrides") or {}})
    ok = [r for r in rows if r["ok"] and all(v for v in r["knees"].values())]
    base = next((r for r in ok if r["name"] == "default"), None)
    if base is None or len(ok) * 2 < len(rows):
        # Too few candidates finished to compare; leave the group
        # unpicked so its persona jobs wait for a re-run.
        print(f"{now()} tuning {group}: {len(ok)}/{len(rows)} usable, not picking",
              flush=True)
        return {}

    def rel(r, key):
        vals = []
        for wl, v in r[key].items():
            b = (base or {}).get(key, {}).get(wl)
            if v and b:
                vals.append(v / b)
        return sum(vals) / len(vals) if vals else 0.0

    for r in ok:
        r["knee_score"] = round(rel(r, "knees"), 3)
        r["standard_score"] = round(rel(r, "standard"), 3)
    best = max(ok, key=lambda r: r["knee_score"], default=None)
    lat = max(ok, key=lambda r: r["standard_score"], default=None)
    out = {"group": group, "picked_at": now(), "candidates": rows,
           "winner": best["name"] if best else None,
           "job_overrides": (best or {}).get("overrides", {})}
    if lat and best and lat is not best and lat["standard_score"] > 1.1 * best["standard_score"]:
        out["latency_alternative"] = {"name": lat["name"], "job_overrides": lat["overrides"]}
    tdir = q.state_dir / "tuned"
    tdir.mkdir(exist_ok=True)
    (tdir / f"{group}.json").write_text(json.dumps(out, indent=1))
    return out


def _tuned_overrides(q: Queue, job: dict) -> dict | None:
    key = job.get("tune_key")
    if not key:
        return {}
    f = q.state_dir / "tuned" / f"{key}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("job_overrides") or {}


def _in_start_gap(q: Queue, running: dict, job: dict) -> bool:
    """start_gap_s: keep a job's engine load clear of other starts.

    Nemotron Super NVFP4 loads its weights in ~50s alone but took
    24-27 min with three or four 2-GPU loads at once on the XE7740, past
    the 30-min health timeout. The gap applies both ways: the job
    waits for recent starts, and nothing starts during its own gap.
    """
    defs = {j["id"]: j for j in q.jobs}
    t = time.time()
    for jid in running:
        try:
            started = dt.datetime.strptime(
                q.status[jid]["started"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=dt.timezone.utc).timestamp()
        except (KeyError, ValueError):
            continue
        gap = max(int(job.get("start_gap_s") or 0),
                  int(defs.get(jid, {}).get("start_gap_s") or 0))
        if t - started < gap:
            return True
    return False


def _over_host_mem(q: Queue, running: dict, job: dict) -> bool:
    """host_mem_gb against the queue's host_mem_budget_gb.

    vLLM's KV offload maps its host buffer in /dev/shm (1008 GiB on the
    XE7740). Eight 1-GPU jobs at 128 GiB each did not fit, and the
    four that failed at start-up auto-held the queue for 8 hours.
    """
    need = float(job.get("host_mem_gb") or 0)
    budget = float(q.doc.get("host_mem_budget_gb") or 0)
    if not need or not budget:
        return False
    defs = {j["id"]: j for j in q.jobs}
    used = sum(float(defs.get(jid, {}).get("host_mem_gb") or 0) for jid in running)
    return used + need > budget


def _maybe_pick(q: Queue) -> None:
    groups = {j["tune_group"] for j in q.jobs if j.get("tune_group")}
    for g in groups:
        if (q.state_dir / "tuned" / f"{g}.json").exists():
            continue
        states = [q.status.get(j["id"], {}).get("state")
                  for j in q.jobs if j.get("tune_group") == g]
        if states and all(s in ("done", "failed") for s in states):
            pick_tuned(q, g)


def run(q: Queue) -> None:
    check_numa(q)
    free = q.total_devices()
    running: dict[str, tuple[subprocess.Popen | None, list[int]]] = {}
    adopted: dict[str, int] = {}
    stop = {"flag": False}
    hold = q.state_dir / "HOLD"
    fast_fails = 0
    # The host-wide setting in force, kept across runner restarts.
    box_setting = _load_box_setting(q)
    qmtime = q.path.stat().st_mtime

    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    # A job left "running" by a dead runner: adopt it while its process
    # lives (it was started in its own session and survives the
    # runner), else start it over.
    for j in q.jobs:
        st = q.status.get(j["id"])
        if st and st.get("state") == "running":
            if _alive(st.get("pid")):
                adopted[j["id"]] = int(st["pid"])
                running[j["id"]] = (None, list(st.get("devices") or []))
                free = [d for d in free if d not in (st.get("devices") or [])]
            else:
                st["state"] = "pending"
    q.save()

    while True:
        # Pick up queue edits (new or reordered jobs) without a restart.
        try:
            m = q.path.stat().st_mtime
            if m != qmtime:
                qmtime = m
                fresh = Queue(q.path)
                q.jobs, q.doc = fresh.jobs, fresh.doc
        except Exception as e:  # noqa: BLE001 - a bad edit must not kill the runner
            print(f"{now()} queue reload failed: {e}", flush=True)
        # Reap.
        for jid, (proc, devs) in list(running.items()):
            if proc is None:
                if _alive(adopted[jid]):
                    continue
                state, rc = _job_outcome(q, jid), None
            else:
                rc = proc.poll()
                if rc is None:
                    continue
                state = "done" if rc == 0 and _finished_ok(q, jid) else "failed"
            q.status[jid].update({"state": state, "rc": rc, "finished": now()})
            # Several jobs failing within seconds of starting means the
            # rig is broken, not the job: hold the queue instead of
            # burning through it.
            try:
                started = dt.datetime.strptime(q.status[jid]["started"], "%Y-%m-%dT%H:%M:%SZ")
                quick = state == "failed" and (
                    dt.datetime.utcnow() - started).total_seconds() < 120
            except (KeyError, ValueError):
                quick = False
            fast_fails = fast_fails + 1 if quick else 0
            if fast_fails >= 4 and not hold.exists():
                hold.write_text(f"{now()} auto-hold: {fast_fails} jobs failed within 2 min of start\n")
                print(f"{now()} auto-hold after {fast_fails} fast failures", flush=True)
            free = sorted(set(free) | set(devs))
            del running[jid]
            q.save()
        _maybe_pick(q)
        if stop["flag"]:
            for jid, (proc, _) in running.items():
                if proc is not None:
                    os.killpg(proc.pid, signal.SIGINT)
            for jid, (proc, _) in running.items():
                if proc is None:
                    continue
                try:
                    proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            return
        # Start in queue order; a job that does not fit blocks those
        # behind it (no back-filling), so whole-box jobs are not starved.
        # HOLD in the state dir: start nothing new.
        pending = [j for j in q.jobs
                   if q.status.get(j["id"], {}).get("state") in (None, "pending")
                   and not j.get("manual")]
        for job in ([] if hold.exists() else pending):
            # barrier: wait until everything before it has finished.
            if job.get("barrier") and running:
                break
            # after: run only once that job is done (a model's load
            # probe); skipped if it failed. Others may start meanwhile.
            dep = job.get("after")
            if dep:
                dstate = q.status.get(dep, {}).get("state")
                if dstate in ("failed", "skipped"):
                    q.status[job["id"]] = {"state": "skipped", "reason": f"{dep} {dstate}",
                                           "finished": now()}
                    q.save()
                    continue
                if dstate != "done":
                    continue
            # box_setting: a host-wide change (CPU clock cap). Jobs
            # under one setting run together; switching waits for the
            # box to empty, then resets the old setting and applies the
            # new one.
            setting = job.get("box_setting")
            if setting != box_setting.get("name"):
                if running:
                    break
                _switch_box_setting(q, box_setting, setting)
            tuned = _tuned_overrides(q, job)
            if tuned is None:
                break  # its model's tuning has not been picked yet
            # The winner's settings apply except where the job pins its
            # own (a job that tests one setting keeps its value).
            pins = set(job.get("pin") or [])
            tuned = {k: v for k, v in tuned.items() if k not in pins}
            job = deep_merge(job, tuned)
            if _in_start_gap(q, running, job) or _over_host_mem(q, running, job):
                break
            need = int(job.get("devices") or 1)
            devs = pick_devices(free, need, q.box)
            if devs is None:
                break
            # A job that cannot even start (a bad config, a model the
            # engine refuses) fails alone instead of stopping the runner.
            try:
                proc = launch(q, job, devs)
            except Exception as exc:  # noqa: BLE001
                q.status[job["id"]] = {"state": "failed", "started": now(), "finished": now(),
                                       "reason": f"launch: {type(exc).__name__}: {exc}"[:500]}
                q.save()
                continue
            free = [d for d in free if d not in devs]
            running[job["id"]] = (proc, devs)
            q.status[job["id"]] = {"state": "running", "devices": devs,
                                   "started": now(), "pid": proc.pid,
                                   "tuned": tuned or None}
            q.save()
        if not running and not hold.exists() and not [
                j for j in q.jobs
                if q.status.get(j["id"], {}).get("state") in (None, "pending")
                and not j.get("manual")]:
            _switch_box_setting(q, box_setting, None)
            return
        time.sleep(10)


def status(q: Queue) -> None:
    for j in q.jobs:
        st = q.status.get(j["id"], {})
        state = st.get("state", "pending")
        if j.get("manual") and state == "pending":
            state = "manual"
        print(f"{j['id']:<48} {state:<8} "
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
