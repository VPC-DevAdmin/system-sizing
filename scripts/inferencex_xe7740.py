#!/usr/bin/env python3
"""InferenceX-style fixed-sequence sweep on the XE7740, with GPU power.

Reuses the P11 survey driver (p11_us_survey.py: engine launch in Docker, AIPerf
load, per-rung records) at InferenceX's fixed-sequence shapes so the result can
be read against inferencex.semianalysis.com rows for the same model:

  * 8k1k: 8,192 input / 1,024 output tokens (InferenceX's active single-turn
    scenario). 1k1k: 1,024 / 1,024 (deprecated there in 2026, but the only
    shape with recent gpt-oss-120b rows on every platform).
  * Closed loop at a fixed concurrency, ignore_eos, as InferenceX does. Their
    concurrency is per engine instance; ours is per server, so per-GPU values
    are what get compared. Every rung is recorded; no latency target here.
  * GPU board power is sampled with nvidia-smi once a second for the whole
    rung and averaged over AIPerf's measurement window, giving W per GPU and
    J per output token like InferenceX's avg_power_w / joules_per_output_token.
  * Server input (wall) power is sampled from the iDRAC at 1 Hz over the same
    window (idrac_power.py, sum of PSU input watts) and reported as wall_w,
    wall_j_per_output_token and gen_tok_s_per_wall_kw: the facility-side
    number MLPerf power results and InferenceX's power model aim at.

Differences from InferenceX's client (infx.bench_serving, a benchmark_serving
fork): AIPerf with fixed prompt lengths instead of a random range; a time
window instead of 10 x concurrency prompts.

    inferencex_xe7740.py list
    inferencex_xe7740.py sweep --workload 8k1k <config>...
    inferencex_xe7740.py screen --workload 8k1k --concurrency 8 <config>
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.environ.get("P11_DIR", "/data/capsim/p11"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p11_us_survey as p  # noqa: E402

p.OUT = Path(os.environ.get("IX_OUT", "/data/capsim/inferencex/results"))
WORKLOADS = {"8k1k": (8192, 1024), "1k1k": (1024, 1024)}
# Server-wide concurrency. Per GPU this is 1 .. 256 for eight TP1 replicas.
LADDER = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
p.MAX_MODEL_LEN = 10240  # 8k prompt + 1k answer + chat template, for both shapes
p.SHARDS = 4
p.max_seqs = lambda c: 1024
p.WARMUP_S = 30
p.GRACE_S = 120  # 8k1k answers take 15-60 s each; let the window drain


def window_s(concurrency):
    # Long enough for a few complete answers per stream at low concurrency.
    return 300 if concurrency <= 32 else 180 if concurrency <= 256 else 120


LLAMA_FP8 = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
QWEN35 = "nvidia/Qwen3.5-397B-A17B-NVFP4"  # the checkpoint in InferenceX's rtx6000pro rows
p.CONFIGS = {
    "gpt-oss-120b-tp1": p.cfg("openai/gpt-oss-120b", "vllm", 1, 8),
    "gpt-oss-120b-tp2": p.cfg("openai/gpt-oss-120b", "vllm", 2, 4),
    # 68 GB of FP8 weights leaves too little KV for 8k prompts on one GPU.
    "llama-70b-fp8-tp2": p.cfg(LLAMA_FP8, "vllm", 2, 4),
    "llama-70b-fp8-tp4": p.cfg(LLAMA_FP8, "vllm", 4, 2),
    # InferenceX's own RTX PRO 6000 rows: SGLang, TP4, with and without MTP.
    # Qwen3.5 is a hybrid (gated DeltaNet linear attention): SGLang sizes its
    # per-request linear-attention state pool from --max-running-requests, and
    # 1,024 of them left no room for KV on 96 GB. 256 per replica covers the ladder.
    "qwen3.5-397b-nvfp4-sglang-tp4": p.cfg(QWEN35, "sglang", 4, 2, args=["--max-running-requests", "128", "--mem-fraction-static", "0.85", "--cuda-graph-max-bs-decode", "64",
                                                                     "--moe-runner-backend", "flashinfer_cutlass"]),  # 256 at 0.9: NCCL OOM during graph capture
    "qwen3.5-397b-nvfp4-sglang-tp4-mtp": p.cfg(
        QWEN35, "sglang", 4, 2,
        args=["--max-running-requests", "64", "--mem-fraction-static", "0.85", "--cuda-graph-max-bs-decode", "32",
              "--moe-runner-backend", "flashinfer_cutlass",  # 128/256 at 0.9: "no GPU memory for the KV cache" with draft weights
              "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
              "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"],
    ),
}


class PowerSampler:
    """nvidia-smi board power, one sample per GPU per second, for one rung."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w")
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,power.draw,utilization.gpu,clocks.sm",
             "--format=csv,noheader,nounits", "-lms", "1000"],
            stdout=self.f, stderr=subprocess.DEVNULL,
        )
        self.t0 = time.time()

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.f.close()
        return self.t0, time.time()

    @staticmethod
    def summarize(path, start, end, gpu_count):
        """Mean board power per GPU over [start, end] (epoch seconds)."""
        per_gpu, n = {}, 0
        for row in csv.reader(open(path)):
            if len(row) < 3:
                continue
            try:
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                watts = float(row[2])
            except ValueError:
                continue
            if start <= ts <= end:
                per_gpu.setdefault(row[1].strip(), []).append(watts)
                n += 1
        if not per_gpu:
            return None
        means = {k: sum(v) / len(v) for k, v in per_gpu.items()}
        total = sum(means.values())
        return {"samples": n, "gpus_sampled": len(means), "gpu_count": gpu_count,
                "avg_w_per_gpu": total / len(means), "total_w": total,
                "per_gpu_w": {k: round(v, 1) for k, v in sorted(means.items())}}


IDRAC_ENV = os.environ.get("IDRAC_ENV", "/data/capsim/idrac/idrac.env")
IDRAC_POLLER = Path(os.environ.get("IDRAC_POLLER", "/data/capsim/idrac/idrac_power.py"))


class WallSampler:
    """Server input power from the iDRAC (sum of PSU input watts) at 1 Hz, via
    idrac_power.py. Optional: skipped, with a note in the record, when the
    credential file or poller is missing."""

    def __init__(self, path):
        self.path = path
        self.proc = None
        if Path(IDRAC_ENV).is_file() and IDRAC_POLLER.is_file():
            self.proc = subprocess.Popen(
                [sys.executable, str(IDRAC_POLLER), "--env", IDRAC_ENV, "--interval", "1", "-o", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    @staticmethod
    def summarize(path, start, end):
        """Mean wall watts over [start, end] (epoch seconds), from epoch,watts[,chassis] rows."""
        vals = []
        if not Path(path).is_file():
            return None
        for row in csv.reader(open(path)):
            try:
                ts, w = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue  # blank watts = a failed poll, recorded as such
            if start <= ts <= end:
                vals.append(w)
        if not vals:
            return None
        return {"samples": len(vals), "avg_w": sum(vals) / len(vals),
                "min_w": min(vals), "max_w": max(vals)}


def aiperf_window(rung_dir, t0, concurrency):
    """AIPerf's measurement window as epoch seconds. Each shard writes its
    request records with timestamps; the profiling phase bounds are taken from
    them. Falls back to warmup/window arithmetic from the sampler start."""
    starts, ends = [], []
    for f in rung_dir.glob("s*/profile_export.jsonl"):
        for line in open(f):
            rec = json.loads(line)
            if rec["metadata"].get("benchmark_phase") == "warmup":
                continue
            s, e = rec["metadata"].get("request_start_ns"), rec["metadata"].get("request_end_ns")
            if s and e:
                starts.append(s / 1e9)
                ends.append(e / 1e9)
    if starts:
        return min(starts), max(ends), "aiperf_records"
    begin = t0 + 15 + p.WARMUP_S
    return begin, begin + window_s(concurrency), "estimated"


_docker_cmd = p.docker_cmd


def docker_cmd_no_prefix_cache(name, c, gpus, port):
    """AIPerf cycles a small synthetic prompt set, so the engine's prefix cache
    ends up serving the whole 8k prefill (vLLM logged a 100% hit rate and ~10
    prompt tok/s at 8k/1k). InferenceX's random prompts never repeat, so
    disabling the cache is the equivalent and the prefill is really computed."""
    cmd = _docker_cmd(name, c, gpus, port)
    flag = "--no-enable-prefix-caching" if c["engine"] == "vllm" else "--disable-radix-cache"
    return cmd if flag in cmd else cmd + [flag]  # newer p11_us_survey.py adds it itself


p.docker_cmd = docker_cmd_no_prefix_cache

_rung = p.aiperf_rung


def aiperf_rung_with_power(c, urls, concurrency, rung_dir):
    p.RUNG_S = window_s(concurrency)
    rung_dir.mkdir(parents=True, exist_ok=True)
    sampler = PowerSampler(rung_dir / "power.csv")
    wall = WallSampler(rung_dir / "power_wall.csv")
    try:
        row = _rung(c, urls, concurrency, rung_dir)
    finally:
        t0, t1 = sampler.stop()
        wall.stop()
    gpu_count = c["tp"] * c["replicas"]
    start, end, source = aiperf_window(rung_dir, t0, concurrency)
    power = PowerSampler.summarize(rung_dir / "power.csv", max(start, t0), min(end, t1), gpu_count)
    row["power"] = {"window_source": source, "window_s": round(min(end, t1) - max(start, t0), 1),
                    **(power or {"samples": 0})}
    wall_power = WallSampler.summarize(rung_dir / "power_wall.csv", max(start, t0), min(end, t1))
    row["wall_power"] = wall_power or {"samples": 0, "note": "no iDRAC samples (credential/poller missing or polls failed)"}
    if power and row.get("gen_tok_s"):
        row["avg_w_per_gpu"] = round(power["avg_w_per_gpu"], 1)
        row["j_per_output_token"] = round(power["total_w"] / row["gen_tok_s"], 4)
        total_tok_s = row["gen_tok_s"] * (p.ISL + p.OSL) / p.OSL
        row["j_per_total_token"] = round(power["total_w"] / total_tok_s, 4)
    if wall_power and row.get("gen_tok_s"):
        # Whole-server (PSU input) energy: the number a facility actually provisions and bills.
        row["wall_w"] = round(wall_power["avg_w"], 1)
        row["wall_j_per_output_token"] = round(wall_power["avg_w"] / row["gen_tok_s"], 4)
        row["wall_j_per_total_token"] = round(wall_power["avg_w"] / (row["gen_tok_s"] * (p.ISL + p.OSL) / p.OSL), 4)
        row["gen_tok_s_per_wall_kw"] = round(row["gen_tok_s"] / (wall_power["avg_w"] / 1000), 1)
        if power:
            # GPU boards' share of the wall; the rest is host, fans, PSU loss. Only
            # meaningful when all 8 GPUs are in use by this config.
            row["gpu_share_of_wall"] = round(power["total_w"] / wall_power["avg_w"], 3)
    row["gen_tok_s_per_gpu"] = round(row["gen_tok_s"] / gpu_count, 1) if row.get("gen_tok_s") else None
    row["concurrency_per_gpu"] = concurrency / gpu_count
    # InferenceX reads curves at p90 interactivity: the per-request output
    # rate (tokens per second per user) that 90% of requests meet or exceed.
    itl = []
    for f in rung_dir.glob("s*/profile_export.jsonl"):
        for line in open(f):
            rec = json.loads(line)
            m = rec.get("metrics", {})
            if rec["metadata"].get("benchmark_phase") == "warmup" or rec.get("error"):
                continue
            if "time_to_first_token" in m and "request_latency" in m:
                itl.append((m["request_latency"]["value"] - m["time_to_first_token"]["value"]) / (p.OSL - 1))
    if itl:
        row["itl_p90_ms"] = p.pct(itl, 0.90)
        row["intvty_p90_tok_s"] = round(1000 / row["itl_p90_ms"], 2)
        row["intvty_median_tok_s"] = round(1000 / p.pct(itl, 0.50), 2)
    return row


p.aiperf_rung = aiperf_rung_with_power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "screen", "sweep"])
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--workload", choices=sorted(WORKLOADS), default="8k1k")
    ap.add_argument("--concurrency", type=int, default=64, help="screen load (server-wide)")
    ap.add_argument("--ladder", help="comma-separated server-wide concurrencies")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_intermixed_args()
    if a.cmd == "list":
        for k, v in p.CONFIGS.items():
            print(f"{k:36} {v['model']:45} {v['engine']:7} tp{v['tp']} x{v['replicas']}")
        return
    p.ISL, p.OSL = WORKLOADS[a.workload]
    ladder = [int(x) for x in a.ladder.split(",")] if a.ladder else LADDER
    for cid in a.configs:
        if a.cmd == "screen":
            p.run(cid, [a.concurrency], False, f"{a.workload}-screen", a.keep)
        else:
            p.run(cid, ladder, False, a.workload, a.keep, a.resume)


if __name__ == "__main__":
    main()
