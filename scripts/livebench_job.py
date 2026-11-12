"""Score one on-prem model and mode on LiveBench, on the lab's own engine.

The campaign runner starts this for jobs with ``kind: livebench``. It builds
the engine exactly as a capacity job would (same image, quantization, flags
and tuned overrides), so the score is for the model as the lab serves it.

  1. Launch the engine and wait for it to be healthy.
  2. Write a LiveBench model config for this run (its thinking toggle,
     sampling and token limit) and run inference only, against the engine.
  3. Shut the engine down (always, in ``finally``).
  4. Grade in a container with no network: coding answers are executed.
  5. Write ``livebench_summary.json`` (scores per category and overall, how
     many answers hit the token limit) and ``manifest.json`` (every setting a
     later cloud run must match).

The runner marks the job done only when the summary exists with status ok.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LB_ROOT = Path(os.environ.get("CAPSIM_LIVEBENCH_ROOT", "/data/capsim-planner/livebench"))
LB_REPO = LB_ROOT / "LiveBench"
LB_PY = LB_ROOT / "venv" / "bin" / "python"
LB_COMMIT = "1d1006a"
# Newest release whose questions are public in every category (2026-11-05
# check: later releases publish no coding questions and fewer others).
LB_RELEASE = "2024-11-25"
GRADE_IMAGE = "python:3.11-slim"
UV_PYTHON = Path.home() / ".local" / "share" / "uv" / "python"
# NLTK refuses data under a group-writable parent, so it lives in home.
NLTK_DATA = Path.home() / "nltk_data"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def display_name(job: dict) -> str:
    return "capsim-" + job["id"].lower()


def model_config_yaml(job: dict, served: str) -> str:
    lb = job["livebench"]
    kwargs = dict(lb.get("api_kwargs") or {})
    kwargs["temperature"] = float(lb["temperature"])
    if lb.get("top_p") is not None:
        kwargs["top_p"] = float(lb["top_p"])
    import yaml
    return yaml.safe_dump({
        "display_name": display_name(job),
        "api_name": {"local": served},
        "default_provider": "local",
        "api_kwargs": {"local": kwargs},
    }, sort_keys=False)


def answers(display: str) -> list[dict]:
    rows = []
    for f in (LB_REPO / "livebench" / "data").glob(f"live_bench/*/*/model_answer/{display}.jsonl"):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return rows


def run_inference(job: dict, base_url: str, served: str, log) -> int:
    lb = job["livebench"]
    cfg_file = LB_REPO / "livebench" / "model" / "model_configs" / f"{display_name(job)}.yml"
    cfg_file.write_text(model_config_yaml(job, served))
    cmd = [str(LB_PY), "run_livebench.py", "--model", display_name(job),
           "--bench-name", "live_bench", "--livebench-release-option", LB_RELEASE,
           "--api-base", base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/v1") else "/v1"),
           "--api-key", "EMPTY", "--max-tokens", str(int(lb["max_tokens"])),
           "--parallel-requests", str(int(lb.get("parallel_requests", 64))),
           "--skip-grading", "--resume", "--venv", str(LB_ROOT / "venv" / "bin" / "activate")]
    # --retry-failures re-asks every $ERROR$ answer. A top-up run that should
    # only fill answers removed by hand (client timeouts) turns it off, so
    # replies that ran out of tokens keep their first, scored attempt.
    if lb.get("retry_failures", True):
        cmd.insert(cmd.index("--venv"), "--retry-failures")
    if lb.get("question_end"):
        # A short check: the first N questions of every task.
        cmd += ["--question-end", str(int(lb["question_end"]))]
    # run_livebench re-launches its steps in the virtualenv it is told to
    # activate; without --venv it takes the caller's (system-sizing's).
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}
    env.update(HF_HOME=str(LB_ROOT / "hf"),
               PATH=f"{LB_ROOT / 'venv' / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}")
    log.write(f"{now()} inference: {' '.join(cmd)}\n"); log.flush()
    return subprocess.run(cmd, cwd=LB_REPO / "livebench", stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def run_grading(job: dict, log) -> int:
    """Grade in a throwaway container: no network, LiveBench mounted from
    the host, its virtualenv and interpreter read-only."""
    # LiveBench appends every model's judgments to one file per task, so
    # jobs running side by side take turns grading.
    # Run as the lab user, not root: NLTK refuses data owned by another
    # account, and graded code gets no more rights than the files it reads.
    cmd = ["flock", str(LB_ROOT / "grade.lock"), "docker", "run", "--rm", "--network", "none",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "--name", f"lb-grade-{job['id'].lower()}"[:60],
           "-v", f"{LB_ROOT}:{LB_ROOT}", "-v", f"{UV_PYTHON}:{UV_PYTHON}:ro",
           "-e", f"HF_HOME={LB_ROOT / 'hf'}", "-e", "HF_DATASETS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1",
           "-v", f"{NLTK_DATA}:{NLTK_DATA}:ro", "-e", f"NLTK_DATA={NLTK_DATA}",
           "-w", str(LB_REPO / "livebench"), GRADE_IMAGE,
           str(LB_PY), "run_livebench.py", "--model", display_name(job),
           "--bench-name", "live_bench", "--livebench-release-option", LB_RELEASE,
           "--skip-inference", "--parallel-grading", "16", "--venv", str(LB_ROOT / "venv" / "bin" / "activate")]
    if job["livebench"].get("question_end"):
        cmd += ["--question-end", str(int(job["livebench"]["question_end"]))]
    log.write(f"{now()} grading: {' '.join(cmd)}\n"); log.flush()
    return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def scores(display: str) -> dict:
    """Per-category and overall scores from LiveBench's own judgment files:
    the mean of task scores within a category, and the mean of categories."""
    latest: dict[tuple, tuple] = {}
    for f in (LB_REPO / "livebench" / "data").glob("live_bench/*/*/model_judgment/ground_truth_judgment.jsonl"):
        for x in f.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r.get("model") == display:
                    # A re-asked question leaves its old judgment behind; the
                    # newest judgment per question is the one that counts.
                    key = (f.parts[-4], f.parts[-3], r.get("question_id"))
                    ts = float(r.get("tstamp") or 0)
                    if key not in latest or ts >= latest[key][0]:
                        latest[key] = (ts, float(r["score"]))
    rows = [(cat, task, sc) for (cat, task, _), (_, sc) in latest.items()]
    by_task: dict[tuple, list[float]] = {}
    for cat, task, sc in rows:
        by_task.setdefault((cat, task), []).append(sc)
    cats: dict[str, list[float]] = {}
    for (cat, _), v in by_task.items():
        cats.setdefault(cat, []).append(100 * statistics.mean(v))
    per_cat = {c: round(statistics.mean(v), 2) for c, v in cats.items()}
    return {"graded": len(rows), "tasks": len(by_task), "categories": per_cat,
            "overall": round(statistics.mean(per_cat.values()), 2) if per_cat else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-file", required=True)
    a = ap.parse_args(argv)
    job = json.loads(Path(a.job_file).read_text())
    out = Path(a.job_file).parent
    log = open(out / "livebench.log", "a")
    sys.path.insert(0, str(REPO))
    from simulator.config import load_config
    from simulator.engines import make_engine

    cfg = load_config(a.config)
    lb = job["livebench"]
    manifest = {
        "job": job["id"], "model": job["model"], "mode": lb.get("mode"),
        "livebench_commit": LB_COMMIT, "release": LB_RELEASE,
        "categories": "all six; no agentic coding in this release",
        "livebench_patches": ["gen_ground_truth_judgment.py: dedupe old instruction-following "
                              "questions by question_id (upstream builds a set of dicts and crashes)"],
        "max_tokens": lb["max_tokens"], "temperature": lb["temperature"], "top_p": lb.get("top_p"),
        "api_kwargs": lb.get("api_kwargs") or {},
        "engine": {"type": cfg.engine.type, "image": getattr(cfg.engine, "gpu_image", None),
                   "quant": getattr(cfg.engine, "model_quant", None),
                   "kv_cache_dtype": getattr(cfg.engine, "kv_cache_dtype", None),
                   "max_model_len": getattr(cfg.engine, "max_model_len", None),
                   "tp": getattr(cfg.engine, "tensor_parallel_size", None),
                   "extra_flags": getattr(cfg.engine, "vllm_extra_flags", None)},
        "started": now(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    summary = {"status": "failed", "job": job["id"]}
    rc = 1
    engine = make_engine(cfg.engine.type, cfg.engine)
    try:
        t0 = time.time()
        engine.launch()
        log.write(f"{now()} engine up in {time.time() - t0:.0f}s at {engine.base_url}\n"); log.flush()
        rc = run_inference(job, engine.base_url, engine.api_model_name, log)
    finally:
        engine.shutdown()
        log.write(f"{now()} engine down\n"); log.flush()
    rows = answers(display_name(job))
    limit = int(lb["max_tokens"])
    toks = [int(r.get("total_output_tokens") or r.get("output_tokens") or 0) for r in rows]
    # LiveBench writes $ERROR$ when every retry came back with no answer
    # text. With a reasoning parser that means the reply ran out of tokens
    # while still reasoning (finish_reason "length"), so it counts as cut off.
    # (LiveBench reads reasoning from `reasoning_content`; this vLLM returns
    # `reasoning`, so a thinking toggle is checked by output length instead.)
    def errored(r: dict) -> bool:
        turns = (r.get("choices") or [{}])[0].get("turns") or [""]
        return not str(turns[0]).strip() or str(turns[0]).startswith("$ERROR$")
    errors = sum(1 for r in rows if errored(r))
    summary.update({"answers": len(rows), "inference_rc": rc, "errored_answers": errors,
                    "hit_token_limit": sum(1 for t in toks if t and t >= limit - 1),
                    "output_tokens_median": statistics.median(toks) if toks else None})
    if rc == 0 and rows:
        grc = run_grading(job, log)
        summary["grading_rc"] = grc
        if grc == 0:
            summary.update(scores(display_name(job)))
            if summary.get("overall") is not None and summary["graded"] >= 0.95 * len(rows):
                summary["status"] = "ok"
    summary["finished"] = now()
    summary["hit_token_limit_share"] = round((summary["hit_token_limit"] + summary.get("errored_answers", 0))
                                             / max(1, summary["answers"]), 4)
    (out / "livebench_summary.json").write_text(json.dumps(summary, indent=1))
    log.write(f"{now()} summary: {json.dumps(summary)}\n")
    log.close()
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
