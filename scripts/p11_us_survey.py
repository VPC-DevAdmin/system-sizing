#!/usr/bin/env python3
"""P11 US model survey driver for the XE7740 (docs/p11-us-stack-scoping.md).

Standalone on purpose (stdlib only, Python 3.10): launches each model on vLLM
or SGLang in Docker, drives load with AIPerf, runs a short exact-match quality
check, and writes one result JSON per config. A config fills the box with
replicas of the smallest tensor-parallel size that fits; AIPerf spreads
requests round-robin over the replicas, so throughput is measured per server,
not multiplied up from one GPU.

    p11_us_survey.py list
    p11_us_survey.py screen <config>...    # one load rung, picks the engine
    p11_us_survey.py sweep <config>...     # full ladder + quality check
    p11_us_survey.py auto <model>...       # screen each engine, sweep the winner
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(os.environ.get("P11_OUT", "/data/capsim/p11/results"))
AIPERF = os.environ.get(
    "P11_AIPERF", "/data/system-sizing/runs/benchmark-tools/aiperf-0.13.0/bin/aiperf"
)
REPO = Path(os.environ.get("P11_REPO", "/data/system-sizing"))
HF_CACHE = "/data2/capsim/huggingface"
OVERFLOW = "/data3/capsim/hf-overflow"

VLLM = "vllm/vllm-openai:latest"  # v0.29.0 on the box; digest recorded per run
SGLANG = "lmsysorg/sglang:latest"  # v0.5.20 on the box
VLLM_BEAM = "vllm/vllm-openai:reflection-beam"  # Beam is not in a vLLM release yet

# Workload: same for every model. AIPerf synthetic prompts (natural English
# text), fixed lengths, ignore_eos so every model writes the same token count.
ISL, OSL = 1024, 256
LADDER = [8, 32, 128, 256, 512, 1024, 2048, 4096]
SCREEN_CONCURRENCY = 1024
# Slow giants need a window longer than one request (Nemotron-Ultra at 1,024 users
# took ~5 min per answer, so a 120 s window recorded nothing): override per run.
RUNG_S = int(os.environ.get("P11_RUNG_S", 120))
GRACE_S = int(os.environ.get("P11_GRACE_S", 60))
WARMUP_S = 30
SHARDS = 4  # parallel AIPerf processes per rung
# No latency target is applied here: every rung is recorded and targets are
# applied afterwards. Only real errors (AIPerf failure, >5% failed requests)
# stop a ladder.
MAX_ERROR_FRACTION = 0.05
MAX_MODEL_LEN = 40960

# The recipe pins the drafter to FLASH_ATTN, which refuses an FP8 KV cache on
# SM120; leave the choice to vLLM (P11_DFLASH_BACKEND forces one if needed).
_dflash = {
    "method": "dflash",
    "model": "reflection/Beam-1.0-501B-A23B-DFlash2",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": True,
}
if os.environ.get("P11_DFLASH_BACKEND"):
    _dflash["attention_backend"] = os.environ["P11_DFLASH_BACKEND"]
if os.environ.get("P11_DFLASH_ADAPTIVE") == "0":  # FLASHINFER cannot do adaptive verification
    del _dflash["enable_adaptive_verification"]
DFLASH = json.dumps(_dflash)


def cfg(model, engine, tp, replicas, image=None, args=(), env=None, reasoning=None):
    return {
        "model": model,
        "engine": engine,
        "tp": tp,
        "replicas": replicas,
        "image": image or (VLLM if engine == "vllm" else SGLANG),
        "args": list(args),
        "env": env or {},
        "reasoning_effort": reasoning,
    }


# SGLang's default MoE runner refuses NVFP4 experts on SM120 and names this one.
SGL_NVFP4_MOE = ["--moe-runner-backend", "flashinfer_cutlass"]
BEAM = "reflection/Beam-1.0-501B-A23B-NVFP4"
BEAM_ARGS = ["--kv-cache-dtype", "fp8", "--reasoning-parser", "beam"]
CONFIGS = {
    "gpt-oss-20b-vllm": cfg("openai/gpt-oss-20b", "vllm", 1, 8, reasoning="low"),
    "gpt-oss-20b-sglang": cfg("openai/gpt-oss-20b", "sglang", 1, 8, reasoning="low"),
    "gpt-oss-120b-vllm": cfg("openai/gpt-oss-120b", "vllm", 1, 8, reasoning="low"),
    "gpt-oss-120b-sglang": cfg("openai/gpt-oss-120b", "sglang", 1, 8, reasoning="low"),
    "gemma-4-26b-vllm": cfg("nvidia/Gemma-4-26B-A4B-NVFP4", "vllm", 1, 8),
    "gemma-4-26b-sglang": cfg("nvidia/Gemma-4-26B-A4B-NVFP4", "sglang", 1, 8, args=SGL_NVFP4_MOE),
    "gemma-4-31b-vllm": cfg("nvidia/Gemma-4-31B-IT-NVFP4", "vllm", 1, 8),
    "gemma-4-31b-sglang": cfg("nvidia/Gemma-4-31B-IT-NVFP4", "sglang", 1, 8),
    "nemotron-3-super-vllm": cfg("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "vllm", 2, 4),
    "nemotron-3-super-sglang": cfg(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "sglang", 2, 4, args=SGL_NVFP4_MOE
    ),
    "nemotron-3-ultra-vllm": cfg(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "vllm", 8, 1,
        env={"VLLM_ENGINE_READY_TIMEOUT_S": "3600"},
    ),
    "nemotron-3-ultra-sglang": cfg(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", "sglang", 8, 1, args=SGL_NVFP4_MOE
    ),
    "beam-vllm": cfg(
        BEAM, "vllm", 8, 1, image=VLLM_BEAM, args=BEAM_ARGS,
        env={"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}, reasoning="low",
    ),
    "beam-dflash-vllm": cfg(
        BEAM, "vllm", 8, 1, image=VLLM_BEAM,
        args=BEAM_ARGS + ["--speculative-config", DFLASH],
        env={"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}, reasoning="low",
    ),
}


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), msg, flush=True)


def sh(cmd, check=True, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def max_seqs(c):
    # Engines size warm-up activations by this, so a single 8-GPU copy of a giant
    # cannot take all 4,096 (Beam OOMed at startup); load above it queues.
    return min(max(LADDER) // c["replicas"], 1024)


def docker_cmd(name, c, gpus, port):
    gpu_arg = "device=" + ",".join(map(str, gpus))
    cmd = [
        "docker", "run", "-d", "--name", name, "--gpus", f'"{gpu_arg}"' if len(gpus) > 1 else gpu_arg,
        "--ipc=host", "--network", "host", "--shm-size", "32g",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface", "-v", f"{OVERFLOW}:{OVERFLOW}",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "--label", "capsim.p11=1",
    ]
    for k, v in c["env"].items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(c["image"])
    if c["engine"] == "vllm":
        cmd += [
            "--model", c["model"], "--host", "0.0.0.0", "--port", str(port),
            "--tensor-parallel-size", str(c["tp"]), "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.90", "--max-num-seqs", str(max_seqs(c)),
            "--served-model-name", c["model"],
        ]
    else:
        cmd += [
            "python3", "-m", "sglang.launch_server", "--model-path", c["model"],
            "--host", "0.0.0.0", "--port", str(port), "--tp", str(c["tp"]),
            "--context-length", str(MAX_MODEL_LEN), "--mem-fraction-static", "0.90",
            "--max-running-requests", str(max_seqs(c)), "--enable-metrics",
            "--nccl-port", str(port + 1000),  # replicas share the host network
        ]
    return cmd + c["args"]


def stop_all():
    ids = sh(["docker", "ps", "-aq", "--filter", "label=capsim.p11=1"]).stdout.split()
    if ids:
        sh(["docker", "rm", "-f", *ids], check=False)
    for _ in range(60):  # wait for GPU memory to drain before the next launch
        used = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if all(int(x) < 1000 for x in used.stdout.split()):
            return
        time.sleep(5)


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def reuse(cid, c):
    urls = [f"http://127.0.0.1:{8100 + 10 * r}" for r in range(c["replicas"])]
    try:
        for u in urls:
            post(u + "/v1/chat/completions", {"model": c["model"], "max_tokens": 4,
                 "messages": [{"role": "user", "content": "Say OK."}]}, timeout=30)
        names = sh(["docker", "ps", "--filter", "label=capsim.p11=1", "--format", "{{.Names}}"]).stdout.split()
        return urls if sorted(names) == sorted(f"p11-{cid}-{r}" for r in range(c["replicas"])) else None
    except Exception:
        return None


def launch(cid, c, run_dir):
    urls = reuse(cid, c)
    if urls:
        log(f"{cid} reusing running replicas")
        return urls, 0
    stop_all()
    urls, names = [], []
    for r in range(c["replicas"]):
        gpus = list(range(r * c["tp"], (r + 1) * c["tp"]))
        port, name = 8100 + 10 * r, f"p11-{cid}-{r}"
        cmd = docker_cmd(name, c, gpus, port)
        (run_dir / f"launch-{r}.json").write_text(json.dumps(cmd, indent=1))
        sh(cmd)
        urls.append(f"http://127.0.0.1:{port}")
        names.append(name)
    t0 = time.time()
    pending = dict(zip(names, urls))
    while pending:
        for name, url in list(pending.items()):
            running = sh(["docker", "inspect", "-f", "{{.State.Running}}", name], check=False)
            if running.stdout.strip() != "true":
                tail = sh(["docker", "logs", "--tail", "60", name], check=False)
                raise RuntimeError(f"{name} exited during startup:\n{tail.stdout}{tail.stderr}")
            try:
                post(url + "/v1/chat/completions", {
                    "model": c["model"], "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Say OK."}],
                }, timeout=60)
                del pending[name]
                log(f"{name} ready after {time.time() - t0:.0f}s")
            except Exception:
                pass
        if time.time() - t0 > 5400:
            raise RuntimeError(f"startup timeout: {sorted(pending)}")
        time.sleep(15)
    return urls, round(time.time() - t0)


def save_logs(run_dir, c, cid):
    for r in range(c["replicas"]):
        out = sh(["docker", "logs", f"p11-{cid}-{r}"], check=False)
        (run_dir / f"engine-{r}.log").write_text(out.stdout[-2_000_000:] + out.stderr[-2_000_000:])


def stat(doc, key, s):
    v = doc.get(key, {})
    return v.get(s) if isinstance(v, dict) else None


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def engine_running(shard_dirs):
    """Median requests the engines actually ran, summed over distinct replicas.
    If this sits well below the offered concurrency, the client is the limit."""
    seen = {}
    for d in shard_dirs:
        f = d / "server_metrics_export.json"
        if not f.is_file():
            continue
        metrics = json.loads(f.read_text()).get("metrics", {})
        for key in ("vllm:num_requests_running", "sglang:num_running_reqs"):
            for s in metrics.get(key, {}).get("series", []):
                p50 = s.get("stats", {}).get("p50")
                if p50 is not None:
                    seen[s.get("endpoint_url")] = p50
    return sum(seen.values()) if seen else None


def aiperf_rung(c, urls, concurrency, rung_dir):
    """One load rung. A single AIPerf process tops out near 400 requests/s (its
    timing manager pins a core), so the load is split over SHARDS processes,
    each driving its own replicas, and per-request records are merged."""
    rung_dir.mkdir(parents=True, exist_ok=True)
    k = min(SHARDS, concurrency)
    procs = []
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HOME=HF_CACHE)
    for i in range(k):
        shard_dir = rung_dir / f"s{i}"
        shard_dir.mkdir(exist_ok=True)
        cmd = [
            AIPERF, "profile", "--model", c["model"], "--endpoint-type", "chat", "--streaming",
            "--tokenizer", c["model"], "--use-server-token-count",
            "--synthetic-input-tokens-mean", str(ISL), "--synthetic-input-tokens-stddev", "0",
            "--output-tokens-mean", str(OSL), "--output-tokens-stddev", "0",
            "--extra-inputs", "ignore_eos:true",
            "--concurrency", str(concurrency // k + (i < concurrency % k)),
            "--benchmark-duration", str(RUNG_S),
            "--benchmark-grace-period", str(GRACE_S), "--warmup-duration", str(WARMUP_S),
            "--random-seed", str(42 + i), "--artifact-dir", str(shard_dir), "--ui", "none",
            "--no-gpu-telemetry", "--workers-max", "16",
        ]
        for u in (urls[i::k] if len(urls) >= k else urls):
            cmd += ["--url", u]
        log_f = open(shard_dir / "aiperf.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env),
                      shard_dir, log_f))
    docs, ttft, itl, failed = [], [], [], 0
    for p, shard_dir, log_f in procs:
        try:
            rc = p.wait(timeout=RUNG_S + GRACE_S + 1800)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
        log_f.close()
        files = sorted(shard_dir.rglob("*aiperf.json"))
        if rc or not files:
            return {"concurrency": concurrency, "error": f"aiperf shard {shard_dir.name} rc={rc}"}
        docs.append(json.loads(files[-1].read_text()))
        for line in open(shard_dir / "profile_export.jsonl"):
            rec = json.loads(line)
            if rec["metadata"].get("benchmark_phase") == "warmup":
                continue
            m = rec.get("metrics", {})
            if rec.get("error") or "time_to_first_token" not in m or "request_latency" not in m:
                failed += 1
                continue
            first = m["time_to_first_token"]["value"]
            ttft.append(first)
            # Every answer is exactly OSL tokens (ignore_eos), so this is AIPerf's ITL.
            itl.append((m["request_latency"]["value"] - first) / (OSL - 1))
    ok = len(ttft)
    total = lambda key: sum(stat(d, key, "avg") or 0 for d in docs)  # noqa: E731
    row = {
        "concurrency": concurrency,
        "window_s": RUNG_S,
        "grace_s": GRACE_S,
        "shards": k,
        "requests": ok,
        "error_fraction": failed / (ok + failed) if ok + failed else 1.0,
        "gen_tok_s": total("output_token_throughput"),
        "req_s": total("request_throughput"),
        "ttft_p50_ms": pct(ttft, 0.50),
        "ttft_p95_ms": pct(ttft, 0.95),
        "itl_p50_ms": pct(itl, 0.50),
        "itl_p95_ms": pct(itl, 0.95),
        "engine_running_p50": engine_running([s for _, s, _ in procs]),
    }
    row["tok_s_per_user_p50"] = 1000 / row["itl_p50_ms"] if row["itl_p50_ms"] else None
    return row


def answer_text(message):
    """The answer the grader sees. Engines served without a reasoning parser put
    the model's reasoning in content: a <think> block (Nemotron) or raw Harmony
    channels (gpt-oss on SGLang). Keep only the final answer, and unwrap an
    optional ```json fence. The JSON itself must still match exactly."""
    text = (message.get("content") or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "<|channel|>final<|message|>" in text:
        text = text.rsplit("<|channel|>final<|message|>", 1)[1]
        for end in ("<|return|>", "<|end|>", "<|call|>"):
            text = text.split(end, 1)[0]
        text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text


def quality(c, urls, run_dir, n=32):
    """Exact-match JSON extraction from synthetic documents at ~8K and ~32K
    tokens (simulator/infrastructure/workload.py). Narrow by design: it checks
    long-document retrieval, not broad model quality. Raw responses are kept."""
    sys.path.insert(0, str(REPO))
    from simulator.infrastructure.workload import build_corpus, grade

    out = {}
    for label, records in (("8k", 110), ("32k", 440)):
        corpus = build_corpus(n, records, 42)

        def one(i_doc):
            i, doc = i_doc
            body = {
                "model": c["model"], "temperature": 0, "max_tokens": 6144,
                "messages": [{"role": "user", "content": doc["prompt"]}],
            }
            if c["reasoning_effort"]:
                body["reasoning_effort"] = c["reasoning_effort"]
            try:
                r = post(urls[i % len(urls)] + "/v1/chat/completions", body, timeout=900)
                choice = r["choices"][0]
                ok = grade(answer_text(choice["message"]), doc["expected"])
                rec = {"id": doc["id"], "correct": ok, "finish_reason": choice.get("finish_reason"),
                       "usage": r.get("usage"), "message": choice["message"]}
                return ok, r.get("usage", {}).get("prompt_tokens"), rec
            except Exception as e:
                return False, repr(e)[:200], {"id": doc["id"], "error": repr(e)[:500]}

        with cf.ThreadPoolExecutor(16) as pool:
            res = list(pool.map(one, enumerate(corpus)))
        with open(run_dir / f"quality-{label}.jsonl", "w") as f:
            for _, _, rec in res:
                f.write(json.dumps(rec) + "\n")
        prompt_tokens = [t for _, t, _ in res if isinstance(t, int)]
        out[label] = {
            "correct": sum(ok for ok, _, _ in res),
            "total": n,
            "accuracy": sum(ok for ok, _, _ in res) / n,
            "mean_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens))
            if prompt_tokens else None,
            "truncated": sum(rec.get("finish_reason") == "length" for _, _, rec in res),
            "errors": [t for _, t, _ in res if isinstance(t, str)][:5],
        }
        log(f"quality {label}: {out[label]['correct']}/{n}")
    return out


def provenance(c):
    img = sh(["docker", "image", "inspect", c["image"], "--format",
              "{{.Id}} {{index .RepoDigests 0}} {{index .Config.Labels \"org.opencontainers.image.version\"}}"],
             check=False).stdout.split()
    snap = None
    for root in (Path(HF_CACHE) / "hub", Path(OVERFLOW) / "hub"):
        ref = root / f"models--{c['model'].replace('/', '--')}" / "refs" / "main"
        if ref.is_file():
            snap = ref.read_text().strip()
            break
    return {"image": c["image"], "image_id": img[0] if img else None,
            "image_digest": img[1] if len(img) > 1 else None,
            "image_version": img[2] if len(img) > 2 else None, "model_revision": snap}


def run(cid, ladder, do_quality, label, keep=False, resume=False):
    c = CONFIGS[cid]
    run_dir = OUT / label / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    prior = json.loads((run_dir / "result.json").read_text()) if resume and (run_dir / "result.json").is_file() else {}
    done = {r["concurrency"] for r in prior.get("rungs", []) if not r.get("error")}
    result = {"config": cid, **c, "isl": ISL, "osl": OSL, "rung_s": RUNG_S,
              "provenance": provenance(c), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if prior:
        result["quality"] = prior.get("quality")
        result["resumed_from"] = prior.get("started")
    try:
        urls, startup_s = launch(cid, c, run_dir)
        result["startup_s"] = startup_s
        rows = [r for r in prior.get("rungs", []) if r["concurrency"] in done]
        for conc in [x for x in ladder if x not in done]:
            log(f"{cid} rung c={conc}")
            row = aiperf_rung(c, urls, conc, run_dir / f"c{conc}")
            rows.append(row)
            log(f"{cid} c={conc}: {json.dumps(row)}")
            result["rungs"] = sorted(rows, key=lambda r: r["concurrency"])
            (run_dir / "result.json").write_text(json.dumps(result, indent=1))
            if row.get("error") or row["error_fraction"] > MAX_ERROR_FRACTION:
                break
        result["rungs"] = sorted(rows, key=lambda r: r["concurrency"])
        if do_quality and not result.get("quality"):
            result["quality"] = quality(c, urls, run_dir)
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[-4000:]
        log(f"{cid} FAILED: {str(e)[-800:]}")
    finally:
        save_logs(run_dir, c, cid)
        if not keep:
            stop_all()
    good = [r for r in result.get("rungs", []) if not r.get("error")]
    result["peak_gen_tok_s"] = max((r["gen_tok_s"] or 0 for r in good), default=None)
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    log(f"{cid} {result['status']}: peak {result['peak_gen_tok_s']} gen tok/s over "
        f"{len(good)} rungs")
    return result


def requality(cid):
    """Re-run only the quality check for a finished sweep (fresh engine launch)."""
    c = CONFIGS[cid]
    run_dir = OUT / "sweep" / cid
    result = json.loads((run_dir / "result.json").read_text())
    (run_dir / "requality").mkdir(exist_ok=True)
    try:
        urls, _ = launch(cid, c, run_dir / "requality")
        result.setdefault("quality_history", []).append(result.get("quality"))
        result["quality"] = quality(c, urls, run_dir)
    finally:
        stop_all()
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    log(f"{cid} requality: {json.dumps(result['quality'])}")


def regrade(cid):
    """Re-grade a sweep's saved quality responses with the current answer_text."""
    sys.path.insert(0, str(REPO))
    from simulator.infrastructure.workload import build_corpus, grade

    run_dir = OUT / "sweep" / cid
    result = json.loads((run_dir / "result.json").read_text())
    for label, records in (("8k", 110), ("32k", 440)):
        f = run_dir / f"quality-{label}.jsonl"
        if not f.is_file():
            continue
        expected = {d["id"]: d["expected"] for d in build_corpus(32, records, 42)}
        recs = [json.loads(line) for line in open(f)]
        for r in recs:
            r["correct"] = bool(r.get("message")) and grade(answer_text(r["message"]), expected[r["id"]])
        f.write_text("".join(json.dumps(r) + "\n" for r in recs))
        q = result.setdefault("quality", {}).setdefault(label, {})
        q["correct"] = sum(r["correct"] for r in recs)
        q["accuracy"] = q["correct"] / len(recs)
        q["regraded"] = True
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    log(f"{cid} regrade: " + json.dumps({k: v.get("correct") for k, v in result["quality"].items()}))


def model_key(cid):
    return cid.rsplit("-", 1)[0]


def auto(key, keep):
    """Screen every engine for one model at the same load, sweep the faster."""
    cands = [cid for cid in CONFIGS if model_key(cid) == key]
    if not cands:
        raise SystemExit(f"unknown model key {key}")
    best = cands[0]
    if len(cands) > 1:
        scored = []
        for cid in cands:
            done = OUT / "screen" / cid / "result.json"
            r = json.loads(done.read_text()) if done.is_file() else run(cid, [SCREEN_CONCURRENCY], False, "screen")
            scored.append((r.get("peak_gen_tok_s") or 0, cid))
        best = max(scored)[1]
        log(f"{key}: screen {scored} -> sweep {best}")
        if not max(scored)[0]:
            log(f"{key}: no engine served the model; skipping the sweep")
            return
    run(best, LADDER, True, "sweep", keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "screen", "sweep", "auto", "requality", "regrade"])
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--keep", action="store_true", help="leave the engines running (debugging)")
    ap.add_argument("--resume", action="store_true", help="sweep: run only rungs not yet recorded")
    ap.add_argument("--concurrency", type=int, default=SCREEN_CONCURRENCY, help="screen load")
    a = ap.parse_intermixed_args()
    if a.cmd == "list":
        for k, v in CONFIGS.items():
            print(f"{k:28} {v['model']:50} {v['engine']:7} tp{v['tp']} x{v['replicas']}")
        return
    for cid in a.configs:
        if a.cmd == "auto":
            auto(cid, a.keep)
        elif a.cmd == "requality":
            requality(cid)
        elif a.cmd == "regrade":
            regrade(cid)
        elif a.cmd == "screen":
            run(cid, [a.concurrency], False, "screen", a.keep)
        else:
            run(cid, LADDER, not a.no_quality, "sweep", a.keep, a.resume)


if __name__ == "__main__":
    main()
