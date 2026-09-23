"""KTransformers hot/cold expert placement experiment.

The question: with a model larger than its GPUs' share of it, does
putting each layer's MOST-USED experts on the GPU (and the rest on the
CPU) move enough of the expert work off the CPU to matter -- and does a
second replica, one per socket, add throughput or only split the same
two Xeons?

Why a driver of its own rather than a roofline cell:

* Routing depends on content. The headline workload's prompts are ten
  English paragraphs about capacity planning; placement learned on text
  that narrow would look far better than it would on traffic. This
  driver serves a multi-domain prompt set (prompt_sets.py) and keeps
  the half used to LEARN the hot experts apart from the half it
  MEASURES on, and measures shifted mixes too (code-heavy, Chinese-
  heavy), where the learned hot set is partly cold.

* The CPU's share of expert work is measured, not assumed: the fork's
  recorder counts every routed token per layer and expert, and records
  which experts sat on the GPU, so each window reports the share of
  activations the CPU served. Throughput is then checked against the
  ceiling it implies (CPU expert rate / CPU share).

* The rate is counted where it is made: the driver streams every
  response and counts tokens arriving inside the window -- no finish-
  time counter to quantise into waves (headline_sweep.wave_bound) --
  and records SGLang's gen_throughput gauge beside it.

A plan is JSON (see ``example_plan``); results land in
``<out_dir>/kt_placement.json`` after every window, so a watcher sees
progress and a crash loses one window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from .prompt_sets import MIXES, MixSampler, calibration_items, load_prompt_set

log = logging.getLogger(__name__)

RESULTS_NAME = "kt_placement.json"
# The recorder keeps the last 1000 forward passes and a dump resets it;
# at ~1 s a decode step that is ~16 minutes, so dump well inside it.
DUMP_EVERY_S = 240


def example_plan() -> dict:
    """The TP4 x DP2 placement test on an XE7740 with Kimi-K2-Thinking."""
    kt = {"engine": "ktransformers", "model_id": "moonshotai/Kimi-K2-Thinking",
          "tp": 4, "gpu_memory_utilization": 0.95, "max_model_len": 2048,
          "max_num_seqs": 128, "kv_cache_dtype": "auto",
          "ktransformers_max_total_tokens": 131072,
          "ktransformers_gpu_experts": "auto"}
    return {
        "prompt_set": "/data/capsim/datasets/kt_placement/prompts.jsonl",
        "out_dir": "/data/system-sizing/runs/kt_placement",
        "max_tokens": 256,
        "warmup_s": 120,
        "measure_s": 300,
        "calibrate_s": 1500,
        "calibrate_concurrency": 128,
        "calibrate_max_tokens": 128,
        "mixes": ["balanced", "code_heavy", "chinese_heavy"],
        "configs": [
            {"name": "dp2_uniform", "calibrate": True, "concurrency": [128, 256],
             "custom": {**kt, "replicas": 2, "ktransformers_numa_pin": True,
                        "ktransformers_expert_placement": "uniform",
                        "ktransformers_record_experts": True}},
            {"name": "dp2_frequency", "concurrency": [128, 256],
             "custom": {**kt, "replicas": 2, "ktransformers_numa_pin": True,
                        "ktransformers_expert_placement": "frequency",
                        "ktransformers_expert_freq_path": "@calibration",
                        "ktransformers_record_experts": True}},
            {"name": "dp1_frequency", "concurrency": [128, 256],
             "custom": {**kt, "replicas": 1, "max_num_seqs": 256,
                        "ktransformers_max_total_tokens": 262144,
                        "ktransformers_expert_placement": "frequency",
                        "ktransformers_expert_freq_path": "@calibration",
                        "ktransformers_record_experts": True}},
            {"name": "sglang_tp8_nvfp4", "concurrency": [128, 256, 1024],
             "custom": {"engine": "sglang_cuda",
                        "model_id": "nvidia/Kimi-K2-Thinking-NVFP4",
                        "tp": 8, "replicas": 1, "placement": "span",
                        "gpu_memory_utilization": 0.95, "max_model_len": 2048,
                        "max_num_seqs": 1024, "kv_cache_dtype": "fp8"}},
        ],
    }


# ── Plan resolution ───────────────────────────────────────────────────

def resolve_gpu_experts(custom: dict, cache: Path | None = None) -> dict:
    """``ktransformers_gpu_experts: "auto"`` -> the most experts per layer
    the replica's cards hold beside its KV pool (roofline.kt_gpu_expert_fit,
    with the pool at ktransformers_max_total_tokens)."""
    if custom.get("ktransformers_gpu_experts") != "auto":
        return custom
    from .roofline import kt_expert_budget, kt_gpu_expert_fit
    budget = kt_expert_budget(custom["model_id"], cache)
    if budget is None:
        raise ValueError(f"{custom['model_id']}: no staged native checkpoint "
                         "to size GPU experts from")
    from . import arena
    vram = float(arena.hardware().get("vram_per_gpu_gb") or 0)
    tokens = int(custom.get("ktransformers_max_total_tokens") or 0) or (
        int(custom["max_num_seqs"]) * int(custom["max_model_len"]))
    fit = kt_gpu_expert_fit(budget, tp=int(custom.get("tp") or 1), seqs=tokens,
                            vram_gb=vram,
                            share=float(custom.get("gpu_memory_utilization") or 0.95),
                            ctx=1)
    return {**custom, "ktransformers_gpu_experts": fit}


def build_engine(custom: dict, out_dir: Path):
    """An Engine for a custom shape, built by the same code the roofline
    and the benchmark form use (engines/custom.py)."""
    import yaml

    from .config import load_config
    from .engines import make_engine
    from .engines.custom import config_doc, custom_engine
    doc = config_doc(custom_engine(custom), out_dir)
    path = out_dir / f"engine_{custom['engine']}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = load_config(path)
    return make_engine(cfg.engine.type, cfg.engine)


# ── Load ──────────────────────────────────────────────────────────────

@dataclass
class Request:
    domain: str
    replica: int
    started: float
    ttft: float | None = None
    ended: float | None = None
    tokens: int = 0
    error: str | None = None
    arrivals: list[float] = field(default_factory=list)


async def _one(client: httpx.AsyncClient, url: str, model: str, item: dict,
               max_tokens: int, req: Request) -> Request:
    """Stream one request into ``req`` -- filled in place, so a request
    cancelled at a window's end keeps the tokens that arrived inside
    it."""
    body = {"model": model, "messages": item["messages"], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    try:
        async with client.stream("POST", f"{url}/chat/completions", json=body) as r:
            if r.status_code != 200:
                req.error = f"HTTP {r.status_code}"
                await r.aread()
                return req
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        now = time.monotonic()
                        if req.ttft is None:
                            req.ttft = now - req.started
                        req.arrivals.append(now)
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    req.tokens = int(usage["completion_tokens"])
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        req.error = type(e).__name__
    req.ended = time.monotonic()
    if not req.tokens:
        req.tokens = len(req.arrivals)
    return req


async def closed_loop(urls: list[str], model: str, sampler, *, concurrency: int,
                      until: float, max_tokens: int, done: list[Request],
                      timeout_s: float = 600.0) -> None:
    """``concurrency`` streams, each sending its next prompt as soon as
    the last one finishes, round-robin over the replicas, until the
    monotonic deadline. Requests still running at the deadline are
    cancelled; their arrivals up to then still count."""
    limits = httpx.Limits(max_connections=concurrency + 8,
                          max_keepalive_connections=concurrency + 8)
    counter = iter(range(10**9))
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        async def stream() -> None:
            while time.monotonic() < until:
                i = next(counter)
                item = sampler.next()
                req = Request(domain=item["domain"], replica=i % len(urls),
                              started=time.monotonic())
                done.append(req)
                try:
                    await asyncio.wait_for(
                        _one(client, urls[i % len(urls)], model, item,
                             max_tokens, req),
                        max(0.1, until - time.monotonic()))
                except asyncio.TimeoutError:
                    return
        await asyncio.gather(*(stream() for _ in range(concurrency)))


# ── Engine-side sampling ──────────────────────────────────────────────

def _proc_stat() -> dict[int, tuple[int, int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            vals = [int(x) for x in f[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            out[int(f[0][3:])] = (sum(vals), idle)
    return out


def node_busy(a: dict, b: dict, nodes: dict[int, list[int]]) -> dict[int, float]:
    """Busy share of each NUMA node's CPUs between two /proc/stat reads."""
    out = {}
    for n, cpus in nodes.items():
        tot = sum(b[c][0] - a[c][0] for c in cpus if c in a and c in b)
        idle = sum(b[c][1] - a[c][1] for c in cpus if c in a and c in b)
        out[n] = round(1 - idle / tot, 3) if tot > 0 else None
    return out


def _gpu_sample() -> list[dict]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in r.stdout.splitlines():
        try:
            i, u, p = [x.strip() for x in line.split(",")]
            out.append({"gpu": int(i), "util": float(u), "power_w": float(p)})
        except ValueError:
            continue
    return out


async def sample_engine(engine, until: float, every_s: float = 2.0) -> dict:
    """gen_throughput (summed over replicas), running/queued, GPU util
    and power, per-node CPU busy -- means over the window."""
    from .engines.ktransformers_v2 import node_cpulist, numa_node_count, parse_cpulist
    nodes = {}
    for n in range(numa_node_count() or 0):
        cl = node_cpulist(n)
        if cl:
            nodes[n] = parse_cpulist(cl)
    gauge, running, queue, util, power = [], [], [], [], []
    stat0 = _proc_stat() if nodes else {}
    while time.monotonic() < until:
        m = await asyncio.to_thread(engine.get_metrics)
        if m.get("gen_throughput") is not None:
            gauge.append(float(m["gen_throughput"]))
        if m.get("num_running") is not None:
            running.append(float(m["num_running"]))
        if m.get("queue_depth") is not None:
            queue.append(float(m["queue_depth"]))
        g = await asyncio.to_thread(_gpu_sample)
        if g:
            util.append(statistics.fmean(x["util"] for x in g))
            power.append(sum(x["power_w"] for x in g))
        await asyncio.sleep(every_s)
    stat1 = _proc_stat() if nodes else {}

    def mean(xs):
        return round(statistics.fmean(xs), 1) if xs else None
    return {"gauge_tok_s": mean(gauge), "running": mean(running),
            "queue": mean(queue), "gpu_util_pct": mean(util),
            "gpu_power_w": mean(power),
            "cpu_busy_by_node": node_busy(stat0, stat1, nodes) if nodes else {}}


# ── The expert recorder ───────────────────────────────────────────────

def _admin_urls(engine) -> list[str]:
    return [u[:-3] if u.endswith("/v1") else u for u in engine.replica_urls]


async def recorder(engine, action: str) -> None:
    """start / stop / dump the fork's expert recorder on every replica."""
    async with httpx.AsyncClient(timeout=120) as c:
        for base in _admin_urls(engine):
            r = await c.post(f"{base}/{action}_expert_distribution_record")
            r.raise_for_status()


AGGREGATE_PY = r'''
import json, sys, torch
mode, out = sys.argv[1], sys.argv[2]
pairs = json.loads(sys.argv[3])       # [[counts.pt, masks.pt | null], ...]
total = None
gpu_hits = None
for counts_path, masks_path in pairs:
    c = torch.load(counts_path, map_location="cpu", weights_only=True)["logical_count"].to(torch.int64)
    total = c.sum(0) if total is None else total + c.sum(0)
    if masks_path:
        m = torch.load(masks_path, map_location="cpu", weights_only=True)["gpu_expert_masks"]
        n = min(c.shape[0], m.shape[0])
        hits = (c[-n:] * m[-n:].to(torch.int64)).sum(0)
        gpu_hits = hits if gpu_hits is None else gpu_hits + hits
if mode == "freq":
    torch.save({"logical_count": total.unsqueeze(0)}, out)
    print(json.dumps({"tokens_routed": int(total.sum())}))
else:
    per_layer = total.sum(1)
    moe = per_layer > 0
    share = None
    layers = []
    if gpu_hits is not None:
        g = gpu_hits.sum(1)
        share = float(1 - g[moe].sum() / per_layer[moe].sum())
        layers = [round(float(1 - g[i] / per_layer[i]), 4) if per_layer[i] > 0 else None
                  for i in range(len(per_layer))]
    top = total[moe].float()
    top = top / top.sum(1, keepdim=True)
    k = max(1, top.shape[1] // 4)
    hot_mass = float(top.sort(1, descending=True).values[:, :k].sum(1).mean())
    print(json.dumps({"cpu_share": share, "cpu_share_by_layer": layers,
                      "top_quarter_mass": round(hot_mass, 4),
                      "tokens_routed": int(total.sum())}))
'''


def recorded_pairs(record_dir: Path, since: set[str]) -> list[list]:
    """(counts, masks) dump pairs written since ``since``, in time order.
    One dump call writes both files; they pair by order."""
    new = sorted(p for p in record_dir.rglob("*.pt") if str(p) not in since)
    counts = [p for p in new if p.name.startswith("expert_distribution_recorder_")]
    masks = [p for p in new if p.name.startswith("gpu_expert_distribution_")]
    return [[str(c), str(masks[i]) if i < len(masks) else None]
            for i, c in enumerate(counts)]


def aggregate(image: str, mode: str, pairs: list[list], out: Path) -> dict:
    """Run AGGREGATE_PY in the engine image (it has torch; capsim does
    not) over dump pairs; ``freq`` writes a placement file to ``out``,
    ``share`` returns the CPU's share of the routed activations."""
    dirs = sorted({str(Path(p).parent) for pr in pairs for p in pr if p}
                  | {str(out.parent)})
    cmd = ["docker", "run", "--rm", "--entrypoint", "python"]
    for d in dirs:
        cmd += ["-v", f"{d}:{d}"]
    cmd += [image, "-c", AGGREGATE_PY, mode, str(out), json.dumps(pairs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"aggregate {mode} failed: {r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def seen_files(record_dir: Path | None) -> set[str]:
    if record_dir is None or not record_dir.exists():
        return set()
    return {str(p) for p in record_dir.rglob("*.pt")}


# ── Windows ───────────────────────────────────────────────────────────

def window_stats(reqs: list[Request], t0: float, t1: float) -> dict:
    """Client-side rate and outcomes for a measurement window: tokens
    that ARRIVED inside it, over its length; outcomes of requests that
    finished inside it."""
    arrived = sum(1 for r in reqs for t in r.arrivals if t0 <= t < t1)
    finished = [r for r in reqs if r.ended is not None and t0 <= r.ended < t1]
    ok = [r for r in finished if not r.error]
    ttft = sorted(r.ttft for r in ok if r.ttft is not None)

    def pct(xs, q):
        return round(xs[min(len(xs) - 1, int(q * len(xs)))] * 1000, 1) if xs else None
    tpot = sorted((r.ended - r.started - r.ttft) / max(1, r.tokens - 1)
                  for r in ok if r.ttft is not None and r.tokens > 1)
    by_domain: dict[str, int] = {}
    for r in ok:
        by_domain[r.domain] = by_domain.get(r.domain, 0) + 1
    return {"stream_tok_s": round(arrived / max(1e-3, t1 - t0), 1),
            "finished": len(finished), "succeeded": len(ok),
            "success_rate": round(len(ok) / len(finished), 3) if finished else None,
            "ttft_p50_ms": pct(ttft, 0.5), "ttft_p95_ms": pct(ttft, 0.95),
            "tpot_p50_ms": pct(tpot, 0.5), "tpot_p95_ms": pct(tpot, 0.95),
            "answers_by_domain": by_domain}


@dataclass
class Results:
    plan: dict
    started: str
    calibration: dict = field(default_factory=dict)
    windows: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def save(self, out_dir: Path) -> None:
        tmp = out_dir / (RESULTS_NAME + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False))
        tmp.replace(out_dir / RESULTS_NAME)


async def measure(engine, model: str, items: list[dict], mix: str, split: str,
                  concurrency: int, *, warmup_s: float, measure_s: float,
                  max_tokens: int, record_dir: Path | None, image: str | None,
                  out_dir: Path, seed: int) -> dict:
    """One window: warm up, then measure with the recorder (if any)
    reset at the window's start and dumped at its end."""
    sampler = MixSampler(items, MIXES[mix], split, seed)
    urls = engine.replica_urls
    done: list[Request] = []
    t_start = time.monotonic()
    w0 = t_start + warmup_s
    w1 = w0 + measure_s
    load = asyncio.create_task(closed_loop(urls, model, sampler, concurrency=concurrency,
                                           until=w1, max_tokens=max_tokens, done=done))
    await asyncio.sleep(max(0.0, w0 - time.monotonic()))
    before = seen_files(record_dir)
    if record_dir is not None:
        await recorder(engine, "start")
    eng = await sample_engine(engine, w1)
    share = None
    if record_dir is not None:
        await recorder(engine, "dump")
        await recorder(engine, "stop")
        await asyncio.sleep(3)
        pairs = recorded_pairs(record_dir, before)
        if pairs and image:
            share = aggregate(image, "share", pairs, out_dir / "share.tmp")
    await load
    return {"mix": mix, "split": split, "concurrency": concurrency,
            "measure_s": measure_s, **window_stats(done, w0, w1),
            "engine": eng, "experts": share}


async def calibrate(engine, model: str, items: list[dict], *, concurrency: int,
                    seconds: float, max_tokens: int, record_dir: Path,
                    image: str, out: Path) -> dict:
    """Serve the calibration half (balanced, round-robin by domain) with
    the recorder on, dumping every DUMP_EVERY_S, and sum the dumps into
    a frequency placement file."""
    order = calibration_items(items)

    class Seq:
        i = 0

        def next(self):
            item = order[self.i % len(order)]
            self.i += 1
            return item
    before = seen_files(record_dir)
    await recorder(engine, "start")
    done: list[Request] = []
    until = time.monotonic() + seconds
    load = asyncio.create_task(closed_loop(engine.replica_urls, model, Seq(),
                                           concurrency=concurrency, until=until,
                                           max_tokens=max_tokens, done=done))
    while time.monotonic() < until:
        await asyncio.sleep(min(DUMP_EVERY_S, max(0.0, until - time.monotonic())))
        await recorder(engine, "dump")
    await load
    await recorder(engine, "stop")
    await asyncio.sleep(3)
    pairs = recorded_pairs(record_dir, before)
    info = aggregate(image, "freq", pairs, out)
    return {"freq_path": str(out), "dumps": len(pairs),
            "answers": sum(1 for r in done if not r.error and r.ended),
            "prompts_distinct": min(len(order), len(done)), **info}


def _mkparent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _engine_model(engine) -> str:
    return engine.api_model_name


async def run(plan: dict) -> Path:
    out_dir = Path(plan["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_prompt_set(plan["prompt_set"])
    res = Results(plan=plan, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    freq_path: str | None = None
    for cfg in plan["configs"]:
        name = cfg["name"]
        custom = dict(cfg["custom"])
        if custom.get("ktransformers_expert_freq_path") == "@calibration":
            prior = out_dir / "calibration" / "expert_freq.pt"
            if not freq_path and prior.is_file():
                # A rerun of the placement configs alone (--only) uses
                # the calibration an earlier run made.
                freq_path = str(prior)
            if not freq_path:
                res.errors.append({"config": name, "error": "no calibration file"})
                res.save(out_dir)
                continue
            custom["ktransformers_expert_freq_path"] = freq_path
        record_dir = None
        if custom.get("ktransformers_record_experts"):
            record_dir = out_dir / "record" / name
            custom["ktransformers_record_dir"] = str(record_dir)
        engine = None
        try:
            custom = resolve_gpu_experts(custom)
            engine = build_engine(custom, out_dir)
            image = getattr(engine.cfg, "ktransformers_image", None)
            if custom["engine"] == "ktransformers" and not image:
                from .engines.ktransformers_v2 import DEFAULT_IMAGE
                image = DEFAULT_IMAGE
            log.info("kt_placement: launching %s", name)
            await asyncio.to_thread(engine.launch, out_dir / "logs")
            model = _engine_model(engine)
            if cfg.get("calibrate"):
                res.calibration = await calibrate(
                    engine, model, items,
                    concurrency=int(plan.get("calibrate_concurrency", 128)),
                    seconds=float(plan.get("calibrate_s", 1500)),
                    max_tokens=int(plan.get("calibrate_max_tokens", 128)),
                    record_dir=record_dir, image=image,
                    out=_mkparent(out_dir / "calibration" / "expert_freq.pt"))
                res.calibration["config"] = name
                freq_path = res.calibration["freq_path"]
                res.save(out_dir)
            for mix in plan["mixes"]:
                for conc in cfg["concurrency"]:
                    w = await measure(
                        engine, model, items, mix, "eval", int(conc),
                        warmup_s=float(plan["warmup_s"]),
                        measure_s=float(plan["measure_s"]),
                        max_tokens=int(plan["max_tokens"]),
                        record_dir=record_dir, image=image, out_dir=out_dir,
                        seed=int(plan.get("seed", 7)))
                    w.update({"config": name,
                              "gpu_experts": custom.get("ktransformers_gpu_experts"),
                              "replicas": custom.get("replicas"),
                              "tp": custom.get("tp")})
                    res.windows.append(w)
                    res.save(out_dir)
                    log.info("kt_placement: %s %s c=%d -> %s tok/s (gauge %s), "
                             "cpu share %s", name, mix, conc, w["stream_tok_s"],
                             w["engine"]["gauge_tok_s"],
                             (w["experts"] or {}).get("cpu_share"))
        except Exception as e:  # noqa: BLE001 - one config must not end the test
            log.exception("kt_placement: %s failed", name)
            res.errors.append({"config": name, "error": f"{type(e).__name__}: {e}"[:2000]})
            res.save(out_dir)
        finally:
            if engine is not None:
                await asyncio.to_thread(engine.shutdown)
    res.save(out_dir)
    return out_dir / RESULTS_NAME


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m simulator.kt_placement")
    ap.add_argument("--plan", help="plan JSON (omit to print the example)")
    ap.add_argument("--only", help="comma-separated config names to run")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.plan:
        print(json.dumps(example_plan(), indent=1))
        return
    plan = json.loads(Path(a.plan).read_text())
    if a.only:
        keep = set(a.only.split(","))
        plan["configs"] = [c for c in plan["configs"] if c["name"] in keep]
    random.seed(plan.get("seed", 7))
    print(asyncio.run(run(plan)))


if __name__ == "__main__":
    main()
