"""Profile diagnostics and explicit backend/parity controls for P10."""

import gzip
import json
import math
import re
from collections import defaultdict

import httpx

from ..benchmarks.storage import read_json, write_json


def config(configuration, spec, arm, profile_directory=None):
    from .worker import arm_config

    # Reuse non-speculative controls without recursively selecting P10.
    neutral = spec.model_copy(update={"project_id": "P04", "speculative_concurrencies": []})
    cfg = arm_config(configuration, neutral, "baseline")
    e = cfg["engine"]
    clean = []
    flags = e["vllm_extra_flags"]
    i = 0
    managed = {"--attention-backend", "--kv-cache-dtype", "--dtype", "--profiler-config"}
    while i < len(flags):
        key = flags[i].split("=", 1)[0]
        if key in managed:
            if "=" not in flags[i]:
                i += 1
        else:
            clean.append(flags[i])
        i += 1
    backend = spec.attention_treatment if arm == "treatment" else spec.attention_baseline
    e.update(attention_backend=backend, kv_cache_dtype=spec.attention_kv_dtype)
    e["vllm_extra_flags"] = clean + [
        "--attention-backend",
        backend,
        "--kv-cache-dtype",
        spec.attention_kv_dtype,
        "--dtype",
        "bfloat16",
    ]
    if profile_directory is not None:
        profile_directory.mkdir(parents=True, exist_ok=True)
        e.setdefault("docker_volumes", {})[str(profile_directory.resolve())] = "/capsim-profile"
        e["vllm_extra_flags"] += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": "/capsim-profile",
                    "max_iterations": 4,
                    "ignore_frontend": True,
                    "torch_profiler_record_shapes": False,
                    "torch_profiler_with_stack": False,
                    "torch_profiler_with_memory": False,
                }
            ),
        ]
    return cfg


def verify_backend(directory, expected, expected_dtype=None):
    rows = []
    for f in sorted(directory.glob("replica-*/engine.log")):
        log = f.read_text(errors="replace")
        found = set(re.findall(r"Using AttentionBackendEnum\.([A-Z0-9_]+) backend", log))
        if found != {expected}:
            raise RuntimeError(
                "Attention backend unverified or silently changed: " + str(sorted(found))
            )
        observed = {"replica": f.parent.name, "observed_backend": expected}
        if expected_dtype is not None:
            dtypes = {
                {"float8_e4m3fn": "fp8"}.get(value, value)
                for value in re.findall(r"\bkv_cache_dtype=(?:torch\.)?([a-z0-9_]+)", log)
            }
            compute = set(re.findall(r"\bdtype=torch\.([a-z0-9_]+)", log))
            if dtypes != {expected_dtype} or compute != {"bfloat16"}:
                raise RuntimeError("Attention cache or compute precision unverified")
            observed.update(cache_dtype=expected_dtype, compute_dtype="bfloat16")
        rows.append(observed)
    if not rows:
        raise RuntimeError("Attention runtime logs unavailable")
    write_json(directory / "attention-verification.json", rows)


async def logits(engine, corpus, directory):
    records = []
    async with httpx.AsyncClient(timeout=120) as client:
        for item in corpus[:4]:
            response = await client.post(
                engine.replica_urls[0].rstrip("/").removesuffix("/v1") + "/v1/completions",
                headers={"Authorization": f"Bearer {engine.api_key}"},
                json={
                    "model": engine.api_model_name,
                    "prompt": item["prompt"],
                    "max_tokens": 1,
                    "temperature": 0,
                    "seed": 42,
                    "logprobs": 20,
                },
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            probs = choice["logprobs"]["top_logprobs"][0]
            if len(probs) < 10 or any(not math.isfinite(v) for v in probs.values()):
                raise RuntimeError("Next-token parity evidence incomplete")
            records.append(
                {
                    "task_id": item["id"],
                    "token": choice["logprobs"]["tokens"][0],
                    "top_logprobs": probs,
                }
            )
    write_json(directory / "next-token-logprobs.json", records)


def parity(path, spec):
    pairs = []
    for block in range(spec.paired_blocks):
        a = read_json(path / f"block-{block:02d}-baseline/next-token-logprobs.json")
        b = read_json(path / f"block-{block:02d}-treatment/next-token-logprobs.json")
        if len(a) != 4 or len(b) != 4:
            raise ValueError("Missing next-token parity probes")
        for x, y in zip(a, b, strict=True):
            common = set(x["top_logprobs"]) & set(y["top_logprobs"])
            overlap = len(common) / max(len(x["top_logprobs"]), len(y["top_logprobs"]))
            maximum = max(
                (abs(x["top_logprobs"][k] - y["top_logprobs"][k]) for k in common), default=math.inf
            )
            passed = (
                x["task_id"] == y["task_id"]
                and x["token"] == y["token"]
                and overlap >= 0.95
                and maximum <= spec.attention_logprob_tolerance
            )
            pairs.append(
                {
                    "block": block,
                    "task_id": x["task_id"],
                    "passed": passed,
                    "top_token_equal": x["token"] == y["token"],
                    "top20_overlap_fraction": overlap,
                    "max_common_logprob_difference": maximum if math.isfinite(maximum) else None,
                }
            )
    result = {
        "passed": all(p["passed"] for p in pairs),
        "probes": pairs,
        "scope": "Four first-token top-20 logprob probes per pair; not full hidden-state or broad model equivalence",
        "max_abs_logprob_tolerance": spec.attention_logprob_tolerance,
        "minimum_top20_overlap": 0.95,
    }
    write_json(path / "numerical-parity.json", result)
    return result


async def profile(engine, corpus, spec, directory):
    from .driver import measure_finite

    root = engine.replica_urls[0].rstrip("/").removesuffix("/v1")
    async with httpx.AsyncClient(
        timeout=300, headers={"Authorization": f"Bearer {engine.api_key}"}
    ) as client:
        r = await client.post(root + "/start_profile")
        r.raise_for_status()
        try:
            limited = spec.model_copy(
                update={"concurrency": 1, "max_output_tokens": 32, "min_requests": 1}
            )
            await measure_finite(engine, corpus[:1], limited, directory / "profile-requests.jsonl")
        finally:
            r = await client.post(root + "/stop_profile")
            r.raise_for_status()
    return summarize_profile(directory)


def summarize_profile(directory):
    durations = defaultdict(float)
    count = 0
    files = []
    for file in sorted((directory / "traces").rglob("*")):
        if not file.is_file() or not (
            file.name.endswith(".json") or file.name.endswith(".json.gz")
        ):
            continue
        if file.stat().st_size > 256 * 1024**2:
            raise RuntimeError("Profile exceeds bounded trace budget")
        opener = gzip.open if file.suffix == ".gz" else open
        with opener(file, "rt") as stream:
            text = stream.read(512 * 1024**2 + 1)
            if len(text) > 512 * 1024**2:
                raise RuntimeError("Expanded profile exceeds bounded trace budget")
            trace = json.loads(text)
        for event in trace.get("traceEvents", []):
            if event.get("cat") == "kernel" and event.get("ph") == "X" and event.get("dur", 0) > 0:
                durations[event["name"]] += event["dur"]
                count += 1
        files.append(str(file.relative_to(directory)))
    if not count:
        raise RuntimeError("No CUDA kernel events in profile; end-to-end benchmark not started")
    summary = {
        "kernel_events": count,
        "trace_files": files,
        "top_kernels": sorted(durations.items(), key=lambda x: -x[1])[:20],
        "scope": "One short-output request with max_iterations=4 configured; summed kernel duration can overlap and is not wall time. Profiled timings are excluded from A/B performance.",
    }
    write_json(directory / "profile-summary.json", summary)
    return summary
