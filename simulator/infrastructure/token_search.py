"""Roofline-compatible token-shape measurements without relaunching each shape."""

import asyncio
import concurrent.futures
import math
import resource
import time

import httpx

from ..engines.base import Engine


def ensure_file_limit(streams):
    required = max(16384, streams * 2 + 1024)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(f"Open-file hard limit {hard} is below required {required}")
    if soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    return dict(required=required, soft=resource.getrlimit(resource.RLIMIT_NOFILE)[0], hard=hard)


def aggregate_metrics(samples, roles):
    if len(samples) != len(roles) or not samples:
        return {"replicas_total": len(roles), "replicas_scraped": 0}
    result = {"replicas_total": len(roles), "replicas_scraped": len(samples)}
    for key in (
        "generation_tokens_total",
        "prompt_tokens_total",
        "num_running",
        "queue_depth",
        "preemptions_total",
        "kv_cache_tokens",
    ):
        # Prefill generates an intermediate token as part of handoff; do not count
        # that computation a second time as end-user output. Decode counters are
        # the headline output counter; all workers remain in activity evidence.
        selected = [
            s
            for s, r in zip(samples, roles, strict=True)
            if key != "generation_tokens_total" or r != "prefill"
        ]
        if key == "prompt_tokens_total" and "prefill" in roles:
            selected = [s for s, r in zip(samples, roles, strict=True) if r == "prefill"]
        values = [s.get(key) for s in selected]
        if values and all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values
        ):
            result[key] = sum(values)
    kv = [s["kv_cache_used_pct"] for s in samples if "kv_cache_used_pct" in s]
    if kv:
        result["kv_cache_used_pct"] = sum(kv) / len(kv)
    if "generation_tokens_total" not in result:
        result["replicas_scraped"] = 0
    return result


class CounterView:
    """Read every worker, while exposing the existing engine API to roofline."""

    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def get_metrics(self):
        def fetch(url):
            response = httpx.get(url, timeout=3)
            response.raise_for_status()
            return Engine._parse_prometheus(response.text)

        roles = self.engine.worker_roles or [
            "prefill" if i < self.engine.spec.prefill_workers else "decode"
            for i in range(len(self.engine.spec.gpu_ids))
        ]
        if self.engine.arm == "baseline":
            roles = ["aggregated"] * len(self.engine.spec.gpu_ids)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(roles)) as pool:
                samples = list(pool.map(fetch, self.engine.metrics_urls))
            return aggregate_metrics(samples, roles)
        except (httpx.HTTPError, ValueError):
            return {"replicas_total": len(roles), "replicas_scraped": 0}


async def wait_idle(engine, timeout=300):
    """Do not start a new shape while cancelled requests are still being served."""
    deadline = time.monotonic() + timeout
    consecutive = 0
    while time.monotonic() < deadline:
        metrics = await asyncio.to_thread(CounterView(engine).get_metrics)
        complete = metrics.get("replicas_scraped", 0) == metrics.get("replicas_total", -1)
        idle = complete and metrics.get("num_running") == 0 and metrics.get("queue_depth") == 0
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 2:
            return
        await asyncio.sleep(2)
    raise RuntimeError("Workers did not drain between token shapes; relaunch required")


def usable(rung):
    done = rung.get("samples", 0) + rung.get("no_content", 0) + rung.get("errors", 0)
    return bool(
        rung.get("steady_state")
        and (rung.get("out_tok_s") or 0) > 0
        and done >= 30
        and rung.get("errors", 0) / done <= 0.01
        and not rung.get("scrape_gaps")
    )


def compare(rows):
    pairs, latency, unresolved = [], [], []
    for row in rows:
        if not row["prefill"]:
            continue
        baseline = next(
            (
                b
                for b in rows
                if not b["prefill"] and b["shape"] == row["shape"] and b["block"] == row["block"]
            ),
            None,
        )
        if not baseline:
            continue
        a = [r for r in baseline["rungs"] if usable(r)]
        b = [r for r in row["rungs"] if usable(r)]
        if not a or not b:
            continue
        best_a, best_b = max(a, key=lambda r: r["out_tok_s"]), max(b, key=lambda r: r["out_tok_s"])
        common = {k: row[k] for k in ("shape", "prefill", "placement", "block")}
        faster_unsettled = [
            r
            for r in baseline["rungs"]
            if not r.get("steady_state")
            and (r.get("out_tok_s") or 0) > best_a["out_tok_s"]
            and usable(r | {"steady_state": True})
        ]
        if faster_unsettled:
            unresolved.append(
                dict(
                    common,
                    reason="A faster aggregate rung did not settle; extend baseline measurement before claiming uplift",
                    rungs=faster_unsettled,
                )
            )
        else:
            pairs.append(
                dict(
                    common,
                    baseline=best_a,
                    treatment=best_b,
                    uplift_percent=100 * (best_b["out_tok_s"] / best_a["out_tok_s"] - 1),
                )
            )
        for ar in a:
            for br in b:
                if (
                    ar["concurrency"] == br["concurrency"]
                    and ar.get("ttft_p95_ms")
                    and br.get("ttft_p95_ms")
                ):
                    latency.append(
                        dict(
                            common,
                            concurrency=ar["concurrency"],
                            baseline_p95_ms=ar["ttft_p95_ms"],
                            treatment_p95_ms=br["ttft_p95_ms"],
                            reduction_percent=100 * (1 - br["ttft_p95_ms"] / ar["ttft_p95_ms"]),
                        )
                    )
    return dict(
        qualified=False,
        reason="Exploratory matched token-shape saturation search; peaks are best observed within the tested grid. Fixed concurrency is not matched arrival rate.",
        comparisons=pairs,
        unresolved_baselines=unresolved,
        matched_concurrency_ttft=latency,
        peak_uplift=max(pairs, key=lambda p: p["uplift_percent"], default=None),
        metrics={},
    )
