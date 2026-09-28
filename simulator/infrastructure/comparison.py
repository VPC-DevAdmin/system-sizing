"""Compare paired blocks without dropping failed measurements or selecting best runs."""

import random
import statistics


def compare(results, spec):
    if spec.project_id == "P04" and spec.speculative_concurrencies:
        unexpected = [
            r for r in results if r.get("concurrency") not in spec.speculative_concurrencies
        ]
        if unexpected:
            raise ValueError("Unexpected speculative load cell")
        by_load = {
            str(c): compare(
                [r for r in results if r.get("concurrency") == c],
                spec.model_copy(update={"concurrency": c, "speculative_concurrencies": []}),
            )
            for c in spec.speculative_concurrencies
        }
        return {
            "qualified": all(c["qualified"] for c in by_load.values()),
            "reason": "Separate matched-load comparisons; no best-cell or roofline uplift claim",
            "scope": "Closed-loop concurrency grid; each comparison has its own gates and interval",
            "by_concurrency": by_load,
            "metrics": {},
        }
    pairs = {}
    for r in results:
        pair = pairs.setdefault(r["block"], {})
        if r["arm"] in pair:
            raise ValueError("Duplicate arm in a paired block")
        pair[r["arm"]] = r["metrics"]
    common = {
        "paired_blocks": len(pairs),
        "qualified": False,
        "scope": "One fixed closed-loop concurrency; not maximum capacity",
        "quality_scope": "Synthetic document JSON extraction only",
    }
    if spec.project_id == "P02":
        common["scope"] = (
            "Finite document-session resume cohort after fixed cache pressure; not maximum retained sessions"
        )
    if len(pairs) != spec.paired_blocks or any(
        set(p) != {"baseline", "treatment"} for p in pairs.values()
    ):
        return dict(common, reason="Incomplete paired experiment", metrics={})
    eligible = all(m["gates_passed"] for p in pairs.values() for m in p.values())
    common["qualified"] = eligible and spec.mode == "confirmation"
    common["reason"] = (
        "Confirmation gates passed"
        if common["qualified"]
        else "Pilot only"
        if eligible
        else "Request count, quality, error, or response-SLO gate failed"
    )
    output = {}
    rng = random.Random(spec.seed)
    metrics = [("qualified_per_hour", False), ("latency_p95_s", True)]
    if spec.project_id in ("P02", "P04"):
        metrics.append(("ttft_p95_s", True))
    for metric, reduction in metrics:
        a = [p["baseline"].get(metric) for p in pairs.values()]
        b = [p["treatment"].get(metric) for p in pairs.values()]
        invalid_a = any(x is None or x <= 0 for x in a)
        invalid_b = any(
            x is None or x < 0 or (x == 0 and metric != "qualified_per_hour") for x in b
        )
        if invalid_a or invalid_b:
            output[metric] = {"status": "insufficient_measurement"}
            continue

        def effect(indices, a=a, b=b, reduction=reduction):
            av = statistics.mean(a[i] for i in indices)
            bv = statistics.mean(b[i] for i in indices)
            return 100 * ((av - bv) / av if reduction else (bv - av) / av)

        samples = sorted(effect([rng.randrange(len(a)) for _ in a]) for _ in range(2000))
        output[metric] = {
            "baseline_mean": statistics.mean(a),
            "treatment_mean": statistics.mean(b),
            "change_percent": effect(range(len(a))),
            "positive_means": "improvement",
            "paired_bootstrap_95_interval": [samples[49], samples[1949]] if len(a) >= 5 else None,
            "method": "Percent change of means of paired block metrics; resample complete pairs",
        }
    return dict(common, metrics=output)
