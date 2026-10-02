"""Open-loop saturation, exact occupancy integration and repeatable SLO analysis."""

import math
import random

from .driver import percentile


def analyze_requests(rows, spec):
    start, end = spec.duration_s / 2, spec.duration_s
    cohort = [r for r in rows if start <= r["arrival_s"] < end]

    def occupancy(selected, begin="arrival_s"):
        return sum(
            max(0, min(end, r["finish_s"]) - max(start, r.get(begin, r["finish_s"])))
            for r in selected
            if r.get("error") != "admission_limit"
        ) / (end - start)

    slos = []
    for limit in spec.session_slo_grid_s:
        accepted = [
            r for r in cohort if r["transport_ok"] and r["correct"] and r["elapsed_s"] <= limit
        ]
        slos.append(
            dict(
                response_slo_s=limit,
                arrival_cohort=len(cohort),
                accepted=len(accepted),
                accepted_fraction=len(accepted) / len(cohort) if cohort else 0,
                passed=len(cohort) >= spec.min_requests
                and len(accepted) / len(cohort) >= spec.min_slo_fraction,
            )
        )
    by_doc = {}
    for r in rows:
        key = f"{r.get('session', r.get('task_id', 'unknown'))}:{r.get('generation', 0)}"
        entry = by_doc.setdefault(
            key, dict(attempted=0, correct=0, returns=0, reported_cached_requests=0)
        )
        entry["attempted"] += 1
        entry["correct"] += int(r["transport_ok"] and r["correct"])
        entry["returns"] += int(not r.get("cold", True))
        cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        entry["reported_cached_requests"] += int(type(cached) is int and cached > 0)
    return dict(
        mean_outstanding_requests=occupancy(rows),
        mean_dispatched_requests=occupancy(rows, "dispatch_s"),
        client_queue_p95_s=percentile(
            [r["client_queue_s"] for r in rows if "client_queue_s" in r], 0.95
        ),
        slo_analysis=slos,
        document_outcomes=by_doc,
        resident_document_count=None,
        residency_note="Distinct resident contexts are not exposed by block counters; no document count inferred from bytes.",
        measurement_timeout_s=spec.request_timeout_s,
    )


def _groups(results, spec):
    for block in range(spec.paired_blocks):
        for pop in spec.session_population_grid:
            yield block, pop, [r for r in results if r["block"] == block and r["population"] == pop]


def saturated_pair_tail(rows):
    """Two successive overload points per arm avoid declaring a single noisy dip a knee."""
    for arm in ["baseline", "treatment"]:
        points = sorted(
            [r for r in rows if r["arm"] == arm and not r["metrics"].get("client_limited")],
            key=lambda r: r["rate"],
        )
        if len(points) < 2 or not all(r["metrics"].get("saturation_observed") for r in points[-2:]):
            return False
    return True


def next_cells(results, spec, refinements):
    """Extend paired rates to the bounded ceiling, then bisect each arm's bracket."""
    if not spec.session_arrival_rates or getattr(spec, "mode", "pilot") == "confirmation":
        return []
    for block, pop, rows in _groups(results, spec):
        rates = sorted({r["rate"] for r in rows})
        if not rates:
            continue
        if any(sum(r["rate"] == rate for r in rows) != 2 for rate in rates):
            raise ValueError("Unpaired saturation evidence")
        add = []
        key = f"{block}:{pop}"
        if not saturated_pair_tail(rows):
            limited = [r["rate"] for r in rows if r["metrics"].get("client_limited")]
            if limited:
                backoff = key + ":client-boundary"
                if refinements.get(backoff, 0) < 5:
                    high = min(limited)
                    low = max((rate for rate in rates if rate < high), default=0)
                    if low and high / low > 1.05:
                        add = [round((high + low) / 2, 6)]
                    refinements[backoff] = refinements.get(backoff, 0) + 1
            elif rates[-1] < spec.session_max_arrival_rate:
                add = [min(spec.session_max_arrival_rate, rates[-1] * 2)]
        elif refinements.get(key, 0) < spec.session_refinement_rounds:
            for arm in ["baseline", "treatment"]:
                points = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["rate"])
                last_stable = max(
                    (r["rate"] for r in points if r["metrics"].get("stable_capacity")), default=0
                )
                for high in points:
                    if high["rate"] <= last_stable or not high["metrics"].get(
                        "saturation_observed"
                    ):
                        continue
                    lows = [
                        r
                        for r in points
                        if r["rate"] < high["rate"] and r["metrics"].get("stable_capacity")
                    ]
                    if lows:
                        low = lows[-1]["rate"]
                        if high["rate"] / low > 1.1:
                            add.append(round((low + high["rate"]) / 2, 6))
                        break
            refinements[key] = refinements.get(key, 0) + 1
        add = sorted(set(add) - set(rates))
        if add:
            cells = []
            for rate in add:
                arms = ["baseline", "treatment"]
                random.Random(f"{spec.seed}:{block}:{pop}:{rate}").shuffle(arms)
                cells.extend(
                    dict(
                        block=block,
                        population=pop,
                        rate=rate,
                        per_session_rate=rate / pop,
                        arm=arm,
                        adaptive=True,
                    )
                    for arm in arms
                )
            return cells
    return []


def compare_saturation(results, spec):
    groups = []
    for block, pop, rows in _groups(results, spec):
        arms = {}
        rates = sorted({r["rate"] for r in rows})
        paired = bool(rates) and all(
            len(pair := [r for r in rows if r["rate"] == rate]) == 2
            and {r["arm"] for r in pair} == {"baseline", "treatment"}
            and len({r["metrics"]["trace_sha256"] for r in pair}) == 1
            for rate in rates
        )
        for arm in ["baseline", "treatment"]:
            points = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["rate"])
            stable = [r for r in points if r["metrics"].get("stable_capacity")]
            best = max(
                stable, key=lambda r: r["metrics"]["sustained_correct_requests_s"], default=None
            )
            overload = [r for r in points if r["metrics"].get("saturation_observed")]
            bracket = None
            last_stable = max((r["rate"] for r in stable), default=0)
            for high in overload:
                if high["rate"] <= last_stable:
                    continue
                lows = [r for r in stable if r["rate"] < high["rate"]]
                if lows:
                    bracket = [max(r["rate"] for r in lows), high["rate"]]
                    break
            queue_knee = False
            if bracket:
                low = next(r for r in points if r["rate"] == bracket[0])["metrics"].get(
                    "native_queue_mean_s"
                )
                high = next(r for r in points if r["rate"] == bracket[1])["metrics"].get(
                    "native_queue_mean_s"
                )
                queue_knee = (
                    low is not None and high is not None and high > low + max(0.001, low * 0.5)
                )
            slo_capacities = []
            for limit in spec.session_slo_grid_s:
                eligible = [
                    r
                    for r in stable
                    if any(
                        x["response_slo_s"] == limit and x["passed"]
                        for x in r["metrics"].get("slo_analysis", [])
                    )
                ]
                selected = max(
                    eligible,
                    key=lambda r: r["metrics"]["sustained_correct_requests_s"],
                    default=None,
                )
                slo_capacities.append(dict(response_slo_s=limit, selected=selected))
            arms[arm] = dict(
                best=best,
                knee_bracket_requests_s=bracket,
                native_queue_knee_supported=queue_knee,
                slo_capacities=slo_capacities,
            )
        ready = (
            paired
            and saturated_pair_tail(rows)
            and all(
                a["knee_bracket_requests_s"] and a["native_queue_knee_supported"]
                for a in arms.values()
            )
        )
        slo_pairs = []
        for index, limit in enumerate(spec.session_slo_grid_s):
            a, b = [
                arms[arm]["slo_capacities"][index]["selected"] for arm in ["baseline", "treatment"]
            ]
            eligible = bool(
                spec.paired_blocks >= 2
                and ready
                and a
                and b
                and b["metrics"].get("cpu_reload_observed")
            )
            av = a["metrics"]["mean_outstanding_requests"] if a else 0
            bv = b["metrics"]["mean_outstanding_requests"] if b else 0
            slo_pairs.append(
                dict(
                    response_slo_s=limit,
                    concurrent_request_change_percent=100 * (bv / av - 1)
                    if eligible and av > 0
                    else None,
                    throughput_change_percent=100
                    * (
                        b["metrics"]["sustained_correct_requests_s"]
                        / a["metrics"]["sustained_correct_requests_s"]
                        - 1
                    )
                    if eligible and a["metrics"]["sustained_correct_requests_s"] > 0
                    else None,
                    note="Concurrency is time-weighted outstanding work at sustainable throughput; compare response distributions as well.",
                )
            )
        groups.append(
            dict(
                block=block,
                population=pop,
                arms=arms,
                matched_traces=paired,
                saturation_resolved=bool(ready),
                slo_comparisons=slo_pairs,
            )
        )
    # Separate supported session-population curve, at a common activity level.
    population_curves = []
    cadences = sorted({round(r["rate"] / r["population"], 10) for r in results})
    for cadence in cadences:
        for block in range(spec.paired_blocks):
            for limit in spec.session_slo_grid_s:
                maxima = {}
                for arm in ["baseline", "treatment"]:
                    candidates = [
                        r
                        for r in results
                        if r["arm"] == arm
                        and r["block"] == block
                        and round(r["rate"] / r["population"], 10) == cadence
                        and r["metrics"].get("stable_capacity")
                        and any(
                            x["response_slo_s"] == limit and x["passed"]
                            for x in r["metrics"].get("slo_analysis", [])
                        )
                    ]
                    maxima[arm] = max((r["population"] for r in candidates), default=None)
                population_curves.append(
                    dict(
                        block=block,
                        questions_per_session_s=cadence,
                        response_slo_s=limit,
                        largest_passing_tested_population=maxima,
                    )
                )
    return dict(
        qualified=False,
        reason="Discovery curves; confirm a refined boundary before publication",
        repeated_load_comparisons=repeated_load_comparisons(results, spec),
        cache_work_by_cell=[
            dict(
                block=r["block"],
                arm=r["arm"],
                population=r["population"],
                rate=r["rate"],
                **r["metrics"].get("cache_work", {}),
            )
            for r in results
        ],
        by_cadence=[],
        by_working_set=groups,
        document_population_curves=population_curves,
        resident_document_count=None,
        scope="Open-loop saturation and supported reusable-document populations; physical document residency unavailable",
        metrics={},
    )


def cache_work_metrics(evidence, correct_requests):
    """Exact metric families only. Residual queries are not historical recomputation.

    vLLM excludes preempted queries from ordinary prefix stats. New input also
    contributes to the residual. Never silently relabel it as avoided GPU FLOPs.
    """
    families = (
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "external_prefix_cache_hits_total",
    )
    totals = dict.fromkeys(families, 0.0)
    valid = bool(evidence)
    for replica in evidence:
        by_label = {}
        for key, value in replica.get("worker_deltas", {}).items():
            # Stored key: <endpoint suffix>:vllm:<family>{labels}.
            metric = key.split("vllm:", 1)[-1]
            family = metric.split("{", 1)[0]
            if family not in families:
                continue
            label = key.split("vllm:", 1)[0] + metric[len(family) :]
            by_label.setdefault(label, {})[family] = value
        if not by_label:
            valid = False
        for values in by_label.values():
            # CPU hits must be present in treatment snapshots. Missing counters
            # are unknown, including lazy metrics before their first observation.
            required = families if replica.get("cpu_backend") else families[:2]
            if any(k not in values for k in required):
                valid = False
            q, g, c = (values.get(k, 0) for k in families)
            if (
                any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                    for v in (q, g, c)
                )
                or g + c > q
            ):
                valid = False
            for k in families:
                totals[k] += values.get(k, 0)
    q, g, c = (totals[k] for k in families)
    residual = q - g - c if valid else None
    return dict(
        cache_work=dict(
            prefix_query_tokens=q if valid else None,
            gpu_hit_tokens=g if valid else None,
            cpu_hit_tokens=c if valid else None,
            uncached_prefix_tokens=residual,
            uncached_prefix_tokens_per_correct_request=residual / correct_requests
            if valid and correct_requests
            else None,
            uncached_prefix_share=residual / q if valid and q else None,
            recomputed_historical_tokens_per_correct_request=None,
            recomputed_historical_share=None,
            accounting_valid=valid,
            scope="Ordinary prefix query residual, including new input; excludes separately counted preemptions. Historical recomputation is unavailable without request-level attribution.",
        )
    )


def paired_block_interval(pairs, seed=42):
    """Bootstrap independent blocks, never individual requests from one launch.

    At least five fixed-design pairs are needed for an interval; discovery peak
    selection still prevents these intervals from qualifying a marketing claim.
    """
    import math

    if any(not (math.isfinite(a) and math.isfinite(b) and a > 0 and b >= 0) for a, b in pairs):
        raise ValueError("Invalid paired throughput")
    n = len(pairs)
    observed = 100 * (sum(b for a, b in pairs) / sum(a for a, b in pairs) - 1) if n >= 2 else None
    interval = None
    if n >= 5:
        rng = random.Random(seed)
        samples = []
        for _ in range(5000):
            draw = [pairs[rng.randrange(n)] for _ in range(n)]
            samples.append(100 * (sum(b for a, b in draw) / sum(a for a, b in draw) - 1))
        interval = [percentile(samples, 0.025), percentile(samples, 0.975)]
    return dict(
        independent_pairs=n,
        throughput_change_percent=observed,
        bootstrap_95_percent=interval,
        method="Paired block percentile bootstrap, ratio of mean throughputs; conditional on tested fixed design",
        qualified=False,
    )


def repeated_load_comparisons(results, spec):
    """Summarize the same rate/population across independent launches, not maxima."""
    summaries = []
    for pop, rate in sorted({(r["population"], r["rate"]) for r in results}):
        pairs = []
        excluded = []
        for block in range(spec.paired_blocks):
            rows = [
                r for r in results if (r["population"], r["rate"], r["block"]) == (pop, rate, block)
            ]
            by_arm = {r["arm"]: r["metrics"] for r in rows}
            if (
                len(rows) != 2
                or set(by_arm) != {"baseline", "treatment"}
                or len({m.get("trace_sha256") for m in by_arm.values()}) != 1
                or not all(
                    m.get("trace_sha256") and m.get("stable_capacity") for m in by_arm.values()
                )
                or not by_arm["treatment"].get("cpu_reload_observed")
            ):
                excluded.append(block)
                continue
            pairs.append(
                tuple(by_arm[a]["sustained_correct_requests_s"] for a in ("baseline", "treatment"))
            )
        # Do not discard failing blocks and report an uplift from survivors.
        summary = (
            paired_block_interval(pairs, spec.seed)
            if not excluded
            else paired_block_interval([], spec.seed)
        )
        summaries.append(
            dict(population=pop, arrival_rate=rate, excluded_blocks=excluded, **summary)
        )
    return summaries
