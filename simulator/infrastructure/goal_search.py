"""Bounded joint search. Decisions depend only on retained measured cells."""

import math
import random
import statistics


def rate_grid(spec):
    low, high = spec.p01_request_rates[0], spec.p01_request_rates[-1]
    count = math.floor((high - low) / spec.p01_rate_resolution)
    return sorted(
        set(spec.p01_request_rates)
        | {round(low + i * spec.p01_rate_resolution, 6) for i in range(count + 1)}
    )


def client_concurrency(spec, rate):
    # Admit all arrivals that could be simultaneously inside the completion SLO.
    # Preserve the explicit user floor and bound sockets at the adapter limit.
    ceiling = spec.p01_client_max_concurrency
    horizon = spec.request_timeout_s if ceiling > 512 else spec.response_slo_s
    required = math.ceil(rate * min(spec.duration_s, horizon))
    return min(ceiling, max(spec.concurrency, required))


def shapes(spec):
    return [
        dict(id=f"d{d}-a{a}", records=d, answers=a)
        for d in spec.p01_document_grid
        for a in spec.p01_answer_grid
    ]


def next_rate(grid, measured):
    """Probe endpoints then narrow a monotonic bracket; never extrapolate."""
    if grid[0] not in measured:
        return grid[0]
    if not measured[grid[0]]:
        return None
    if grid[-1] not in measured:
        return grid[-1]
    if measured[grid[-1]]:
        return None
    low = max(i for i, rate in enumerate(grid) if measured.get(rate) is True)
    high = min(i for i, rate in enumerate(grid) if rate > grid[low] and measured.get(rate) is False)
    if high - low <= 1:
        return None
    return grid[(low + high) // 2]


def non_monotonic(measured):
    passing = [rate for rate, passed in measured.items() if passed]
    return bool(passing) and any(not ok and rate < max(passing) for rate, ok in measured.items())


def next_search_rate(spec, measured):
    """Measure shared curve points before refining each capacity boundary."""
    for rate in spec.p01_request_rates:
        if rate not in measured:
            return rate
    if non_monotonic(measured):
        return None
    return next_rate(rate_grid(spec), measured)


def windows(spec, cell, rows):
    ordered = shapes(spec)
    random.Random(spec.seed + cell["block"]).shuffle(ordered)
    for shape in ordered:
        while True:
            observed = {
                r["rate"]: r["metrics"]["gates_passed"]
                for r in rows
                if r["block"] == cell["block"]
                and r["prefill"] == cell["prefill"]
                and r.get("shape") == shape["id"]
                and r.get("placement", "pooled") == cell.get("placement", "pooled")
            }
            rate = next_search_rate(spec, observed)
            if rate is None:
                break
            yield dict(shape, rate=rate)


def compare_goal(rows, spec):
    keys = [
        (r["block"], r["prefill"], r.get("placement", "pooled"), r["shape"], r["rate"])
        for r in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate goal search cell")
    envelopes, capacity, latency = [], [], []
    complete = True
    for shape in shapes(spec):
        by_split = {}
        allocations = [(0, "aggregate")] + [
            (p, placement) for p in spec.p01_prefill_grid for placement in spec.p01_placement_grid
        ]
        for prefill, placement in allocations:
            blocks = []
            for block in range(spec.paired_blocks):
                cells = [
                    r
                    for r in rows
                    if r["shape"] == shape["id"]
                    and r["prefill"] == prefill
                    and r["block"] == block
                    and (prefill == 0 or r.get("placement", "pooled") == placement)
                ]
                observed = {r["rate"]: r["metrics"]["gates_passed"] for r in cells}
                done = next_search_rate(spec, observed) is None
                complete &= done
                passed = [rate for rate, ok in observed.items() if ok]
                low = max(passed) if passed else None
                failed_above = [
                    rate
                    for rate, ok in observed.items()
                    if not ok
                    and low is not None
                    and rate > low
                    and not next(
                        c["metrics"].get("client_limit_detected", False)
                        for c in cells
                        if c["rate"] == rate
                    )
                ]
                blocks.append(
                    dict(
                        passing_rate=low,
                        failing_rate=min(failed_above) if failed_above else None,
                        search_finished=done,
                        non_monotonic=non_monotonic(observed),
                    )
                )
            irregular = any(b["non_monotonic"] for b in blocks)
            valid = not irregular and all(
                b["passing_rate"] is not None and b["search_finished"] for b in blocks
            )
            passing = min(b["passing_rate"] for b in blocks) if valid else None
            bounded = valid and all(b["failing_rate"] is not None for b in blocks)
            env = dict(
                shape=shape["id"],
                document_records=shape["records"],
                answer_records=shape["answers"],
                prefill=prefill,
                placement=placement,
                passing_rate=passing,
                failing_rate=max(b["failing_rate"] for b in blocks) if bounded else None,
                boundary="non_monotonic"
                if irregular
                else "bracketed"
                if bounded
                else "client_limited"
                if valid and passing < rate_grid(spec)[-1]
                else "grid_ceiling"
                if valid
                else "no_passing_boundary",
                blocks=blocks,
            )
            envelopes.append(env)
            by_split[prefill, placement] = env
        base = by_split[0, "aggregate"]
        for p, placement in allocations[1:]:
            treatment = by_split[p, placement]
            if base["passing_rate"] and treatment["passing_rate"]:
                capacity.append(
                    dict(
                        shape=shape["id"],
                        prefill=p,
                        placement=placement,
                        baseline_rate=base["passing_rate"],
                        treatment_rate=treatment["passing_rate"],
                        observed_uplift_percent=100
                        * (treatment["passing_rate"] / base["passing_rate"] - 1),
                        baseline_boundary=base["boundary"],
                        treatment_boundary=treatment["boundary"],
                        conservative_uplift_percent=100
                        * (treatment["passing_rate"] / base["failing_rate"] - 1)
                        if base["failing_rate"]
                        else None,
                    )
                )
            for rate in rate_grid(spec):
                a = [
                    r["metrics"]
                    for r in rows
                    if r["shape"] == shape["id"] and r["prefill"] == 0 and r["rate"] == rate
                ]
                b = [
                    r["metrics"]
                    for r in rows
                    if r["shape"] == shape["id"]
                    and r["prefill"] == p
                    and r["rate"] == rate
                    and r.get("placement", "pooled") == placement
                ]
                if len(a) == len(b) == spec.paired_blocks and all(m["gates_passed"] for m in a + b):
                    av = statistics.mean(m["latency_p95_s"] for m in a)
                    bv = statistics.mean(m["latency_p95_s"] for m in b)
                    latency.append(
                        dict(
                            shape=shape["id"],
                            prefill=p,
                            placement=placement,
                            rate=rate,
                            baseline_p95_s=av,
                            treatment_p95_s=bv,
                            reduction_percent=100 * (1 - bv / av),
                        )
                    )
    # A baseline still passing at the upper grid edge cannot establish an uplift peak.
    candidates = [
        c
        for c in capacity
        if c["conservative_uplift_percent"] is not None and c["conservative_uplift_percent"] > 0
    ]
    absolute = [e for e in envelopes if e["prefill"] and e["passing_rate"]]
    for candidate in [*capacity, *absolute, *latency]:
        matching = [
            r["metrics"]
            for r in rows
            if r["shape"] == candidate["shape"]
            and r["prefill"] == candidate["prefill"]
            and r.get("placement", "pooled") == candidate["placement"]
        ]
        for key in ("prompt_tokens", "output_tokens"):
            lows = [m[key + "_min"] for m in matching if m.get(key + "_min") is not None]
            highs = [m[key + "_max"] for m in matching if m.get(key + "_max") is not None]
            candidate[key + "_range"] = [min(lows), max(highs)] if lows and highs else None
    result = dict(
        qualified=False,
        reason="Exploratory measured peaks within the declared search; token lengths and service targets apply to every claim.",
        metrics={},
        complete_search=complete,
        envelopes=envelopes,
        capacity_comparisons=capacity,
        matched_latency=latency,
        peak_capacity_uplift=max(
            candidates, key=lambda c: c["conservative_uplift_percent"], default=None
        ),
        peak_absolute_capacity=max(absolute, key=lambda e: e["passing_rate"], default=None),
        peak_latency_reduction=max(
            (x for x in latency if x["reduction_percent"] > 0),
            key=lambda c: c["reduction_percent"],
            default=None,
        ),
    )

    if spec.p01_cuda_launch_blocking:
        result.update(
            diagnostic_only=True,
            reason="CUDA launch blocking is enabled for fault localization. These timings are diagnostic only.",
            peak_capacity_uplift=None,
            peak_absolute_capacity=None,
            peak_latency_reduction=None,
        )
    return result
