"""Self-paced document users: population ladders and return surges.

The open-loop replay raises a request rate over a fixed population. These two
modes make population the variable instead:

* ``think``: every user has one outstanding question, then an exponential think
  time before the next one. The offered load follows from the population.
* ``surge``: every user is primed and idle; after a quiet period a share of them
  ask one question inside a short window. Shares use disjoint users.

Think draws and surge membership are seeded per user, so both arms see the same
plan. Arrival times still differ between arms because a slow answer delays the
next question; that delay is the measurement, not an error.
"""

import asyncio
import hashlib
import json
import random
import time
from collections import Counter

import httpx

from ..benchmarks.storage import write_json
from .driver import percentile, request_one, summarize
from .session_capacity import document_item

WAIT_TARGETS_S = (2, 5, 10, 30, 60)
TIMEOUT = "deadline_including_admission"


def pacing_plan(spec, population, block):
    plan = dict(pacing=spec.session_pacing, population=population, block=block, seed=spec.seed)
    if spec.session_pacing == "think":
        plan.update(think_time_mean_s=spec.session_think_s, duration_s=spec.duration_s)
        return plan
    rng = random.Random(spec.seed + block * 100003 + population + 500000)
    users = list(range(population))
    rng.shuffle(users)
    surges, taken = [], 0
    for share in spec.session_surge_shares:
        count = max(1, round(share * population))
        chosen = users[taken : taken + count]
        taken += count
        surges.append(
            dict(
                share=share,
                users=chosen,
                offsets_s=[rng.uniform(0, spec.session_surge_window_s) for _ in chosen],
            )
        )
    plan.update(
        quiet_s=spec.session_surge_quiet_s, window_s=spec.session_surge_window_s, surges=surges
    )
    return plan


def wait_views(rows, spec):
    """Waits over a cohort. Failed or timed-out questions stay in every fraction.

    Percentiles censor a question without a first token at the request deadline,
    so an overloaded arm cannot look better by answering fewer questions.
    """
    n = len(rows)
    good = [r for r in rows if r["transport_ok"] and r["correct"]]

    def censored(key):
        return [
            r[key] if r["transport_ok"] and r.get(key) is not None else spec.request_timeout_s
            for r in rows
        ]

    def within(key):
        return {
            str(t): (sum(r.get(key) is not None and r[key] <= t for r in good) / n if n else 0)
            for t in WAIT_TARGETS_S
        }

    return dict(
        questions=n,
        correct_fraction=len(good) / n if n else 0,
        timeout_fraction=sum(r.get("error") == TIMEOUT for r in rows) / n if n else 0,
        first_token_p50_s=percentile(censored("ttft_s"), 0.5),
        first_token_p95_s=percentile(censored("ttft_s"), 0.95),
        answer_p50_s=percentile(censored("elapsed_s"), 0.5),
        answer_p95_s=percentile(censored("elapsed_s"), 0.95),
        within_first_token=within("ttft_s"),
        within_answer=within("elapsed_s"),
        censored_at_s=spec.request_timeout_s,
    )


def measured(rows, spec):
    """Think mode scores the final half, as the open-loop ladder did; surges score everything."""
    if spec.session_pacing == "surge":
        return list(rows)
    return [r for r in rows if spec.duration_s / 2 <= r["arrival_s"] < spec.duration_s]


def paced_metrics(rows, snapshots, marks, plan, elapsed, started_at, spec):
    result = summarize(rows, elapsed, spec)
    cohort = measured(rows, spec)
    good = [r for r in cohort if r["transport_ok"] and r["correct"]]
    waits = wait_views(cohort, spec)
    window = elapsed if spec.session_pacing == "surge" else spec.duration_s / 2
    result["error_counts"] = dict(
        Counter(
            r.get("error") or "incorrect_answer"
            for r in rows
            if not (r["transport_ok"] and r["correct"])
        )
    )
    quality_ok = (
        bool(cohort)
        and waits["correct_fraction"] >= spec.min_quality
        and sum(not r["transport_ok"] for r in cohort) / len(cohort) <= spec.max_error_fraction
    )
    on_target = sum(
        r.get("ttft_s") is not None and r["ttft_s"] <= spec.session_ttft_slo_s for r in good
    )
    result.update(
        pacing=spec.session_pacing,
        started_at=started_at,
        measured_window_s=window,
        trace_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        scheduled=len(rows),
        cold_requests=0,
        repeat_requests=len(rows),
        pending_at_end=snapshots[-1]["outstanding"] if snapshots else 0,
        max_pending=max((s["outstanding"] for s in snapshots), default=0),
        mean_outstanding_requests=sum(s["outstanding"] for s in snapshots) / len(snapshots)
        if snapshots
        else None,
        client_max_lag_s=0.0,
        client_limited=False,
        backlog_slope_requests_s=None,
        final_window_completions=len(good),
        sustained_correct_requests_s=len(good) / window if window > 0 else None,
        waits=waits,
        slo_analysis=[
            dict(
                response_slo_s=t,
                arrival_cohort=len(cohort),
                accepted=round(waits["within_answer"][str(t)] * len(cohort)),
                accepted_fraction=waits["within_answer"][str(t)],
                passed=len(cohort) >= spec.min_requests
                and waits["within_answer"][str(t)] >= spec.min_slo_fraction,
            )
            for t in WAIT_TARGETS_S
        ],
        overload_observed=waits["timeout_fraction"] >= 0.05,
        saturation_observed=waits["timeout_fraction"] >= 0.05,
        stable_capacity=bool(quality_ok and len(good) >= spec.min_requests),
        measurement_scope="Final half of the window by arrival"
        if spec.session_pacing == "think"
        else "Every surge question; quiet periods carry no questions",
        activity_model="One outstanding question, then exponential think time"
        if spec.session_pacing == "think"
        else "Idle users released inside a short window",
    )
    result["gates_passed"] = bool(
        result["stable_capacity"] and on_target / max(1, len(cohort)) >= spec.min_slo_fraction
    )
    result["service_capacity"] = result["gates_passed"]
    if spec.session_pacing == "think":
        asked = Counter(r["session"] for r in rows)
        result["questions_per_user_min"] = min(
            (asked.get(i, 0) for i in range(plan["population"])), default=0
        )
        result["think_time_mean_s"] = spec.session_think_s
    else:
        result["surges"] = [
            dict(
                mark,
                share=plan["surges"][mark["surge"]]["share"],
                **wait_views([r for r in rows if r.get("surge") == mark["surge"]], spec),
            )
            for mark in marks
        ]
    return result


async def replay_paced(
    engine, spec, population, block, directory, clock=time.monotonic, sleep=asyncio.sleep
):
    directory.mkdir(parents=True, exist_ok=True)
    urls = list(engine.replica_urls)
    if not urls:
        raise ValueError("No serving endpoints")
    plan = pacing_plan(spec, population, block)
    write_json(directory / "trace.json", plan)
    (directory / "requests.jsonl").write_text("")
    rows, snapshots, marks = [], [], []
    state = dict(outstanding=0, completed=0)
    started_at = time.time()
    start = clock()

    async def sample():
        while True:
            snapshots.append(dict(at_s=clock() - start, **state))
            write_json(directory / "progress.json", snapshots[-1])
            await sleep(1)

    async with httpx.AsyncClient(
        timeout=spec.request_timeout_s,
        limits=httpx.Limits(max_connections=max(spec.concurrency, population)),
    ) as client:

        async def ask(slot, turn, **extra):
            arrival = clock() - start
            request_id = f"{slot}-{turn}"
            row = dict(
                request_id=request_id,
                task_id=str(slot),
                transport_ok=False,
                correct=False,
                ttft_s=None,
                output="",
                usage=None,
            )
            state["outstanding"] += 1
            try:
                item = document_item(spec, dict(slot=slot, generation=0, turn=turn), block)
                row = await asyncio.wait_for(
                    request_one(
                        client,
                        urls[slot % len(urls)],
                        engine.api_model_name,
                        item,
                        spec,
                        request_id,
                        engine.api_key,
                    ),
                    spec.request_timeout_s,
                )
            except (TimeoutError, asyncio.TimeoutError):
                row["error"] = TIMEOUT
            except asyncio.CancelledError:
                row["error"] = "cancelled"
                raise
            except Exception as exc:
                row["error"] = type(exc).__name__
            finally:
                state["outstanding"] -= 1
                finish = clock() - start
                row.update(
                    arrival_s=arrival,
                    dispatch_s=arrival,
                    finish_s=finish,
                    elapsed_s=finish - arrival,
                    session=slot,
                    turn=turn,
                    cold=False,
                    generation=0,
                    client_queue_s=0.0,
                    request_ttft_s=row.get("ttft_s"),
                    **extra,
                )
                state["completed"] += int(row["transport_ok"])
                rows.append(row)
                with (directory / "requests.jsonl").open("a") as evidence:
                    evidence.write(json.dumps(row, allow_nan=False) + "\n")

        async def thinker(slot):
            # The first think time staggers users, so they never start in step.
            rng = random.Random(spec.seed + block * 100003 + slot + 700000)
            turn = 1  # turn 0 was the priming question
            while True:
                pause = rng.expovariate(1 / spec.session_think_s)
                if clock() - start + pause >= spec.duration_s:
                    return
                await sleep(pause)
                await ask(slot, turn)
                turn += 1

        async def surge(index, group):
            await sleep(spec.session_surge_quiet_s)
            began = clock() - start

            async def one(slot, offset):
                await sleep(offset)
                await ask(slot, 1, surge=index)

            await asyncio.gather(
                *(one(s, o) for s, o in zip(group["users"], group["offsets_s"], strict=True))
            )
            marks.append(
                dict(
                    surge=index,
                    users=len(group["users"]),
                    start_s=began,
                    time_to_clear_s=clock() - start - began,
                )
            )

        sampler = asyncio.create_task(sample())
        try:
            if spec.session_pacing == "think":
                await asyncio.gather(*(thinker(slot) for slot in range(population)))
                await sleep(max(0, spec.duration_s - (clock() - start)))
            else:
                for index, group in enumerate(plan["surges"]):
                    await surge(index, group)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            write_json(directory / "backlog.json", snapshots)
    return paced_metrics(rows, snapshots, marks, plan, clock() - start, started_at, spec)


def compare_paced(results, spec):
    """Users served per arm at each first-token target: the largest rung that
    passes with every smaller tested rung also passing. Nothing is extrapolated;
    an arm whose top rung passes has an unresolved limit."""
    served = []
    for block in sorted({r["block"] for r in results}):
        for target in WAIT_TARGETS_S:
            arms = {}
            for arm in ("baseline", "treatment"):
                points = sorted(
                    (r for r in results if r["block"] == block and r["arm"] == arm),
                    key=lambda r: r["population"],
                )
                best = None
                for r in points:
                    waits = r["metrics"].get("waits") or {}
                    passed = (
                        waits.get("within_first_token", {}).get(str(target), 0)
                        >= spec.min_slo_fraction
                        and waits.get("correct_fraction", 0) >= spec.min_quality
                    )
                    if not passed:
                        break
                    best = r["population"]
                arms[arm] = dict(
                    users_served=best,
                    tested=[r["population"] for r in points],
                    limit_resolved=bool(points) and best != points[-1]["population"],
                )
            a, b = arms["baseline"]["users_served"], arms["treatment"]["users_served"]
            served.append(
                dict(
                    block=block,
                    first_token_target_s=target,
                    arms=arms,
                    users_ratio=b / a if a and b else None,
                )
            )
    return dict(
        qualified=False,
        reason="Exploratory population ladder; single block, synthetic extraction task",
        scope="Users with one outstanding question and a stated think time, or a stated "
        "return surge. Measured rungs only; no extrapolation beyond the top rung.",
        pacing=spec.session_pacing,
        think_time_mean_s=spec.session_think_s if spec.session_pacing == "think" else None,
        users_served=served,
        surges=[
            dict(block=r["block"], arm=r["arm"], population=r["population"], **s)
            for r in results
            for s in r["metrics"].get("surges", [])
        ],
        by_cadence=[],
        metrics={},
    )
