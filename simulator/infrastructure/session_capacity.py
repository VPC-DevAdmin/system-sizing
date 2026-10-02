"""Scheduled document Q&A capacity, including cold turns and bounded backlog.

Fixtures are synthetic extraction tasks, not a validation of broad document
analysis quality. No artificial eviction phase or response-paced arrival loop.
"""

import asyncio
import functools
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Mapping

import httpx

from ..benchmarks.storage import write_json
from .driver import request_one, summarize
from .workload import build_corpus


def make_trace(spec, population, rate, block):
    """Freeze independent questions after a stable, unique document prefix.

    Total offered rate is independent of completions. A slot changes document
    after session_turns questions; first touches and cold replacements are billed.
    """
    rng = random.Random(spec.seed + block * 100003 + population)
    counts = [int(spec.session_prime_documents)] * population
    cycle = []
    events, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= spec.duration_s:
            break
        if spec.session_prime_documents:
            if not cycle:
                cycle = list(range(population))
                rng.shuffle(cycle)
            slot = cycle.pop()
        else:
            slot = rng.randrange(population)
        turn = counts[slot]
        counts[slot] += 1
        events.append(
            dict(
                at_s=t,
                slot=slot,
                generation=0 if spec.session_prime_documents else turn // spec.session_turns,
                turn=turn if spec.session_prime_documents else turn % spec.session_turns,
            )
        )
        if len(events) > 100000:
            raise ValueError("Trace exceeds 100000 requests")
    if spec.session_prime_documents and min(counts) - 1 < spec.session_min_returns:
        raise ValueError(
            "Frozen trace lacks minimum returns for every document; increase duration or rate"
        )
    return events


@functools.lru_cache(maxsize=8192)
def document_prefix(spec_seed, records, identity):
    """Build a document once; every question on it reuses the identical prefix.

    A 1600-record document costs milliseconds to build, which would stall the
    open-loop client at high arrival rates if rebuilt per request.
    """
    seed = int.from_bytes(hashlib.sha256(f"{spec_seed}-{identity}".encode()).digest()[:4], "big")
    doc = build_corpus(1, records, seed, answer_records=1)[0]
    rows = [json.loads(line) for line in doc["prompt"].split("\nDocument:\n", 1)[1].splitlines()]
    # Put the session identity first so different sessions do not share most context.
    header = f"Document case {identity}.\n"
    records = "\n".join(json.dumps(r) for r in rows)
    return seed, header, records, tuple(rows)  # rows are read-only after this point


PADDING_NOTE = (
    "Background note: this workspace is used for document review. "
    "Preserve the reference records.\n"
)


def document_item(spec, event, block):
    identity = f"b{block}-s{event['slot']}-g{event['generation']}"
    seed, header, records, rows = document_prefix(spec.seed, spec.document_records, identity)
    # Prefix remains identical across independent questions on the same document.
    # Optional neutral padding stands for earlier working notes and grows the
    # context without adding records; the reference records stay adjacent to the
    # question, which is where a worker's active material sits and where exact
    # recall at 96K was measured to hold (mid-context placement lost ~10%).
    selected = random.Random(seed + event["turn"] + 1).sample(rows, spec.session_answer_records)
    prompt = (
        header
        + PADDING_NOTE * spec.session_document_pad_repeats
        + records
        + "\nQuestion "
        + str(event["turn"] + 1)
        + ": Extract these records in order, exactly as written. Return only a JSON array "
        "with every field and no additional prose. Requested record IDs: "
        + json.dumps([r["record_id"] for r in selected])
    )
    return dict(id=identity + f"-q{event['turn']}", prompt=prompt, expected=selected)


def padding_repeats(tokenizer, spec, block):
    """Smallest note count that lifts one document to the requested context size."""
    def length(n):
        probe = spec.model_copy(update={"session_document_pad_repeats": n})
        return prompt_token_count(
            tokenizer, document_item(probe, dict(slot=0, generation=0, turn=0), block)["prompt"]
        )

    target = spec.session_context_tokens
    if length(0) > target:
        raise ValueError("Document already exceeds the requested context; reduce document_records")
    lo, hi = 0, 1
    while length(hi) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if length(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def prompt_token_count(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if (
        not isinstance(encoded, (list, tuple))
        or not encoded
        or any(type(t) is not int for t in encoded)
    ):
        raise ValueError("Unexpected tokenizer input_ids; refusing an invalid context check")
    return len(encoded)


def trace_digest(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()


async def prime_documents(engine, spec, population, block, directory):
    """Bill first touches separately; no eviction phase or cache flush."""
    rows = []
    sem = asyncio.Semaphore(min(16, spec.concurrency))
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=spec.request_timeout_s) as client:

        async def one(slot):
            async with sem:
                item = document_item(spec, dict(slot=slot, generation=0, turn=0), block)
                result = await request_one(
                    client,
                    engine.replica_urls[slot % len(engine.replica_urls)],
                    engine.api_model_name,
                    item,
                    spec,
                    f"prime-{slot}",
                    engine.api_key,
                )
                result["session"] = slot
                rows.append(result)
                with (directory / "priming-requests.jsonl").open("a") as out:
                    out.write(json.dumps(result, allow_nan=False) + "\n")

        await asyncio.gather(*(one(slot) for slot in range(population)))
    metrics = summarize(rows, time.monotonic() - started, spec)
    write_json(directory / "priming-metrics.json", metrics)
    if (
        metrics["quality_fraction"] < spec.min_quality
        or metrics["error_fraction"] > spec.max_error_fraction
    ):
        raise RuntimeError("Document priming failed correctness/transport gate")
    await engine.capture(directory, "steady-before")
    return metrics


async def replay(engine, spec, trace, block, directory, clock=time.monotonic, sleep=asyncio.sleep):
    """Bounded open-loop replay; queue time and rejected arrivals stay visible."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "trace.json", trace)
    urls = list(engine.replica_urls)
    if not urls:
        raise ValueError("No serving endpoints")
    start = clock()
    rows, snapshots, tasks = [], [], set()
    sem = asyncio.Semaphore(spec.concurrency)
    pending = 0
    max_pending = 0
    client_lag = 0.0
    completed = 0

    (directory / "requests.jsonl").write_text("")

    def record(row):
        rows.append(row)
        with (directory / "requests.jsonl").open("a") as evidence:
            evidence.write(json.dumps(row, allow_nan=False) + "\n")

    async def sample():
        while True:
            snapshots.append(dict(at_s=clock() - start, outstanding=pending, completed=completed))
            write_json(directory / "progress.json", snapshots[-1])
            await sleep(min(1, spec.duration_s / 10))

    async with httpx.AsyncClient(
        timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
    ) as client:

        async def one(i, event):
            nonlocal pending, completed
            due = start + event["at_s"]
            row = dict(
                request_id=str(i),
                task_id=str(event["slot"]),
                transport_ok=False,
                correct=False,
                ttft_s=None,
                output="",
                usage=None,
            )
            try:
                # Deadline covers both admission waiting and the streaming request.
                async def send():
                    async with sem:
                        item = document_item(spec, event, block)
                        queued = clock() - due
                        row["dispatch_s"] = clock() - start
                        result = await request_one(
                            client,
                            urls[event["slot"] % len(urls)],
                            engine.api_model_name,
                            item,
                            spec,
                            str(i),
                            engine.api_key,
                        )
                        result["dispatch_s"] = row["dispatch_s"]
                        result["first_token_s"] = (
                            (row["dispatch_s"] + result["ttft_s"])
                            if result.get("ttft_s") is not None
                            else None
                        )
                        result["request_ttft_s"] = result.get("ttft_s")
                        if result.get("ttft_s") is not None:
                            result["ttft_s"] += queued
                        result["client_queue_s"] = queued
                        return result

                remaining = max(0.0, spec.request_timeout_s - (clock() - due))
                row = await asyncio.wait_for(send(), remaining)
            except TimeoutError:
                row["error"] = "deadline_including_admission"
            except asyncio.CancelledError:
                row["error"] = "cancelled"
                raise
            except Exception as exc:
                row["error"] = type(exc).__name__
            finally:
                finished = clock() - start
                row.update(
                    arrival_s=event["at_s"],
                    finish_s=finished,
                    elapsed_s=finished - event["at_s"],
                    session=event["slot"],
                    cold=event["turn"] == 0,
                    generation=event["generation"],
                )
                record(row)
                pending -= 1
                completed += int(row["transport_ok"])

        sampler = asyncio.create_task(sample())
        try:
            for i, event in enumerate(trace):
                await sleep(max(0, start + event["at_s"] - clock()))
                client_lag = max(client_lag, clock() - start - event["at_s"])
                if pending >= spec.session_max_pending:
                    record(
                        dict(
                            request_id=str(i),
                            task_id=str(event["slot"]),
                            transport_ok=False,
                            correct=False,
                            ttft_s=None,
                            output="",
                            usage=None,
                            error="admission_limit",
                            arrival_s=event["at_s"],
                            finish_s=clock() - start,
                            elapsed_s=clock() - start - event["at_s"],
                            cold=event["turn"] == 0,
                            session=event["slot"],
                            generation=event["generation"],
                        )
                    )
                    continue
                pending += 1
                max_pending = max(max_pending, pending)
                task = asyncio.create_task(one(i, event))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            await sleep(max(0, start + spec.duration_s - clock()))
            pending_at_end = pending
            snapshots.append(dict(at_s=spec.duration_s, outstanding=pending, completed=completed))
            if tasks:
                await asyncio.gather(*list(tasks))
        finally:
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*list(tasks), return_exceptions=True)
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            (directory / "requests.jsonl").write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
            )
            write_json(directory / "backlog.json", snapshots)
    elapsed = clock() - start
    result = summarize(rows, elapsed, spec)
    # Use the latter half for sustained-load diagnostics; all-time billing still includes cold turns.
    window = [r for r in rows if spec.duration_s / 2 <= r["finish_s"] <= spec.duration_s]
    good = [r for r in window if r["transport_ok"] and r["correct"]]
    xs = [s for s in snapshots if spec.duration_s / 2 <= s["at_s"] <= spec.duration_s]
    mean_t = sum(x["at_s"] for x in xs) / len(xs) if xs else 0
    denom = sum((x["at_s"] - mean_t) ** 2 for x in xs)
    slope = sum((x["at_s"] - mean_t) * x["outstanding"] for x in xs) / denom if denom else None
    usage = [(r.get("usage") or {}).get("completion_tokens") for r in good]
    result["error_counts"] = dict(
        Counter(
            r.get("error") or "incorrect_answer"
            for r in rows
            if not (r["transport_ok"] and r["correct"])
        )
    )
    result.update(
        trace_sha256=trace_digest(trace),
        scheduled=len(trace),
        cold_requests=sum(bool(r.get("cold")) for r in rows),
        pending_at_end=pending_at_end,
        max_pending=max_pending,
        client_max_lag_s=client_lag,
        backlog_slope_requests_s=slope,
        sustained_correct_requests_s=len(good) / (spec.duration_s / 2),
        sustained_output_tokens_s=sum(usage) / (spec.duration_s / 2)
        if usage and all(type(v) is int and v >= 0 for v in usage)
        else None,
        final_window_completions=len(good),
        output_scope="Completion tokens on correct responses; reasoning may be included by the engine",
    )
    offered = len(trace) / spec.duration_s
    result["repeat_requests"] = len(rows) - result["cold_requests"]
    client_queue_fraction = (
        sum(x["outstanding"] > spec.concurrency for x in xs) / len(xs) if xs else 0
    )
    result["client_queue_sample_fraction"] = client_queue_fraction
    result["client_limited"] = bool(
        client_queue_fraction > 0.1
        or client_lag > spec.session_client_lag_s
        or any(r.get("error") == "admission_limit" for r in rows)
    )
    final_arrivals = sum(spec.duration_s / 2 <= r["arrival_s"] < spec.duration_s for r in rows)
    deadline_failures = sum(
        spec.duration_s / 2 <= r["arrival_s"] < spec.duration_s
        and r.get("error") == "deadline_including_admission"
        for r in rows
    )
    result["final_window_deadline_failures"] = deadline_failures
    # Timed-out requests leave the backlog, so a flat slope alone can hide overload;
    # final-window arrivals that never finished inside the deadline count as well.
    result["overload_observed"] = bool(
        (slope is not None and slope > max(0.01, offered * spec.session_queue_growth_fraction))
        or (final_arrivals and deadline_failures / final_arrivals >= 0.05)
    )
    result["stable_capacity"] = bool(
        slope is not None
        and slope <= max(0.01, offered * spec.session_queue_growth_fraction)
        and not result["client_limited"]
        and not any(r.get("error") == "admission_limit" for r in rows)
        and result["error_fraction"] <= spec.max_error_fraction
        and result["quality_fraction"] >= spec.min_quality
        and len(good) >= spec.min_requests
    )
    result["final_window_offered_requests_s"] = final_arrivals / (spec.duration_s / 2)
    result["completion_to_arrival_ratio"] = len(good) / final_arrivals if final_arrivals else None
    successful = [r for r in rows if r["transport_ok"]]
    correct_completed = sum(r["correct"] for r in successful) / len(successful) if successful else 0
    result["saturation_observed"] = bool(
        result["overload_observed"]
        and not result["client_limited"]
        and final_arrivals
        and len(good) / final_arrivals < 0.95
        and correct_completed >= spec.min_quality
    )
    from .session_saturation import analyze_requests

    result.update(analyze_requests(rows, spec))
    result["service_capacity"] = result["stable_capacity"] and result["gates_passed"]
    return result


def compare_capacity(results, spec):
    """Compare active document populations at an identical question cadence.

    A failed client, missing cell or quality failure cannot establish saturation.
    CPU reload evidence is required to attribute capacity to CPU memory.
    """
    if spec.session_workload == "knowledge_worker":
        from .knowledge_workers import compare_workers

        return compare_workers(results, spec)
    if spec.session_arrival_rates:
        from .session_saturation import compare_saturation

        return compare_saturation(results, spec)
    summaries = []
    for cadence in spec.session_rate_grid:
        blocks = []
        for block in range(spec.paired_blocks):
            arms = {}
            for arm in ["baseline", "treatment"]:
                cells = [
                    x
                    for x in results
                    if x["block"] == block and x["arm"] == arm and x["per_session_rate"] == cadence
                ]
                complete = sorted(x["population"] for x in cells) == spec.session_population_grid
                passing = [x for x in cells if x["metrics"].get("stable_capacity")]
                best = max(passing, key=lambda x: x["population"], default=None)
                top = next(
                    (x for x in cells if x["population"] == max(spec.session_population_grid)), None
                )
                m = top["metrics"] if top else {}
                bounded = bool(
                    complete
                    and best
                    and m.get("overload_observed")
                    and not m.get("client_limited")
                    and m.get("quality_fraction", 0) >= spec.min_quality
                )
                arms[arm] = dict(best=best, boundary_resolved=bounded, complete=complete)
            a, b = arms["baseline"]["best"], arms["treatment"]["best"]
            matched = True
            for pop in spec.session_population_grid:
                pair = [
                    x
                    for x in results
                    if x["block"] == block
                    and x["per_session_rate"] == cadence
                    and x["population"] == pop
                ]
                if (
                    len(pair) != 2
                    or {x["arm"] for x in pair} != {"baseline", "treatment"}
                    or len({x["metrics"]["trace_sha256"] for x in pair}) != 1
                ):
                    matched = False
            attributed = bool(b and b["metrics"].get("cpu_reload_observed"))
            blocks.append(
                dict(
                    block=block,
                    arms=arms,
                    matched_traces=matched,
                    cpu_reload_at_selected_capacity=attributed,
                    exploratory_capacity_uplift_percent=100
                    * (b["population"] / a["population"] - 1)
                    if spec.paired_blocks >= 2 and a and b and matched
                    else None,
                    comparison_ready=bool(
                        matched
                        and attributed
                        and all(x["boundary_resolved"] for x in arms.values())
                    ),
                )
            )
        summaries.append(
            dict(
                per_session_rate=cadence, mean_seconds_between_questions=1 / cadence, blocks=blocks
            )
        )
    return dict(
        qualified=False,
        reason="Exploratory capacity curves; boundary refinement, confirmation and representative task validation required",
        scope="Active document sessions at equal per-session activity; not simultaneous GPU streams. Synthetic extraction, including cold turns.",
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
        by_cadence=summaries,
        metrics={},
    )
