"""Mature knowledge-worker sessions with one outstanding turn per worker.

Synthetic scored reference tasks; no claim of general writing/coding quality.
Priming is billed separately. Measured history grows with actual final answers.
"""

import asyncio
import hashlib
import json
import random
import time

import httpx

from ..benchmarks.storage import write_json
from .driver import percentile, request_one, summarize
from .workload import build_corpus


def token_count(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def make_session(tokenizer, spec, slot, block):
    seed = spec.seed + block * 100003 + slot
    doc = build_corpus(1, spec.document_records, seed, answer_records=1)[0]
    rows = [json.loads(line) for line in doc["prompt"].split("\nDocument:\n", 1)[1].splitlines()]
    prefix = f"Knowledge workspace {block}:{slot}. Reference records:\n" + "\n".join(
        json.dumps(r) for r in rows
    )
    target = spec.session_context_tokens
    # Pad only the initial mature history; subsequent requests preserve its exact bytes.
    note = (
        "\nReference note: preserve the supplied records; answer requests from their exact fields."
    )
    active = getattr(spec, "session_active_records", None)

    def padded(n):
        if active is None:
            return prefix + note * n
        # Explicit recent-working-set variant; older reference records remain.
        return (
            f"Knowledge workspace {block}:{slot}. Earlier workspace notes:\n"
            + note * n
            + "\nArchived reference records:\n"
            + "\n".join(json.dumps(row) for row in rows[:-active])
            + "\nActive reference records (current working set):\n"
            + "\n".join(json.dumps(row) for row in rows[-active:])
        )

    base = token_count(tokenizer, [{"role": "user", "content": padded(0)}])
    if base > target:
        raise ValueError("Reference fixture exceeds target; reduce document_records")
    estimate = max(1, len(tokenizer.encode(note, add_special_tokens=False)))
    lo = max(0, (target - base) // estimate)
    while token_count(tokenizer, [{"role": "user", "content": padded(lo)}]) < target:
        lo += 1
    while lo and token_count(tokenizer, [{"role": "user", "content": padded(lo - 1)}]) >= target:
        lo -= 1
    prefix = padded(lo)
    initial = token_count(tokenizer, [{"role": "user", "content": prefix}])
    if not target <= initial <= target + 64:
        raise ValueError("Initial fixture misses requested context target")
    return dict(prefix=prefix, rows=rows, messages=[], seed=seed, initial_tokens=initial)


def turn_item(session, tokenizer, spec, turn):
    selected = random.Random(session["seed"] + 70001 + turn).sample(
        session["rows"][-spec.session_active_records :]
        if getattr(spec, "session_active_records", None)
        else session["rows"],
        spec.session_answer_records,
    )
    question = (
        "Return only a JSON array containing all fields of these records, in this order: "
        + json.dumps([r["record_id"] for r in selected])
    )
    note = " Working note: retain exact values from the reference records."
    # A fixed small addition; tokenized and reported independently from context.
    while len(tokenizer.encode(question, add_special_tokens=False)) < spec.session_new_input_tokens:
        question = note + question
    if turn == 0:
        question = session["prefix"] + "\n\n" + question
    messages = session["messages"] + [{"role": "user", "content": question}]
    length = token_count(tokenizer, messages)
    if length + spec.max_output_tokens > spec.context_limit:
        raise ValueError("Session reached context limit; no silent compaction or truncation")
    return dict(
        id=f"s{session['seed']}-t{turn}",
        prompt=question,
        messages=messages,
        expected=selected,
        input_tokens=length,
        prompt_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
    )


def advance(session, item, result):
    # Preserve actual final responses, including incorrect ones, as a real session would.
    session["messages"] = item["messages"] + [{"role": "assistant", "content": result["output"]}]


def summarize_workers(rows, elapsed, spec, population):
    result = summarize(rows, elapsed, spec)
    slos = []
    for limit in spec.session_slo_grid_s:
        accepted = [
            r for r in rows if r["transport_ok"] and r["correct"] and r["elapsed_s"] <= limit
        ]
        slos.append(
            dict(
                response_slo_s=limit,
                accepted=len(accepted),
                attempted=len(rows),
                accepted_fraction=len(accepted) / len(rows) if rows else 0,
                passed=len(rows) >= spec.min_requests
                and len(accepted) / len(rows) >= spec.min_slo_fraction,
            )
        )
    result.update(
        population=population,
        activity_model="One outstanding turn, then exponential think time",
        think_time_mean_s=spec.session_think_s,
        slo_analysis=slos,
        final_answer_ttft_p50_s=percentile(
            [r["final_answer_ttft_s"] for r in rows if r.get("final_answer_ttft_s") is not None],
            0.5,
        ),
        final_answer_ttft_p95_s=percentile(
            [r["final_answer_ttft_s"] for r in rows if r.get("final_answer_ttft_s") is not None],
            0.95,
        ),
        mean_outstanding_requests=sum(
            max(0, min(spec.duration_s, r["finish_s"]) - r["arrival_s"]) for r in rows
        )
        / spec.duration_s,
        context_limit_failures=sum(r.get("error") == "context_limit" for r in rows),
        worker_turns={str(i): sum(r["session"] == i for r in rows) for i in range(population)},
        resident_document_count=None,
        measurement_scope="Mature-session phase; priming cost reported separately; drain included in throughput denominator",
        stable_capacity=False,
        service_capacity=False,
        saturation_observed=False,
        client_limited=population > spec.concurrency,
        trace_sha256=None,
    )
    return result


async def replay_workers(engine, spec, population, block, directory, tokenizer):
    directory.mkdir(parents=True, exist_ok=True)
    sessions = []
    for slot in range(population):
        sessions.append(await asyncio.to_thread(make_session, tokenizer, spec, slot, block))
    write_json(
        directory / "worker-fixtures.json",
        dict(
            population=population,
            initial_token_counts=[s["initial_tokens"] for s in sessions],
            context_target=spec.session_context_tokens,
            seed=spec.seed,
            active_reference_records=getattr(spec, "session_active_records", None),
            reference_records=spec.document_records,
            scope="Synthetic long-session extraction proxy; actual answers retained, no semantic-quality claim for general work",
        ),
    )
    if population > spec.concurrency:
        raise ValueError("Worker population exceeds client stream capacity")
    rows = []
    prime = []
    sem = asyncio.Semaphore(min(16, spec.concurrency))
    async with httpx.AsyncClient(
        timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
    ) as client:

        async def send(slot, turn):
            item = await asyncio.to_thread(turn_item, sessions[slot], tokenizer, spec, turn)
            result = await request_one(
                client,
                engine.replica_urls[slot % len(engine.replica_urls)],
                engine.api_model_name,
                item,
                spec,
                f"{slot}:{turn}",
                engine.api_key,
            )
            result.update(
                session=slot,
                turn=turn,
                input_tokens_checked=item["input_tokens"],
                prompt_sha256=item["prompt_sha256"],
            )
            advance(sessions[slot], item, result)
            return result

        prime_start = time.monotonic()

        async def initialize(slot):
            async with sem:
                result = await send(slot, 0)
                prime.append(result)
                with (directory / "prime-requests.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")

        await asyncio.gather(*(initialize(i) for i in range(population)))
        prime_metrics = summarize(prime, time.monotonic() - prime_start, spec)
        write_json(directory / "priming-metrics.json", prime_metrics)
        if (
            prime_metrics["quality_fraction"] < spec.min_quality
            or prime_metrics["error_fraction"] > spec.max_error_fraction
        ):
            raise RuntimeError("Knowledge-worker priming failed correctness/transport gate")
        await engine.capture(directory, "steady-before")
        start = time.monotonic()

        async def worker(slot):
            rng = random.Random(spec.seed + block * 100003 + slot + 900000)
            turn = 1
            while turn < spec.session_turns:
                pause = rng.expovariate(1 / spec.session_think_s)
                if time.monotonic() - start + pause >= spec.duration_s:
                    break
                await asyncio.sleep(pause)
                arrival = time.monotonic() - start
                try:
                    result = await send(slot, turn)
                except ValueError as exc:
                    result = dict(
                        session=slot,
                        turn=turn,
                        transport_ok=False,
                        correct=False,
                        ttft_s=None,
                        output="",
                        usage=None,
                        error="context_limit",
                        detail=str(exc),
                    )
                result.update(arrival_s=arrival, finish_s=time.monotonic() - start)
                result["elapsed_s"] = result["finish_s"] - arrival
                rows.append(result)
                with (directory / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
                write_json(
                    directory / "progress.json",
                    dict(at_s=time.monotonic() - start, completed=len(rows)),
                )
                if result.get("error") == "context_limit":
                    break
                turn += 1

        tasks = [asyncio.create_task(worker(i)) for i in range(population)]
        try:
            await asyncio.gather(*tasks)
            await asyncio.sleep(max(0, spec.duration_s - (time.monotonic() - start)))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start
        await engine.capture(directory, "steady-after")
    result = summarize_workers(rows, elapsed, spec, population)
    result["priming"] = prime_metrics
    return result


def compare_workers(results, spec):
    views = []
    for slo in spec.session_slo_grid_s:
        arms = {}
        for arm in ["baseline", "treatment"]:
            cells = [r for r in results if r["arm"] == arm]
            passing = []
            for row in cells:
                m = row["metrics"]
                view = next(
                    (s for s in m.get("slo_analysis", []) if s["response_slo_s"] == slo), {}
                )
                if (
                    view.get("passed")
                    and m.get("quality_fraction", 0) >= spec.min_quality
                    and m.get("error_fraction", 1) <= spec.max_error_fraction
                    and not m.get("context_limit_failures")
                ):
                    passing.append(row)
            best = max(passing, key=lambda r: r["population"], default=None)
            arms[arm] = dict(
                best_observed_population=best["population"] if best else None,
                correct_turns_per_minute=best["metrics"]["correct_requests_per_s"] * 60
                if best
                else None,
                cpu_reload_observed=best["metrics"].get("cpu_reload_observed", False)
                if best
                else False,
                tested_cells=len(cells),
            )
        views.append(dict(response_slo_s=slo, arms=arms))
    return dict(
        qualified=False,
        reason="Exploratory worker-population grid; confirm boundaries, repeat pairs and review actual prompt growth before claims",
        workload="Mature knowledge-worker proxy with retained actual answers",
        active_reference_records=getattr(spec, "session_active_records", None),
        context_target=spec.session_context_tokens,
        by_worker_slo=views,
        cache_work_by_cell=[
            dict(
                block=r["block"],
                arm=r["arm"],
                population=r["population"],
                **r["metrics"].get("cache_work", {}),
            )
            for r in results
        ],
        metrics={},
        boundary_resolved=False,
        token_savings_claim=None,
        energy_savings_claim=None,
    )


def render_worker_reports(path, manifest, state, comparison):
    from docx import Document

    for output in manifest["project"]["outputs"]:
        folder = path / "outputs" / output["id"]
        folder.mkdir(parents=True, exist_ok=True)
        write_json(
            folder / "claim.json",
            dict(
                status="not_qualified",
                positive_claim_supported=False,
                measurement=comparison,
                reason="Discovery only; no confirmed density, token or energy claim",
            ),
        )
        doc = Document()
        doc.add_heading("Knowledge-worker " + output["title"], 0)
        doc.add_paragraph(
            "Synthetic mature-session workload. Actual final answers are retained between turns. Each worker waits after a completed response. Initial context loading is reported separately. GPU-only and CPU-backed runs use the same model, cache sizing and activity policy; actual arrivals depend on completions."
        )
        doc.add_paragraph(
            "Active reference subset: "
            + str(manifest["spec"].get("session_active_records"))
            + ". When configured, questions target that many recent reference records at the end of the accumulated context. This is not a whole-history recall result."
        )
        doc.add_paragraph(
            "Requested initial context: "
            + str(manifest["spec"]["session_context_tokens"])
            + " tokens. Mean think time: "
            + str(manifest["spec"]["session_think_s"])
            + " seconds. Physical resident session counts and avoided GPU token processing are not inferred from cache blocks or logical prompt tokens."
        )
        if output["id"] == "P02-O3":
            for row in state.get("results", []):
                work = row["metrics"].get("cache_work", {})
                doc.add_paragraph(
                    f"{row['arm']}, {row['population']} workers: uncached prefix tokens per correct turn {work.get('uncached_prefix_tokens_per_correct_request')}; uncached share {work.get('uncached_prefix_share')}. This includes new input; historical recomputation is not yet attributable."
                )
        table = doc.add_table(rows=1, cols=6)
        for c, t in zip(
            table.rows[0].cells,
            ["Arm", "Workers", "Correct turns/min", "Quality", "Response p95 s", "CPU reload"],
            strict=True,
        ):
            c.text = t
        for row in state.get("results", []):
            m = row["metrics"]
            values = [
                row["arm"],
                row["population"],
                round(m["correct_requests_per_s"] * 60, 2),
                m["quality_fraction"],
                m["latency_p95_s"],
                m.get("cpu_reload_observed"),
            ]
            for c, v in zip(table.add_row().cells, values, strict=True):
                c.text = str(v)
        doc.add_paragraph(
            "Status: "
            + str(state.get("phase"))
            + ". "
            + str(
                state.get("error")
                or "Boundary confirmation and independent paired repeats remain required."
            )
        )
        doc.save(folder / "white-paper.docx")
