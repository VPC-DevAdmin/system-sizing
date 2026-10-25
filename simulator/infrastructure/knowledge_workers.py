"""Knowledge-worker sessions with one outstanding turn per worker.

Stage 1 (flat): a mature history at a fixed size with small per-turn growth.
Stage 2 (growing, docs/p02-stage2-compaction.md): each turn adds a work item,
probes sample the worker's whole history, no-compaction arms reset at the
engine limit, compaction arms summarize on the Xeon between a trigger and a
target. Synthetic scored reference tasks; no claim of general writing/coding
quality. Priming is billed separately. Measured history grows with actual
final answers.
"""

import asyncio
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import httpx

from ..benchmarks.storage import write_json
from .driver import percentile, request_one, summarize
from .session_compaction import (
    apply_compaction,
    compact,
    empty_ledger,
    summarizer_timer_delta,
    summarizer_timers,
)
from .workload import build_corpus

PROBE_SCHEMAS = {
    "session_recall": {
        "type": "object",
        "properties": {
            "turn": {"type": "integer"},
            "record_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["turn", "record_ids"],
        "additionalProperties": False,
    },
    "knowledge_update": {
        "type": "object",
        "properties": {
            "record_id": {"type": "string"},
            "allocation": {"type": ["integer", "null"]},
        },
        "required": ["record_id", "allocation"],
        "additionalProperties": False,
    },
}
PROBE_SCHEMAS["abstention"] = PROBE_SCHEMAS["knowledge_update"]

WORK_ITEM_SENTENCE = (
    "Routine workspace activity: reviewed the shipment board, confirmed labels, "
    "filed the inspection notes and cleared the pending queue for this interval."
)


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
    return dict(
        prefix=prefix,
        rows=rows,
        messages=[],
        seed=seed,
        initial_tokens=initial,
        # Stage 2 state. Facts are ground truth that follows the worker, never the window.
        facts=[],
        archive=[],
        summaries=[],
        ledger=empty_ledger(),
        cycle=0,
        resets=0,
        history_start=0,
        compacted_through=-1,
        pending=None,
    )


def growing(spec):
    return getattr(spec, "session_growth_tokens", 0) > 0


def probe_type(session, spec, turn):
    """Rotate through the configured mix; fall back when no fact exists yet."""
    mix = list(getattr(spec, "session_probe_mix", ["reference"]))
    if turn == 0 or not growing(spec):
        return "reference"
    if turn == session.get("prime_turns", 0):
        # The first real request after a synthetic history is the priming request and
        # is gated on the extraction task, so it is always a reference probe.
        return "reference"
    # Offset by worker so every turn carries every probe type across the population.
    kind = mix[(turn - 1 + session.get("seed", 0)) % len(mix)]
    earlier = [f for f in session.get("facts", []) if f["turn"] < turn]
    if kind == "session_recall" and not any(f["kind"] == "requested" for f in earlier):
        return "reference"
    if kind == "knowledge_update" and not any(f["kind"] == "update" for f in earlier):
        return "reference"
    return kind


def work_item(tokenizer, session, spec, turn, rng):
    """New material for the turn: a session note (a fact) plus neutral padding."""
    rows = session["rows"]
    row = rng.choice(rows)
    value = rng.randrange(1, 10000)
    # The allocation is session state, not a reference field, so reference probes are
    # never in conflict with a note.
    note = f"Session note (turn {turn}): the working allocation for record {row['record_id']} is now {value}."
    session["facts"].append(
        dict(turn=turn, kind="update", record_id=row["record_id"], allocation=value)
    )
    target = spec.session_growth_tokens
    sentence = len(tokenizer.encode(WORK_ITEM_SENTENCE, add_special_tokens=False))
    repeats = max(
        0, (target - len(tokenizer.encode(note, add_special_tokens=False))) // max(1, sentence)
    )
    return "Work item:\n" + " ".join([WORK_ITEM_SENTENCE] * repeats) + "\n" + note


def reference_question(session, spec, turn, rng, tokenizer):
    pool = (
        session["rows"][-spec.session_active_records :]
        if getattr(spec, "session_active_records", None)
        else session["rows"]
    )
    selected = rng.sample(pool, spec.session_answer_records)
    question = (
        "Return only a JSON array containing all fields of these records, in this order: "
        + json.dumps([r["record_id"] for r in selected])
    )
    note = " Working note: retain exact values from the reference records."
    # A fixed small addition; tokenized and reported independently from context.
    while len(tokenizer.encode(question, add_special_tokens=False)) < spec.session_new_input_tokens:
        question = note + question
    session["facts"].append(
        dict(turn=turn, kind="requested", record_ids=[r["record_id"] for r in selected])
    )
    return question, selected, None, None


def history_question(session, kind, turn, rng):
    """Probes drawn uniformly over the worker's entire history, lost or not."""
    earlier = [f for f in session["facts"] if f["turn"] < turn]
    if kind == "session_recall":
        fact = rng.choice([f for f in earlier if f["kind"] == "requested"])
        question = (
            f"Which record IDs were requested at turn {fact['turn']} of this session? "
            f'Return only a JSON object {{"turn": {fact["turn"]}, "record_ids": [...]}} '
            "listing them in the order requested."
        )
        return question, dict(turn=fact["turn"], record_ids=fact["record_ids"]), fact["turn"]
    if kind == "knowledge_update":
        updates = [f for f in earlier if f["kind"] == "update"]
        record_id = rng.choice(sorted({f["record_id"] for f in updates}))
        latest = max((f for f in updates if f["record_id"] == record_id), key=lambda f: f["turn"])
        question = (
            f"According to the session notes, what is the current working allocation for record "
            f'{record_id}? Return only a JSON object {{"record_id": "{record_id}", "allocation": N}}; '
            "use null if no session note has set it."
        )
        return (
            question,
            dict(record_id=record_id, allocation=latest["allocation"]),
            latest["turn"],
        )
    if kind == "abstention":
        noted = {f["record_id"] for f in session["facts"] if f["kind"] == "update"}
        candidates = [r["record_id"] for r in session["rows"] if r["record_id"] not in noted]
        record_id = rng.choice(candidates) if candidates else f"R{rng.randrange(9000, 9999)}-9999"
        question = (
            f"According to the session notes, what is the current working allocation for record "
            f'{record_id}? Return only a JSON object {{"record_id": "{record_id}", "allocation": N}}; '
            "use null if no session note has set it."
        )
        return question, dict(record_id=record_id, allocation=None), None
    raise ValueError("Unknown probe type")


def turn_item(session, tokenizer, spec, turn):
    rng = random.Random(session["seed"] + 70001 + turn)
    kind = probe_type(session, spec, turn)
    schema = None
    fact_turn = None
    if kind == "reference":
        question, expected, _, _ = reference_question(session, spec, turn, rng, tokenizer)
    else:
        question, expected, fact_turn = history_question(session, kind, turn, rng)
        schema = PROBE_SCHEMAS[kind]
    body = question
    if growing(spec) and turn > 0:
        body = (
            f"Turn {turn}.\n" + work_item(tokenizer, session, spec, turn, rng) + "\n\n" + question
        )
    elif growing(spec):
        body = f"Turn {turn}.\n" + question
    reset = False
    if not session["messages"]:
        content = session["prefix"] + "\n\n" + body
    else:
        content = body
    messages = session["messages"] + [{"role": "user", "content": content}]
    length = token_count(tokenizer, messages)
    budget = getattr(spec, "session_limit_tokens", None) or (
        spec.context_limit - spec.max_output_tokens
    )
    if length > budget:
        if getattr(spec, "session_limit_policy", "fail") == "new_session" and session["messages"]:
            # Realistic no-compaction behaviour: keep the prefix, drop the history.
            # Facts stay in the log, so probes about them are still asked and scored.
            session["messages"] = []
            session["resets"] += 1
            session["history_start"] = turn
            reset = True
            content = session["prefix"] + "\n\n" + body
            messages = [{"role": "user", "content": content}]
            length = token_count(tokenizer, messages)
        else:
            raise ValueError("Session reached context limit; no silent compaction or truncation")
    return dict(
        id=f"s{session['seed']}-t{turn}",
        prompt=content,
        body=body,
        messages=messages,
        expected=expected,
        schema=schema,
        probe=kind,
        fact_turn=fact_turn,
        fact_age=None if fact_turn is None else turn - fact_turn,
        reset=reset,
        cycle=session.get("cycle", 0),
        resets=session.get("resets", 0),
        history_turns=turn - session.get("history_start", 0),
        input_tokens=length,
        prompt_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
    )


def phase_turns(spec, compaction, rng):
    """How many synthetic turns put this worker at a random point of its cycle."""
    if not getattr(spec, "session_prime_phase", False):
        return 0
    if compaction == "xeon":
        upper = spec.session_compaction_trigger_tokens
    else:
        upper = getattr(spec, "session_limit_tokens", None) or (
            spec.context_limit - spec.max_output_tokens
        )
    per_turn = spec.session_growth_tokens + spec.session_new_input_tokens + 256
    span = max(1, int((upper - spec.session_context_tokens) / per_turn) - 1)
    return rng.randrange(0, span)


def prime_history(session, tokenizer, spec, turns):
    """Synthetic prior turns with fabricated correct answers; facts and archive are real."""
    for turn in range(turns):
        item = turn_item(session, tokenizer, spec, turn)
        if item.get("reset"):
            raise ValueError("Phase priming crossed the context budget; reduce the span")
        advance(session, item, {"output": json.dumps(item["expected"])})
    session["prime_turns"] = turns
    return turns


def advance(session, item, result):
    # Preserve actual final responses, including incorrect ones, as a real session would.
    session["messages"] = item["messages"] + [{"role": "assistant", "content": result["output"]}]
    if "archive" in session:
        session["archive"].append(
            dict(
                turn=int(item["id"].rsplit("-t", 1)[1]),
                user=item.get("body", item["prompt"]),
                assistant=result["output"],
            )
        )


RECALL_PROBES = ("session_recall", "knowledge_update")


def recall_metrics(rows):
    probes = [r for r in rows if r.get("probe")]
    # Whole-history recall counts only probes whose answer is a session fact. Abstention
    # is a confabulation check: after a reset "nothing was set" is correct, so counting it
    # would flatter an arm that has just lost its history.
    history = [r for r in probes if r["probe"] in RECALL_PROBES]

    def fraction(items):
        return sum(bool(r["correct"]) for r in items) / len(items) if items else None

    by_type = {}
    for kind in sorted({r["probe"] for r in probes}):
        items = [r for r in probes if r["probe"] == kind]
        by_type[kind] = dict(
            attempted=len(items),
            correct=sum(bool(r["correct"]) for r in items),
            fraction=fraction(items),
        )

    # Cycle index: compactions so far in a compaction arm, resets so far in a reset arm.
    # Only one of the two is ever non-zero for a given arm.
    def cycle_of(r):
        return (r.get("cycle") or 0) + (r.get("resets") or 0)

    by_cycle = {}
    for cycle in sorted({cycle_of(r) for r in history}):
        items = [r for r in history if cycle_of(r) == cycle]
        by_cycle[str(cycle)] = dict(recall=fraction(items), attempted=len(items))
    # End state: what the user sees at the end of the run, the last four history
    # probes of every worker (about the final tenth of the turns).
    last = {}
    for r in sorted(history, key=lambda r: (r.get("session"), r["turn"])):
        last.setdefault(r.get("session"), []).append(r)
    tail = [r for rows_ in last.values() for r in rows_[-4:]]
    end_state = dict(recall=fraction(tail), attempted=len(tail))
    buckets = [(1, 4), (5, 8), (9, 16), (17, 10**9)]
    by_age = {}
    for lo, hi in buckets:
        items = [r for r in history if r.get("fact_age") is not None and lo <= r["fact_age"] <= hi]
        by_age[f"{lo}-{hi if hi < 10**9 else 'plus'}"] = dict(
            attempted=len(items), fraction=fraction(items)
        )
    before_after = dict(
        before_event=fraction([r for r in history if r.get("event_edge") == "before"]),
        after_event=fraction([r for r in history if r.get("event_edge") == "after"]),
    )
    # Intact versus behind a wall: is the probed fact older than the worker's last reset?
    last_reset = {}
    intact, behind = [], []
    for r in sorted(rows, key=lambda r: (r.get("session"), r["turn"])):
        if r.get("reset"):
            last_reset[r.get("session")] = r["turn"]
        if r.get("probe") in RECALL_PROBES and r.get("fact_turn") is not None:
            (behind if r["fact_turn"] < last_reset.get(r.get("session"), -1) else intact).append(r)
    intact_behind = dict(
        intact=fraction(intact),
        intact_attempted=len(intact),
        behind_reset=fraction(behind),
        behind_reset_attempted=len(behind),
    )
    timeline = []
    for turn in sorted({r.get("window_turn", r["turn"]) for r in rows}):
        at = [r for r in rows if r.get("window_turn", r["turn"]) == turn]
        live = [r.get("input_tokens_checked") for r in at if r.get("input_tokens_checked")]
        timeline.append(
            dict(
                turn=turn,
                live_tokens_mean=sum(live) / len(live) if live else None,
                whole_history_recall=fraction([r for r in at if r.get("probe") in RECALL_PROBES]),
                attempted=len(at),
                resets=sum(bool(r.get("reset")) for r in at),
                compactions_landed=sum(bool(r.get("compaction_landed")) for r in at),
            )
        )
    return dict(
        # The window average blends early intact history with late lost history and is
        # not what a user experiences; the curve (by_cycle, timeline) and end_state are.
        window_average_recall=fraction(history),
        end_state=end_state,
        reference_recall=fraction([r for r in probes if r["probe"] == "reference"]),
        abstention=fraction([r for r in probes if r["probe"] == "abstention"]),
        intact_vs_behind_reset=intact_behind,
        by_probe_type=by_type,
        by_cycle=by_cycle,
        by_fact_age=by_age,
        before_after=before_after,
        timeline=timeline,
        scope="Synthetic session-state probes over the worker's whole history; not general summarization quality",
    )


def compaction_metrics(compactions, rows):
    landed = [c for c in compactions if c["ok"]]
    pending_turns = sum(bool(r.get("compaction_pending")) for r in rows)
    waits = [r.get("compaction_wait_s") or 0.0 for r in rows]
    waited = [w for w in waits if w > 0]
    per_worker = {}
    for r in rows:
        per_worker[r.get("session")] = per_worker.get(r.get("session"), 0.0) + (
            r.get("compaction_wait_s") or 0.0
        )
    return dict(
        wait_turns=len(waited),
        wait_turn_fraction=len(waited) / len(rows) if rows else None,
        wait_p50_s=percentile(waited, 0.5),
        wait_p95_s=percentile(waited, 0.95),
        wait_total_s=sum(waits),
        wait_per_worker_mean_s=_mean(list(per_worker.values())) if per_worker else None,
        attempted=len(compactions),
        landed=len(landed),
        failed=len(compactions) - len(landed),
        lag_p50_s=percentile([c["lag_s"] for c in landed], 0.5),
        lag_p95_s=percentile([c["lag_s"] for c in landed], 0.95),
        wall_p50_s=percentile([c["wall_s"] for c in compactions], 0.5),
        raw_tokens_mean=_mean([(c.get("usage") or {}).get("prompt_tokens") for c in landed]),
        summary_tokens_mean=_mean(
            [(c.get("usage") or {}).get("completion_tokens") for c in landed]
        ),
        post_compaction_prefill_mean=_mean([c.get("post_compaction_tokens") for c in landed]),
        pending_turns=pending_turns,
        pending_turn_fraction=pending_turns / len(rows) if rows else None,
        turns_sent_while_pending_mean=_mean([c.get("turns_while_pending") for c in landed]),
    )


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def slo_views(rows, spec):
    views = []
    for limit in spec.session_slo_grid_s:
        accepted = [
            r for r in rows if r["transport_ok"] and r["correct"] and r["elapsed_s"] <= limit
        ]
        views.append(
            dict(
                response_slo_s=limit,
                accepted=len(accepted),
                attempted=len(rows),
                accepted_fraction=len(accepted) / len(rows) if rows else 0,
                passed=len(rows) >= spec.min_requests
                and len(accepted) / len(rows) >= spec.min_slo_fraction,
            )
        )
    return views


def summarize_workers(rows, elapsed, spec, population, compactions=None):
    result = summarize(rows, elapsed, spec)
    slos = slo_views(rows, spec)
    if growing(spec):
        # Density is gated on the extraction task (reference probes) and the SLO.
        # Session-history recall is the second claim and is reported separately.
        task = [r for r in rows if r.get("probe") == "reference"]
        result["task_quality_fraction"] = (
            sum(bool(r["correct"]) for r in task) / len(task) if task else 0
        )
        result["slo_analysis_task"] = slo_views(task, spec)
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
        session_resets=sum(bool(r.get("reset")) for r in rows),
        turns_per_worker_mean=len(rows) / population if population else None,
        workers_completed=sum(
            1
            for i in range(population)
            if sum(r["session"] == i for r in rows) >= getattr(spec, "session_turns", 0) - 1
        ),
        wall_s_to_complete=max((r.get("finish_s") or 0) for r in rows) if rows else None,
        worker_turns={str(i): sum(r["session"] == i for r in rows) for i in range(population)},
        live_tokens_mean=_mean([r.get("input_tokens_checked") for r in rows]),
        resident_document_count=None,
        measurement_scope="Mature-session phase; priming cost reported separately; drain included in throughput denominator",
        stable_capacity=False,
        service_capacity=False,
        saturation_observed=False,
        client_limited=population > spec.concurrency,
        trace_sha256=None,
    )
    if growing(spec):
        result["recall"] = recall_metrics(rows)
        result["compaction"] = compaction_metrics(compactions or [], rows)
    return result


async def replay_workers(engine, spec, population, block, directory, tokenizer, compaction="none"):
    directory.mkdir(parents=True, exist_ok=True)
    sessions = []
    phase_rng = random.Random(spec.seed + block * 100003 + 424242)
    for slot in range(population):
        session = await asyncio.to_thread(make_session, tokenizer, spec, slot, block)
        turns = phase_turns(spec, compaction, phase_rng)
        if turns:
            await asyncio.to_thread(prime_history, session, tokenizer, spec, turns)
        sessions.append(session)
    write_json(
        directory / "worker-fixtures.json",
        dict(
            population=population,
            initial_token_counts=[s["initial_tokens"] for s in sessions],
            context_target=spec.session_context_tokens,
            seed=spec.seed,
            active_reference_records=getattr(spec, "session_active_records", None),
            reference_records=spec.document_records,
            growth_tokens=getattr(spec, "session_growth_tokens", 0),
            prime_phase=bool(getattr(spec, "session_prime_phase", False)),
            prime_turns=[s.get("prime_turns", 0) for s in sessions],
            probe_mix=list(getattr(spec, "session_probe_mix", ["reference"])),
            limit_policy=getattr(spec, "session_limit_policy", "fail"),
            compaction=compaction,
            compaction_trigger_tokens=getattr(spec, "session_compaction_trigger_tokens", None),
            compaction_target_tokens=getattr(spec, "session_compaction_target_tokens", None),
            scope="Synthetic long-session extraction proxy; actual answers retained, no semantic-quality claim for general work",
        ),
    )
    if population > spec.concurrency:
        raise ValueError("Worker population exceeds client stream capacity")
    rows = []
    prime = []
    compactions = []
    sem = asyncio.Semaphore(min(16, spec.concurrency))
    compacting = compaction == "xeon"
    chunk_counter = itertools.count()
    blocking = getattr(spec, "session_compaction_mode", "blocking") == "blocking"
    fixed_work = growing(spec)
    async with (
        httpx.AsyncClient(
            timeout=spec.request_timeout_s, limits=httpx.Limits(max_connections=spec.concurrency)
        ) as client,
        httpx.AsyncClient(timeout=getattr(spec, "session_compaction_timeout_s", 900)) as summarizer,
    ):

        async def send(slot, turn):
            session = sessions[slot]
            item = await asyncio.to_thread(turn_item, session, tokenizer, spec, turn)
            pending = session.get("pending")
            if pending:
                pending["turns_while_pending"] = pending.get("turns_while_pending", 0) + 1
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
                window_turn=turn - session.get("prime_turns", 0),
                input_tokens_checked=item["input_tokens"],
                prompt_sha256=item["prompt_sha256"],
                probe=item.get("probe"),
                fact_turn=item.get("fact_turn"),
                fact_age=item.get("fact_age"),
                reset=item.get("reset", False),
                cycle=item.get("cycle", 0),
                resets=item.get("resets", 0),
                history_turns=item.get("history_turns"),
                compaction_pending=bool(pending),
                compaction_landed=bool(session.pop("landed_now", False)),
                event_edge=session.pop("event_edge", None),
            )
            advance(session, item, result)
            if item.get("reset"):
                # The turn after a reset pairs with the one before it.
                session["event_edge"] = "after"
            return result

        async def start_compaction(slot, turn, start):
            session = sessions[slot]
            archive = [
                e for e in session.get("archive", []) if e["turn"] > session["compacted_through"]
            ]
            if len(archive) < 2:
                return
            # Keep the most recent completed turn raw; summarize the rest.
            archive = archive[:-1]
            session["pending"] = dict(
                trigger_turn=turn, trigger_at=time.monotonic() - start, turns_while_pending=0
            )

            async def run_compaction():
                row = await compact(summarizer, spec, archive, session["ledger"], chunk_counter)
                row.update(
                    session=slot,
                    trigger_turn=turn,
                    trigger_at=session["pending"]["trigger_at"],
                    landed_at=time.monotonic() - start,
                    turns_while_pending=session["pending"]["turns_while_pending"],
                )
                row["lag_s"] = row["landed_at"] - row["trigger_at"]
                if row["ok"]:
                    apply_compaction(
                        session,
                        row,
                        row["landed_at"],
                        getattr(spec, "session_compaction_ledger", True),
                    )
                    row["post_compaction_tokens"] = await asyncio.to_thread(
                        token_count, tokenizer, session["messages"]
                    )
                    session["landed_now"] = True
                    session["event_edge"] = "after"
                session["pending"] = None
                compactions.append({k: v for k, v in row.items() if k != "raw_output"})
                with (directory / "compactions.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")

            session["compaction_task"] = asyncio.create_task(run_compaction())

        prime_start = time.monotonic()

        async def initialize(slot):
            async with sem:
                # The first real request carries any synthetic prior history (priming cost).
                result = await send(slot, sessions[slot].get("prime_turns", 0))
                prime.append(result)
                with (directory / "prime-requests.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")

        await asyncio.gather(*(initialize(i) for i in range(population)))
        prime_metrics = summarize(prime, time.monotonic() - prime_start, spec)
        write_json(directory / "priming-metrics.json", prime_metrics)
        # Priming catches a broken endpoint before a cell is spent. With synthetic prior
        # histories the priming request is itself a long-context extraction, so the
        # quality floor is loose here; the cell's task gate over every reference probe
        # governs quality.
        prime_floor = 0.8 if getattr(spec, "session_prime_phase", False) else spec.min_quality
        prime_metrics["quality_floor"] = prime_floor
        if (
            prime_metrics["quality_fraction"] < prime_floor
            or prime_metrics["error_fraction"] > spec.max_error_fraction
        ):
            raise RuntimeError("Knowledge-worker priming failed correctness/transport gate")
        await engine.capture(directory, "steady-before")
        timers_before = await summarizer_timers(summarizer, spec) if compacting else None
        timers = None
        start = time.monotonic()
        trigger = getattr(spec, "session_compaction_trigger_tokens", None)

        async def worker(slot):
            session = sessions[slot]
            rng = random.Random(spec.seed + block * 100003 + slot + 900000)
            turn = session.get("prime_turns", 0) + 1
            while turn < spec.session_turns + session.get("prime_turns", 0):
                pause = rng.expovariate(1 / spec.session_think_s)
                if time.monotonic() - start + pause >= spec.duration_s:
                    break
                await asyncio.sleep(pause)
                wait_s = 0.0
                task = session.get("compaction_task")
                if blocking and task is not None and not task.done():
                    # The turn waits for the summary: this is the user-felt compaction cost.
                    waited_from = time.monotonic()
                    await asyncio.gather(task, return_exceptions=True)
                    wait_s = time.monotonic() - waited_from
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
                        compaction_pending=bool(session.get("pending")),
                    )
                result.update(arrival_s=arrival, finish_s=time.monotonic() - start)
                result["elapsed_s"] = result["finish_s"] - arrival
                result["compaction_wait_s"] = wait_s
                rows.append(result)
                with (directory / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
                write_json(
                    directory / "progress.json",
                    dict(at_s=time.monotonic() - start, completed=len(rows)),
                )
                if result.get("error") == "context_limit":
                    break
                if (
                    compacting
                    and trigger
                    and not session.get("pending")
                    and result.get("input_tokens_checked", 0) >= trigger
                ):
                    # This turn is the one before the event; the post-landing turn pairs with it.
                    result["event_edge"] = "before"
                    await start_compaction(slot, turn, start)
                turn += 1

        tasks = [asyncio.create_task(worker(i)) for i in range(population)]
        try:
            await asyncio.gather(*tasks)
            if not fixed_work:
                await asyncio.sleep(max(0, spec.duration_s - (time.monotonic() - start)))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Compactions already in flight are evidence; wait briefly, then record what landed.
            inflight = [
                s.get("compaction_task")
                for s in sessions
                if s.get("compaction_task") and not s["compaction_task"].done()
            ]
            if inflight:
                await asyncio.wait(
                    inflight, timeout=min(60, getattr(spec, "session_compaction_timeout_s", 60))
                )
                for task in inflight:
                    task.cancel()
                await asyncio.gather(*inflight, return_exceptions=True)
        elapsed = time.monotonic() - start
        await engine.capture(directory, "steady-after")
        if compacting:
            # Engine-side split of each summarizer call: waiting for a batch slot
            # versus prefill and decode once running.
            timers = summarizer_timer_delta(
                timers_before, await summarizer_timers(summarizer, spec)
            )
            write_json(directory / "summarizer-timers.json", timers)
    result = summarize_workers(rows, elapsed, spec, population, compactions)
    result["priming"] = prime_metrics
    result["compaction_mode"] = compaction
    result["summarizer_timers"] = timers
    if growing(spec):
        write_json(
            directory / "fact-logs.json",
            {str(i): s.get("facts", []) for i, s in enumerate(sessions)},
        )
        write_json(
            directory / "session-state.json",
            {
                str(i): dict(
                    cycle=s.get("cycle"),
                    resets=s.get("resets"),
                    summaries=s.get("summaries"),
                    ledger=s.get("ledger"),
                )
                for i, s in enumerate(sessions)
            },
        )
    return result


ARM_LABELS = {
    ("baseline", "none"): "A: GPU-only",
    ("treatment", "none"): "B: Xeon KV tier",
    ("baseline", "xeon"): "C: Xeon compaction",
    ("treatment", "xeon"): "D: Xeon KV tier + compaction",
}


def arm_key(row):
    return (row["arm"], row.get("compaction", "none"))


def compare_workers(results, spec):
    views = []
    keys = sorted({arm_key(r) for r in results} | {("baseline", "none"), ("treatment", "none")})
    for slo in spec.session_slo_grid_s:
        arms = {}
        for key in keys:
            cells = [r for r in results if arm_key(r) == key]
            passing = []
            for row in cells:
                m = row["metrics"]
                task_gate = growing(spec) and "slo_analysis_task" in m
                gate_views = m.get("slo_analysis_task" if task_gate else "slo_analysis", [])
                quality = m.get("task_quality_fraction" if task_gate else "quality_fraction", 0)
                view = next((s for s in gate_views if s["response_slo_s"] == slo), {})
                if (
                    view.get("passed")
                    and quality >= spec.min_quality
                    and m.get("error_fraction", 1) <= spec.max_error_fraction
                    and not m.get("context_limit_failures")
                ):
                    passing.append(row)
            best = max(passing, key=lambda r: r["population"], default=None)
            arms["/".join(key)] = dict(
                label=ARM_LABELS.get(key, "/".join(key)),
                best_observed_population=best["population"] if best else None,
                correct_turns_per_minute=best["metrics"]["correct_requests_per_s"] * 60
                if best
                else None,
                cpu_reload_observed=best["metrics"].get("cpu_reload_observed", False)
                if best
                else False,
                end_state_recall=((best["metrics"].get("recall") or {}).get("end_state") or {}).get(
                    "recall"
                )
                if best
                else None,
                task_quality_fraction=best["metrics"].get("task_quality_fraction")
                if best
                else None,
                session_resets=best["metrics"].get("session_resets") if best else None,
                compactions_landed=(best["metrics"].get("compaction") or {}).get("landed")
                if best
                else None,
                tested_cells=len(cells),
            )

        def pop(name, arms=arms):
            return arms.get(name, {}).get("best_observed_population")

        ratios = {}
        for label, num, den in (
            ("C_over_A", "baseline/xeon", "baseline/none"),
            ("B_over_A", "treatment/none", "baseline/none"),
            ("D_over_A", "treatment/xeon", "baseline/none"),
            ("D_over_B", "treatment/xeon", "treatment/none"),
        ):
            ratios[label] = pop(num) / pop(den) if pop(num) and pop(den) else None
        views.append(dict(response_slo_s=slo, arms=arms, density_ratios=ratios))
    return dict(
        qualified=False,
        reason="Exploratory worker-population grid; confirm boundaries, repeat pairs and review actual prompt growth before claims",
        workload="Mature knowledge-worker proxy with retained actual answers"
        + (" ; growing sessions with whole-history probes" if growing(spec) else ""),
        active_reference_records=getattr(spec, "session_active_records", None),
        context_target=spec.session_context_tokens,
        growth_tokens=getattr(spec, "session_growth_tokens", 0),
        compaction_band=[
            getattr(spec, "session_compaction_trigger_tokens", None),
            getattr(spec, "session_compaction_target_tokens", None),
        ],
        by_worker_slo=views,
        cache_work_by_cell=[
            dict(
                block=r["block"],
                arm=r["arm"],
                compaction=r.get("compaction", "none"),
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


def timeline_svg(cells, title, key, unit):
    """Per-turn mean of a timeline metric for each arm; hand-rolled like report.chart."""
    from html import escape

    colors = ["#3B5265", "#087D8C", "#B8541A", "#6B3FA0"]
    width, height, left, top, plot_w, plot_h = 760, 320, 60, 40, 660, 220
    series = []
    for row in cells:
        points = [
            (t["turn"], t[key])
            for t in (row["metrics"].get("recall") or {}).get("timeline", [])
            if t.get(key) is not None
        ]
        if points:
            series.append(
                (
                    ARM_LABELS.get(arm_key(row), row["arm"]) + f" ({row['population']} workers)",
                    points,
                )
            )
    if not series:
        return None
    max_x = max(x for _, pts in series for x, _ in pts) or 1
    max_y = max(y for _, pts in series for _, y in pts) or 1
    if key == "whole_history_recall":
        max_y = 1
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="white"/><g font-family="Arial" fill="#12212c">',
        f'<text x="12" y="24" font-size="16">{escape(title[:90])}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#888"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#888"/>',
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 8}" font-size="12" text-anchor="middle">turn</text>',
        f'<text x="14" y="{top + 10}" font-size="12">{escape(unit)}</text>',
    ]
    for i, (name, pts) in enumerate(series):
        color = colors[i % len(colors)]
        path = " ".join(
            f"{left + plot_w * x / max_x:.1f},{top + plot_h - plot_h * y / max_y:.1f}"
            for x, y in pts
        )
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
        out.append(
            f'<rect x="{left + 10 + (i % 2) * 330}" y="{top + plot_h + 14 + (i // 2) * 16}" width="12" height="12" fill="{color}"/>'
            f'<text x="{left + 26 + (i % 2) * 330}" y="{top + plot_h + 25 + (i // 2) * 16}" font-size="12">{escape(name[:48])}</text>'
        )
    out.append("</g></svg>")
    return "\n".join(out)


def reanalyze_workers(path):
    """Recompute stage 2 metrics and the four-arm comparison from retained request rows.

    Engine-derived fields (cache work, native counters) are kept from the original
    metrics; everything derived from requests.jsonl and compactions.jsonl is rebuilt.
    The original run is not rewritten: results go under analyses/stage2/.
    """
    import copy

    from ..benchmarks.storage import read_json
    from .spec import ExperimentSpec

    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    state = read_json(path / "state.json")
    spec = ExperimentSpec(**manifest["spec"])
    results = copy.deepcopy(state["results"])
    for row in results:
        cell = path / row["evidence"]
        rows = [json.loads(line) for line in (cell / "requests.jsonl").read_text().splitlines()]
        compactions = (
            [json.loads(line) for line in (cell / "compactions.jsonl").read_text().splitlines()]
            if (cell / "compactions.jsonl").exists()
            else []
        )
        old = row["metrics"]
        rebuilt = summarize_workers(
            rows,
            old.get("elapsed_s_including_drain") or spec.duration_s,
            spec,
            row["population"],
            compactions,
        )
        for key in ("cache_work", "cpu_reload_observed", "priming", "compaction_mode"):
            if key in old:
                rebuilt[key] = old[key]
        for key, value in old.items():
            if key.startswith("native_queue") or key.startswith("queue"):
                rebuilt[key] = value
        row["metrics"] = rebuilt
    comparison = compare_workers(results, spec)
    target = path / "analyses" / "stage2"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "comparison.json", comparison)
    write_json(target / "results.json", results)
    render_worker_reports(path, manifest, dict(state, results=results), comparison, target)
    return comparison


def render_worker_reports(path, manifest, state, comparison, root=None):
    from docx import Document

    spec = manifest["spec"]
    stage2 = spec.get("session_growth_tokens", 0) > 0
    root = Path(root) if root else path / "outputs"
    for output in manifest["project"]["outputs"]:
        folder = root / output["id"]
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
        if stage2:
            doc.add_paragraph(
                f"Stage 2: growing sessions, about {spec['session_growth_tokens']} new tokens per turn. "
                f"No-compaction arms apply the '{spec.get('session_limit_policy')}' policy at the engine limit. "
                f"Compaction arms summarize on the Xeon between {spec.get('session_compaction_trigger_tokens')} and "
                f"{spec.get('session_compaction_target_tokens')} tokens; the reference prefix is never rewritten. "
                "Probes sample each worker's whole history; a probe about a lost fact is scored wrong, never skipped. "
                "Recall loss under compaction is bounded, not zero."
            )
        doc.add_paragraph(
            "Active reference subset: "
            + str(spec.get("session_active_records"))
            + ". When configured, questions target that many recent reference records at the end of the accumulated context."
        )
        doc.add_paragraph(
            "Requested initial context: "
            + str(spec["session_context_tokens"])
            + " tokens. Mean think time: "
            + str(spec["session_think_s"])
            + " seconds. Physical resident session counts and avoided GPU token processing are not inferred from cache blocks or logical prompt tokens."
        )
        if output["id"] == "P02-O3":
            for row in state.get("results", []):
                work = row["metrics"].get("cache_work", {})
                doc.add_paragraph(
                    f"{row['arm']}/{row.get('compaction', 'none')}, {row['population']} workers: uncached prefix tokens per correct turn {work.get('uncached_prefix_tokens_per_correct_request')}; uncached share {work.get('uncached_prefix_share')}. This includes new input; historical recomputation is not yet attributable."
                )
        headers = ["Arm", "Workers", "Correct turns/min", "Quality", "Response p95 s", "CPU reload"]
        if stage2:
            headers += [
                "Task quality",
                "End-state recall",
                "Resets",
                "Compactions",
                "Lag p95 s",
                "Live tokens",
            ]
        table = doc.add_table(rows=1, cols=len(headers))
        for c, t in zip(table.rows[0].cells, headers, strict=True):
            c.text = t
        for row in state.get("results", []):
            m = row["metrics"]
            values = [
                ARM_LABELS.get(arm_key(row), row["arm"]),
                row["population"],
                round(m["correct_requests_per_s"] * 60, 2),
                m["quality_fraction"],
                m["latency_p95_s"],
                m.get("cpu_reload_observed"),
            ]
            if stage2:
                recall = m.get("recall") or {}
                comp = m.get("compaction") or {}
                values += [
                    m.get("task_quality_fraction"),
                    (recall.get("end_state") or {}).get("recall"),
                    m.get("session_resets"),
                    comp.get("landed"),
                    comp.get("lag_p95_s"),
                    None if m.get("live_tokens_mean") is None else round(m["live_tokens_mean"]),
                ]
            for c, v in zip(table.add_row().cells, values, strict=True):
                c.text = str(v)
        for view in comparison.get("by_worker_slo", []):
            ratios = view.get("density_ratios") or {}
            if any(v for v in ratios.values()):
                doc.add_paragraph(
                    f"Response SLO {view['response_slo_s']} s, best supported populations: "
                    + "; ".join(
                        f"{a['label']} {a['best_observed_population']}"
                        for a in view["arms"].values()
                    )
                    + ". Ratios: "
                    + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items() if v)
                    + ". Exploratory; not a qualified claim."
                )
        if stage2 and state.get("results"):
            best = {}
            for row in state["results"]:
                key = arm_key(row)
                if key not in best or row["population"] > best[key]["population"]:
                    best[key] = row
            cells = list(best.values())
            for name, key, unit, title in (
                (
                    "live-context.svg",
                    "live_tokens_mean",
                    "live tokens",
                    "Live context per turn (sawtooth under compaction, resets without)",
                ),
                (
                    "whole-history-recall.svg",
                    "whole_history_recall",
                    "recall",
                    "Whole-history recall per turn",
                ),
            ):
                svg = timeline_svg(cells, title, key, unit)
                if svg:
                    (folder / name).write_text(svg)
                    doc.add_paragraph(f"Chart: {name} (largest tested population per arm).")
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
