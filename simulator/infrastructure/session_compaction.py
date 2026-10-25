"""Xeon context compaction for growing knowledge-worker sessions (P02 stage 2).

Context compaction, not KV compaction: the summarizer rewrites the trailing
raw turns of a session into one prose summary plus a structured ledger. The
reference prefix stays byte-identical at the head of the first message so its
KV remains a prefix-cache hit; only the new summary text is prefilled.

Every summary is written once from raw turns. The summarizer never sees a
previous summary; only the structured ledger is carried forward, by merging.
"""

import asyncio
import json
import re
import time

SUMMARIZER_SYSTEM = (
    "You compact the transcript of a work session into a faithful summary and a "
    "structured ledger. Preserve every record ID, turn number and stated allocation "
    "exactly. Do not invent facts. Output exactly one <summary> block followed by "
    "one <ledger> block and nothing else."
)

SUMMARIZER_INSTRUCTIONS = (
    "Write <summary>...</summary> containing at most two sentences of prose about what "
    "the session covered; do not repeat per-turn details or record values there, the "
    "ledger carries them. "
    "Then write <ledger>...</ledger> containing one line per fact, nothing else, in "
    "transcript order, using exactly these two line forms:\n"
    "requested turn <N>: <record id>, <record id>\n"
    "allocation <record id>: <number>\n"
    "Write a 'requested' line for every turn whose user message asks to return records, "
    "listing them in the order requested; a question about an earlier turn or about a "
    "working allocation is not a request. Write an 'allocation' line only for a sentence "
    "that begins 'Session note' and sets a working allocation, with that note's integer "
    "value. Fields of the reference records or of an assistant answer (location, part, "
    "quantity, status and the like) are never allocations. Write each line once. Omit "
    "nothing from the ledger."
)

ACKNOWLEDGEMENT = "Acknowledged. The compacted session state above is restored."


def empty_ledger():
    return {"requested_by_turn": {}, "working_allocations": {}}


def merge_ledger(prior, new):
    """Structured state is merged; prose summaries are never merged or rewritten."""
    out = {
        "requested_by_turn": dict(prior.get("requested_by_turn", {})),
        "working_allocations": dict(prior.get("working_allocations", {})),
    }
    for turn, ids in (new.get("requested_by_turn") or {}).items():
        if isinstance(ids, list):
            out["requested_by_turn"][str(turn)] = [str(i) for i in ids]
    for rid, value in (new.get("working_allocations") or {}).items():
        out["working_allocations"][str(rid)] = value
    return out


REQUESTED_LINE = re.compile(r"^\s*requested\s+turn\s+(\d+)\s*:\s*(.*?)\s*$", re.I)
QUANTITY_LINE = re.compile(
    r"^\s*(?:allocation|quantity)\s+([A-Za-z0-9_-]+)\s*:\s*(-?\d+|null|none)\s*$", re.I
)


NON_NUMERIC_ALLOCATION = re.compile(r"^\s*(?:allocation|quantity)\s+[A-Za-z0-9_-]+\s*:", re.I)


def parse_summary(text):
    """Summary prose plus a line-based ledger; JSON is built here, never by the model."""
    summary = re.search(r"<summary>(.*?)</summary>", text, re.S)
    ledger = re.search(r"<ledger>(.*?)</ledger>", text, re.S)
    if not summary or not ledger:
        raise ValueError("Summarizer output lacks summary or ledger block")
    parsed = empty_ledger()
    lines = [line for line in ledger.group(1).splitlines() if line.strip()]
    for line in lines:
        requested = REQUESTED_LINE.match(line)
        quantity = QUANTITY_LINE.match(line)
        if not requested and not quantity and NON_NUMERIC_ALLOCATION.match(line):
            # An allocation is always an integer from a session note; a non-numeric value
            # was copied from a record field and carries no session state. Drop the line.
            parsed["dropped_lines"] = parsed.get("dropped_lines", 0) + 1
            continue
        if requested:
            ids = [i.strip() for i in requested.group(2).split(",") if i.strip()]
            parsed["requested_by_turn"][requested.group(1)] = ids
        elif quantity:
            value = quantity.group(2).lower()
            parsed["working_allocations"][quantity.group(1)] = (
                None if value in ("null", "none") else int(value)
            )
        else:
            raise ValueError("Ledger line has an unknown form: " + line.strip()[:80])
    if not lines:
        raise ValueError("Ledger is empty")
    return summary.group(1).strip(), parsed


def transcript(archive):
    """Raw turns for the summarizer. Only raw turns, never a prior summary."""
    lines = []
    for entry in archive:
        lines.append(f"Turn {entry['turn']} user: {entry['user']}")
        lines.append(f"Turn {entry['turn']} assistant: {entry['assistant']}")
    return "\n".join(lines)


def render_header(prefix, summaries, ledger, include_ledger=True):
    """First user message after compaction: prefix bytes first, then the ledger, then
    the short per-cycle summaries. Ledger first because it carries the facts; the prose
    is kept short so accumulated summaries do not crowd the reference records."""
    parts = [prefix, "", "Compacted session history."]
    if include_ledger:
        parts.append("Session ledger (current structured state, JSON):")
        parts.append(json.dumps(ledger, sort_keys=True))
    if summaries:
        parts.append("Cycle summaries, each written once from the raw turns it names:")
    for s in summaries:
        parts.append(
            f"Summary {s['cycle']} (turns {s['first_turn']}-{s['last_turn']}): {s['text']}"
        )
    parts.append("")
    parts.append("Acknowledge the restored session state.")
    return "\n".join(parts)


def restore_messages(session, include_ledger=True):
    """Rebuild live messages: header, acknowledgement, then raw turns after the compaction."""
    header = render_header(
        session["prefix"], session["summaries"], session["ledger"], include_ledger
    )
    if not header.startswith(session["prefix"]):
        raise RuntimeError("Compacted header lost prefix alignment")
    messages = [
        {"role": "user", "content": header},
        {"role": "assistant", "content": ACKNOWLEDGEMENT},
    ]
    for entry in session["archive"]:
        if entry["turn"] > session["compacted_through"]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
    return messages


def chunks(archive, chunk_turns):
    chunk_turns = max(1, int(chunk_turns or 1))
    return [archive[i : i + chunk_turns] for i in range(0, len(archive), chunk_turns)]


def endpoints(spec):
    return [u.strip() for u in (spec.session_compaction_url or "").split(",") if u.strip()]


async def compact_chunk(client, spec, archive, url):
    text = transcript(archive)
    body = {
        "model": spec.session_compaction_model,
        "messages": [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": "Transcript:\n" + text + "\n\n" + SUMMARIZER_INSTRUCTIONS},
        ],
        "temperature": 0,
        "seed": spec.seed,
        "max_tokens": spec.session_compaction_budget_tokens,
        "stream": False,
    }
    start = time.monotonic()
    row = dict(
        url=url,
        first_turn=archive[0]["turn"],
        last_turn=archive[-1]["turn"],
        raw_turns=len(archive),
        raw_chars=len(text),
        ok=False,
        error=None,
        summary=None,
        ledger=None,
        usage=None,
        finish_reason=None,
    )
    try:
        response = await client.post(
            url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
            json=body,
            timeout=spec.session_compaction_timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        row["usage"] = data.get("usage")
        row["finish_reason"] = choice.get("finish_reason")
        content = choice["message"]["content"] or ""
        row["raw_output"] = content
        row["summary"], row["ledger"] = parse_summary(content)
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001 - every failure is evidence, never a retry loop
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        body_text = getattr(getattr(exc, "response", None), "text", None)
        if body_text:
            row["error_body"] = body_text[:300]
    row["wall_s"] = time.monotonic() - start
    return row


async def compact(client, spec, archive, prior_ledger, counter=None):
    """One compaction: the cycle's raw turns, summarized in parallel chunks.

    Each chunk is a separate summarizer call over raw turns only. Chunk ledgers are
    merged in turn order and chunk summaries are concatenated; nothing is re-summarized.
    Returns evidence whether or not it succeeded.
    """
    urls = endpoints(spec) or [spec.session_compaction_url]
    parts = chunks(archive, getattr(spec, "session_compaction_chunk_turns", 1))
    # A shared counter assigns chunks to endpoints one at a time across every
    # compaction, so an odd chunk count cannot pile two thirds onto one socket.
    slots = [next(counter) if counter is not None else i for i in range(len(parts))]
    start = time.monotonic()
    results = await asyncio.gather(
        *(
            compact_chunk(client, spec, part, urls[slot % len(urls)])
            for slot, part in zip(slots, parts, strict=True)
        )
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for r in results:
        for key in usage:
            value = (r.get("usage") or {}).get(key)
            usage[key] = (
                None if usage[key] is None or not isinstance(value, int) else usage[key] + value
            )
    row = dict(
        first_turn=archive[0]["turn"],
        last_turn=archive[-1]["turn"],
        raw_turns=len(archive),
        raw_chars=sum(r["raw_chars"] for r in results),
        chunk_count=len(parts),
        chunks=[{k: v for k, v in r.items() if k != "raw_output"} for r in results],
        ok=all(r["ok"] for r in results),
        error="; ".join(r["error"] for r in results if r["error"]) or None,
        summary=None,
        ledger=None,
        usage=usage,
        finish_reason=",".join(str(r["finish_reason"]) for r in results),
        wall_s=None,
    )
    row["ledger_dropped_lines"] = sum(
        (r.get("ledger") or {}).get("dropped_lines", 0) for r in results
    )
    if row["ok"]:
        ledger = prior_ledger
        for r in results:
            ledger = merge_ledger(ledger, r["ledger"])
        row["ledger"] = ledger
        row["cycle_ledger"] = results[0]["ledger"] if len(results) == 1 else None
        row["summary"] = " ".join(r["summary"] for r in results)
    else:
        row["raw_output"] = "\n---\n".join(r.get("raw_output", "") for r in results)
    row["wall_s"] = time.monotonic() - start
    return row


# Engine request timers, by the names vLLM and SGLang export on /metrics. A missing
# timer stays None; the client-side chunk walls in compactions.jsonl remain the total.
SUMMARIZER_TIMERS = {
    "queue": ("vllm:request_queue_time_seconds", "sglang:queue_time_seconds"),
    "first_token": ("vllm:time_to_first_token_seconds", "sglang:time_to_first_token_seconds"),
    "prefill": ("vllm:request_prefill_time_seconds",),
    "decode": ("vllm:request_decode_time_seconds",),
    "end_to_end": ("vllm:e2e_request_latency_seconds", "sglang:e2e_request_latency_seconds"),
}


def metric_total(text, name, kind):
    """Sum a histogram's _sum or _count across label sets; None when absent."""
    pattern = rf"^{re.escape(name)}_{kind}(?:{{[^}}]*}})? ([0-9.eE+-]+)$"
    values = [float(v) for v in re.findall(pattern, text, re.M)]
    return sum(values) if values else None


async def summarizer_timers(client, spec):
    """Cumulative request timers per summarizer endpoint; None where unavailable."""
    out = {}
    for url in endpoints(spec):
        base = url.rstrip("/").removesuffix("/v1")
        try:
            response = await client.get(base + "/metrics", timeout=10)
            response.raise_for_status()
            text = response.text
        except Exception:  # noqa: BLE001 - an engine without metrics records None
            out[url] = None
            continue
        values = {}
        for short, names in SUMMARIZER_TIMERS.items():
            for kind in ("sum", "count"):
                found = [metric_total(text, n, kind) for n in names]
                values[f"{short}_{kind}"] = next((v for v in found if v is not None), None)
        out[url] = values
    return out


def summarizer_timer_delta(before, after):
    """Per-endpoint request counts and mean seconds for each timer the engine exports."""
    out = {}
    for url, end in (after or {}).items():
        begin = (before or {}).get(url)
        if not end or not begin:
            out[url] = None
            continue
        row = {"requests": None}
        for short in SUMMARIZER_TIMERS:
            if end.get(f"{short}_sum") is None or end.get(f"{short}_count") is None:
                row[f"{short}_mean_s"] = None
                continue
            total = end[f"{short}_sum"] - (begin.get(f"{short}_sum") or 0)
            count = end[f"{short}_count"] - (begin.get(f"{short}_count") or 0)
            if short == "end_to_end":
                row["requests"] = count
            row[f"{short}_mean_s"] = total / count if count else None
        out[url] = row
    return out


def apply_compaction(session, row, landed_at, include_ledger=True):
    """Install a landed summary. Raw turns stay in the archive; nothing is deleted."""
    session["summaries"].append(
        dict(
            cycle=session["cycle"] + 1,
            first_turn=row["first_turn"],
            last_turn=row["last_turn"],
            text=row["summary"],
        )
    )
    session["ledger"] = row["ledger"]
    session["compacted_through"] = row["last_turn"]
    session["cycle"] += 1
    session["history_start"] = row["last_turn"] + 1
    session["messages"] = restore_messages(session, include_ledger)
    session["landed_at"] = landed_at
