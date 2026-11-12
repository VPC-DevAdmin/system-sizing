"""Measure how long an on-prem model reasons on the planner's workloads (T1).

The campaign runner starts this for jobs with ``kind: reasoning_probe``. It
builds the engine exactly as a capacity job would, then sends the cloud token
probe's sessions (``cloud_token_probe.build_sessions``: real public text shaped
like the planner's seven H-wave workloads, with their shared system prompts) to
it, one session at a time per worker, and records for every turn how many
tokens went to reasoning and how many to the answer.

The capacity reruns (T2) take the **mean** reasoning tokens per workload as
their allowance: throughput depends on the average, and sizing on a high
percentile would hide spare capacity. Median, p90 and the share of turns that
hit the token cap are kept as evidence. A capped turn makes the mean too low,
so each turn gets far more room than the answer needs (``reasoning_room``).

  1. Launch the engine and wait for it to be healthy.
  2. Run every session, ``concurrency`` at a time, streaming each turn.
     Reasoning is what the engine's reasoning parser returns in
     ``reasoning`` / ``reasoning_content``; its tokens are the reply's
     ``completion_tokens`` minus the answer's tokens (counted with the
     model's own tokenizer), or the reasoning text's tokens when the
     engine reports no usage.
  3. Shut the engine down (always, in ``finally``).
  4. Write ``probe_calls.jsonl`` (one line per turn) and
     ``reasoning_probe_summary.json``. Status is ok only when at least 95%
     of turns succeeded and, for a variant that should reason, reasoning
     was actually separated from the answer.

Job settings (``job["probe"]``)::

    temperature, top_p      the variant's recommended sampling
    extra_body              e.g. {"reasoning_effort": "low"}
    expect_reasoning        default true; false for a thinking-off baseline
    reasoning_room          tokens above the answer cap (default 32768)
    concurrency             sessions in flight (default 16)
    workloads               optional list of probe workload ids to run
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import cloud_token_probe as ctp  # noqa: E402

SUMMARY = "reasoning_probe_summary.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hwave_id(workload: str) -> str:
    """Probe workload id -> H-wave persona id (``document-qa`` -> ``h_document_qa``)."""
    return "h_" + workload.replace("-", "_")


def turn_cap(turn: "ctp.Turn", room: int, prompt_tokens: int, max_model_len: int | None) -> int:
    """Answer cap plus reasoning room, kept inside the context window."""
    cap = turn.max_output - ctp.REASONING_ROOM + room
    if max_model_len:
        cap = min(cap, max_model_len - prompt_tokens - 256)
    return max(256, cap)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(len(s) - 1, int(k) + 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(rows: list[dict], expect_reasoning: bool) -> dict:
    per: dict = {}
    for wl in dict.fromkeys(r["workload"] for r in rows):
        rs = [r for r in rows if r["workload"] == wl]
        ok = [r for r in rs if "error" not in r]
        reason = [r["reasoning_tokens"] for r in ok]
        answer = [r["answer_tokens"] for r in ok]
        per[hwave_id(wl)] = {
            "probe_workload": wl, "turns": len(rs), "errors": len(rs) - len(ok),
            "reasoning_mean": round(statistics.mean(reason), 1) if reason else None,
            "reasoning_median": round(statistics.median(reason), 1) if reason else None,
            "reasoning_p90": round(pct(reason, 0.9), 1) if reason else None,
            "answer_mean": round(statistics.mean(answer), 1) if answer else None,
            "answer_target_mean": round(statistics.mean(r["answer_target"] for r in ok), 1) if ok else None,
            "hit_cap": sum(1 for r in ok if r["hit_cap"]),
            "hit_cap_share": round(sum(1 for r in ok if r["hit_cap"]) / max(1, len(ok)), 4),
            "no_answer": sum(1 for r in ok if r["answer_tokens"] == 0),
            "first_answer_s_median": round(statistics.median(r["first_answer_s"] for r in ok
                                                             if r["first_answer_s"] is not None), 2)
            if any(r["first_answer_s"] is not None for r in ok) else None,
        }
    total = len(rows)
    ok = sum(1 for r in rows if "error" not in r)
    # A turn reasoned when the parser returned reasoning text; a reply's
    # control tokens alone (end-of-turn, empty think tags) do not count.
    separated = sum(1 for r in rows if r.get("reasoning_text_tokens", r.get("reasoning_tokens", 0)) > 0)
    status = "ok"
    reason = None
    if total == 0 or ok < 0.95 * total:
        status, reason = "failed", f"{ok} of {total} turns succeeded"
    elif expect_reasoning and separated < 0.5 * ok:
        status, reason = "failed", (f"reasoning separated on only {separated} of {ok} turns: "
                                    "check the reasoning parser and thinking switch")
    elif not expect_reasoning and separated > 0.05 * ok:
        status, reason = "failed", (f"a thinking-off variant reasoned on {separated} of {ok} turns: "
                                    "check the thinking switch")
    return {"status": status, "reason": reason, "turns": total, "ok_turns": ok,
            "turns_with_reasoning": separated, "workloads": per}


class Counter:
    """Token counts with the model's tokenizer (falls back to a heuristic)."""

    def __init__(self, model_id: str):
        sys.path.insert(0, str(REPO))
        from simulator.tokenizer_corpus import TokenCorpus
        self._c = TokenCorpus(model_id)
        self._lock = threading.Lock()

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            return self._c.count(text)


def run_turn(client, served: str, messages: list[dict], cap: int, probe: dict) -> dict:
    t0 = time.time()
    first_any = first_answer = None
    reasoning, content, usage, finish = [], [], None, None
    stream = client.chat.completions.create(
        model=served, messages=messages, max_tokens=cap, stream=True,
        stream_options={"include_usage": True},
        temperature=float(probe.get("temperature", 1.0)),
        top_p=float(probe.get("top_p", 1.0)),
        extra_body=probe.get("extra_body") or None)
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        d = ch.delta
        extra = getattr(d, "model_extra", None) or {}
        r = extra.get("reasoning") or extra.get("reasoning_content") or getattr(d, "reasoning_content", None)
        if r:
            first_any = first_any or time.time()
            reasoning.append(r)
        if d.content:
            first_any = first_any or time.time()
            first_answer = first_answer or time.time()
            content.append(d.content)
        if ch.finish_reason:
            finish = ch.finish_reason
    return {"reasoning_text": "".join(reasoning), "content": "".join(content),
            "completion_tokens": usage.completion_tokens if usage else None,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "finish": finish, "seconds": round(time.time() - t0, 2),
            "first_token_s": round(first_any - t0, 2) if first_any else None,
            "first_answer_s": round(first_answer - t0, 2) if first_answer else None}


def run_session(client, served: str, s: "ctp.Session", probe: dict, count: Counter,
                max_model_len: int | None, log_rows: list, lock: threading.Lock, out) -> None:
    room = int(probe.get("reasoning_room", 32768))
    msgs = [{"role": "system", "content": s.system}]
    for i, t in enumerate(s.turns):
        msgs.append({"role": "user", "content": t.text})
        prompt_est = sum(ctp.est_tokens(m["content"]) for m in msgs)
        cap = turn_cap(t, room, prompt_est, max_model_len)
        row = {"workload": s.workload, "session": s.index, "turn": i, "cap": cap,
               "answer_target": t.answer_tokens}
        try:
            r = run_turn(client, served, msgs, cap, probe)
        except Exception as exc:  # noqa: BLE001 - record, end this session
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            with lock:
                log_rows.append(row)
                out.write(json.dumps(row) + "\n"); out.flush()
            return
        answer = count(r["content"])
        if r["completion_tokens"] is not None:
            reason_tok = max(0, r["completion_tokens"] - answer)
        else:
            reason_tok = count(r["reasoning_text"])
        row.update({"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                    "reasoning_tokens": reason_tok, "answer_tokens": answer,
                    "reasoning_text_tokens": count(r["reasoning_text"]),
                    "hit_cap": r["finish"] == "length", "finish": r["finish"],
                    "seconds": r["seconds"], "first_token_s": r["first_token_s"],
                    "first_answer_s": r["first_answer_s"]})
        if row["hit_cap"]:
            # Evidence for why a turn never answered (a loop, or long work).
            row["reasoning_head"] = r["reasoning_text"][:1500]
            row["reasoning_tail"] = r["reasoning_text"][-3000:]
        with lock:
            log_rows.append(row)
            out.write(json.dumps(row) + "\n"); out.flush()
        # History keeps the answer only, as a deployed chat client does.
        msgs.append({"role": "assistant", "content": r["content"] or "(no answer)"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-file", required=True)
    a = ap.parse_args(argv)
    job = json.loads(Path(a.job_file).read_text())
    out_dir = Path(a.job_file).parent
    probe = job.get("probe") or {}
    log = open(out_dir / "reasoning_probe.log", "a")
    sys.path.insert(0, str(REPO))
    from simulator.config import load_config
    from simulator.engines import make_engine
    import openai

    cfg = load_config(a.config)
    sessions = ctp.build_sessions()
    # Optional: probe only some workloads (probe workload ids, e.g. coding-agent).
    if probe.get("workloads"):
        sessions = [s for s in sessions if s.workload in set(probe["workloads"])]
        if not sessions:
            raise SystemExit(f"no probe sessions for {probe['workloads']}")
    manifest ={"job": job["id"], "model": job["model"], "probe": probe,
                "engine": {"type": cfg.engine.type, "image": getattr(cfg.engine, "gpu_image", None),
                           "tp": getattr(cfg.engine, "tensor_parallel_size", None),
                           "max_model_len": getattr(cfg.engine, "max_model_len", None),
                           "extra_flags": getattr(cfg.engine, "vllm_extra_flags", None)},
                "sessions": len(sessions), "turns": sum(len(s.turns) for s in sessions),
                "started": now()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    rows: list[dict] = []
    lock = threading.Lock()
    calls = open(out_dir / "probe_calls.jsonl", "w")
    engine = make_engine(cfg.engine.type, cfg.engine)
    try:
        t0 = time.time()
        engine.launch()
        log.write(f"{now()} engine up in {time.time() - t0:.0f}s at {engine.base_url}\n"); log.flush()
        base = engine.base_url.rstrip("/")
        base = base if base.endswith("/v1") else base + "/v1"
        client = openai.OpenAI(base_url=base, api_key=getattr(engine, "api_key", None) or "x",
                               timeout=3600, max_retries=0)
        count = Counter(job["model"])
        mml = getattr(cfg.engine, "max_model_len", None)
        # Longest sessions first, so the tail is short ones.
        order = sorted(sessions, key=lambda s: -len(s.turns))
        with cf.ThreadPoolExecutor(int(probe.get("concurrency", 16))) as pool:
            futs = [pool.submit(run_session, client, engine.api_model_name, s, probe, count, mml,
                                rows, lock, calls) for s in order]
            for f in cf.as_completed(futs):
                f.result()
        log.write(f"{now()} {len(rows)} turns in {time.time() - t0:.0f}s\n"); log.flush()
    except Exception as exc:  # noqa: BLE001 - summary records the failure
        log.write(f"{now()} failed: {type(exc).__name__}: {exc}\n"); log.flush()
    finally:
        engine.shutdown()
        calls.close()
        log.write(f"{now()} engine down\n"); log.flush()
    summary = summarize(rows, bool(probe.get("expect_reasoning", True)))
    summary.update({"job": job["id"], "model": job["model"], "finished": now()})
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=1))
    log.write(f"{now()} summary status {summary['status']} {summary.get('reason') or ''}\n")
    log.close()
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
