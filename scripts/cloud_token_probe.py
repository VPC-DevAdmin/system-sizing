"""Measure what the planner's cloud comparison models really bill.

The AI capacity planner prices each plan's tokens on the cloud model it is
compared with. It counts input, cached input and the reply, but a model that
reasons bills those tokens as output too, and the planner cannot see them.
This probe sends realistic tasks shaped like the planner's seven workloads to
each compared model, at the reasoning setting its Artificial Analysis score
was measured in, and records the billed usage of every call: uncached input,
cache reads and writes, visible output and reasoning.

Prompts come from public text already on this machine: Python's own
language reference (``pydoc_data``) for documents and its standard library
for code. Nothing private is sent.

  uv run python scripts/cloud_token_probe.py plan
      Dry run. Builds every prompt, prices the run three ways and makes no
      call. Needs no SDK and no key.
  uv run --with anthropic --with openai --with google-genai \\
      python scripts/cloud_token_probe.py run --budget-usd 25 --models gpt6-luna
      Live run. Keys come from OPENAI_API_KEY, ANTHROPIC_API_KEY and
      GEMINI_API_KEY. Before each call the script adds that call's worst
      case (prompt at full price, max_output at the output price) to what it
      has spent; if that would pass the budget, it stops.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import random
import sys
import sysconfig
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The compared models. Prices per million tokens, checked 2026-11-04 against
# each provider's pricing page; the planner's content/data/ai-cloud-pricing.json
# carries the same figures. Only these three are ever matched by the planner's
# Auto rule (scanned across every level, speed, mix and pinned model); each
# provider's Fast / Priority tier is the same model at a higher price, so it
# bills the same tokens and needs no run of its own.
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "gpt6-luna": {
        "provider": "openai", "model": "gpt-6-luna",
        # Scored with no reasoning; the API default is medium.
        "reasoning_effort": "none",
        "input": 0.10, "cached": 0.01, "cache_write": 0.10, "output": 0.50,
        "long": {"over": 272_000, "input": 2.0, "output": 1.5},
    },
    # Compared with reasoning on-prem models (AA 21.53 at low).
    "gpt6-luna-low": {
        "provider": "openai", "model": "gpt-6-luna",
        "reasoning_effort": "low",
        "input": 0.10, "cached": 0.01, "cache_write": 0.10, "output": 0.50,
        "long": {"over": 272_000, "input": 2.0, "output": 1.5},
    },
    # Not compared by the planner: GPT-6 Luna at its API default, to show
    # what a buyer who leaves reasoning on would pay. Run only on request.
    "gpt6-luna-medium": {
        "provider": "openai", "model": "gpt-6-luna",
        "reasoning_effort": "medium", "optional": True,
        "input": 0.10, "cached": 0.01, "cache_write": 0.10, "output": 0.50,
        "long": {"over": 272_000, "input": 2.0, "output": 1.5},
    },
    "claude-haiku": {
        "provider": "anthropic", "model": "claude-haiku-5-5",
        # Scored at low effort; adaptive thinking is on by default.
        "effort": "low",
        "input": 0.10, "cached": 0.01, "cache_write": 0.125, "output": 0.50,
        "long": {"over": 100_000, "input": 5.0, "output": 5.0},
    },
    "gemini-flash-lite": {
        "provider": "google", "model": "gemini-3.5-flash-lite",
        # Scored at its default, minimal thinking.
        "thinking_level": "minimal",
        "input": 0.30, "cached": 0.03, "cache_write": 0.30, "output": 2.50,
        "long": None,
    },
}

# ---------------------------------------------------------------------------
# Workloads: the lab's H-wave shapes (config/personas/hwave.yaml) and system
# prompt sizes, with how many sessions and turns the probe samples. Agent
# sessions run 10-120 turns in the planner; the probe measures their first
# turns, which carry the full system prompt and starting context.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Workload:
    id: str
    system_tokens: int
    context: tuple | None      # (median, sigma, min, max) starting material
    input: tuple               # per-turn input
    output: tuple              # per-turn answer
    sessions: int
    turns: int
    kind: str                  # which task builder


WORKLOADS = [
    Workload("assistant", 2000, None, (400, .6, 20, 4000), (300, .6, 20, 2000), 10, 3, "chat"),
    Workload("document-qa", 1500, None, (8000, .5, 1000, 32000), (400, .4, 50, 1500), 15, 2, "docqa"),
    Workload("service-agent", 10000, (1500, .5, 200, 6000), (600, .7, 50, 4000), (150, .6, 20, 1500), 5, 6, "service"),
    Workload("research-agent", 20000, (1000, .3, 200, 4000), (1200, .9, 100, 16000), (300, .7, 30, 3000), 3, 10, "research"),
    Workload("coding-agent", 24000, (8000, .4, 2000, 24000), (3000, .8, 200, 24000), (600, .7, 50, 4000), 3, 10, "coding"),
    Workload("document-pipeline", 2000, None, (32000, .6, 4000, 120000), (1500, .5, 200, 6000), 30, 1, "pipeline"),
    Workload("deep-analysis", 2000, None, (6000, .6, 500, 32000), (6000, .5, 500, 16000), 20, 1, "analysis"),
]
# Room for reasoning above each turn's answer cap.
REASONING_ROOM = 8000
CHARS_PER_TOKEN = 4.0


def est_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def lognormal(rng: random.Random, spec: tuple) -> int:
    med, sigma, lo, hi = spec
    return int(min(hi, max(lo, med * math.exp(rng.gauss(0, sigma)))))


# ---------------------------------------------------------------------------
# Public source text
# ---------------------------------------------------------------------------
class Corpus:
    def __init__(self) -> None:
        import pydoc_data.topics as topics
        self.docs = [(k, v) for k, v in sorted(topics.topics.items()) if len(v) > 4000]
        std = sysconfig.get_paths()["stdlib"]
        files = sorted(f for pkg in ("asyncio", "email", "http", "json", "logging", "concurrent/futures")
                       for f in glob.glob(f"{std}/{pkg}/*.py") if os.path.getsize(f) > 3000)
        self.code = [(os.path.relpath(f, std), Path(f).read_text(errors="replace")) for f in files]

    def text(self, rng: random.Random, tokens: int) -> tuple[str, str]:
        """Consecutive reference topics from a random start, cut to size."""
        i, parts, title = rng.randrange(len(self.docs)), [], None
        while est_tokens("\n\n".join(parts)) < tokens:
            name, body = self.docs[i % len(self.docs)]
            title = title or name
            parts.append(f"## {name}\n\n{body}")
            i += 1
        return title, "\n\n".join(parts)[: int(tokens * CHARS_PER_TOKEN)]

    def files(self, rng: random.Random, tokens: int) -> str:
        out, used = [], 0
        for path, src in rng.sample(self.code, len(self.code)):
            take = src[: int((tokens - used) * CHARS_PER_TOKEN)]
            out.append(f"--- {path} ---\n{take}")
            used += est_tokens(take)
            if used >= tokens:
                break
        return "\n\n".join(out)


def pad_to(text: str, tokens: int, filler: str) -> str:
    """Grow a system prompt to the workload's size with real reference text."""
    need = tokens - est_tokens(text)
    return text if need <= 0 else text + "\n\n# Reference\n\n" + filler[: int(need * CHARS_PER_TOKEN)]


ASSISTANT_ASKS = [
    "Draft a short email to a supplier asking to move our delivery from Tuesday to Thursday.",
    "Explain the difference between a mutex and a semaphore to a new hire.",
    "Summarise the trade-offs of renting versus buying office space for a 40-person team.",
    "Write three subject lines for a product update newsletter about faster search.",
    "What questions should I ask in a reference check for a senior data engineer?",
    "Turn these notes into a meeting agenda: budget review, hiring plan, Q3 roadmap, office move.",
    "How do I explain a 6% price increase to long-standing customers without losing them?",
    "Give me a checklist for preparing a quarterly business review.",
    "What is the difference between gross margin and contribution margin?",
    "Rewrite this sentence to sound less defensive: 'We did exactly what the contract said.'",
    "Suggest a fair way to split on-call duty across five engineers in three time zones.",
    "Outline a one-page onboarding plan for a new account manager.",
]
FOLLOW_UPS = [
    "Make it shorter and more direct.", "Can you give a concrete example?",
    "What would you change for a European audience?", "Now put that in a table.",
    "What are the risks with that approach?", "Turn the main points into next steps with owners.",
]
DOC_QUESTIONS = [
    "According to this document, what exactly happens, step by step, in the main operation it describes?",
    "List every rule or restriction this document states, quoting the sentence each comes from.",
    "What does this document say about errors or exceptions, and when are they raised?",
    "Summarise this section for an engineer who has five minutes, keeping the key edge cases.",
    "Which statements in this document would surprise someone used to another language, and why?",
]
SERVICE_TURNS = [
    "Hi, I was charged twice for my subscription this month. Order 48213.",
    "I already tried the refund form and it said my account was not eligible.",
    "Can you just cancel the second charge and keep my plan as it is?",
    "Fine. How long will that take and will I get an email?",
    "Also my colleague cannot log in since your update yesterday. Same company account.",
    "She gets 'session expired' every time, on two different browsers.",
    "Is there anything you can do today? We have a client demo at 3pm.",
    "OK thanks. Please summarise what you have done so I can forward it.",
]
RESEARCH_GOALS = [
    "Compare how this language's reference describes iteration, generators and the iterator protocol, and identify where the three concepts depend on each other.",
    "Work out what the reference guarantees about evaluation order and side effects, with the sections that support each claim.",
    "Build a brief on how names are bound and resolved (scopes, global, nonlocal, class bodies), citing the sections you used.",
]
CODING_TASKS = [
    "Add a timeout parameter to the main request function in these files, thread it through every caller, and update the docstrings.",
    "There is a bug report: under heavy concurrency some callbacks run twice. Find the most likely cause in this code and propose a fix.",
    "Refactor the largest function here into smaller helpers without changing behaviour, and list the tests you would add.",
]
PIPELINE_ASKS = [
    "Extract every defined term in this document into JSON: [{\"term\", \"definition\", \"section\"}]. Return only JSON.",
    "Produce a structured outline of this document as JSON with headings, a one-sentence summary per section, and any code examples' purpose.",
    "Find every statement that contains a requirement (must, must not, is required, is an error) and return them as JSON with the section name.",
]
ANALYSIS_ASKS = [
    "Write a detailed technical report on this material for an engineering review board: background, key mechanisms, risks, open questions and recommendations.",
    "Write a long-form comparison of the approaches described here, with worked examples, trade-offs and a recommendation for a team adopting them.",
]


@dataclasses.dataclass
class Turn:
    text: str
    answer_tokens: int
    max_output: int


@dataclasses.dataclass
class Session:
    workload: str
    index: int
    system: str
    turns: list[Turn]


def build_sessions(seed: int = 7) -> list[Session]:
    rng, corpus = random.Random(seed), Corpus()
    filler = "\n\n".join(body for _, body in corpus.docs)
    tools_note = ("You can call these tools by writing a TOOL line; the user replies with its output.\n"
                  "TOOL read_file(path) - returns a file\nTOOL search(query) - returns matching passages\n"
                  "TOOL run_tests(target) - runs tests and returns the report\n")
    sessions: list[Session] = []
    for w in WORKLOADS:
        # One system prompt per workload, shared by every session, as the
        # lab's load generator does (simulator/arrivals.py) and as a deployed
        # service does: later sessions can read it from the provider's cache,
        # just as on-prem engines reuse it from the prefix cache.
        system = None
        for s in range(w.sessions):
            base = {
                "chat": "You are a helpful assistant for staff at a mid-sized company.",
                "docqa": "You answer questions about the document the user provides, quoting it where you can.",
                "service": "You are a customer service agent for a software subscription company. Follow the policy below. Escalate when policy requires it.",
                "research": "You are a research agent. Plan, call tools, read results and build a cited brief.\n" + tools_note,
                "coding": "You are a coding agent working in the repository below. Make careful, minimal changes.\n" + tools_note,
                "pipeline": "You are a document-processing service. Follow the output format exactly.",
                "analysis": "You are a senior analyst. Write thorough, well-structured reports.",
            }[w.kind]
            if system is None:
                system = pad_to(base, w.system_tokens, filler[rng.randrange(len(filler) // 2):])
            turns: list[Turn] = []
            context = corpus.files(rng, lognormal(rng, w.context)) if w.kind == "coding" and w.context else (
                corpus.text(rng, lognormal(rng, w.context))[1] if w.context else "")
            for t in range(w.turns):
                n_in, n_out = lognormal(rng, w.input), lognormal(rng, w.output)
                cap = w.output[3] + REASONING_ROOM
                if w.kind == "chat":
                    body = rng.choice(ASSISTANT_ASKS) if t == 0 else rng.choice(FOLLOW_UPS)
                    if n_in > 200:
                        body += "\n\nContext from my notes:\n" + corpus.text(rng, n_in - 40)[1]
                elif w.kind == "docqa":
                    title, doc = corpus.text(rng, n_in)
                    body = f"Document: {title}\n\n{doc}\n\nQuestion: {rng.choice(DOC_QUESTIONS)}"
                elif w.kind == "service":
                    body = SERVICE_TURNS[t % len(SERVICE_TURNS)]
                    if t == 0:
                        body = f"Customer account notes:\n{context}\n\nCustomer: {body}"
                    elif n_in > 300:
                        body += "\n\nCRM lookup result:\n" + corpus.text(rng, n_in - 60)[1]
                elif w.kind == "research":
                    body = (f"Goal: {rng.choice(RESEARCH_GOALS)}\n\nNotes so far:\n{context}" if t == 0
                            else "Tool output (search):\n" + corpus.text(rng, n_in)[1])
                elif w.kind == "coding":
                    body = (f"Repository files:\n{context}\n\nTask: {rng.choice(CODING_TASKS)}" if t == 0
                            else "Tool output (read_file):\n" + corpus.files(rng, n_in))
                elif w.kind == "pipeline":
                    title, doc = corpus.text(rng, n_in)
                    body = f"{rng.choice(PIPELINE_ASKS)}\n\nDocument: {title}\n\n{doc}"
                else:
                    title, doc = corpus.text(rng, n_in)
                    body = f"{rng.choice(ANALYSIS_ASKS)}\n\nMaterial: {title}\n\n{doc}"
                body += f"\n\n(Reply in about {max(20, round(n_out * 0.75))} words.)"
                turns.append(Turn(body, n_out, cap))
            sessions.append(Session(w.id, s, system, turns))
    return sessions


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------
def call_cost(m: dict, prompt: int, cache_read: int, cache_write: int, output: int) -> float:
    """Dollars for one call. ``prompt`` is the whole prompt; reads and writes
    are the parts billed at the cache prices. Output includes reasoning."""
    rule = m.get("long")
    k_in = k_out = 1.0
    if rule and prompt > rule["over"]:
        k_in, k_out = rule["input"], rule["output"]
    uncached = max(0, prompt - cache_read - cache_write)
    return (uncached * m["input"] * k_in + cache_read * m["cached"] * k_in
            + cache_write * m["cache_write"] * k_in + output * m["output"] * k_out) / 1e6


def plan(sessions: list[Session], reasoning_share: float = 1.0) -> dict:
    """Three prices per model and workload, all from the built prompts:
    low - answers only, the shared prefix read from cache after the first call;
    expected - answers plus reasoning of ``reasoning_share`` x the answer, no cache;
    ceiling - every call at its max_output, no cache (what the budget guard allows for)."""
    out: dict = {}
    for mid, m in MODELS.items():
        if m.get("optional"):
            continue
        rows = {}
        for w in WORKLOADS:
            low = exp = ceil = 0.0
            calls = prompt_total = 0
            for s in (x for x in sessions if x.workload == w.id):
                history, sys_tok = 0, est_tokens(s.system)
                for i, t in enumerate(s.turns):
                    p = sys_tok + history + est_tokens(t.text)
                    cached = sys_tok + history if (i > 0 or s.index > 0) else 0
                    low += call_cost(m, p, cached, 0, t.answer_tokens)
                    exp += call_cost(m, p, 0, 0, round(t.answer_tokens * (1 + reasoning_share)))
                    ceil += call_cost(m, p, 0, 0, t.max_output)
                    history += est_tokens(t.text) + t.answer_tokens
                    calls += 1
                    prompt_total += p
            rows[w.id] = {"calls": calls, "prompt_tokens": prompt_total,
                          "low": round(low, 4), "expected": round(exp, 4), "ceiling": round(ceil, 4)}
        out[mid] = rows
    return out


def print_plan(p: dict) -> None:
    for mid, rows in p.items():
        tot = {k: sum(r[k] for r in rows.values()) for k in ("calls", "prompt_tokens", "low", "expected", "ceiling")}
        print(f"\n{mid} ({MODELS[mid]['model']})")
        print(f"  {'workload':20} {'calls':>5} {'prompt tok':>11} {'low $':>8} {'expected $':>11} {'ceiling $':>10}")
        for wid, r in list(rows.items()) + [("total", tot)]:
            print(f"  {wid:20} {r['calls']:>5} {r['prompt_tokens']:>11,} {r['low']:>8.2f} {r['expected']:>11.2f} {r['ceiling']:>10.2f}")
    print(f"\nAll models: low ${sum(sum(r['low'] for r in v.values()) for v in p.values()):.2f}, "
          f"expected ${sum(sum(r['expected'] for r in v.values()) for v in p.values()):.2f}, "
          f"ceiling ${sum(sum(r['ceiling'] for r in v.values()) for v in p.values()):.2f}")


# ---------------------------------------------------------------------------
# Live calls: one adapter per provider, each returning the same usage record.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Usage:
    prompt: int
    cache_read: int
    cache_write: int
    output: int          # billed output, reasoning included
    reasoning: int
    text: str
    stop: str
    first_token_s: float | None = None   # first streamed delta of any kind
    first_text_s: float | None = None    # first visible answer text


class _Clock:
    """Times one streamed call: the first delta of any kind, and the first
    delta of visible answer text. Hidden reasoning (OpenAI, Gemini) streams
    nothing, so there the two are the same moment."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.first_token: float | None = None
        self.first_text: float | None = None

    def token(self, visible: bool) -> None:
        now = round(time.perf_counter() - self.t0, 3)
        if self.first_token is None:
            self.first_token = now
        if visible and self.first_text is None:
            self.first_text = now


def call_openai(client, m: dict, system: str, msgs: list[dict], max_out: int) -> Usage:
    clock, final, parts = _Clock(), None, []
    events = client.responses.create(
        model=m["model"], instructions=system, input=msgs, max_output_tokens=max_out,
        reasoning={"effort": m["reasoning_effort"]}, store=False, stream=True)
    for ev in events:
        if ev.type == "response.output_text.delta":
            clock.token(True)
            parts.append(ev.delta)
        elif ev.type.endswith(".delta"):
            clock.token(False)
        elif ev.type in ("response.completed", "response.incomplete", "response.failed"):
            final = ev.response
    if final is None:
        raise RuntimeError("stream ended without a final response")
    u = final.usage
    det_in, det_out = u.input_tokens_details, u.output_tokens_details
    return Usage(u.input_tokens, det_in.cached_tokens or 0, getattr(det_in, "cache_write_tokens", 0) or 0,
                 u.output_tokens, det_out.reasoning_tokens or 0, "".join(parts), final.status or "",
                 clock.first_token, clock.first_text)


def call_anthropic(client, m: dict, system: str, msgs: list[dict], max_out: int) -> Usage:
    # Streamed: the SDK refuses a non-streaming call whose max_tokens could
    # run past ten minutes, and deep-analysis caps are 24K.
    clock = _Clock()
    with client.messages.stream(
            model=m["model"], max_tokens=max_out, system=system, messages=msgs,
            output_config={"effort": m["effort"]}, cache_control={"type": "ephemeral"}) as stream:
        for ev in stream:
            if ev.type == "content_block_delta":
                clock.token(ev.delta.type == "text_delta")
        r = stream.get_final_message()
    u = r.usage
    read, write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
    thinking = (u.output_tokens_details.thinking_tokens if u.output_tokens_details else 0) or 0
    text = "".join(b.text for b in r.content if b.type == "text")
    return Usage(u.input_tokens + read + write, read, write, u.output_tokens, thinking, text, r.stop_reason or "",
                 clock.first_token, clock.first_text)


def call_google(client, m: dict, system: str, msgs: list[dict], max_out: int) -> Usage:
    from google.genai import types
    contents = [types.Content(role="user" if x["role"] == "user" else "model", parts=[types.Part(text=x["content"])])
                for x in msgs]
    clock, parts, u, stop = _Clock(), [], None, ""
    for chunk in client.models.generate_content_stream(
            model=m["model"], contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=max_out,
                thinking_config=types.ThinkingConfig(thinking_level=m["thinking_level"].upper()))):
        cand = chunk.candidates[0] if chunk.candidates else None
        text = "".join(p.text for p in (cand.content.parts if cand and cand.content and cand.content.parts else [])
                       if p.text and not p.thought)
        if text:
            clock.token(True)
            parts.append(text)
        if cand and cand.finish_reason:
            stop = cand.finish_reason.name
        if chunk.usage_metadata:
            u = chunk.usage_metadata
    if u is None:
        raise RuntimeError("stream ended without usage")
    thoughts = u.thoughts_token_count or 0
    return Usage(u.prompt_token_count or 0, u.cached_content_token_count or 0, 0,
                 (u.candidates_token_count or 0) + thoughts, thoughts, "".join(parts), stop,
                 clock.first_token, clock.first_text)


KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}


def make_client(provider: str):
    if provider == "openai":
        import openai
        return openai.OpenAI(max_retries=3)
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(max_retries=3)
    from google import genai
    return genai.Client(api_key=os.environ[KEYS["google"]])


CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}


def run(sessions: list[Session], model_ids: list[str], budget: float, out_dir: Path) -> None:
    missing = [KEYS[MODELS[m]["provider"]] for m in model_ids if not os.environ.get(KEYS[MODELS[m]["provider"]])]
    if missing:
        raise SystemExit(f"missing keys: {', '.join(sorted(set(missing)))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "calls.jsonl").open("a")
    spent = 0.0
    for mid in model_ids:
        m = MODELS[mid]
        client, call = make_client(m["provider"]), CALLERS[m["provider"]]
        for s in sessions:
            msgs: list[dict] = []
            for i, t in enumerate(s.turns):
                msgs.append({"role": "user", "content": t.text})
                worst = call_cost(m, est_tokens(s.system) + sum(est_tokens(x["content"]) for x in msgs), 0, 0, t.max_output)
                if spent + worst > budget:
                    print(f"budget: ${spent:.2f} spent, next call could cost ${worst:.2f}; stopping", flush=True)
                    log.close()
                    return
                t0 = time.time()
                try:
                    u = call(client, m, s.system, msgs, t.max_output)
                except Exception as exc:  # noqa: BLE001 - record and move on to the next session
                    log.write(json.dumps({"model": mid, "workload": s.workload, "session": s.index, "turn": i,
                                          "error": f"{type(exc).__name__}: {exc}"[:400]}) + "\n")
                    log.flush()
                    break
                cost = call_cost(m, u.prompt, u.cache_read, u.cache_write, u.output)
                spent += cost
                visible = max(0, u.output - u.reasoning)
                log.write(json.dumps({
                    "model": mid, "workload": s.workload, "session": s.index, "turn": i,
                    "prompt": u.prompt, "cache_read": u.cache_read, "cache_write": u.cache_write,
                    "output": u.output, "reasoning": u.reasoning, "visible": visible,
                    "answer_target": t.answer_tokens, "max_output": t.max_output, "stop": u.stop,
                    "cost": round(cost, 6), "seconds": round(time.time() - t0, 2),
                    "first_token_s": u.first_token_s, "first_text_s": u.first_text_s}) + "\n")
                log.flush()
                msgs.append({"role": "assistant", "content": u.text or "(no text)"})
        print(f"{mid}: done, ${spent:.2f} spent so far", flush=True)
    log.close()


def summarize(out_dir: Path) -> dict:
    rows = [json.loads(x) for x in (out_dir / "calls.jsonl").read_text().splitlines() if x.strip()]
    out: dict = {}
    for r in rows:
        k = out.setdefault(r["model"], {}).setdefault(r["workload"], {
            "calls": 0, "errors": 0, "reasoning": 0, "visible": 0, "answer_target": 0,
            "prompt": 0, "cache_read": 0, "cost": 0.0, "hit_max": 0,
            "first_text_s": [], "seconds": [], "answer_tok_s": []})
        if "error" in r:
            k["errors"] += 1
            continue
        k["calls"] += 1
        for f in ("reasoning", "visible", "answer_target", "prompt", "cache_read", "cost"):
            k[f] += r[f]
        k["hit_max"] += int(r["output"] >= r["max_output"])
        k["seconds"].append(r["seconds"])
        ft = r.get("first_text_s")
        if ft is not None:
            k["first_text_s"].append(ft)
            if r["visible"] > 1 and r["seconds"] - ft > 0.05:
                k["answer_tok_s"].append(r["visible"] / (r["seconds"] - ft))
    for ws in out.values():
        for k in ws.values():
            for f in ("first_text_s", "seconds", "answer_tok_s"):
                v = sorted(k.pop(f))
                k[f] = {"n": len(v), "median": round(v[len(v) // 2], 2) if v else None,
                        "p90": round(v[min(len(v) - 1, int(len(v) * 0.9))], 2) if v else None}
    for mid, ws in out.items():
        print(f"\n{mid}")
        print(f"  {'workload':20} {'calls':>5} {'err':>4} {'reason/turn':>11} {'reason/answer':>13} {'cached':>7} {'$':>7} {'hit cap':>7}")
        for wid, k in ws.items():
            n = max(1, k["calls"])
            print(f"  {wid:20} {k['calls']:>5} {k['errors']:>4} {k['reasoning']/n:>11.0f} "
                  f"{k['reasoning']/max(1, k['visible']):>13.2f} {k['cache_read']/max(1, k['prompt']):>7.0%} "
                  f"{k['cost']:>7.2f} {k['hit_max']:>7}")
        print(f"  {'workload':20} {'answer starts, median / p90 s':>30} {'whole reply, median s':>22} {'answer tok/s, median':>21}")
        for wid, k in ws.items():
            ft, sec, ts = k["first_text_s"], k["seconds"], k["answer_tok_s"]
            fts = f"{ft['median']} / {ft['p90']}" if ft["n"] else "not timed"
            print(f"  {wid:20} {fts:>30} {sec['median'] or '':>22} {ts['median'] or '':>21}")
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--reasoning-share", type=float, default=1.0,
                   help="expected reasoning as a multiple of the answer (default 1.0)")
    p.add_argument("--json", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--models", default=",".join(k for k, v in MODELS.items() if not v.get("optional")))
    r.add_argument("--budget-usd", type=float, required=True)
    r.add_argument("--out", default="runs/cloud-token-probe")
    s = sub.add_parser("summary")
    s.add_argument("--out", default="runs/cloud-token-probe")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        summarize(Path(a.out))
        return
    sessions = build_sessions()
    if a.cmd == "plan":
        result = plan(sessions, a.reasoning_share)
        print(json.dumps(result, indent=1)) if a.json else print_plan(result)
        return
    ids = [x.strip() for x in a.models.split(",") if x.strip()]
    unknown = [x for x in ids if x not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choose from {list(MODELS)}")
    run(sessions, ids, a.budget_usd, Path(a.out))
    summarize(Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
