"""Multi-domain prompt sets for the KTransformers expert-placement study.

KTransformers can pin a subset of MoE experts on GPU and leave the rest
on CPU. Which experts are "hot" is not a property of the model alone:
the router picks experts per token from the content, so a code prompt
lights up different experts than a Chinese chat turn. A placement is
therefore only as good as the traffic it was learned from, and the
experiment has to answer two questions separately:

* **Does placement help at all?** Learn the hot set on one half of each
  domain (CALIBRATION) and measure on the other half (EVALUATION).
  Measuring on the prompts that chose the hot set would grade the
  placement on its own homework — the same text routes to the same
  experts, so the hit rate would be flattering by construction.

* **How brittle is it?** Measure the same placement under different
  domain MIXES. A placement calibrated on balanced traffic and then hit
  with 70% code (or 70% Chinese) shows what distribution shift costs,
  which is the number an operator actually needs before trusting a
  static placement in production.

Everything comes from public datasets already downloaded on the host
(``hf_hub_download(..., local_dir=root/"<org>__<name>")``), rendered as
chat-completion messages. Sampling is seeded per domain, so adding a
domain or changing ``per_domain`` for one run never reshuffles another
domain's calibration/evaluation split.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

DOMAINS: tuple[str, ...] = ("chat", "code", "math", "science", "chinese", "extract")

DEFAULT_ROOT = "/data/capsim/datasets/kt_placement"
DEFAULT_PER_DOMAIN = 400
DEFAULT_SEED = 20260923
DEFAULT_MAX_CHARS = 6000
# Chinese runs at roughly 1.5 characters per token against ~4 for
# English, so the same character cap would admit ~2.5x the tokens.
# 2500 chars keeps a Chinese prompt in the same token range as a
# 6000-char English one.
MAX_CHARS_BY_DOMAIN = {"chinese": 2500}

_CHAT_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing", "classification"}
# Extraction-shaped work: the answer is mostly in the supplied context.
# These route differently from open generation (long prefill, short
# decode), which is why they are their own domain and not part of chat.
_EXTRACT_CATEGORIES = {"information_extraction", "summarization", "closed_qa"}
# MMLU has 57 subjects and roughly half are humanities/social science,
# which would dilute "science" into general knowledge. Substring match
# keeps college_physics, high_school_statistics and medical_genetics;
# clinical_knowledge and professional_law fall out.
_STEM_MARKERS = (
    "physics", "chemistry", "biology", "math", "computer", "engineering",
    "astronomy", "statistics", "medicine", "anatomy", "genetics",
    "electrical", "machine_learning", "virology", "nutrition",
)

_MATH_SUFFIX = "\n\nShow your reasoning, then give the final answer."
_SCIENCE_SUFFIX = "\n\nExplain your reasoning and give the letter of the answer."

Message = dict[str, str]
# (source, stable key within that source, messages)
Candidate = tuple[str, str, list[Message]]


# --------------------------------------------------------------------------
# Readers


def _read_parquet(path: Path) -> list[dict]:
    """Rows of a parquet file as plain dicts.

    pyarrow is preferred because ``to_pylist`` yields Python lists and
    dicts; pandas yields numpy arrays for list columns, which callers
    normalise with ``_as_list``. Neither is a declared dependency of the
    package, so absence surfaces as ImportError and the source is
    skipped rather than failing the whole build.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        return pq.read_table(path).to_pylist()
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("reading parquet needs pyarrow or pandas") from exc
    return pd.read_parquet(path).to_dict("records")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_json_array(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _as_list(value: Any) -> list:
    """Normalise a list-ish cell (list, tuple, numpy array, None)."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _user(content: str) -> list[Message]:
    return [{"role": "user", "content": content}]


def _join(head: Any, tail: Any) -> str:
    head, tail = _text(head).strip(), _text(tail).strip()
    return f"{head}\n\n{tail}" if tail else head


def _lettered(question: str, options: list[tuple[str, str]]) -> str:
    body = "\n".join(f"{label}. {text}" for label, text in options)
    return f"{question.strip()}\n\n{body}{_SCIENCE_SUFFIX}"


# --------------------------------------------------------------------------
# Sources. Each yields candidates in file order; keys are stable for a
# given download so ids survive a rebuild.


def _dolly(root: Path, categories: set[str]) -> list[Candidate]:
    path = root / "databricks__databricks-dolly-15k" / "databricks-dolly-15k.jsonl"
    out = []
    for i, row in enumerate(_read_jsonl(path)):
        if row.get("category") in categories:
            out.append(("dolly", str(i), _user(_join(row.get("instruction"), row.get("context")))))
    return out


def _mbpp(root: Path) -> list[Candidate]:
    base = root / "google-research-datasets__mbpp" / "full"
    out = []
    found = False
    for split in ("train", "test", "validation", "prompt"):
        path = base / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            continue
        found = True
        for i, row in enumerate(_read_parquet(path)):
            tests = _as_list(row.get("test_list"))
            text = _text(row.get("text")).strip()
            msg = f"Write a Python function for the following task.\n\n{text}"
            if tests:
                msg += f"\n\nIt must pass:\n{tests[0]}"
            key = _text(row.get("task_id")) or f"{split}-{i}"
            out.append(("mbpp", key, _user(msg)))
    if not found:
        raise FileNotFoundError(base)
    return out


def _humaneval(root: Path) -> list[Candidate]:
    path = root / "openai__openai_humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet"
    out = []
    for i, row in enumerate(_read_parquet(path)):
        msg = f"Complete the following Python function.\n\n```python\n{_text(row.get('prompt'))}```"
        key = _text(row.get("task_id")).replace("/", "-") or str(i)
        out.append(("humaneval", key, _user(msg)))
    return out


def _gsm8k(root: Path) -> list[Candidate]:
    path = root / "openai__gsm8k" / "main" / "test-00000-of-00001.parquet"
    return [
        ("gsm8k", str(i), _user(_text(row.get("question")).strip() + _MATH_SUFFIX))
        for i, row in enumerate(_read_parquet(path))
    ]


def _math500(root: Path) -> list[Candidate]:
    path = root / "HuggingFaceH4__MATH-500" / "test.jsonl"
    return [
        ("math500", str(i), _user(_text(row.get("problem")).strip() + _MATH_SUFFIX))
        for i, row in enumerate(_read_jsonl(path))
    ]


def _mmlu_stem(root: Path) -> list[Candidate]:
    path = root / "cais__mmlu" / "all" / "test-00000-of-00001.parquet"
    out = []
    for i, row in enumerate(_read_parquet(path)):
        subject = _text(row.get("subject"))
        if not any(m in subject for m in _STEM_MARKERS):
            continue
        choices = [_text(c) for c in _as_list(row.get("choices"))]
        options = list(zip("ABCDEFGHIJ", choices, strict=False))
        out.append(("mmlu", str(i), _user(_lettered(_text(row.get("question")), options))))
    return out


def _arc(root: Path) -> list[Candidate]:
    path = root / "allenai__ai2_arc" / "ARC-Challenge" / "test-00000-of-00001.parquet"
    out = []
    for i, row in enumerate(_read_parquet(path)):
        choices = row.get("choices") or {}
        texts = [_text(t) for t in _as_list(choices.get("text"))]
        labels = [_text(lab) for lab in _as_list(choices.get("label"))]
        # ARC mixes A-D with 1-4 labelling; normalise to letters so the
        # instruction ("give the letter") is always answerable.
        if len(labels) != len(texts) or not all(lab.isalpha() for lab in labels):
            labels = list("ABCDEFGHIJ"[: len(texts)])
        key = _text(row.get("id")) or str(i)
        out.append(("arc", key, _user(_lettered(_text(row.get("question")), list(zip(labels, texts, strict=True))))))
    return out


def _alpaca_zh(root: Path) -> list[Candidate]:
    path = root / "shibing624__alpaca-zh" / "alpaca_gpt4_data_zh.json"
    return [
        ("alpaca_zh", str(i), _user(_join(row.get("instruction"), row.get("input"))))
        for i, row in enumerate(_read_json_array(path))
    ]


def _normalise_messages(raw: Any) -> list[Message]:
    """json-mode-eval ``prompt`` → plain role/content dicts.

    Via pyarrow it is a list of dicts; via pandas a numpy object array
    of dicts; some exports store it as a JSON string. All three land
    here.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    out = []
    for m in _as_list(raw):
        if isinstance(m, dict):
            out.append({"role": _text(m.get("role")), "content": _text(m.get("content"))})
    return out


def _json_mode(root: Path) -> list[Candidate]:
    path = root / "NousResearch__json-mode-eval" / "data" / "train-00000-of-00001.parquet"
    return [
        ("json_mode", str(i), _normalise_messages(row.get("prompt")))
        for i, row in enumerate(_read_parquet(path))
    ]


# Source order within a domain is the round-robin order.
_SOURCES: dict[str, list[tuple[str, Callable[[Path], list[Candidate]]]]] = {
    "chat": [("dolly", lambda r: _dolly(r, _CHAT_CATEGORIES))],
    "code": [("mbpp", _mbpp), ("humaneval", _humaneval)],
    "math": [("gsm8k", _gsm8k), ("math500", _math500)],
    "science": [("mmlu", _mmlu_stem), ("arc", _arc)],
    "chinese": [("alpaca_zh", _alpaca_zh)],
    "extract": [("dolly", lambda r: _dolly(r, _EXTRACT_CATEGORIES)), ("json_mode", _json_mode)],
}


# --------------------------------------------------------------------------
# Build


def _truncate(messages: list[Message], max_chars: int) -> list[Message]:
    # A hard cut, no ellipsis: an appended "..." is extra tokens that
    # route through experts too, and it would be identical across every
    # long prompt — a small, systematic bias toward whatever it lights.
    return [{"role": m["role"], "content": m["content"][:max_chars]} for m in messages]


def _has_user_content(messages: list[Message]) -> bool:
    users = [m for m in messages if m.get("role") == "user"]
    return bool(users) and bool(users[-1]["content"].strip())


def _round_robin(groups: Iterable[list]) -> list:
    """Interleave lists one element at a time; shorter lists drop out.

    HumanEval has 164 rows against MBPP's ~1000: after 164 rounds the
    remainder is MBPP only, so a 400-item code domain is 164 + 236
    rather than an MBPP-only sample that happens to be larger.
    """
    queues = [list(g) for g in groups if g]
    out = []
    i = 0
    while queues:
        out.extend(q[i] for q in queues)
        i += 1
        queues = [q for q in queues if len(q) > i]
    return out


def _domain_items(root: Path, domain: str, per_domain: int, seed: int) -> list[dict]:
    max_chars = MAX_CHARS_BY_DOMAIN.get(domain, DEFAULT_MAX_CHARS)
    # A string seed keys the RNG on (seed, domain) alone, so each
    # domain's split is independent of which other domains loaded.
    rng = random.Random(f"{seed}:{domain}")
    groups = []
    for name, loader in _SOURCES[domain]:
        try:
            cands = loader(root)
        except (FileNotFoundError, ImportError) as exc:
            log.warning("prompt_sets: %s/%s skipped: %s", domain, name, exc)
            continue
        kept = []
        for source, key, messages in cands:
            messages = _truncate(messages, max_chars)
            if _has_user_content(messages):
                kept.append((source, key, messages))
        rng.shuffle(kept)
        groups.append(kept)
    picked = _round_robin(groups)[:per_domain]
    n_calib = len(picked) // 2
    return [
        {
            "id": f"{domain}:{source}:{key}",
            "domain": domain,
            "source": source,
            "split": "calib" if i < n_calib else "eval",
            "messages": messages,
        }
        for i, (source, key, messages) in enumerate(picked)
    ]


def build_prompt_set(
    root: str | Path, *, per_domain: int = DEFAULT_PER_DOMAIN, seed: int = DEFAULT_SEED
) -> list[dict]:
    """All domains' items, calib and eval, in domain order.

    Round-robin interleaving happens before the split, so both halves
    carry every source in the same proportion.
    """
    root = Path(root)
    items: list[dict] = []
    for domain in DOMAINS:
        got = _domain_items(root, domain, per_domain, seed)
        if not got:
            log.warning("prompt_sets: domain %s has no items; omitted", domain)
        items.extend(got)
    return items


def write_prompt_set(items: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_prompt_set(path: str | Path) -> list[dict]:
    return _read_jsonl(Path(path))


# --------------------------------------------------------------------------
# Mixes and sampling


def _skewed(heavy: str, share: float = 0.70) -> dict[str, float]:
    rest = (1.0 - share) / (len(DOMAINS) - 1)
    return {d: (share if d == heavy else rest) for d in DOMAINS}


# balanced is the calibration-like traffic; the two skews are the
# distribution shifts. Code and Chinese are chosen because they are the
# domains whose routing is most unlike English chat.
MIXES: dict[str, dict[str, float]] = {
    "balanced": {d: 1.0 / len(DOMAINS) for d in DOMAINS},
    "code_heavy": _skewed("code"),
    "chinese_heavy": _skewed("chinese"),
}


class MixSampler:
    """Endless stream of prompts from one split, drawn by domain weight.

    Domain is drawn per request; within a domain items cycle in a
    shuffled order and reshuffle on exhaustion, so a long run visits
    every prompt before repeating one — repeats would hit the prefix
    cache and flatter the measurement.
    """

    def __init__(self, items: list[dict], mix: dict[str, float], split: str, seed: int):
        self._rng = random.Random(seed)
        by_domain: dict[str, list[dict]] = {}
        for item in items:
            if item.get("split") == split:
                by_domain.setdefault(item["domain"], []).append(item)
        self._domains = [d for d in sorted(by_domain) if mix.get(d, 0.0) > 0]
        if not self._domains:
            raise ValueError(f"no items in split {split!r} for any domain in the mix")
        total = sum(mix[d] for d in self._domains)
        # Renormalised: a domain absent on this host must not turn into
        # a silent share of dropped draws.
        self.weights = {d: mix[d] / total for d in self._domains}
        self._pools = {d: by_domain[d] for d in self._domains}
        self._queues: dict[str, list[dict]] = {d: [] for d in self._domains}

    def next(self) -> dict:
        domain = self._rng.choices(self._domains, weights=[self.weights[d] for d in self._domains])[0]
        queue = self._queues[domain]
        if not queue:
            queue.extend(self._pools[domain])
            self._rng.shuffle(queue)
        return queue.pop()


def calibration_items(items: list[dict]) -> list[dict]:
    """Calib items interleaved across domains.

    A calibration pass may be cut short (it is expensive on a 1T model);
    interleaving means any prefix still covers every domain evenly
    rather than being all chat.
    """
    by_domain: dict[str, list[dict]] = {}
    for item in items:
        if item.get("split") == "calib":
            by_domain.setdefault(item["domain"], []).append(item)
    order = [d for d in DOMAINS if d in by_domain] + sorted(set(by_domain) - set(DOMAINS))
    return _round_robin(by_domain[d] for d in order)


# --------------------------------------------------------------------------
# CLI


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simulator.prompt_sets")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="build a prompt set JSONL from downloaded datasets")
    build.add_argument("--root", default=DEFAULT_ROOT)
    build.add_argument("--out", required=True)
    build.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    items = build_prompt_set(args.root, per_domain=args.per_domain, seed=args.seed)
    write_prompt_set(items, args.out)
    counts = Counter((i["domain"], i["split"]) for i in items)
    sources = Counter((i["domain"], i["source"]) for i in items)
    for domain in DOMAINS:
        c, e = counts.get((domain, "calib"), 0), counts.get((domain, "eval"), 0)
        if not c + e:
            print(f"{domain:8s}  (none)")
            continue
        per_src = ", ".join(f"{s}={n}" for (d, s), n in sorted(sources.items()) if d == domain)
        print(f"{domain:8s}  calib={c:4d}  eval={e:4d}  [{per_src}]")
    print(f"total {len(items)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
