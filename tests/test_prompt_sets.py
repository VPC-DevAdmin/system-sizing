"""Tests for simulator.prompt_sets.

Parquet sources are served through a monkeypatched ``_read_parquet`` so
the suite runs without pyarrow/pandas (neither is a package
dependency); one test round-trips real parquet when pyarrow exists.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from simulator import prompt_sets as ps

DOLLY_CATS = [
    "open_qa", "general_qa", "brainstorming", "creative_writing", "classification",
    "information_extraction", "summarization", "closed_qa",
]

MBPP_DIR = ("google-research-datasets__mbpp", "full")
HE_PATH = ("openai__openai_humaneval", "openai_humaneval", "test-00000-of-00001.parquet")
GSM_PATH = ("openai__gsm8k", "main", "test-00000-of-00001.parquet")
ARC_PATH = ("allenai__ai2_arc", "ARC-Challenge", "test-00000-of-00001.parquet")
MMLU_PATH = ("cais__mmlu", "all", "test-00000-of-00001.parquet")
JSON_MODE_PATH = ("NousResearch__json-mode-eval", "data", "train-00000-of-00001.parquet")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _parquet_tables(n_mbpp: int = 30, n_he: int = 5) -> dict[tuple, list[dict]]:
    mbpp_train = [
        {"task_id": i, "text": f"task {i}", "code": "", "test_list": [f"assert f({i}) == {i}", "assert 2"]}
        for i in range(n_mbpp)
    ]
    mbpp_test = [
        {"task_id": 1000 + i, "text": f"test task {i}", "code": "", "test_list": [f"assert g({i})"]}
        for i in range(3)
    ]
    return {
        MBPP_DIR + ("train-00000-of-00001.parquet",): mbpp_train,
        MBPP_DIR + ("test-00000-of-00001.parquet",): mbpp_test,
        HE_PATH: [
            {"task_id": f"HumanEval/{i}", "prompt": f"def h{i}():\n    pass\n", "canonical_solution": "",
             "test": "", "entry_point": f"h{i}"}
            for i in range(n_he)
        ],
        GSM_PATH: [{"question": f"What is {i}+1?", "answer": str(i + 1)} for i in range(10)],
        ARC_PATH: [
            {"id": "arc-a", "question": "Which is a gas?",
             "choices": {"text": ["rock", "air", "wood", "iron"], "label": ["A", "B", "C", "D"]},
             "answerKey": "B"},
            {"id": "arc-n", "question": "Numbered?",
             "choices": {"text": ["one", "two", "three"], "label": ["1", "2", "3"]}, "answerKey": "2"},
        ],
        MMLU_PATH: [
            {"question": "Force equals?", "subject": "college_physics", "choices": ["ma", "mv", "m/a", "a/m"],
             "answer": 0},
            {"question": "Who wrote it?", "subject": "high_school_european_history",
             "choices": ["a", "b", "c", "d"], "answer": 1},
            {"question": "Gene?", "subject": "medical_genetics", "choices": ["w", "x", "y", "z"], "answer": 2},
        ],
        JSON_MODE_PATH: [
            {"prompt": [
                {"role": "system", "content": "Output JSON matching schema {\"a\": int}"},
                {"role": "user", "content": "Give me a"},
            ], "completion": "{}", "schema": "{}"},
        ],
    }


def _make_root(tmp_path: Path, monkeypatch, tables: dict | None = None, *, skip: tuple = ()) -> Path:
    root = tmp_path / "ds"
    tables = _parquet_tables() if tables is None else tables
    fake = {}
    for parts, rows in tables.items():
        if parts in skip:
            continue
        p = root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")  # existence only; rows come from the fake reader
        fake[p] = rows

    def fake_read(path):
        path = Path(path)
        if path not in fake:
            raise FileNotFoundError(path)
        return fake[path]

    monkeypatch.setattr(ps, "_read_parquet", fake_read)

    dolly = []
    for i, cat in enumerate(DOLLY_CATS * 5):
        dolly.append({"instruction": f"instr {i} {cat}", "context": f"ctx {i}" if i % 2 else "",
                      "response": "", "category": cat})
    dolly.append({"instruction": "   ", "context": "", "response": "", "category": "open_qa"})  # empty
    _write_jsonl(root / "databricks__databricks-dolly-15k" / "databricks-dolly-15k.jsonl", dolly)
    _write_jsonl(root / "HuggingFaceH4__MATH-500" / "test.jsonl",
                 [{"problem": f"Solve x={i}", "solution": "", "answer": str(i), "subject": "Algebra",
                   "level": 1, "unique_id": f"u{i}"} for i in range(6)])
    zh = root / "shibing624__alpaca-zh" / "alpaca_gpt4_data_zh.json"
    zh.parent.mkdir(parents=True, exist_ok=True)
    zh_rows = [{"instruction": f"写一首诗 {i}", "input": "关于春天" if i % 2 else "", "output": ""} for i in range(8)]
    zh_rows.append({"instruction": "长" * 5000, "input": "", "output": ""})
    zh.write_text(json.dumps(zh_rows, ensure_ascii=False), encoding="utf-8")
    return root


def _user_text(item: dict) -> str:
    return [m for m in item["messages"] if m["role"] == "user"][-1]["content"]


def _by(items, **kw):
    return [i for i in items if all(i[k] == v for k, v in kw.items())]


# --------------------------------------------------------------------------


def test_formatting_per_domain(tmp_path, monkeypatch):
    root = _make_root(tmp_path, monkeypatch)
    items = ps.build_prompt_set(root, per_domain=1000)
    assert {i["domain"] for i in items} == set(ps.DOMAINS)
    for it in items:
        assert set(it) == {"id", "domain", "source", "split", "messages"}
        assert it["id"].startswith(f"{it['domain']}:{it['source']}:")
        assert it["split"] in ("calib", "eval")

    chat = _by(items, domain="chat")
    assert all(i["source"] == "dolly" for i in chat)
    assert len(chat) == 25  # 5 categories x 5, empty-instruction row dropped
    texts = {_user_text(i) for i in chat}
    assert "instr 1 general_qa\n\nctx 1" in texts
    assert "instr 0 open_qa" in texts
    assert not any("closed_qa" in t for t in texts)

    mbpp = next(i for i in items if i["id"] == "code:mbpp:3")
    assert _user_text(mbpp) == "Write a Python function for the following task.\n\ntask 3\n\nIt must pass:\nassert f(3) == 3"
    assert any(i["id"] == "code:mbpp:1001" for i in items)  # other splits included
    he = next(i for i in items if i["id"] == "code:humaneval:HumanEval-2")
    assert _user_text(he) == "Complete the following Python function.\n\n```python\ndef h2():\n    pass\n```"

    math = _by(items, domain="math")
    assert {i["source"] for i in math} == {"gsm8k", "math500"}
    assert all(_user_text(i).endswith("\n\nShow your reasoning, then give the final answer.") for i in math)
    assert any(_user_text(i).startswith("Solve x=") for i in math)

    sci = _by(items, domain="science")
    mmlu = _by(sci, source="mmlu")
    assert len(mmlu) == 2  # history subject filtered out
    phys = next(i for i in mmlu if "Force" in _user_text(i))
    assert _user_text(phys) == ("Force equals?\n\nA. ma\nB. mv\nC. m/a\nD. a/m"
                                "\n\nExplain your reasoning and give the letter of the answer.")
    arc = {i["id"]: _user_text(i) for i in _by(sci, source="arc")}
    assert "B. air" in arc["science:arc:arc-a"]
    assert "A. one\nB. two\nC. three" in arc["science:arc:arc-n"]  # numeric labels -> letters

    zh = _by(items, domain="chinese")
    assert "写一首诗 1\n\n关于春天" in {_user_text(i) for i in zh}

    extract = _by(items, domain="extract")
    assert {i["source"] for i in extract} == {"dolly", "json_mode"}
    dolly_x = [_user_text(i) for i in _by(extract, source="dolly")]
    assert len(dolly_x) == 15
    assert all(any(c in t for c in ("information_extraction", "summarization", "closed_qa")) for t in dolly_x)


def test_json_mode_messages_passed_through(tmp_path, monkeypatch):
    root = _make_root(tmp_path, monkeypatch)
    items = ps.build_prompt_set(root, per_domain=1000)
    jm = _by(items, source="json_mode")
    assert len(jm) == 1
    assert jm[0]["messages"] == [
        {"role": "system", "content": "Output JSON matching schema {\"a\": int}"},
        {"role": "user", "content": "Give me a"},
    ]


def test_normalise_messages_handles_arrays_and_strings():
    class FakeArray:  # stands in for a numpy object array from pandas
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return list(self._data)

    msgs = [{"role": "user", "content": "hi", "extra": 1}]
    assert ps._normalise_messages(FakeArray(msgs)) == [{"role": "user", "content": "hi"}]
    assert ps._normalise_messages(json.dumps(msgs)) == [{"role": "user", "content": "hi"}]
    assert ps._normalise_messages(None) == []


def test_truncation_limits(tmp_path, monkeypatch):
    tables = _parquet_tables()
    tables[GSM_PATH] = [{"question": "q" * 10000, "answer": "1"}]
    root = _make_root(tmp_path, monkeypatch, tables)
    items = ps.build_prompt_set(root, per_domain=1000)
    long_math = next(i for i in items if i["id"] == "math:gsm8k:0")
    assert _user_text(long_math) == "q" * 6000  # hard cut, suffix gone, no ellipsis
    long_zh = next(i for i in _by(items, domain="chinese") if _user_text(i).startswith("长"))
    assert _user_text(long_zh) == "长" * 2500


def test_split_halves_and_disjoint(tmp_path, monkeypatch):
    root = _make_root(tmp_path, monkeypatch)
    items = ps.build_prompt_set(root, per_domain=20)
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    for domain in ps.DOMAINS:
        d = _by(items, domain=domain)
        calib, ev = _by(d, split="calib"), _by(d, split="eval")
        assert len(calib) == len(d) // 2
        assert len(ev) == len(d) - len(d) // 2
        assert not {i["id"] for i in calib} & {i["id"] for i in ev}
    # capped domains: 20 of 25 chat, 20 of 38 code
    assert len(_by(items, domain="chat")) == 20
    assert len(_by(items, domain="code")) == 20
    # small domain uses everything: 6 math500 + 10 gsm8k
    assert len(_by(items, domain="math")) == 16


def test_determinism(tmp_path, monkeypatch):
    root = _make_root(tmp_path, monkeypatch)
    a = ps.build_prompt_set(root, per_domain=12, seed=7)
    b = ps.build_prompt_set(root, per_domain=12, seed=7)
    c = ps.build_prompt_set(root, per_domain=12, seed=8)
    assert a == b
    assert [i["id"] for i in a] != [i["id"] for i in c]


def test_code_interleaves_sources(tmp_path, monkeypatch):
    root = _make_root(tmp_path, monkeypatch, _parquet_tables(n_mbpp=30, n_he=5))
    code = _by(ps.build_prompt_set(root, per_domain=20), domain="code")
    sources = [i["source"] for i in code]
    # round-robin until HumanEval's 5 run out, then MBPP fills the rest
    assert sources[:10] == ["mbpp", "humaneval"] * 5
    assert sources[10:] == ["mbpp"] * 10
    # both halves carry HumanEval
    assert {"mbpp", "humaneval"} <= {i["source"] for i in _by(code, split="calib")}


def test_missing_sources_tolerated(tmp_path, monkeypatch, caplog):
    root = _make_root(tmp_path, monkeypatch, skip=(HE_PATH, ARC_PATH, MMLU_PATH))
    (root / "shibing624__alpaca-zh" / "alpaca_gpt4_data_zh.json").unlink()
    with caplog.at_level("WARNING", logger="simulator.prompt_sets"):
        items = ps.build_prompt_set(root, per_domain=100)
    domains = {i["domain"] for i in items}
    assert "science" not in domains and "chinese" not in domains
    assert {i["source"] for i in _by(items, domain="code")} == {"mbpp"}
    assert "humaneval" in caplog.text and "chinese" in caplog.text


def test_empty_root(tmp_path):
    assert ps.build_prompt_set(tmp_path / "nothing") == []


def test_write_load_roundtrip(tmp_path, monkeypatch):
    root = _make_root(tmp_path, monkeypatch)
    items = ps.build_prompt_set(root, per_domain=10)
    out = tmp_path / "out" / "set.jsonl"
    ps.write_prompt_set(items, out)
    assert "写一首诗" in out.read_text(encoding="utf-8")  # ensure_ascii=False
    assert ps.load_prompt_set(out) == items


def test_mixes_defined():
    assert set(ps.MIXES) == {"balanced", "code_heavy", "chinese_heavy"}
    for mix in ps.MIXES.values():
        assert set(mix) == set(ps.DOMAINS)
        assert sum(mix.values()) == pytest.approx(1.0)
    assert ps.MIXES["code_heavy"]["code"] == pytest.approx(0.70)
    assert ps.MIXES["code_heavy"]["math"] == pytest.approx(0.06)
    assert ps.MIXES["chinese_heavy"]["chinese"] == pytest.approx(0.70)
    assert ps.MIXES["balanced"]["chat"] == pytest.approx(1 / 6)


def _synthetic_items(n_per_domain: int = 10, domains=ps.DOMAINS) -> list[dict]:
    items = []
    for d in domains:
        for k in range(n_per_domain):
            items.append({"id": f"{d}:s:{k}", "domain": d, "source": "s",
                          "split": "calib" if k < n_per_domain // 2 else "eval",
                          "messages": [{"role": "user", "content": f"{d} {k}"}]})
    return items


def test_mix_sampler_weights_and_split():
    items = _synthetic_items()
    sampler = ps.MixSampler(items, ps.MIXES["code_heavy"], "eval", seed=1)
    draws = [sampler.next() for _ in range(3000)]
    assert all(d["split"] == "eval" for d in draws)
    frac = {d: n / 3000 for d, n in Counter(x["domain"] for x in draws).items()}
    assert frac["code"] == pytest.approx(0.70, abs=0.03)
    for d in ps.DOMAINS:
        if d != "code":
            assert frac[d] == pytest.approx(0.06, abs=0.02)
    # deterministic
    again = ps.MixSampler(items, ps.MIXES["code_heavy"], "eval", seed=1)
    assert [again.next()["id"] for _ in range(3000)] == [d["id"] for d in draws]


def test_mix_sampler_cycles_before_repeating():
    items = _synthetic_items(n_per_domain=10, domains=("code",))
    sampler = ps.MixSampler(items, {"code": 1.0}, "calib", seed=3)
    first = [sampler.next()["id"] for _ in range(5)]
    second = [sampler.next()["id"] for _ in range(5)]
    assert sorted(first) == sorted(second) == sorted(f"code:s:{k}" for k in range(5))


def test_mix_sampler_renormalises_missing_domains():
    items = _synthetic_items(domains=("chat", "code"))
    sampler = ps.MixSampler(items, ps.MIXES["chinese_heavy"], "eval", seed=2)
    assert sampler.weights == pytest.approx({"chat": 0.5, "code": 0.5})
    assert {sampler.next()["domain"] for _ in range(200)} == {"chat", "code"}
    with pytest.raises(ValueError):
        ps.MixSampler(items, {"chinese": 1.0}, "eval", seed=2)


def test_calibration_items_round_robin():
    items = _synthetic_items(n_per_domain=10, domains=("code", "chat", "math"))
    items += [{"id": "math:s:x", "domain": "math", "source": "s", "split": "calib", "messages": []}]
    cal = ps.calibration_items(items)
    assert all(i["split"] == "calib" for i in cal)
    assert len(cal) == 16
    # DOMAINS order (chat, code, math), one each per round; math's extra last
    assert [i["domain"] for i in cal[:6]] == ["chat", "code", "math"] * 2
    assert cal[-1]["id"] == "math:s:x"


def test_cli_build(tmp_path, monkeypatch, capsys):
    root = _make_root(tmp_path, monkeypatch)
    out = tmp_path / "cli.jsonl"
    assert ps._main(["build", "--root", str(root), "--out", str(out), "--per-domain", "8", "--seed", "5"]) == 0
    printed = capsys.readouterr().out
    assert "code" in printed and "calib=   4" in printed
    assert len(ps.load_prompt_set(out)) == 5 * 8 + 4  # science has only 4 fixtures


def test_real_parquet_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "ds"
    tables = _parquet_tables(n_mbpp=4, n_he=2)
    for parts, rows in tables.items():
        p = root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), p)
    items = ps.build_prompt_set(root, per_domain=100)
    assert {i["source"] for i in _by(items, domain="code")} == {"mbpp", "humaneval"}
    assert _by(items, source="json_mode")[0]["messages"][1] == {"role": "user", "content": "Give me a"}
    assert "B. air" in next(_user_text(i) for i in items if i["id"] == "science:arc:arc-a")
