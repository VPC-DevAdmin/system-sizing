from types import SimpleNamespace

import pytest

from simulator.infrastructure.knowledge_workers import (
    advance,
    compare_workers,
    make_session,
    turn_item,
)


class Tokenizer:
    def encode(self, text, **kw):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kw):
        return list(range(sum(len(m["content"].split()) for m in messages) + len(messages)))


def settings():
    return SimpleNamespace(
        seed=42,
        document_records=4,
        session_context_tokens=1024,
        session_answer_records=2,
        session_new_input_tokens=32,
        max_output_tokens=128,
        context_limit=4096,
    )


def test_history_is_preserved_and_workers_are_distinct():
    tokenizer = Tokenizer()
    spec = settings()
    a = make_session(tokenizer, spec, 0, 0)
    b = make_session(tokenizer, spec, 1, 0)
    assert a["prefix"] != b["prefix"]
    first = turn_item(a, tokenizer, spec, 0)
    advance(a, first, {"output": "Actual model answer retained"})
    second = turn_item(a, tokenizer, spec, 1)
    assert second["messages"][:2] == first["messages"] + [
        {"role": "assistant", "content": "Actual model answer retained"}
    ]
    assert second["input_tokens"] > first["input_tokens"]
    assert 1024 <= a["initial_tokens"] <= 1088
    spec.context_limit = 1050
    with pytest.raises(ValueError, match="context limit"):
        turn_item(a, tokenizer, spec, 2)


def test_no_density_claim_from_empty_results():
    spec = SimpleNamespace(
        session_context_tokens=96000,
        session_slo_grid_s=[60],
        min_quality=0.95,
        max_error_fraction=0.01,
    )
    report = compare_workers([], spec)
    assert not report["qualified"]
    assert report["token_savings_claim"] is None


def test_worker_replay_never_overlaps_turns_and_retains_answers(monkeypatch, tmp_path):
    import asyncio

    from simulator.infrastructure import knowledge_workers as kw
    from simulator.infrastructure.spec import ExperimentSpec

    spec = ExperimentSpec(
        project_id="P02",
        session_mode="capacity",
        warmup_s=0,
        session_gpu_gb=None,
        duration_s=300,
        min_requests=1,
    ).model_copy(
        update={"duration_s": 0.12, "session_think_s": 0.001, "session_turns": 4, "concurrency": 4}
    )
    active = set()
    calls = []
    monkeypatch.setattr(
        kw,
        "make_session",
        lambda tokenizer, spec, slot, block: dict(messages=[], seed=slot, initial_tokens=96000),
    )

    def item(session, tokenizer, spec, turn):
        return dict(
            id=f"{session['seed']}:{turn}",
            prompt="q",
            messages=session["messages"] + [dict(role="user", content="q")],
            input_tokens=96000 + turn,
            prompt_sha256="test",
            expected=[],
        )

    monkeypatch.setattr(kw, "turn_item", item)

    async def request(client, url, model, item, spec, request_id, key):
        slot = int(request_id.split(":")[0])
        assert slot not in active
        active.add(slot)
        calls.append((request_id, list(item["messages"])))
        await asyncio.sleep(0.001)
        active.remove(slot)
        return dict(
            transport_ok=True,
            correct=True,
            elapsed_s=0.001,
            ttft_s=0.0001,
            final_answer_ttft_s=0.0002,
            output="[]",
            usage=dict(prompt_tokens=96000, completion_tokens=2),
        )

    monkeypatch.setattr(kw, "request_one", request)

    async def capture(*args):
        pass

    engine = SimpleNamespace(
        replica_urls=["http://one", "http://two"],
        api_model_name="m",
        api_key="EMPTY",
        capture=capture,
    )
    result = asyncio.run(kw.replay_workers(engine, spec, 2, 0, tmp_path, None))
    assert result["priming"]["completed"] == 2
    assert result["completed"] == 6
    assert len([r for r in calls if r[0].endswith(":0")]) == 2
    assert next(history for rid, history in calls if rid == "0:1")[1] == dict(
        role="assistant", content="[]"
    )


def test_recent_working_set_keeps_archive_and_context_target():
    tokenizer = Tokenizer()
    spec = settings()
    spec.session_active_records = 2
    session = make_session(tokenizer, spec, 0, 0)
    assert "Archived reference records:" in session["prefix"]
    assert "Active reference records (current working set):" in session["prefix"]
    for row in session["rows"]:
        assert row["record_id"] in session["prefix"]
    assert 1024 <= session["initial_tokens"] <= 1088
    for turn in range(8):
        expected = turn_item(session, tokenizer, spec, turn)["expected"]
        assert all(row in session["rows"][-2:] for row in expected)
