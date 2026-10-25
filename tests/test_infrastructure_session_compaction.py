"""P02 stage 2: growing sessions, whole-history probes, resets and Xeon compaction."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from simulator.infrastructure import knowledge_workers as kw
from simulator.infrastructure.compaction_calibration import capacity_verdict
from simulator.infrastructure.session_compaction import (
    ACKNOWLEDGEMENT,
    apply_compaction,
    empty_ledger,
    merge_ledger,
    parse_summary,
    render_header,
)


class Tokenizer:
    def encode(self, text, **kw):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kw):
        return list(range(sum(len(m["content"].split()) for m in messages) + len(messages)))


def settings(**overrides):
    base = dict(
        seed=42,
        document_records=4,
        session_context_tokens=256,
        session_answer_records=2,
        session_new_input_tokens=16,
        session_growth_tokens=40,
        session_probe_mix=["reference", "session_recall", "knowledge_update", "abstention"],
        session_limit_policy="fail",
        max_output_tokens=16,
        context_limit=100000,
        session_active_records=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def stage2_spec(**overrides):
    from simulator.infrastructure.spec import ExperimentSpec

    fields = dict(
        project_id="P02",
        session_mode="capacity",
        session_workload="knowledge_worker",
        warmup_s=0,
        session_gpu_gb=None,
        duration_s=300,
        min_requests=1,
        context_limit=131072,
        session_context_tokens=48000,
        session_growth_tokens=6000,
        session_probe_mix=["reference", "session_recall", "knowledge_update", "abstention"],
        session_limit_policy="new_session",
        session_compaction_arms=["none", "xeon"],
        session_compaction_url="http://127.0.0.1:18000/v1",
        session_compaction_model="Qwen/Qwen3-4B",
        session_compaction_trigger_tokens=96000,
        session_compaction_target_tokens=32000,
        concurrency=256,
        session_population_grid=[32, 64],
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


def test_spec_requires_summarizer_growth_and_band():
    stage2_spec()
    with pytest.raises(ValueError, match="summarizer endpoint"):
        stage2_spec(session_compaction_url=None)
    with pytest.raises(ValueError, match="growing sessions"):
        stage2_spec(
            session_growth_tokens=0, session_probe_mix=["reference"], session_limit_policy="fail"
        )
    with pytest.raises(ValueError, match="below the trigger"):
        stage2_spec(session_compaction_target_tokens=96000)
    with pytest.raises(ValueError, match="new-session"):
        stage2_spec(
            session_growth_tokens=0,
            session_compaction_arms=["none"],
            session_probe_mix=["reference"],
        )


def test_schedule_has_four_arms_per_population(tmp_path, monkeypatch):
    from simulator.infrastructure.campaign import create, verify

    monkeypatch.setattr(
        "simulator.infrastructure.campaign.recorded_config",
        lambda *a: {"engine": {"type": "vllm_cuda", "model_id": "test/model"}},
    )
    spec = stage2_spec(paired_blocks=1, session_rate_grid=[1 / 60])
    manifest = verify(create(tmp_path, {"model": "test/model"}, spec, "c" * 32))
    cells = [(c["arm"], c["compaction"], c["population"]) for c in manifest["schedule"]]
    assert len(cells) == 8
    for population in (32, 64):
        assert {(a, c) for a, c, p in cells if p == population} == {
            ("baseline", "none"),
            ("treatment", "none"),
            ("baseline", "xeon"),
            ("treatment", "xeon"),
        }


def test_summary_parse_merge_and_prefix_alignment():
    text = (
        "<summary>Turn 1 requested R0000-0001.</summary>"
        "<ledger>\nrequested turn 1: R0000-0001, R0000-0004\nquantity R0000-0002: 7\n"
        "Quantity R0000-0003: null\n</ledger>"
    )
    summary, ledger = parse_summary(text)
    assert summary.startswith("Turn 1")
    assert ledger == {
        "requested_by_turn": {"1": ["R0000-0001", "R0000-0004"]},
        "working_allocations": {"R0000-0002": 7, "R0000-0003": None},
    }
    merged = merge_ledger(
        {"requested_by_turn": {"0": ["R0000-0000"]}, "working_allocations": {"R0000-0002": 3}},
        ledger,
    )
    assert merged["requested_by_turn"] == {"0": ["R0000-0000"], "1": ["R0000-0001", "R0000-0004"]}
    assert merged["working_allocations"]["R0000-0002"] == 7
    with pytest.raises(ValueError):
        parse_summary("<summary>no ledger</summary>")
    with pytest.raises(ValueError, match="unknown form"):
        parse_summary("<summary>x</summary><ledger>turn 1 was great</ledger>")
    header = render_header(
        "PREFIX BYTES", [dict(cycle=1, first_turn=0, last_turn=6, text=summary)], merged
    )
    assert header.startswith("PREFIX BYTES")
    assert "Summary 1 (turns 0-6)" in header
    # Ledger precedes the prose: it carries the facts, the prose is a short narrative.
    assert header.index("Session ledger") < header.index("Summary 1")


def test_chunked_compaction_merges_in_turn_order(monkeypatch):
    from simulator.infrastructure import session_compaction as sc

    calls = []

    async def fake_chunk(client, spec, archive, url):
        calls.append((url, [e["turn"] for e in archive]))
        first = archive[0]["turn"]
        return dict(
            url=url,
            first_turn=first,
            last_turn=archive[-1]["turn"],
            raw_turns=len(archive),
            raw_chars=10,
            ok=True,
            error=None,
            summary=f"chunk {first}.",
            ledger={"requested_by_turn": {str(first): ["r"]}, "working_allocations": {"R1": first}},
            usage=dict(prompt_tokens=100, completion_tokens=10),
            finish_reason="stop",
            wall_s=0.01,
        )

    monkeypatch.setattr(sc, "compact_chunk", fake_chunk)
    spec = SimpleNamespace(
        session_compaction_url="http://a/v1, http://b/v1",
        session_compaction_chunk_turns=2,
        session_compaction_model="m",
        seed=1,
        session_compaction_budget_tokens=10,
        session_compaction_timeout_s=10,
    )
    archive = [dict(turn=t, user="u", assistant="a") for t in range(1, 6)]
    row = asyncio.run(sc.compact(None, spec, archive, empty_ledger()))
    assert row["ok"] and row["chunk_count"] == 3
    assert [c[1] for c in calls] == [[1, 2], [3, 4], [5]]
    assert [c[0] for c in calls] == ["http://a/v1", "http://b/v1", "http://a/v1"]
    assert row["summary"] == "chunk 1. chunk 3. chunk 5."
    # Later chunks override earlier quantities: turn order is preserved in the merge.
    assert row["ledger"]["working_allocations"]["R1"] == 5
    assert row["usage"] == dict(prompt_tokens=300, completion_tokens=30)


def test_growing_turns_add_facts_and_probe_the_whole_history():
    tokenizer = Tokenizer()
    spec = settings()
    session = kw.make_session(tokenizer, spec, 0, 0)
    seen = []
    for turn in range(9):
        item = kw.turn_item(session, tokenizer, spec, turn)
        seen.append(item["probe"])
        kw.advance(session, item, {"output": json.dumps(item["expected"])})
        if turn > 0:
            assert "Session note (turn" in item["body"]
    assert seen[0] == "reference"
    assert {"session_recall", "knowledge_update", "abstention"} <= set(seen)
    kinds = {f["kind"] for f in session["facts"]}
    assert kinds == {"requested", "update"}
    assert len(session["archive"]) == 9
    item = kw.turn_item(session, tokenizer, spec, 9)
    if item["probe"] == "session_recall":
        assert item["fact_turn"] < 9 and item["schema"]["properties"]["record_ids"]
    # Mix rotates on turn-1: turns 10, 11, 12 are session_recall, knowledge_update, abstention.
    recall = [kw.turn_item(session, tokenizer, spec, t) for t in (10, 11, 12)]
    assert any(i["probe"] == "session_recall" and i["expected"]["record_ids"] for i in recall)
    assert any(i["probe"] == "abstention" and i["expected"]["allocation"] is None for i in recall)


def test_new_session_policy_resets_window_but_keeps_facts():
    tokenizer = Tokenizer()
    spec = settings(session_limit_policy="new_session", context_limit=520)
    session = kw.make_session(tokenizer, spec, 0, 0)
    first = kw.turn_item(session, tokenizer, spec, 0)
    kw.advance(session, first, {"output": "[]"})
    resets, turn = 0, 1
    while resets == 0 and turn < 40:
        item = kw.turn_item(session, tokenizer, spec, turn)
        resets = session["resets"]
        kw.advance(session, item, {"output": "[]"})
        turn += 1
    assert resets == 1 and item["reset"]
    assert item["messages"][0]["content"].startswith(session["prefix"])
    assert len(item["messages"]) == 1
    # Ground truth survives the reset even though the window did not.
    assert any(f["turn"] == 0 and f["kind"] == "requested" for f in session["facts"])
    assert session["history_start"] == turn - 1
    spec_fail = settings(context_limit=520)
    session2 = kw.make_session(tokenizer, spec_fail, 1, 0)
    kw.advance(session2, kw.turn_item(session2, tokenizer, spec_fail, 0), {"output": "[]"})
    with pytest.raises(ValueError, match="context limit"):
        for t in range(1, 40):
            kw.advance(session2, kw.turn_item(session2, tokenizer, spec_fail, t), {"output": "[]"})


def test_apply_compaction_keeps_prefix_archive_and_recent_turns():
    tokenizer = Tokenizer()
    spec = settings()
    session = kw.make_session(tokenizer, spec, 0, 0)
    for turn in range(6):
        item = kw.turn_item(session, tokenizer, spec, turn)
        kw.advance(session, item, {"output": json.dumps(item["expected"])})
    row = dict(
        first_turn=0,
        last_turn=4,
        summary="Turns 0-4 condensed.",
        ledger=merge_ledger(empty_ledger(), {"requested_by_turn": {"0": ["x"]}}),
    )
    apply_compaction(session, row, landed_at=12.0)
    assert session["cycle"] == 1 and session["compacted_through"] == 4
    assert session["messages"][0]["content"].startswith(session["prefix"])
    assert session["messages"][1] == {"role": "assistant", "content": ACKNOWLEDGEMENT}
    # Turn 5 stayed raw after the header; the archive still holds every turn.
    assert session["messages"][2]["content"] == session["archive"][5]["user"]
    assert len(session["archive"]) == 6
    nxt = kw.turn_item(session, tokenizer, spec, 6)
    assert nxt["cycle"] == 1 and nxt["history_turns"] == 1
    assert nxt["messages"][0]["content"].startswith(session["prefix"])


def test_replay_with_xeon_compaction_cycles_and_reports(monkeypatch, tmp_path):
    spec = stage2_spec().model_copy(
        update={
            "duration_s": 0.4,
            "session_think_s": 0.001,
            "session_turns": 12,
            "concurrency": 4,
            "session_compaction_trigger_tokens": 400,
            "session_context_tokens": 256,
            "session_growth_tokens": 40,
            "session_new_input_tokens": 16,
            "max_output_tokens": 16,
            "document_records": 4,
            "session_slo_grid_s": [60],
        }
    )
    tokenizer = Tokenizer()

    async def request(client, url, model, item, spec, request_id, key):
        await asyncio.sleep(0.001)
        return dict(
            transport_ok=True,
            correct=True,
            elapsed_s=0.001,
            ttft_s=0.0001,
            final_answer_ttft_s=0.0002,
            output=json.dumps(item["expected"]),
            usage=dict(prompt_tokens=item["input_tokens"], completion_tokens=2),
        )

    async def fake_compact(client, spec, archive, prior, counter=None):
        await asyncio.sleep(0.002)
        return dict(
            first_turn=archive[0]["turn"],
            last_turn=archive[-1]["turn"],
            raw_turns=len(archive),
            ok=True,
            error=None,
            summary="Condensed " + str(len(archive)) + " turns.",
            ledger=merge_ledger(prior, {"requested_by_turn": {str(archive[0]["turn"]): ["r"]}}),
            usage=dict(prompt_tokens=400, completion_tokens=50),
            finish_reason="stop",
            wall_s=0.002,
        )

    monkeypatch.setattr(kw, "request_one", request)
    monkeypatch.setattr(kw, "compact", fake_compact)

    async def capture(*args):
        pass

    engine = SimpleNamespace(
        replica_urls=["http://one"], api_model_name="m", api_key="EMPTY", capture=capture
    )
    result = asyncio.run(kw.replay_workers(engine, spec, 2, 0, tmp_path, tokenizer, "xeon"))
    assert result["compaction_mode"] == "xeon"
    assert result["compaction"]["landed"] >= 1
    assert result["context_limit_failures"] == 0
    assert result["recall"]["timeline"] and "end_state" in result["recall"]
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert any(r["compaction_landed"] for r in rows)
    assert max(r["cycle"] for r in rows) >= 1
    landed = [
        json.loads(line) for line in (tmp_path / "compactions.jsonl").read_text().splitlines()
    ]
    assert landed and all(c["lag_s"] >= 0 for c in landed)
    assert (tmp_path / "fact-logs.json").exists() and (tmp_path / "session-state.json").exists()


def test_recall_metrics_score_lost_facts_wrong():
    rows = []
    for turn in range(1, 9):
        rows.append(
            dict(
                turn=turn,
                session=0,
                probe="session_recall",
                correct=turn < 5,  # after the reset at turn 5 every lost fact is wrong
                cycle=0,
                fact_age=turn,
                input_tokens_checked=1000 * turn,
                reset=turn == 5,
                event_edge="before" if turn == 4 else "after" if turn == 5 else None,
                transport_ok=True,
            )
        )
    rows.append(
        dict(
            turn=9,
            session=0,
            probe="abstention",
            correct=True,
            cycle=0,
            fact_age=None,
            input_tokens_checked=9000,
            reset=False,
            transport_ok=True,
        )
    )
    rows.append(
        dict(
            turn=10,
            session=0,
            probe="session_recall",
            correct=False,
            cycle=0,
            fact_age=8,
            fact_turn=2,
            input_tokens_checked=10000,
            reset=False,
            transport_ok=True,
        )
    )
    for r in rows[:8]:
        r["fact_turn"] = 0
    metrics = kw.recall_metrics(rows)
    # Abstention is excluded from whole-history recall; the extra wrong probe makes 4/9.
    assert metrics["window_average_recall"] == pytest.approx(4 / 9)
    assert metrics["end_state"]["recall"] == 0.0  # the last four probes of the worker are all lost
    assert metrics["by_cycle"]["0"]["attempted"] == 9
    assert metrics["abstention"] == 1.0
    assert metrics["intact_vs_behind_reset"]["intact"] == 1.0
    assert metrics["intact_vs_behind_reset"]["behind_reset"] == 0.0
    assert metrics["intact_vs_behind_reset"]["behind_reset_attempted"] == 5
    assert metrics["before_after"] == dict(before_event=1.0, after_event=0.0)
    assert metrics["by_fact_age"]["1-4"]["fraction"] == 1.0
    assert metrics["by_fact_age"]["5-8"]["fraction"] == 0.0
    assert [t["resets"] for t in metrics["timeline"]][4] == 1


def test_compare_workers_reports_four_arm_density_ratios():
    spec = SimpleNamespace(
        session_context_tokens=48000,
        session_slo_grid_s=[60],
        min_quality=0.95,
        max_error_fraction=0.01,
        session_growth_tokens=6000,
    )

    def cell(arm, compaction, population):
        return dict(
            block=0,
            arm=arm,
            compaction=compaction,
            population=population,
            metrics=dict(
                slo_analysis=[dict(response_slo_s=60, passed=False)],
                slo_analysis_task=[dict(response_slo_s=60, passed=True)],
                task_quality_fraction=0.98,
                quality_fraction=0.85,
                error_fraction=0,
                context_limit_failures=0,
                correct_requests_per_s=1,
                recall=dict(end_state=dict(recall=0.9, attempted=200)),
                compaction=dict(landed=3),
                session_resets=0,
            ),
        )

    results = [
        cell("baseline", "none", 48),
        cell("treatment", "none", 96),
        cell("baseline", "xeon", 64),
        cell("treatment", "xeon", 128),
    ]
    report = kw.compare_workers(results, spec)
    ratios = report["by_worker_slo"][0]["density_ratios"]
    assert ratios == dict(C_over_A=64 / 48, B_over_A=2.0, D_over_A=128 / 48, D_over_B=128 / 96)
    assert report["by_worker_slo"][0]["arms"]["treatment/xeon"]["label"].startswith("D:")
    assert not report["qualified"]


def test_calibration_verdict_requires_two_x_headroom():
    results = [
        dict(concurrency=4, ok=4, attempted=4, compactions_per_s=0.3, wall_p50_s=12.0),
        dict(concurrency=8, ok=7, attempted=8, compactions_per_s=0.9, wall_p50_s=9.0),
    ]
    verdict = capacity_verdict(results, dict(population=96, cycle_minutes=8))
    assert verdict["required_compactions_per_s"] == pytest.approx(0.2)
    assert verdict["best_concurrency"] == 4 and not verdict["passes"]
    results[0]["compactions_per_s"] = 0.5
    assert capacity_verdict(results, dict(population=96, cycle_minutes=8))["passes"]


def test_density_gate_uses_task_quality_for_growing_sessions():
    spec = stage2_spec().model_copy(update={"min_requests": 1, "session_slo_grid_s": [60]})
    rows = []
    for i in range(20):
        rows.append(
            dict(
                turn=1 + i,
                session=0,
                probe="reference" if i % 2 == 0 else "session_recall",
                correct=i % 2 == 0,  # every reference probe right, every history probe wrong
                transport_ok=True,
                elapsed_s=1.0,
                ttft_s=0.1,
                request_ttft_s=0.1,
                final_answer_ttft_s=0.1,
                arrival_s=float(i),
                finish_s=float(i) + 1,
                usage=dict(prompt_tokens=1000, completion_tokens=10),
                input_tokens_checked=1000,
                cycle=0,
            )
        )
    metrics = kw.summarize_workers(rows, 100.0, spec, 1, [])
    assert metrics["quality_fraction"] == 0.5
    assert metrics["task_quality_fraction"] == 1.0
    assert metrics["slo_analysis_task"][0]["passed"]
    assert metrics["recall"]["window_average_recall"] == 0.0
    result = dict(
        block=0,
        arm="baseline",
        compaction="none",
        population=8,
        metrics=dict(metrics, error_fraction=0, context_limit_failures=0, correct_requests_per_s=1),
    )
    report = kw.compare_workers([result], spec)
    assert report["by_worker_slo"][0]["arms"]["baseline/none"]["best_observed_population"] == 8


def test_context_budget_resets_before_the_engine_limit():
    tokenizer = Tokenizer()
    spec = settings(
        session_limit_policy="new_session", context_limit=100000, session_limit_tokens=560
    )
    session = kw.make_session(tokenizer, spec, 0, 0)
    kw.advance(session, kw.turn_item(session, tokenizer, spec, 0), {"output": "[]"})
    turn = 1
    while session["resets"] == 0 and turn < 40:
        item = kw.turn_item(session, tokenizer, spec, turn)
        kw.advance(session, item, {"output": "[]"})
        turn += 1
    assert session["resets"] == 1 and item["reset"] and item["input_tokens"] < 560
    with pytest.raises(ValueError, match="context budget"):
        stage2_spec(session_limit_tokens=30000)
    assert stage2_spec(session_limit_tokens=96000).session_limit_tokens == 96000


def test_phase_priming_builds_synthetic_history_and_starts_at_its_depth(monkeypatch, tmp_path):
    spec = stage2_spec(session_prime_phase=True, session_limit_tokens=96000).model_copy(
        update={
            "duration_s": 0.2,
            "session_think_s": 0.001,
            "session_turns": 3,
            "concurrency": 4,
            "session_context_tokens": 256,
            "session_growth_tokens": 40,
            "session_new_input_tokens": 16,
            "max_output_tokens": 16,
            "document_records": 4,
            "session_slo_grid_s": [60],
            "session_compaction_trigger_tokens": 2000,
            "session_limit_tokens": 3000,
            "concurrency": 8,
        }
    )
    tokenizer = Tokenizer()
    sent = []

    async def request(client, url, model, item, spec, request_id, key):
        sent.append(
            (int(request_id.split(":")[0]), int(request_id.split(":")[1]), len(item["messages"]))
        )
        return dict(
            transport_ok=True,
            correct=True,
            elapsed_s=0.001,
            ttft_s=0.0001,
            final_answer_ttft_s=0.0002,
            output=json.dumps(item["expected"]),
            usage=dict(prompt_tokens=item["input_tokens"], completion_tokens=2),
        )

    monkeypatch.setattr(kw, "request_one", request)

    async def capture(*args):
        pass

    engine = SimpleNamespace(
        replica_urls=["http://one"], api_model_name="m", api_key="EMPTY", capture=capture
    )
    asyncio.run(kw.replay_workers(engine, spec, 6, 0, tmp_path, tokenizer, "none"))
    fixtures = json.loads((tmp_path / "worker-fixtures.json").read_text())
    assert fixtures["prime_phase"] and max(fixtures["prime_turns"]) > 0
    firsts = {slot: (turn, n) for slot, turn, n in sent if turn == fixtures["prime_turns"][slot]}
    assert len(firsts) == 6
    # The first real request of a primed worker carries its synthetic history.
    for slot, (_turn, n) in firsts.items():
        assert n == 2 * fixtures["prime_turns"][slot] + 1
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert all(r["window_turn"] == r["turn"] - fixtures["prime_turns"][r["session"]] for r in rows)
    primes = [
        json.loads(line) for line in (tmp_path / "prime-requests.jsonl").read_text().splitlines()
    ]
    assert primes and all(r["probe"] == "reference" for r in primes)
    assert all(r["window_turn"] >= 1 for r in rows)


def test_chunks_alternate_endpoints_across_compactions():
    import asyncio
    import itertools
    from types import SimpleNamespace

    from simulator.infrastructure import session_compaction as sc

    seen = []

    async def fake_chunk(client, spec, part, url):
        seen.append(url)
        return dict(
            ok=True,
            error=None,
            raw_chars=1,
            usage=None,
            finish_reason="stop",
            summary="s",
            ledger=sc.empty_ledger(),
        )

    original = sc.compact_chunk
    sc.compact_chunk = fake_chunk
    try:
        spec = SimpleNamespace(
            session_compaction_url="http://a/v1,http://b/v1", session_compaction_chunk_turns=2
        )
        archive = [dict(turn=t, user="u", assistant="a") for t in range(6)]  # three chunks
        counter = itertools.count()
        for _ in range(4):
            asyncio.run(sc.compact(None, spec, archive, sc.empty_ledger(), counter))
    finally:
        sc.compact_chunk = original
    assert seen.count("http://a/v1") == seen.count("http://b/v1") == 6


def test_summarizer_timer_delta_means():
    from simulator.infrastructure.session_compaction import metric_total, summarizer_timer_delta

    text = (
        'vllm:request_queue_time_seconds_sum{engine="0"} 100.0\n'
        'vllm:request_queue_time_seconds_count{engine="0"} 10.0\n'
        'sglang:e2e_request_latency_seconds_sum{a="1"} 150.0\n'
        'sglang:e2e_request_latency_seconds_sum{a="2"} 50.0\n'
        'sglang:e2e_request_latency_seconds_count{a="1"} 10.0\n'
    )
    assert metric_total(text, "sglang:e2e_request_latency_seconds", "sum") == 200
    assert metric_total(text, "vllm:request_decode_time_seconds", "sum") is None
    before = {
        "u": {"queue_sum": 0.0, "queue_count": 0.0, "end_to_end_sum": 0.0, "end_to_end_count": 0.0}
    }
    after = {
        "u": {
            "queue_sum": 100.0,
            "queue_count": 10.0,
            "end_to_end_sum": 200.0,
            "end_to_end_count": 10.0,
            "decode_sum": None,
            "decode_count": None,
        }
    }
    row = summarizer_timer_delta(before, after)["u"]
    assert row["requests"] == 10
    assert row["queue_mean_s"] == 10 and row["end_to_end_mean_s"] == 20
    assert row["decode_mean_s"] is None


def test_parser_drops_non_numeric_allocation_lines():
    from simulator.infrastructure.session_compaction import parse_summary

    text = (
        "<summary>Two turns.</summary><ledger>\n"
        "requested turn 4: R0000-0001, R0000-0002\n"
        "allocation R0000-0001: Seattle\n"
        "allocation R0000-0003: 512\n"
        "allocation R0000-0003: 512\n"
        "</ledger>"
    )
    _, ledger = parse_summary(text)
    assert ledger["requested_by_turn"] == {"4": ["R0000-0001", "R0000-0002"]}
    assert ledger["working_allocations"] == {"R0000-0003": 512}
    assert ledger["dropped_lines"] == 1


def test_ledger_score_counts_exact_and_invented():
    from simulator.infrastructure.compaction_calibration import ledger_score

    facts = [
        dict(turn=3, kind="requested", record_ids=["A", "B"]),
        dict(turn=3, kind="update", record_id="A", allocation=5),
        dict(turn=4, kind="update", record_id="A", allocation=7),
        dict(turn=4, kind="update", record_id="C", allocation=9),
    ]
    ledger = {
        "requested_by_turn": {"3": ["A", "B"], "9": ["Z"]},
        "working_allocations": {"A": 7, "C": 8},
    }
    assert ledger_score(ledger, facts) == dict(facts=3, exact=2, invented=1)
