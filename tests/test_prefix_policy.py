"""Prefix-cache policy: template floor, rung flags, workload flags."""

from simulator.export_sizing import curve_point
from simulator.headline_sweep import prefix_hit_rate
from simulator.prefix_policy import (
    PRODUCTION_REUSE_CEILING,
    TEMPLATE_TOLERANCE,
    block_tokens,
    headline_reuse,
    template_floor,
    workload_class,
    workload_reuse,
)


class FakeTokenizer:
    """Chat template = a fixed 40-token preamble, then one token per
    word of the user text, then a 3-token assistant header."""

    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = list(range(1000, 1040))
        for m in messages:
            ids += [hash((m["role"], w)) % 50000 for w in m["content"].split()]
        return ids + [7, 8, 9]


def test_template_floor_rounds_down_to_whole_blocks():
    texts = ["a b c d e f g h", "z y x w v u t s"]
    # 40 shared tokens -> 32 in whole 16-token blocks, of 40 + 8 + 3 = 51.
    assert template_floor(FakeTokenizer(), texts, 16) == round(32 / 51, 4)
    # Token-level matching (SGLang) keeps all 40.
    assert template_floor(FakeTokenizer(), texts, 1) == round(40 / 51, 4)


def test_template_floor_reads_a_batch_encoding():
    from collections import UserDict

    class Encoding(FakeTokenizer):
        def apply_chat_template(self, *a, **kw):
            ids = super().apply_chat_template(*a, **kw)
            return UserDict({"input_ids": ids, "attention_mask": [1] * len(ids)})

    texts = ["a b c d e f g h", "z y x w v u t s"]
    assert template_floor(Encoding(), texts, 16) == round(32 / 51, 4)


def test_template_floor_counts_a_fixed_system_prompt():
    texts = ["a b c d", "z y x w"]
    system = " ".join(f"s{i}" for i in range(24))
    # 40 preamble + 24 system tokens shared = 64 (4 blocks) of 64 + 4 + 3.
    assert template_floor(FakeTokenizer(), texts, 16, system_prompt=system) == round(64 / 71, 4)


def test_template_floor_without_a_template_is_unknown():
    class Bare:
        chat_template = None
    assert template_floor(Bare(), ["a", "b"], 16) is None
    assert template_floor(None, ["a", "b"], 16) is None


def test_block_tokens_by_engine():
    assert block_tokens("vllm_cuda_multi") == 16
    assert block_tokens("sglang_cuda") == 1


def test_headline_reuse_flags_reuse_beyond_the_template():
    floor = 0.33
    assert headline_reuse(floor, floor) == "template"
    assert headline_reuse(floor + TEMPLATE_TOLERANCE, floor) == "template"
    # The 2026-10-26 AIPerf run: 98.5% against a ~33% gpt-oss template.
    assert headline_reuse(0.985, floor) == "above_template"
    assert headline_reuse(None, floor) is None
    assert headline_reuse(0.5, None) is None


def test_workload_reuse_against_the_production_ceiling():
    assert workload_reuse(0.40) == "within_production"   # Mooncake conversation
    assert workload_reuse(PRODUCTION_REUSE_CEILING["chat"]) == "within_production"
    assert workload_reuse(0.80) == "above_production"
    assert workload_reuse(None) is None


def test_long_running_agents_have_their_own_ceiling():
    # Coding agents resend a growing history: ~94% in production traces.
    assert workload_class("h_coding_agent") == "agent"
    assert workload_class("conversational") == "chat"
    assert workload_class("software_engineering") == "chat"   # a mix
    assert workload_reuse(0.90, workload_class("h_coding_agent")) == "within_production"
    assert workload_reuse(0.90, workload_class("document_qa")) == "above_production"
    assert workload_reuse(0.97, "agent") == "above_production"


def test_rescore_labels_every_capacity_point():
    from simulator.rescore import _label_reuse
    entry = {"knee_point": {"prefix_hit_rate": 0.9},
             "tiers": {"Interactive": {"basis": "interpolated", "prefix_hit_rate": 0.5},
                       "Batch": {"basis": "fails_at_lowest_rate", "rate_per_min": None}}}
    _label_reuse(entry, "coding_agent")
    assert entry["prefix_workload_class"] == "agent"
    assert entry["knee_point"]["prefix_reuse"] == "within_production"
    assert entry["tiers"]["Interactive"]["prefix_reuse"] == "within_production"
    assert "prefix_reuse" not in entry["tiers"]["Batch"]
    chat = {"knee_point": {"prefix_hit_rate": 0.9}, "tiers": {}}
    _label_reuse(chat, "conversational")
    assert chat["knee_point"]["prefix_reuse"] == "above_production"


def test_prefix_hit_rate_from_counter_deltas():
    m0 = {"prefix_cache_hits": 1000.0, "prefix_cache_queries": 4000.0}
    m1 = {"prefix_cache_hits": 1600.0, "prefix_cache_queries": 6000.0}
    assert prefix_hit_rate(m0, m1, []) == 0.3


def test_prefix_hit_rate_falls_back_to_the_engine_gauge():
    assert prefix_hit_rate({}, {}, [0.2, 0.4]) == 0.3
    assert prefix_hit_rate({}, {}, []) is None
    # No new queries in the chunk: nothing to measure from the counters.
    flat = {"prefix_cache_hits": 5.0, "prefix_cache_queries": 10.0}
    assert prefix_hit_rate(flat, flat, []) is None


def test_export_curve_point_carries_reuse():
    pt = curve_point({"concurrency": 64, "samples": 100, "out_tok_s": 1.0,
                      "prefix_hit_rate": 0.31, "prefix_reuse": "template"})
    assert pt["prefix_cache_hit_rate"] == 0.31
    assert pt["prefix_reuse"] == "template"
    old = curve_point({"concurrency": 64, "samples": 100, "out_tok_s": 1.0})
    assert old["prefix_cache_hit_rate"] is None


# ── The reuse level a headline sweep is built to reach ────────────────

import pytest  # noqa: E402

from simulator.prefix_policy import (  # noqa: E402
    REUSE_LEVELS,
    choose_system_tokens,
    resolve_reuse,
)


def test_default_reuse_is_the_published_business_api_level():
    level = resolve_reuse(None)
    assert level["setting"] == "industry" and level["target"] == 0.54
    assert "ATC" in level["source"] and level["within_production"]
    assert resolve_reuse("template")["target"] is None
    assert resolve_reuse("Consumer_Chat")["target"] == REUSE_LEVELS["consumer_chat"][0]


def test_reuse_accepts_a_share_and_says_when_it_is_above_production():
    assert resolve_reuse(0.45) == resolve_reuse("45%") == resolve_reuse("0.45")
    assert resolve_reuse(45)["target"] == 0.45
    assert resolve_reuse(0.45)["setting"] == "0.45"
    assert resolve_reuse(0.45)["within_production"]
    assert not resolve_reuse(0.9)["within_production"]
    assert resolve_reuse(0)["setting"] == "template"
    for bad in ("lots", 0.99, -0.1, True):
        with pytest.raises(ValueError):
            resolve_reuse(bad)


def test_system_prompt_length_lands_closest_to_the_target():
    # 16-token blocks over a 144-token prompt whose template shares 4 tokens.
    def floor_at(n):
        return ((4 + n) // 16 * 16) / 144

    n = choose_system_tokens(floor_at, 96, 0.54)
    assert abs(floor_at(n) - 0.54) <= min(abs(floor_at(m) - 0.54)
                                          for m in range(0, 97, 8))
    assert floor_at(n) == 80 / 144
    assert choose_system_tokens(floor_at, 96, None) == 0       # "template"
    assert choose_system_tokens(lambda n: 0.6, 96, 0.54) == 0  # template alone reaches it
    assert choose_system_tokens(floor_at, 32, 0.9) == 32       # capped: as close as it gets
    assert choose_system_tokens(lambda n: None, 96, 0.54) == 0  # no chat template


def _plan(monkeypatch, n_in=128, **sim):
    """_prefix_plan against a one-token-per-word tokenizer."""
    from types import SimpleNamespace

    import simulator.headline_sweep as hs
    import simulator.tokenizer_corpus as tc

    class Tok(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    monkeypatch.setattr(tc, "_load_tokenizer", lambda model_id: Tok())
    monkeypatch.setattr(hs, "_cohort_shape", lambda cohort: {"input_tokens": n_in})
    cfg = SimpleNamespace(
        simulation=SimpleNamespace(
            prefix_reuse=sim.get("prefix_reuse", "industry"),
            system_prompt_tokens=sim.get("system_prompt_tokens")),
        engine=SimpleNamespace(type="vllm", model_id="fake/model"))
    cohort = SimpleNamespace(persona_weights={"headline_generation": 1.0})
    return hs._prefix_plan(cfg, cohort)


def test_headline_plan_splits_the_prompt_to_reach_the_level(monkeypatch):
    plan = _plan(monkeypatch)
    assert plan["setting"] == "industry" and plan["target"] == 0.54
    # The shared prompt comes out of the prompt length, not on top of it.
    assert plan["system_prompt_actual_tokens"] + plan["user_tokens"] == 128
    assert plan["system_prompt_actual_tokens"] > 0
    assert abs(plan["template_floor"] - 0.54) <= 0.06
    assert headline_reuse(plan["template_floor"], plan["template_floor"]) == "template"
    assert headline_reuse(0.95, plan["template_floor"]) == "above_template"


def test_headline_plan_template_level_sends_no_system_prompt(monkeypatch):
    plan = _plan(monkeypatch, prefix_reuse="template")
    assert plan["system_prompt_tokens"] == 0 and plan["user_tokens"] == 128
    assert plan["template_floor"] == round(32 / 171, 4)  # 40-token preamble


def test_headline_plan_keeps_an_explicit_system_prompt(monkeypatch):
    plan = _plan(monkeypatch, system_prompt_tokens={"headline_generation": 40})
    assert plan["setting"] == "explicit" and plan["system_prompt_tokens"] == 40


def test_reuse_level_is_part_of_a_roofline_cell():
    from simulator.roofline import cell_key, cell_overrides
    cell = {"model": "m", "engine": "vllm", "max_num_seqs": 256,
            "output_tokens": 256, "input_tokens": 128, "prefix_reuse": "industry"}
    assert cell_key(cell) != cell_key({**cell, "prefix_reuse": "template"})
    assert "prefix_reuse" not in cell_overrides(cell)  # not an engine launch field


def test_config_defaults_to_the_industry_level():
    from simulator.config import SimulationConfig
    assert SimulationConfig().prefix_reuse == "industry"
