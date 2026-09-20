"""The test arena: hardware-derived feasibility, full-space defaults,
subtractive selection, restart-aware ordering, and the new engine
dimensions (kv_cache_dtype, expert_parallel)."""

from __future__ import annotations

import pytest
import yaml

from simulator import arena as arena_mod
from simulator.arena import (
    build_space_doc,
    feasible_tps,
    full_arena,
    summarize_space_doc,
)

CATALOG = [
    {"id": "org/Small-30B", "family": "small-30b", "quant": "bf16",
     "min_vram_gb": 70, "approx_size_gb": 61, "moe": True,
     "gated": False, "engine_args": [], "notes": ""},
    {"id": "org/Small-30B-FP8", "family": "small-30b", "quant": "fp8",
     "min_vram_gb": 40, "approx_size_gb": 31, "moe": True,
     "gated": False, "engine_args": [], "notes": ""},
    {"id": "org/Huge-235B", "family": "huge-235b", "quant": "bf16",
     "min_vram_gb": 560, "approx_size_gb": 470, "moe": True,
     "gated": False, "engine_args": [], "notes": ""},
    {"id": "org/Dense-32B", "family": "dense-32b", "quant": "bf16",
     "min_vram_gb": 75, "approx_size_gb": 66, "moe": False,
     "gated": False, "engine_args": [], "notes": ""},
]


@pytest.fixture
def xe7740(monkeypatch, tmp_path):
    """8× 96 GB in two 4-GPU domains (the XE7740 shape)."""
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)


def test_feasible_tps_derived_from_vram() -> None:
    tps = [1, 2, 4]
    assert feasible_tps({"min_vram_gb": 70}, tps, 96.0) == [1, 2, 4]
    assert feasible_tps({"min_vram_gb": 300}, tps, 96.0) == [4]
    assert feasible_tps({"min_vram_gb": 560}, tps, 96.0) == []
    # Unknown VRAM or size: allow everything, validate at launch.
    assert feasible_tps({}, tps, 96.0) == tps
    assert feasible_tps({"min_vram_gb": 70}, tps, None) == tps


def test_full_arena_shape(xe7740) -> None:
    a = full_arena(CATALOG)
    hw = a["hardware"]
    assert hw["count"] == 8 and hw["vram_per_gpu_gb"] == 96.0
    assert len(hw["device_groups"]) == 2
    # TP capped by the largest DOMAIN (4), dp by the box (8).
    assert a["dimensions"]["tp"] == [1, 2, 4]
    assert a["dimensions"]["dp"] == [1, 2, 4, 8]
    assert a["dimensions"]["kv_cache_dtype"] == ["auto", "fp8", "nvfp4"]
    by_id = {m["id"]: m for m in a["models"]}
    assert by_id["org/Small-30B"]["feasible_tps"] == [1, 2, 4]
    # 235B bf16 needs 560 GB; the largest domain gives 4×96=384 —
    # simply not runnable here, and shown as such.
    assert by_id["org/Huge-235B"]["feasible"] is False


def test_build_space_defaults_to_everything_feasible(xe7740) -> None:
    doc = build_space_doc({}, CATALOG, budget=60)
    # The infeasible 235B is excluded automatically; the rest are in.
    assert set(doc["model_variants"]) == {
        "small-30b-bf16", "small-30b-fp8", "dense-32b-bf16"}
    assert doc["vram_per_gpu_gb"] == 96.0
    assert doc["dimensions"]["tp"] == [1, 2, 4]
    assert doc["search"]["budget"] == 60
    # The coverage stage takes what refinement doesn't (24), floors
    # at 8, so a bigger budget widens the screen.
    assert doc["search"]["initial_samples"] == 36
    # min_vram + moe ride into the variants for validate/normalize.
    assert doc["model_variants"]["small-30b-bf16"]["moe"] is True
    assert doc["model_variants"]["dense-32b-bf16"]["moe"] is False

    # The doc round-trips through the real space loader.
    import tempfile

    from simulator.search import load_space
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        space = load_space(f.name)
    assert space.vram_per_gpu_gb == 96.0


def test_build_space_subtractive_selection(xe7740) -> None:
    doc = build_space_doc({
        "models": ["org/Small-30B-FP8"],
        "dims": {"tp": [1, 2], "kv_cache_dtype": ["fp8"]},
    }, CATALOG)
    assert list(doc["model_variants"]) == ["small-30b-fp8"]
    assert doc["dimensions"]["tp"] == [1, 2]
    assert doc["dimensions"]["kv_cache_dtype"] == ["fp8"]
    with pytest.raises(ValueError, match="cannot run here"):
        build_space_doc({"models": ["org/Huge-235B"]}, CATALOG)
    with pytest.raises(ValueError, match="not in the arena"):
        build_space_doc({"dims": {"tp": [16]}}, CATALOG)


def test_vram_pruning_in_validate(xe7740) -> None:
    """A dense 32B (needs 75 GB) is valid at tp=1 on 96 GB cards, but
    a hypothetical 300 GB model only validates at tp=4."""
    import tempfile

    from simulator.search import load_space, validate_candidate
    catalog = CATALOG + [{
        "id": "org/Mid-150B", "family": "mid-150b", "quant": "bf16",
        "min_vram_gb": 300, "approx_size_gb": 280, "moe": False,
        "gated": False, "engine_args": [], "notes": ""}]
    doc = build_space_doc({}, catalog)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        space = load_space(f.name)
    ok, _ = validate_candidate(
        {"model_variant": "mid-150b-bf16", "tp": 1}, space)
    assert not ok
    ok, why = validate_candidate(
        {"model_variant": "mid-150b-bf16", "tp": 4}, space)
    assert ok, why


def test_expert_parallel_gating_and_args(xe7740) -> None:
    import tempfile

    from simulator.search import candidate_summary, load_space, normalize
    doc = build_space_doc({}, CATALOG)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        space = load_space(f.name)
    # EP forced off for dense models and for tp=1 — infeasible combos
    # dedupe instead of wasting evaluations.
    n = normalize({"model_variant": "dense-32b-bf16", "tp": 2,
                   "expert_parallel": "on"}, space)
    assert n["expert_parallel"] == "off"
    n = normalize({"model_variant": "small-30b-bf16", "tp": 1,
                   "expert_parallel": "on"}, space)
    assert n["expert_parallel"] == "off"
    # MoE at tp>1: EP survives and reaches the engine args.
    s = candidate_summary({"model_variant": "small-30b-bf16", "tp": 2,
                           "expert_parallel": "on",
                           "kv_cache_dtype": "fp8"}, space)
    assert "--enable-expert-parallel" in s["engine_args"]
    assert "--kv-cache-dtype" in s["engine_args"]
    s = candidate_summary({"model_variant": "small-30b-bf16", "tp": 2,
                           "kv_cache_dtype": "auto"}, space)
    assert "--kv-cache-dtype" not in s["engine_args"]


def test_batches_grouped_by_model(xe7740) -> None:
    """Restart-cost ordering: initial batches come out grouped by
    model variant so weight loads amortize."""
    import random
    import tempfile

    from simulator.search import SearchState, load_space, next_batch
    doc = build_space_doc({}, CATALOG, budget=40)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        space = load_space(f.name)
    state = SearchState(space_hash=space.space_hash())
    kind, batch = next_batch(state, space, random.Random(42))
    assert kind == "initial" and len(batch) >= 6
    variants = [p["model_variant"] for p in batch]
    # Grouped: the variant sequence never returns to an earlier one.
    seen, blocks = set(), []
    for v in variants:
        if not blocks or blocks[-1] != v:
            assert v not in seen, f"variant {v} split across blocks: {variants}"
            seen.add(v)
            blocks.append(v)


def test_summarize_space_doc_counts(xe7740) -> None:
    doc = build_space_doc({"models": ["org/Small-30B-FP8"],
                           "dims": {"tp": [1], "dp": [1, 2],
                                    "placement": ["pack"],
                                    "expert_parallel": ["off"]}},
                          CATALOG, budget=10)
    s = summarize_space_doc(doc)
    # tp1 × dp{1,2} × pack × ep-off × one variant = 2 launch shapes.
    assert s["launch_shapes"] == 2
    assert s["total_combinations"] == 2 * 4 * 3 * 3   # seqs×mbt×kv
    assert s["budget"] == 10
    assert s["estimated_cold_weight_loads"] >= 1


def test_arena_api(xe7740, tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from simulator.service import create_app
    monkeypatch.setattr(
        "simulator.model_catalog.load_model_catalog",
        lambda user_dir=None: CATALOG if user_dir is None else CATALOG,
    )
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(tmp_path / "runs")) as client:
        a = client.get("/api/arena").json()
        assert a["hardware"]["count"] == 8
        assert any(m["id"] == "org/Small-30B" for m in a["models"])

        r = client.post("/api/arena/preview",
                        json={"mode": "arena", "arena": {}, "budget": 24})
        assert r.status_code == 200, r.text
        assert r.json()["budget"] == 24
        assert r.json()["launch_shapes"] > 0

        r = client.post("/api/arena/preview",
                        json={"mode": "arena",
                              "arena": {"models": ["org/Huge-235B"]}})
        assert r.status_code == 422


def test_size_class_grouping() -> None:
    """30B and 32B land in the same operator-level size bucket."""
    from simulator.arena import size_class
    assert size_class(30.5) == size_class(32.8) == "16–45B"
    assert size_class(14.7) == "≤ 15B"
    assert size_class(70.6) == "46–90B"
    assert size_class(235) == "> 90B"
    assert size_class(None) == "unknown"


def test_catalog_carries_series_params_specialty(tmp_path) -> None:
    from simulator.model_catalog import load_model_catalog
    by_id = {e["id"]: e for e in load_model_catalog(user_dir=tmp_path / "x")}
    q = by_id["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert q["series"] == "Qwen3" and q["params_b"] == 30.5
    assert q["specialty"] == "instruct"
    assert by_id["Qwen/Qwen3-Coder-30B-A3B-Instruct"]["specialty"] == "coder"
    # Same size bucket for 30B and 32B — the dropdown's grouping.
    from simulator.arena import size_class
    assert size_class(q["params_b"]) == \
        size_class(by_id["Qwen/Qwen3-32B"]["params_b"])


def test_recommend_search_math(xe7740) -> None:
    from simulator.arena import build_space_doc, recommend_search
    # 2 values ×3 dims: OFAT floor = 3·1+1 = 4; coverage padded to
    # max(2·2, ceil(1.25·4)) = 5; +24 refinement.
    rec = recommend_search({"a": [1, 2], "b": ["x", "y"], "c": [0, 1],
                            "fixed": [1]})
    assert rec["ofat_min"] == 4
    assert rec["recommended"] == 5 + 24
    assert rec["screening"] < rec["recommended"] < rec["thorough"]

    # No explicit budget → the space is sized to the recommendation.
    doc = build_space_doc({}, CATALOG)
    rec2 = recommend_search(doc["dimensions"])
    assert doc["search"]["budget"] == rec2["recommended"]


def test_summarize_carries_recommendation_tiers(xe7740) -> None:
    from simulator.arena import build_space_doc, summarize_space_doc
    s = summarize_space_doc(build_space_doc({}, CATALOG))
    rec = s["recommendation"]
    names = [t["name"] for t in rec["tiers"]]
    assert names == ["screening", "recommended", "thorough"]
    assert all(t["hours"] > 0 for t in rec["tiers"])
    assert rec["tiers"][0]["budget"] < rec["tiers"][2]["budget"]


def test_hardware_flags_config_claiming_more_gpus_than_detected(
        monkeypatch, tmp_path) -> None:
    """A copied arena.yaml on a smaller (or GPU-less) host keeps its
    shape but must say so: source carries the detected count and
    detected_count is what nvidia-smi actually saw."""
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)

    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [])
    hw = arena_mod.hardware()
    assert hw["count"] == 8 and hw["detected_count"] == 0
    assert hw["source"] == "config (unverified: detected 0)"

    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 4)
    hw = arena_mod.hardware()
    assert hw["source"] == "config (unverified: detected 4)"

    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    assert arena_mod.hardware()["source"] == "config"


def test_hardware_without_config_is_detection_only(
        monkeypatch, tmp_path) -> None:
    """No config/arena.yaml (the shipped state — only the .example is
    committed): detection alone decides, so a laptop reports 0 GPUs."""
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", tmp_path / "absent.yaml")
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [])
    hw = arena_mod.hardware()
    assert hw["count"] == 0 and hw["device_groups"] == []
    assert hw["source"] == "detected"
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [48.0, 48.0])
    hw = arena_mod.hardware()
    assert hw["count"] == 2 and hw["device_groups"] == [[0, 1]]
    assert hw["vram_per_gpu_gb"] == 48.0


def test_example_config_is_shipped_and_real_one_is_not() -> None:
    """The XE7740 map ships as the example only; the loader's path is
    the per-host file that the operator copies into place."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "config" / "arena.example.yaml").exists()
    assert arena_mod.ARENA_CONFIG == Path("config/arena.yaml")
    doc = yaml.safe_load((root / "config" / "arena.example.yaml").read_text())
    assert doc["device_groups"] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_hardware_reports_host_ram_for_the_ktransformers_budget(
        monkeypatch, tmp_path) -> None:
    """host_ram_gb comes from /proc/meminfo on Linux, from arena.yaml
    when a laptop describes the box, and is None otherwise -- the
    roofline treats None as 'nothing is beyond VRAM here'."""
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", tmp_path / "absent.yaml")
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    monkeypatch.setattr(arena_mod, "detect_host_ram", lambda: 2015.6)
    assert arena_mod.hardware()["host_ram_gb"] == 2015.6
    monkeypatch.setattr(arena_mod, "detect_host_ram", lambda: None)
    assert arena_mod.hardware()["host_ram_gb"] is None
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("host_ram_gb: 2048\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)
    assert arena_mod.hardware()["host_ram_gb"] == 2048.0


def test_detect_host_ram_parses_meminfo(monkeypatch, tmp_path) -> None:
    import builtins
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       2113929216 kB\nMemFree: 1 kB\n")
    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            return real_open(meminfo, *a, **kw)
        return real_open(path, *a, **kw)
    monkeypatch.setattr(builtins, "open", fake_open)
    assert arena_mod.detect_host_ram() == 2016.0
    monkeypatch.setattr(builtins, "open", lambda *a, **kw: (_ for _ in ()).throw(
        FileNotFoundError()))
    assert arena_mod.detect_host_ram() is None
