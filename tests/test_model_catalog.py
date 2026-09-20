"""Model catalog: packaged starter set + local overlays, one-line
additions, quant-sibling suggestion, and catalog-driven search spaces."""

from __future__ import annotations

import textwrap

import pytest

from simulator import model_catalog as mc
from simulator.model_catalog import (
    CatalogError,
    add_catalog_model,
    catalog_families,
    infer_family,
    infer_quant,
    load_model_catalog,
    sibling_candidates,
    suggest_quant_siblings,
)


def test_packaged_catalog_loads_and_is_sane(tmp_path) -> None:
    entries = load_model_catalog(user_dir=tmp_path / "none")
    assert entries, "packaged catalog must ship a starter set"
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    for e in entries:
        assert "/" in e["id"]
        assert e["family"] and e["quant"] in mc.KNOWN_QUANTS
    # The validated baseline family carries both precisions.
    fams = catalog_families(user_dir=tmp_path / "none")
    quants = {e["quant"] for e in fams["qwen3-30b-a3b"]}
    assert quants == {"bf16", "fp8"}


def test_infer_quant_and_family() -> None:
    assert infer_quant("Qwen/Qwen3-32B-FP8") == "fp8"
    assert infer_quant("org/Model-AWQ") == "awq"
    assert infer_quant("org/Model-GPTQ-Int4") == "gptq-int4"
    assert infer_quant("RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16") == "gptq-int4"
    assert infer_quant("meta-llama/Llama-3.3-70B-Instruct") == "bf16"
    assert infer_family("Qwen/Qwen3-32B-FP8") == "qwen3-32b"
    assert infer_family("meta-llama/Llama-3.3-70B-Instruct") == "llama-3.3-70b-instruct"


def test_add_model_writes_local_overlay(tmp_path) -> None:
    user = tmp_path / "models"
    entry, created = add_catalog_model("org/New-Model-7B", user_dir=user)
    assert created is True
    assert entry["quant"] == "bf16" and entry["family"] == "new-model-7b"
    assert (user / "local.yaml").exists()

    # Idempotent: same id returns the existing entry untouched.
    entry2, created2 = add_catalog_model("org/New-Model-7B", user_dir=user)
    assert created2 is False and entry2["id"] == entry["id"]

    # Merged view carries packaged + local.
    ids = {e["id"] for e in load_model_catalog(user_dir=user)}
    assert "org/New-Model-7B" in ids
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in ids


def test_add_model_rejects_bad_ids(tmp_path) -> None:
    for bad in ("no-slash", "a/b/c", "org/mo del", ""):
        with pytest.raises(CatalogError):
            add_catalog_model(bad, user_dir=tmp_path / "m")
    with pytest.raises(CatalogError):
        add_catalog_model("org/ok", quant="float42", user_dir=tmp_path / "m")


def test_local_overlay_wins_by_id(tmp_path) -> None:
    user = tmp_path / "models"
    user.mkdir()
    (user / "override.yaml").write_text(textwrap.dedent("""\
        models:
          - id: Qwen/Qwen3-30B-A3B-Instruct-2507
            family: qwen3-30b-a3b
            quant: bf16
            engine_args: ["--enable-prefix-caching"]
    """))
    by_id = {e["id"]: e for e in load_model_catalog(user_dir=user)}
    assert by_id["Qwen/Qwen3-30B-A3B-Instruct-2507"]["engine_args"] == \
        ["--enable-prefix-caching"]


def test_sibling_candidates_only_for_base_models() -> None:
    sibs = sibling_candidates("meta-llama/Llama-3.3-70B-Instruct")
    ids = {s["id"] for s in sibs}
    assert "meta-llama/Llama-3.3-70B-Instruct-FP8" in ids
    assert "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic" in ids
    # A quant artifact has no siblings to suggest.
    assert sibling_candidates("Qwen/Qwen3-32B-FP8") == []


def test_suggest_siblings_offline_and_filtering(tmp_path, monkeypatch) -> None:
    """Verified-absent candidates drop; unverified (offline) survive
    as exists=None; already-cataloged ones are flagged."""
    calls = {}

    def fake_exists(model_id, timeout=5.0):
        calls[model_id] = True
        if model_id.endswith("-AWQ"):
            return False
        if model_id.endswith("-FP8"):
            return True
        return None
    monkeypatch.setattr(mc, "hub_model_exists", fake_exists)

    out = suggest_quant_siblings(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        user_dir=tmp_path / "none",
    )
    by_id = {s["id"]: s for s in out}
    fp8 = by_id["Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"]
    assert fp8["exists"] is True
    assert fp8["in_catalog"] is True       # packaged catalog has it
    assert not any(s["id"].endswith("-AWQ") for s in out)
    assert any(s["exists"] is None for s in out)

    # verify=False never touches the network.
    calls.clear()
    out = suggest_quant_siblings(
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        verify=False, user_dir=tmp_path / "none",
    )
    assert not calls and all(s["exists"] is None for s in out)


def test_referenced_models_includes_catalog(tmp_path, monkeypatch) -> None:
    """A model that no profile or space mentions is still stageable
    once it's in the catalog — that's the whole point."""
    from simulator.models import referenced_models
    monkeypatch.chdir(tmp_path)          # no config/ at all
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    entries = referenced_models()
    by_id = {e["model"]: e for e in entries}
    gpt = by_id["openai/gpt-oss-120b"]
    assert gpt["referenced_by"] == ["catalog:gpt-oss-120b"]
    assert gpt["quant"] == "mxfp4" and gpt["approx_size_gb"]
    assert by_id["meta-llama/Llama-3.3-70B-Instruct"]["gated"] is True


def test_search_space_catalog_families(tmp_path, monkeypatch) -> None:
    from simulator.search import SearchSpaceError, load_space
    monkeypatch.chdir(tmp_path)          # packaged catalog only
    space_file = tmp_path / "space.yaml"
    space_file.write_text(textwrap.dedent("""\
        name: fam-test
        engine: vllm_cuda
        device_groups: [[0, 1]]
        catalog_families: [qwen3-30b-a3b]
        dimensions:
          tp: [1, 2]
    """))
    space = load_space(space_file)
    assert set(space.model_variants) == {
        "qwen3-30b-a3b-bf16", "qwen3-30b-a3b-fp8",
    }
    assert space.model_variants["qwen3-30b-a3b-fp8"]["model"] == \
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    # The implicit model_variant dimension covers every family member.
    assert set(space.dimensions["model_variant"]) == set(space.model_variants)

    # Growing the family changes the fingerprint → resume refuses.
    h1 = space.space_hash()
    add_catalog_model("Qwen/Fake-30B-A3B-NVFP4", family="qwen3-30b-a3b",
                      user_dir=tmp_path / "config" / "models")
    h2 = load_space(space_file).space_hash()
    assert h1 != h2

    # Unknown family: named error listing what exists.
    space_file.write_text(textwrap.dedent("""\
        name: bad
        engine: vllm_cuda
        device_groups: [[0]]
        catalog_families: [no-such-family]
        dimensions:
          tp: [1]
    """))
    with pytest.raises(SearchSpaceError, match="no-such-family"):
        load_space(space_file)


def test_models_add_api(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from simulator.service import create_app
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(mc, "hub_model_exists", lambda m, timeout=5.0: None)

    with TestClient(create_app(tmp_path / "runs")) as client:
        r = client.post("/api/models/add",
                        json={"model": "org/Fresh-9B", "check_hub": False})
        assert r.status_code == 200, r.text
        doc = r.json()
        assert doc["created"] is True
        assert doc["entry"]["family"] == "fresh-9b"
        assert (tmp_path / "config" / "models" / "local.yaml").exists()

        # Now listed — and therefore downloadable per the gate.
        models = client.get("/api/models").json()["models"]
        assert any(m["model"] == "org/Fresh-9B" for m in models)

        # Bad id → 422 with the reason.
        r = client.post("/api/models/add",
                        json={"model": "not-an-id", "check_hub": False})
        assert r.status_code == 422


def test_gguf_companion_parsing(tmp_path) -> None:
    """KTransformers loads weights from GGUF only, so a catalog entry
    may name its GGUF companion; both precisions of a family share
    the same one because it is the same weights."""
    by_id = {e["id"]: e for e in load_model_catalog(user_dir=tmp_path / "none")}
    bf16 = by_id["Qwen/Qwen3-30B-A3B-Instruct-2507"]["gguf"]
    fp8 = by_id["Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"]["gguf"]
    assert bf16 == fp8
    assert bf16["repo"] == "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
    assert bf16["file"].endswith("Q4_K_M.gguf") and bf16["size_gb"] == 18.6
    # Entries without one carry an explicit None, never a missing key.
    assert by_id["Qwen/Qwen3-32B"]["gguf"] is None

    user = tmp_path / "models"
    user.mkdir()
    (user / "a.yaml").write_text(textwrap.dedent("""\
        models:
          - id: org/Thing
            gguf: {repo: org/Thing-GGUF, file: Q4_K_M/Thing-Q4_K_M.gguf}
    """))
    e = {x["id"]: x for x in load_model_catalog(user_dir=user)}["org/Thing"]
    assert e["gguf"] == {"repo": "org/Thing-GGUF",
                         "file": "Q4_K_M/Thing-Q4_K_M.gguf", "size_gb": None}

    for bad in ("gguf: {repo: not-a-repo, file: x.gguf}",
                "gguf: {repo: org/R, file: weights.safetensors}",
                "gguf: {repo: org/R, file: ../escape.gguf}",
                "gguf: just-a-string"):
        (user / "a.yaml").write_text(f"models:\n  - id: org/Thing\n    {bad}\n")
        with pytest.raises(CatalogError, match="gguf"):
            load_model_catalog(user_dir=user)


def test_kt_only_and_host_ram_and_shard_directories(tmp_path) -> None:
    """The large end of the catalog: a model whose weights exceed the
    GPUs is kt_only (GPU engines never run it), says how much host
    RAM its GGUF needs, and its companion is a shard DIRECTORY --
    unsloth splits a 387 GB quant into eight files."""
    by_id = {e["id"]: e for e in load_model_catalog(user_dir=tmp_path / "none")}
    v31 = by_id["deepseek-ai/DeepSeek-V3.1"]
    assert v31["kt_only"] is True and v31["host_ram_gb"] == 450.0
    assert v31["gguf"] == {"repo": "unsloth/DeepSeek-V3.1-GGUF",
                           "file": "UD-Q4_K_XL", "size_gb": 386.9}
    # V3.2's sparse-attention architecture is not served by the
    # v0.3.2 image: no companion, and the notes say so.
    v32 = by_id["deepseek-ai/DeepSeek-V3.2"]
    assert v32["gguf"] is None and v32["kt_only"] is False
    assert "KTransformers" in v32["notes"] and "V3.1" in v32["notes"]
    # Qwen3-235B is GPU-fitting AND KTransformers-eligible: both
    # precisions share the sharded Q4_K_M, neither is kt_only.
    bf16 = by_id["Qwen/Qwen3-235B-A22B-Instruct-2507"]
    fp8 = by_id["Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"]
    assert bf16["gguf"] == fp8["gguf"]
    assert bf16["gguf"]["file"] == "Q4_K_M" and bf16["gguf"]["size_gb"] == 142.2
    assert bf16["kt_only"] is False and bf16["host_ram_gb"] == 170.0
    # Every entry carries the keys Track I keys off, never a missing one.
    for e in by_id.values():
        assert "kt_only" in e and "host_ram_gb" in e
        if e["kt_only"]:
            assert e["gguf"], f"{e['id']} is kt_only without a companion"

    user = tmp_path / "models"
    user.mkdir()
    ok = ("gguf: {repo: org/R, file: UD-Q4_K_XL}\n    kt_only: true\n"
          "    host_ram_gb: 420")
    (user / "a.yaml").write_text(f"models:\n  - id: org/Thing\n    {ok}\n")
    e = {x["id"]: x for x in load_model_catalog(user_dir=user)}["org/Thing"]
    assert e["gguf"]["file"] == "UD-Q4_K_XL" and e["kt_only"] is True
    assert e["host_ram_gb"] == 420.0

    for bad, msg in (
        ("kt_only: true", "kt_only but names no gguf"),
        ("gguf: {repo: org/R, file: x.gguf}\n    host_ram_gb: lots", "host_ram_gb"),
        ("gguf: {repo: org/R, file: x.gguf}\n    host_ram_gb: 0", "host_ram_gb"),
        ("gguf: {repo: org/R, file: weights.safetensors}", "gguf"),
        ("gguf: {repo: org/R, file: ./Q4}", "gguf"),
        ("gguf: {repo: org/R, file: Q4//x.gguf}", "gguf"),
    ):
        (user / "a.yaml").write_text(f"models:\n  - id: org/Thing\n    {bad}\n")
        with pytest.raises(CatalogError, match=msg):
            load_model_catalog(user_dir=user)
