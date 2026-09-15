"""Live model discovery: Hub listings → sized, box-validated
candidates (Hub stubbed — no network in tests)."""

from __future__ import annotations

from simulator import discovery as disc
from simulator.discovery import _candidate, discover_models

LISTING = [
    {"modelId": "org/New-40B-A4B-Instruct"},
    {"modelId": "org/New-40B-A4B-Instruct-GGUF"},      # excluded
    {"modelId": "org/Big-700B-Chat"},
    {"modelId": "org/Embedder-embed"},                  # excluded
]

DETAILS = {
    "org/New-40B-A4B-Instruct": {
        "id": "org/New-40B-A4B-Instruct", "gated": False,
        "downloads": 12345, "lastModified": "2026-09-01T00:00:00.000Z",
        "createdAt": "2026-08-20T00:00:00.000Z",
        "safetensors": {"total": 40_000_000_000,
                        "parameters": {"BF16": 40_000_000_000}},
    },
    "org/Big-700B-Chat": {
        "id": "org/Big-700B-Chat", "gated": True,
        "downloads": 99, "lastModified": "2026-07-01T00:00:00.000Z",
        "safetensors": {"total": 700_000_000_000,
                        "parameters": {"F8_E4M3": 690_000_000_000,
                                       "BF16": 10_000_000_000}},
    },
}


def _fake_hub(url: str, timeout: float = 15.0):
    if "?author=" in url:
        return LISTING
    return DETAILS.get(url.rsplit("/api/models/", 1)[-1])


def test_candidate_filter() -> None:
    assert _candidate("Qwen/Qwen3.6-35B-A3B")
    assert _candidate("zai-org/GLM-5.3-Flash")
    assert not _candidate("Qwen/Qwen3-VL-8B-Instruct")
    assert not _candidate("org/Model-Instruct-GGUF")
    assert not _candidate("org/Model-embed-large")


def test_discover_sizes_and_validates(monkeypatch) -> None:
    monkeypatch.setattr(disc, "_hub_get", _fake_hub)
    out = discover_models(orgs=("org",), vram_per_gpu_gb=96.0, max_tp=4,
                          known_ids={"org/Big-700B-Chat"})
    by_id = {e["id"]: e for e in out}
    assert set(by_id) == {"org/New-40B-A4B-Instruct", "org/Big-700B-Chat"}

    small = by_id["org/New-40B-A4B-Instruct"]
    assert small["quant"] == "bf16" and small["moe"] is True
    assert small["params_b"] == 40.0
    # bf16 40B ≈ 83GB weights → min_vram ~96 → tp1 on 96GB cards.
    assert small["feasible"] is True and 1 in small["feasible_tps"]
    assert "MoE" in small["capabilities"]
    assert small["in_catalog"] is False

    big = by_id["org/Big-700B-Chat"]
    assert big["quant"] == "fp8"           # native F8 dtype detected
    assert big["gated"] is True
    assert big["feasible"] is False        # 700B fp8 > 4×96GB
    assert big["in_catalog"] is True
    # Newest first.
    assert out[0]["id"] == "org/New-40B-A4B-Instruct"


def test_discover_api(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from simulator import arena as arena_mod
    from simulator.service import create_app
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(disc, "_hub_get", _fake_hub)
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)

    with TestClient(create_app(tmp_path / "runs")) as client:
        doc = client.get("/api/models/discover?orgs=org").json()
        assert doc["hardware"]["count"] == 8
        assert len(doc["models"]) == 2
        # Add with the discovery metadata → catalog carries it.
        m = doc["models"][0]
        r = client.post("/api/models/add", json={
            "model": m["id"], "check_hub": False, "quant": m["quant"],
            "moe": m["moe"], "params_b": m["params_b"],
            "approx_size_gb": m["approx_size_gb"],
            "min_vram_gb": m["min_vram_gb"],
        })
        assert r.status_code == 200, r.text
        entry = r.json()["entry"]
        assert entry["moe"] is True and entry["min_vram_gb"] == m["min_vram_gb"]


def test_discover_sizes_4bit_checkpoints(monkeypatch) -> None:
    """NVFP4/MXFP4 checkpoints pack two weights per stored byte —
    sizing must come from the dtype table, not bytes-per-param
    heuristics, or a 122B model reads as 244GB instead of 83."""
    listing = [{"modelId": "nvidia/Foo-122B-A10B-NVFP4"}]
    details = {"nvidia/Foo-122B-A10B-NVFP4": {
        "id": "nvidia/Foo-122B-A10B-NVFP4", "gated": False,
        "downloads": 5, "lastModified": "2026-09-01T00:00:00.000Z",
        "safetensors": {"total": 64_600_000_000, "parameters": {
            "U8": 57_400_000_000,          # packed 4-bit pairs
            "BF16": 6_000_000_000,
            "F8_E4M3": 1_200_000_000,      # block scales
        }},
    }}

    def fake(url, timeout=15.0):
        return listing if "?author=" in url \
            else details.get(url.rsplit("/api/models/", 1)[-1])
    monkeypatch.setattr(disc, "_hub_get", fake)
    out = discover_models(orgs=("nvidia",), vram_per_gpu_gb=96.0, max_tp=4)
    (e,) = out
    assert e["quant"] == "nvfp4"
    assert e["approx_size_gb"] == 71.0          # bytes, not 2x params
    assert e["params_b"] == 122.0               # unpacked param count
    assert e["feasible"] is True and 1 in e["feasible_tps"]
    assert "4-bit NVFP4" in e["capabilities"]
