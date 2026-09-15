"""Model staging (simulator/models.py + /api/models): cache status,
referenced-model discovery, and one-click downloads via a stub hf CLI."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from simulator import models as models_mod
from simulator.models import model_status, referenced_models
from simulator.service import create_app


def _fake_cached_model(cache: Path, model_id: str, size: int = 1000) -> None:
    d = cache / "hub" / ("models--" + model_id.replace("/", "--"))
    rev = d / "snapshots" / "abc123"
    rev.mkdir(parents=True)
    (rev / "model.safetensors").write_bytes(b"x" * size)


def test_model_status(tmp_path) -> None:
    cache = tmp_path / "hf"
    st = model_status("org/absent", cache)
    assert st["cached"] is False and st["partial"] is False

    _fake_cached_model(cache, "org/present")
    st = model_status("org/present", cache)
    assert st["cached"] is True and st["size_gb"] >= 0

    # An .incomplete blob marks a partial download.
    d = cache / "hub" / "models--org--present" / "blobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "xyz.incomplete").write_bytes(b"y")
    st = model_status("org/present", cache)
    assert st["cached"] is False and st["partial"] is True


def test_referenced_models_finds_profiles_and_spaces(monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).parent.parent)
    entries = referenced_models()
    ids = {e["model"] for e in entries}
    # GPU profile + both search-space variants.
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in ids
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" in ids
    by_id = {e["model"]: e for e in entries}
    refs = by_id["Qwen/Qwen3-30B-A3B-Instruct-2507"]["referenced_by"]
    assert any(r.startswith("profile:") for r in refs)
    assert any(r.startswith("space:") for r in refs)
    # CPU pre-staged /models paths are not HF downloads.
    assert not any(m.startswith("/") for m in ids)


def test_models_api_download_flow(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))

    # Stub hf CLI: "downloads" by creating the snapshot layout.
    stub = tmp_path / "hf-stub"
    stub.write_text(
        "#!/bin/sh\n"
        'MODEL_DIR="$HF_HOME/hub/models--$(echo "$2" | sed s#/#--#g)"\n'
        'mkdir -p "$MODEL_DIR/snapshots/rev1"\n'
        'echo weights > "$MODEL_DIR/snapshots/rev1/model.safetensors"\n'
        'echo "done $2"\n'
    )
    stub.chmod(0o755)
    monkeypatch.setattr(
        models_mod, "download_command",
        lambda model: ([str(stub), "download", model],
                       {"HF_HOME": str(cache)}),
    )
    monkeypatch.setattr(
        models_mod, "referenced_models",
        lambda **kw: [
            {**model_status("org/tiny", cache), "referenced_by": ["space:t/x"]},
        ],
    )

    with TestClient(create_app(tmp_path / "runs")) as client:
        doc = client.get("/api/models").json()
        assert doc["models"][0]["model"] == "org/tiny"
        assert doc["models"][0]["cached"] is False

        # Unknown model refused — not a general download proxy.
        assert client.post("/api/models/download",
                           json={"model": "org/other"}).status_code == 404

        r = client.post("/api/models/download", json={"model": "org/tiny"})
        assert r.status_code == 202, r.text
        deadline = time.time() + 10
        while time.time() < deadline:
            doc = client.get("/api/models").json()
            dl = doc["downloads"].get("org/tiny")
            if dl and not dl["running"]:
                break
            time.sleep(0.1)
        assert dl["exit_code"] == 0
        assert "done org/tiny" in dl["log_tail"]
        assert doc["models"][0]["cached"] is True


def test_storage_api(tmp_path, monkeypatch) -> None:
    """Storage step: filesystem listing, current-cache resolution, and
    persisting a chosen location that downstream resolution honors."""
    import simulator.models as models_mod

    monkeypatch.delenv("OPTIMIZER_HF_CACHE", raising=False)
    monkeypatch.setattr(models_mod, "STORAGE_CONFIG",
                        tmp_path / "cfg" / "storage.json")

    with TestClient(create_app(tmp_path / "runs")) as client:
        doc = client.get("/api/storage").json()
        assert doc["hf_cache_source"] in ("default", "data-layout")
        assert isinstance(doc["filesystems"], list) and doc["filesystems"]
        fs = doc["filesystems"][0]
        assert {"mountpoint", "free_gb", "total_gb", "fstype"} <= set(fs)

        # Relative path refused with a human-readable reason.
        r = client.post("/api/storage", json={"hf_cache": "relative/path"})
        assert r.status_code == 422 and "absolute" in r.json()["detail"]

        # A good choice persists and resolution follows it.
        target = tmp_path / "bigdisk" / "capsim" / "huggingface"
        r = client.post("/api/storage", json={"hf_cache": str(target)})
        assert r.status_code == 200, r.text
        assert r.json()["resolved"] == str(target)
        assert target.exists()
        assert models_mod.hf_cache_dir() == target
        assert models_mod.hf_cache_source() == "configured"
        doc = client.get("/api/storage").json()
        assert doc["hf_cache"] == str(target)

        # Env override wins and blocks UI changes with an explanation.
        monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "env"))
        r = client.post("/api/storage", json={"hf_cache": str(target)})
        assert r.status_code == 422 and "OPTIMIZER_HF_CACHE" in r.json()["detail"]
