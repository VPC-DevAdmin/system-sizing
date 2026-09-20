"""Model staging (simulator/models.py + /api/models): cache status,
referenced-model discovery, and one-click downloads via a stub hf CLI."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
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


def test_gguf_status_points_at_the_snapshot_directory(tmp_path) -> None:
    """``path`` is the DIRECTORY holding the .gguf -- KTransformers'
    --gguf_path takes a directory -- and a file in a subfolder of a
    sharded repo resolves to that subfolder."""
    from simulator.models import gguf_status
    cache = tmp_path / "hf"
    spec = {"repo": "org/M-GGUF", "file": "Q4/M-Q4.gguf", "size_gb": 1.5}
    st = gguf_status({"id": "org/M", "gguf": spec}, cache)
    assert st == {"repo": "org/M-GGUF", "file": "Q4/M-Q4.gguf",
                  "size_gb": 1.5, "sharded": False, "cached": False, "path": None}
    assert gguf_status({"id": "org/M", "gguf": None}, cache) is None

    rev = cache / "hub" / "models--org--M-GGUF" / "snapshots" / "r1"
    (rev / "Q4").mkdir(parents=True)
    (rev / "Q4" / "M-Q4.gguf").write_bytes(b"")     # zero bytes: not staged
    assert gguf_status(spec, cache)["cached"] is False
    (rev / "Q4" / "M-Q4.gguf").write_bytes(b"gguf")
    st = gguf_status(spec, cache)
    assert st["cached"] is True and st["path"] == str(rev / "Q4")

    # model_status carries the companion when handed the spec.
    row = model_status("org/M", cache, gguf=spec)
    assert row["gguf"]["cached"] is True
    assert model_status("org/M", cache)["gguf"] is None


def test_download_command_companion(tmp_path, monkeypatch) -> None:
    from simulator.models import download_command
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    argv, env = download_command("Qwen/Qwen3-30B-A3B-Instruct-2507",
                                 companion="gguf")
    assert argv[1:] == ["download", "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                        "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"]
    assert env["HF_HOME"] == str(tmp_path / "hf")
    with pytest.raises(ValueError, match="no GGUF companion"):
        download_command("Qwen/Qwen3-32B", companion="gguf")
    with pytest.raises(ValueError, match="unknown companion"):
        download_command("Qwen/Qwen3-32B", companion="onnx")


def test_models_api_gguf_companion_download(tmp_path, monkeypatch) -> None:
    """The companion is staged under its own key ("<model>#gguf"), the
    row reports it, and a model with no companion is refused (422)."""
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    spec = {"repo": "org/tiny-GGUF", "file": "tiny-Q4.gguf", "size_gb": 0.1}

    stub = tmp_path / "hf-stub"
    stub.write_text(
        "#!/bin/sh\n"
        'REPO_DIR="$HF_HOME/hub/models--$(echo "$2" | sed s#/#--#g)"\n'
        'mkdir -p "$REPO_DIR/snapshots/rev1"\n'
        'echo gguf > "$REPO_DIR/snapshots/rev1/$3"\n'
        'echo "done $2 $3"\n'
    )
    stub.chmod(0o755)
    monkeypatch.setattr(
        models_mod, "download_command",
        lambda model, companion=None: (
            [str(stub), "download", spec["repo"], spec["file"]],
            {"HF_HOME": str(cache)}),
    )
    monkeypatch.setattr(
        models_mod, "referenced_models",
        lambda **kw: [
            {**model_status("org/tiny", cache, gguf=spec),
             "referenced_by": ["catalog:tiny"]},
            {**model_status("org/plain", cache),
             "referenced_by": ["catalog:plain"]},
        ],
    )

    with TestClient(create_app(tmp_path / "runs")) as client:
        rows = {m["model"]: m for m in client.get("/api/models").json()["models"]}
        assert rows["org/tiny"]["gguf"] == {**spec, "sharded": False, "cached": False,
                                            "path": None, "downloading": False}
        assert rows["org/plain"]["gguf"] is None

        r = client.post("/api/models/download",
                        json={"model": "org/plain", "companion": "gguf"})
        assert r.status_code == 422 and "no GGUF companion" in r.json()["detail"]
        r = client.post("/api/models/download",
                        json={"model": "org/tiny", "companion": "onnx"})
        assert r.status_code == 422

        r = client.post("/api/models/download",
                        json={"model": "org/tiny", "companion": "gguf"})
        assert r.status_code == 202, r.text
        assert r.json()["key"] == "org/tiny#gguf"
        deadline = time.time() + 10
        while time.time() < deadline:
            doc = client.get("/api/models").json()
            dl = doc["downloads"].get("org/tiny#gguf")
            if dl and not dl["running"]:
                break
            time.sleep(0.1)
        assert dl["exit_code"] == 0
        assert "done org/tiny-GGUF tiny-Q4.gguf" in dl["log_tail"]
        row = next(m for m in doc["models"] if m["model"] == "org/tiny")
        assert row["gguf"]["cached"] is True and row["gguf"]["downloading"] is False
        assert row["gguf"]["path"] == str(
            cache / "hub" / "models--org--tiny-GGUF" / "snapshots" / "rev1")
        # The safetensors weights are a separate key, untouched.
        assert row["cached"] is False and "org/tiny" not in doc["downloads"]


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
        lambda model, companion=None: ([str(stub), "download", model],
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


def _fake_config_only(cache: Path, model_id: str, arch: str = "deepseek_v3",
                      tokenizer: str | None = "tokenizer.json") -> Path:
    rev = cache / "hub" / ("models--" + model_id.replace("/", "--")) \
        / "snapshots" / "cfg1"
    rev.mkdir(parents=True, exist_ok=True)
    (rev / "config.json").write_text(f'{{"model_type": "{arch}"}}')
    if tokenizer:
        (rev / tokenizer).write_text("{}")
    return rev


def test_sharded_gguf_directory_is_staged_only_when_every_shard_is(tmp_path) -> None:
    """A companion whose ``file`` is a directory: --gguf_path is that
    directory, and it counts as staged only once all -0000i-of-0000N
    shards are present -- a download killed at five of eight would
    otherwise read as ready and fail at load."""
    from simulator.models import gguf_status
    cache = tmp_path / "hf"
    spec = {"repo": "org/Big-GGUF", "file": "UD-Q4_K_XL", "size_gb": 386.9}
    st = gguf_status(spec, cache)
    assert st["sharded"] is True and st["cached"] is False and st["path"] is None

    d = cache / "hub" / "models--org--Big-GGUF" / "snapshots" / "r1" / "UD-Q4_K_XL"
    d.mkdir(parents=True)
    assert gguf_status(spec, cache)["cached"] is False       # empty dir
    (d / "Big-UD-Q4_K_XL-00001-of-00003.gguf").write_bytes(b"g")
    (d / "Big-UD-Q4_K_XL-00003-of-00003.gguf").write_bytes(b"g")
    assert gguf_status(spec, cache)["cached"] is False       # shard 2 missing
    (d / "Big-UD-Q4_K_XL-00002-of-00003.gguf").write_bytes(b"")
    assert gguf_status(spec, cache)["cached"] is False       # shard 2 empty
    (d / "Big-UD-Q4_K_XL-00002-of-00003.gguf").write_bytes(b"g")
    st = gguf_status(spec, cache)
    assert st["cached"] is True and st["path"] == str(d)
    # A single-file companion reports sharded=False as before.
    assert gguf_status({"repo": "org/S", "file": "s.gguf"}, cache)["sharded"] is False


def test_model_status_config_only(tmp_path, monkeypatch) -> None:
    """A kt_only model is cached once config.json and a tokenizer are
    staged (KTransformers takes the weights from the GGUF); the row
    carries the architecture the launcher keys its optimize rule on."""
    cache = tmp_path / "hf"
    st = model_status("org/Big", cache, config_only=True)
    assert st["cached"] is False and st["config_only"] is True and st["arch"] is None

    rev = _fake_config_only(cache, "org/Big", tokenizer=None)
    st = model_status("org/Big", cache, config_only=True)
    assert st["cached"] is False and st["partial"] is True    # no tokenizer
    assert st["arch"] == "deepseek_v3"
    (rev / "tokenizer.model").write_text("spm")
    st = model_status("org/Big", cache, config_only=True)
    assert st["cached"] is True and st["partial"] is False
    # A regular (weights) model reports its arch too.
    _fake_config_only(cache, "org/Small", arch="qwen3_moe")
    assert model_status("org/Small", cache, config_only=False)["arch"] == "qwen3_moe"

    # config_only=None consults the catalog, so callers that only know
    # the id (HF_HUB_OFFLINE gating, the roofline) agree with Prepare.
    monkeypatch.setattr(models_mod, "catalog_entry",
                        lambda mid: {"id": mid, "kt_only": mid == "org/Big"})
    assert model_status("org/Big", cache)["config_only"] is True
    assert model_status("org/Small", cache)["config_only"] is False


def test_download_command_config_only_and_sharded(tmp_path, monkeypatch) -> None:
    """A kt_only entry downloads config + tokenizer only -- never the
    687 GB of safetensors nobody will load -- and a shard-directory
    companion downloads the whole directory."""
    from simulator.models import CONFIG_ONLY_INCLUDE, download_command
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "hf"))
    argv, env = download_command("deepseek-ai/DeepSeek-V3.1")
    # One --include per pattern: hf 1.x's typer CLI repeats the flag,
    # and a bare second pattern would be read as a positional filename.
    assert argv[1:] == ["download", "deepseek-ai/DeepSeek-V3.1"] + [
        a for pat in CONFIG_ONLY_INCLUDE for a in ("--include", pat)]
    assert "*.json" in CONFIG_ONLY_INCLUDE and "tokenizer*" in CONFIG_ONLY_INCLUDE
    assert not any(p.endswith("safetensors") for p in CONFIG_ONLY_INCLUDE)
    argv, _ = download_command("deepseek-ai/DeepSeek-V3.1", companion="gguf")
    assert argv[1:] == ["download", "unsloth/DeepSeek-V3.1-GGUF",
                        "--include", "UD-Q4_K_XL/*.gguf"]
    # GPU-fitting entries keep the plain full download.
    argv, _ = download_command("Qwen/Qwen3-235B-A22B-Instruct-2507")
    assert argv[1:] == ["download", "Qwen/Qwen3-235B-A22B-Instruct-2507"]
    argv, _ = download_command("Qwen/Qwen3-235B-A22B-Instruct-2507", companion="gguf")
    assert argv[1:] == ["download", "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF",
                        "--include", "Q4_K_M/*.gguf"]


def test_referenced_models_rows_carry_kt_fields(tmp_path, monkeypatch) -> None:
    """Prepare's rows (and Track I's roofline) read kt_only,
    host_ram_gb and arch straight off referenced_models."""
    cache = tmp_path / "hf"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    _fake_config_only(cache, "deepseek-ai/DeepSeek-V3.1")
    by_id = {e["model"]: e for e in referenced_models()}
    v31 = by_id["deepseek-ai/DeepSeek-V3.1"]
    assert v31["kt_only"] is True and v31["host_ram_gb"] == 450.0
    assert v31["cached"] is True and v31["config_only"] is True
    assert v31["arch"] == "deepseek_v3"
    assert v31["gguf"]["sharded"] is True and v31["gguf"]["cached"] is False
    small = by_id["Qwen/Qwen3-32B"]
    assert small["kt_only"] is False and small["host_ram_gb"] is None
    assert small["arch"] is None and small["config_only"] is False
