"""Engines run offline against the hub when the weights are staged.

Observed on the XE7740 (2026-09-20): every roofline cell died in
startup with ``Temporary failure in name resolution`` because vLLM
lists the model repo on the hub even when every file is in the
mounted cache, and the box had no outbound DNS. Prepare stages the
weights precisely so launches do not depend on the network.
"""
from __future__ import annotations

from pathlib import Path

from simulator.config import EngineConfig
from simulator.engines.base import hub_env_args
from simulator.engines.vllm_cuda_multi import VllmCudaMultiEngine


def _stage(cache: Path, model_id: str, complete: bool = True) -> None:
    d = cache / "hub" / f"models--{model_id.replace('/', '--')}"
    snap = d / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    if not complete:
        (d / "blobs").mkdir()
        (d / "blobs" / "x.incomplete").write_text("")


def _env_pairs(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "-e"]


def test_staged_weights_launch_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _stage(tmp_path, "org/M")
    assert hub_env_args("org/M") == ["-e", "HF_HUB_OFFLINE=1"]


def test_missing_or_partial_weights_keep_the_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hub_env_args("org/Missing") == []
    _stage(tmp_path, "org/Partial", complete=False)
    assert hub_env_args("org/Partial") == []
    assert hub_env_args(None) == []
    assert hub_env_args("/models/local-dir") == []


def test_explicit_env_wins_and_token_still_passes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    _stage(tmp_path, "org/M")
    assert hub_env_args("org/M") == ["-e", "HF_TOKEN=hf_abc", "-e", "HF_HUB_OFFLINE=0"]


def test_replica_command_carries_offline_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _stage(tmp_path, "org/M")
    eng = VllmCudaMultiEngine(EngineConfig(
        type="vllm_cuda_multi", model_id="org/M", port=9100,
        replica_devices=[[0], [1]]))
    cmd = eng.build_replica_command(0, [0], "vllm-r0-x")
    assert "HF_HUB_OFFLINE=1" in _env_pairs(cmd)


def test_cache_overflow_symlinks_are_mounted_at_their_own_path(tmp_path, monkeypatch):
    """A cache spread over drives with symlinked model dirs: the engine
    mounts each overflow root at its host path so the links resolve
    inside the container (the XE7740 keeps 1 TB of NVFP4 giants on a
    second drive this way)."""
    from simulator.models import cache_mount_args, cache_overflow_roots

    cache = tmp_path / "cache"
    (cache / "hub" / "models--org--Local").mkdir(parents=True)
    overflow = tmp_path / "overflow"
    (overflow / "hub" / "models--org--Giant").mkdir(parents=True)
    (cache / "hub" / "models--org--Giant").symlink_to(overflow / "hub" / "models--org--Giant")
    (cache / "hub" / "models--org--Dangling").symlink_to(tmp_path / "nowhere")
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(cache))
    assert cache_overflow_roots(cache) == [overflow]
    args = cache_mount_args()
    assert args == ["-v", f"{cache}:/root/.cache/huggingface",
                    "-v", f"{overflow}:{overflow}"]


def test_trtllm_names_the_staged_tokenizer_directory(tmp_path, monkeypatch) -> None:
    """trtllm-serve resolves a hub id's tokenizer through the hub API
    even with the snapshot cached, which offline mode refuses: the
    server came up tokenizer-less and answered HTTP 400 (XE7740,
    Llama-3.3-70B NVFP4). The launcher points --tokenizer at the
    snapshot as the container sees it; an unstaged model keeps the
    hub id and the network."""
    from simulator.engines.trtllm import TrtLlmEngine
    from simulator.models import staged_snapshot_in_container

    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _stage(tmp_path, "org/M")
    _stage(tmp_path, "org/Partial", complete=False)
    assert staged_snapshot_in_container("org/M") == (
        "/root/.cache/huggingface/hub/models--org--M/snapshots/abc123")
    assert staged_snapshot_in_container("org/Partial") is None
    assert staged_snapshot_in_container("org/Missing") is None
    assert staged_snapshot_in_container("/models/local") is None

    def _cmd(model_id):
        eng = TrtLlmEngine(EngineConfig(
            type="trtllm", model_id=model_id, port=9100,
            replica_devices=[[0]], max_model_len=2048))
        return eng.build_replica_command(0, [0], "trtllm-r0-x")

    cmd = _cmd("org/M")
    assert cmd[cmd.index("--tokenizer") + 1] == (
        "/root/.cache/huggingface/hub/models--org--M/snapshots/abc123")
    assert cmd[cmd.index("serve") + 1] == "org/M"      # weights still by id
    assert "HF_HUB_OFFLINE=1" in _env_pairs(cmd)
    assert "--tokenizer" not in _cmd("org/Missing")


def test_hub_token_comes_from_env_then_files(tmp_path, monkeypatch) -> None:
    """One resolver for the engines, the downloader and the hub
    lookups: env first, then HF_TOKEN_PATH, then the login file under
    HF_HOME / ~/.cache/huggingface, then capsim's own cache. ``hf auth
    login`` on the box is enough."""
    from simulator.models import hf_token
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()
    for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    assert hf_token() is None
    (tmp_path / "cache" / "token").write_text("hf_cache\n")
    assert hf_token() == "hf_cache"
    login = tmp_path / "home" / ".cache" / "huggingface"
    login.mkdir(parents=True)
    (login / "token").write_text("hf_login")
    assert hf_token() == "hf_login"
    (tmp_path / "explicit").write_text("hf_path")
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "explicit"))
    assert hf_token() == "hf_path"
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_env2")
    assert hf_token() == "hf_env2"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hf_token() == "hf_env"


def test_engine_containers_and_downloads_carry_the_login_token(tmp_path, monkeypatch) -> None:
    from simulator.models import download_command
    monkeypatch.setenv("OPTIMIZER_HF_CACHE", str(tmp_path))
    for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HOME", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    login = tmp_path / "home" / ".cache" / "huggingface"
    login.mkdir(parents=True)
    (login / "token").write_text("hf_login")
    assert _env_pairs(hub_env_args("org/Missing")) == ["HF_TOKEN=hf_login"]
    _, env = download_command("org/M")
    assert env["HF_TOKEN"] == "hf_login" and env["HF_HOME"] == str(tmp_path)
