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
