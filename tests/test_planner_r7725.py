"""The R7725 planner queue: one AMD MI350P, every job one GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import planner_campaign as pc  # noqa: E402
import planner_queues as pq  # noqa: E402


def test_r7725_queue_is_single_gpu_and_unique():
    for tiers in ((1, 2), (1, 2, 3)):
        doc = pq.r7725(tiers)
        ids = [j["id"] for j in doc["jobs"]]
        assert len(ids) == len(set(ids))
        assert all(j["devices"] == 1 and j["tp"] == 1 for j in doc["jobs"])
        # No NVIDIA-only checkpoints on the AMD card.
        assert not any("nvfp4" in j["model"].lower() for j in doc["jobs"])
        # The box's own HF cache, not the XE7740's /data2 layout.
        assert all(j["engine_extra"]["docker_volumes"] == pq.R7725_HF for j in doc["jobs"])
    for tiers in ((1, 2), (1, 2, 3)):
        jobs = pq.r7725(tiers)["jobs"]
        groups = {j["tune_group"] for j in jobs if j.get("tune_group")}
        # Every tune_key must be filled by a tuning group in the same queue,
        # or the runner waits on it forever.
        assert {j["tune_key"] for j in jobs if j.get("tune_key")} <= groups
    by = {j["id"]: j for j in pq.r7725((1, 2))["jobs"]}
    assert "tune_key" not in by["H1-qwen36-1gpu-h_assistant"]
    kv = by["K-qwen30b-1gpu-conversational-kvoff"]
    assert "--kv-offloading-backend" in kv["engine_extra"]["vllm_extra_flags"]
    assert kv["host_mem_gb"] <= pq.r7725()["host_mem_budget_gb"]
    # Tuned models wait for their pick; setup and tuning runs do not.
    assert by["G1.1-qwen30b-1gpu-writer"]["tune_key"] == "qwen30b-gpu"
    assert "tune_key" not in by["G0-qwen30b-1gpu-conversational"]
    assert by["T1-qwen30b-1gpu-default"]["tune_group"] == "qwen30b-gpu"
    assert by["H1-gemma26b-1gpu-h_research_agent"]["max_model_len"] == 131072
    assert by["TW-gptoss20b-1gpu-t_agent_support"]["max_model_len"] == pq.T_LEN


def test_r7725_runner_treats_it_as_a_gpu_box(tmp_path):
    doc = pq.r7725((1,))
    doc["state_dir"], doc["runs_dir"] = str(tmp_path / "s"), str(tmp_path / "r")
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump(doc))
    q = pc.Queue(path)
    assert q.total_devices() == [0]
    assert pc.pick_devices([0], 1, "r7725") == [0]
    assert pc.pick_devices([], 1, "r7725") is None
    job = q.jobs[0]
    cfg = yaml.safe_load(pc.build_config(q, job, [0], "g0", tmp_path / "run").read_text())
    # One replica on one GPU is the single-engine path, as on the XE7740;
    # its stock image becomes the ROCm build at launch (gpu_vendor).
    assert cfg["engine"]["type"] == "vllm_cuda"
    assert cfg["engine"]["gpu_image"] == "vllm/vllm-openai:latest"
    # The MI350P's VRAM, not the XE7740 card's 95.6 GB.
    assert cfg["engine"]["vram_per_gpu_gb"] == 143.7
    assert cfg["telemetry"]["enable_gpu"] is True
    assert cfg["simulation"]["open_loop_max_workers"] >= 8
