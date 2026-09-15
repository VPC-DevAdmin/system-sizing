"""Optimizer-through-the-service (UI backend): catalog listing,
subprocess lifecycle, results surfacing, and mutual exclusion with
capacity runs. Uses a stub optimizer script — the real one needs
Docker + engines; its registry shapes are covered by --list-json."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from simulator.service import create_app

STUB = """#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path

CATALOG = {
    "profiles": {"stub_gpu": [
        {"name": "baseline", "description": "b", "expected_outcome": "", "replicas": 1},
        {"name": "variant", "description": "v", "expected_outcome": "", "replicas": 1},
    ]},
    "cells": [{"name": "cell_a", "input_tokens": 128, "output_tokens": 128,
               "concurrency": 1}],
    "default_profile": "stub_gpu",
}

p = argparse.ArgumentParser()
p.add_argument("--list-json", action="store_true")
p.add_argument("--out", type=Path)
p.add_argument("--profile")
p.add_argument("--new-run", action="store_true")
p.add_argument("--only", nargs="+")
p.add_argument("--search", type=Path)
p.add_argument("--search-out", type=Path)
args = p.parse_args()

if args.list_json:
    print(json.dumps(CATALOG)); sys.exit(0)

if args.search:
    args.search_out.parent.mkdir(parents=True, exist_ok=True)
    args.search_out.write_text(json.dumps({
        "kind": "search", "space": args.search.stem,
        "summary": {"evaluated": 3, "ok": 3, "failed": 0,
                     "done_reason": "budget",
                     "best": {"key": "tp=2", "params": {"tp": 2},
                              "score": 5000.0, "config_name": "s001"},
                     "top": [], "iterations": []},
        "state": {"space_hash": "x", "iterations": [],
                   "evaluated": {}, "done_reason": "budget"},
    }))
    sys.exit(0)

args.out.parent.mkdir(parents=True, exist_ok=True)
names = args.only or ["baseline", "variant"]
args.out.write_text(json.dumps({
    "profile": args.profile, "model": "stub", "generated_at": "now",
    "configs": [
        {"name": n, "description": "", "status": "ok", "launch_seconds": 1.0,
         "failure_reason": "",
         "cells": [{"cell_name": "cell_a", "samples": 4, "errors": 0,
                    "timeouts": 0, "ttft_p50_ms": 100.0, "ttft_p95_ms": 150.0,
                    "tpot_p50_ms": 10.0, "tpot_p95_ms": 12.0,
                    "throughput_out_tok_s": 500.0, "tier_errors": {}}]}
        for n in names
    ],
}))
"""


def _make_app(tmp_path: Path):
    stub = tmp_path / "stub_optimizer.py"
    stub.write_text(STUB)
    return create_app(tmp_path / "runs", optimizer_script=stub)


def test_optimizer_catalog_and_lifecycle(tmp_path) -> None:
    with TestClient(_make_app(tmp_path)) as client:
        status = client.get("/api/optimizer").json()
        assert status["running"] is False
        assert "stub_gpu" in status["catalog"]["profiles"]
        assert status["results"] is None

        # Unknown profile refused.
        r = client.post("/api/optimizer/start", json={"profile": "nope"})
        assert r.status_code == 404

        # Start with a config subset; stub exits quickly.
        r = client.post("/api/optimizer/start", json={
            "profile": "stub_gpu", "only": ["baseline"], "new_run": True,
        })
        assert r.status_code == 202, r.text
        deadline = time.time() + 15
        while time.time() < deadline:
            status = client.get("/api/optimizer").json()
            if not status["running"] and status["results"]:
                break
            time.sleep(0.2)
        assert status["results"]["profile"] == "stub_gpu"
        assert [c["name"] for c in status["results"]["configs"]] == ["baseline"]
        assert status["active"]["exit_code"] == 0
        assert Path(status["active"]["log"]).exists()

        # Stop with nothing running → 409.
        assert client.post("/api/optimizer/stop").status_code == 409


def test_optimizer_excludes_capacity_runs(tmp_path, monkeypatch) -> None:
    """A running optimizer blocks run starts and vice versa — they
    contend for the same engines/GPUs/ports."""
    with TestClient(_make_app(tmp_path)) as client:
        # Fake an alive optimizer process.
        class FakeProc:
            pid = 999999
            def poll(self):
                return None
        app = client.app
        app.state.optimizer = {"proc": FakeProc(), "profile": "stub_gpu",
                               "started_at": 0.0, "log": "x.log"}
        r = client.post("/api/runs", json={
            "config": "config/default.yaml",
            "workload": {"kind": "cohort", "id": "chat_heavy"},
        })
        assert r.status_code == 409
        assert "optimizer" in r.json()["detail"]

        # And a fake active capacity run blocks the optimizer.
        app.state.optimizer = None

        class FakeTask:                  # duck-typed asyncio.Task
            def done(self):
                return False
        from simulator.service import ActiveRun
        app.state.active = ActiveRun(
            task=FakeTask(), workload={"kind": "cohort", "id": "x"},
            config_path="c", started_at=0.0,
        )
        r = client.post("/api/optimizer/start", json={"profile": "stub_gpu"})
        assert r.status_code == 409
        assert "capacity run" in r.json()["detail"]
        app.state.active = None


def test_optimizer_search_mode(tmp_path) -> None:
    """Search mode resolves a space by name, launches the driver with
    --search, and surfaces search.json through the status endpoint."""
    import yaml as _yaml

    space_dir = tmp_path / "spaces"
    space_dir.mkdir()
    (space_dir / "tiny.yaml").write_text(_yaml.safe_dump({
        "engine": "vllm_cuda", "device_groups": [[0]],
        "model_variants": {"bf16": {"model": "m"}},
        "dimensions": {"tp": [1]},
    }))
    import simulator.search as search_mod
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        import unittest.mock as mock
        with mock.patch.object(
            search_mod, "list_spaces",
            lambda directory=None: {"tiny": str(space_dir / "tiny.yaml")},
        ):
            status = client.get("/api/optimizer").json()
            assert "tiny" in status["spaces"]

            r = client.post("/api/optimizer/start", json={
                "mode": "search", "space": "nope"})
            assert r.status_code == 404

            r = client.post("/api/optimizer/start", json={
                "mode": "search", "space": "tiny", "new_run": True})
            assert r.status_code == 202, r.text
            deadline = time.time() + 15
            while time.time() < deadline:
                status = client.get("/api/optimizer").json()
                if not status["running"] and status["search_results"]:
                    break
                time.sleep(0.2)
            assert status["search_results"]["summary"]["best"]["score"] == 5000.0
            assert status["active"]["mode"] == "search"
