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


def test_optimizer_start_rechecks_after_its_awaits(tmp_path, monkeypatch) -> None:
    """optimizer_start awaits (space doc, seed file, catalog) between
    its guard and Popen. An optimizer that appears during those awaits
    -- a CLI launch taking the flock -- must be seen before anything
    is spawned."""
    import fcntl
    import json
    import os

    from simulator import arena as arena_mod
    from simulator import service as svc

    runs = tmp_path / "runs"
    held: list = []

    def take_the_lock(selection, catalog, budget):
        p = runs / "engine_optimizer" / ".optimizer.lock"
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(json.dumps({"pid": os.getpid(), "started_at": 0,
                             "argv": ["--search", "x"]}))
        fh.flush()
        held.append(fh)
        return {"name": "arena", "engine": "vllm_cuda",
                "device_groups": [[0]], "model_variants": {},
                "dimensions": {"tp": [1]}}
    monkeypatch.setattr(arena_mod, "build_space_doc", take_the_lock)

    def no_spawn(*a, **kw):
        raise AssertionError("Popen must not run once another optimizer holds the lock")
    monkeypatch.setattr(svc.subprocess, "Popen", no_spawn)
    try:
        with TestClient(_make_app(tmp_path)) as client:
            r = client.post("/api/optimizer/start",
                            json={"mode": "arena", "arena": {}})
            assert r.status_code == 409, r.text
            assert "already running" in r.json()["detail"]
            assert held                          # the await did happen
            assert client.app.state.optimizer is None
    finally:
        for fh in held:
            fh.close()


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


def test_attaches_to_external_optimizer(tmp_path, monkeypatch) -> None:
    """A serve restart orphans a running optimizer — the service must
    still report it as running (via the flock) and refuse a second
    start; releasing the lock clears the state."""
    import fcntl
    import json as _json

    from fastapi.testclient import TestClient

    from simulator.service import create_app

    runs = tmp_path / "runs"
    lock_dir = runs / "engine_optimizer"
    lock_dir.mkdir(parents=True)
    (lock_dir / "optimizer_20260101T000000.log").write_text("log\n")
    lock = open(lock_dir / ".optimizer.lock", "w")
    lock.write(_json.dumps({
        "pid": 999999, "started_at": 123.0,
        "argv": ["--search", "runs/engine_optimizer/arena_space.yaml"],
    }))
    lock.flush()
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    stub = tmp_path / "opt.py"
    stub.write_text("import json; print(json.dumps("
                    "{'profiles': {}, 'cells': [], 'default_profile': 'x'}))")
    try:
        with TestClient(create_app(runs, optimizer_script=stub)) as client:
            doc = client.get("/api/optimizer").json()
            assert doc["running"] is True
            active = doc["active"]
            assert active["external"] is True
            assert active["mode"] == "arena"
            assert active["pid"] == 999999
            assert active["log"].endswith("optimizer_20260101T000000.log")

            # Mutual exclusion holds across the attach boundary.
            r = client.post("/api/optimizer/start",
                            json={"mode": "registry", "profile": "x"})
            assert r.status_code == 409

            # Lock released (runner exited) -> no longer running.
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            doc = client.get("/api/optimizer").json()
            assert doc["running"] is False
    finally:
        if not lock.closed:
            lock.close()


def test_search_history_archive_and_promote(tmp_path, monkeypatch) -> None:
    """A fresh run archives the previous search (results + space copy,
    space_file re-pointed) so optimizer runs have history — and an
    archived winner can still be promoted."""
    import json
    import textwrap

    import yaml as _yaml
    from fastapi.testclient import TestClient

    from simulator.search import load_space
    from simulator.service import create_app

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    opt_dir = runs / "engine_optimizer"
    opt_dir.mkdir(parents=True)

    space_path = opt_dir / "arena_space.yaml"
    space_path.write_text(textwrap.dedent("""\
        name: histspace
        engine: vllm_cuda
        device_groups: [[0, 1]]
        model_variants:
          bf16: {model: org/M, served_name: m}
        dimensions:
          tp: [1]
          dp: [2]
    """))
    space = load_space(space_path)
    (opt_dir / "search.json").write_text(json.dumps({
        "kind": "search", "space": "histspace",
        "space_file": str(space_path),
        "space_hash": space.space_hash(),
        "generated_at": "2026-09-15T20:00:00+00:00",
        "summary": {"evaluated": 5, "done_reason": "converged",
                    "best": {"key": "k", "score": 9000.0,
                             "params": {"model_variant": "bf16",
                                        "tp": 1, "dp": 2}}},
        "state": {"space_hash": space.space_hash()},
    }))

    stub = tmp_path / "opt.py"
    stub.write_text("import json; print(json.dumps("
                    "{'profiles': {}, 'cells': [], 'default_profile': 'x'}))")
    with TestClient(create_app(runs, optimizer_script=stub)) as client:
        # No history yet.
        assert client.get("/api/optimizer/history").json() == []

        # A fresh arena start archives the old results first. (The
        # start itself fails later or not — archive happens first;
        # use a bad selection so no subprocess launches.)
        client.post("/api/optimizer/start", json={
            "mode": "arena", "new_run": True,
            "arena": {"models": ["org/NopeNotInCatalog"]},
        })
        hist = client.get("/api/optimizer/history").json()
        assert len(hist) == 1
        entry = hist[0]
        assert entry["space"] == "histspace"
        assert entry["evaluated"] == 5 and entry["best_score"] == 9000.0

        # The archived doc's space_file points at the archived COPY.
        doc = client.get(
            f"/api/optimizer/history/{entry['file']}").json()
        assert "history" in doc["space_file"]
        assert _yaml.safe_load(open(doc["space_file"]))["name"] == "histspace"

        # Promote straight from history — hash still verifies.
        r = client.post("/api/optimizer/promote",
                        json={"source": "search", "file": entry["file"]})
        assert r.status_code == 200, r.text
        assert r.json()["profile"] == "optimized-histspace"

        # Path traversal refused.
        assert client.get(
            "/api/optimizer/history/..%2Fsearch.json").status_code in (404, 422)


def test_combined_group_view(tmp_path, monkeypatch) -> None:
    """Runs over the SAME model set merge into one ranking (dedup by
    candidate key, best score wins); a run with a different
    measurement is excluded by fingerprint, never silently mixed."""
    import json
    import textwrap

    from fastapi.testclient import TestClient

    from simulator.search import load_space
    from simulator.service import create_app

    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    hist = runs / "engine_optimizer" / "history"
    hist.mkdir(parents=True)

    space_yaml = textwrap.dedent("""\
        name: g
        engine: vllm_cuda
        device_groups: [[0, 1, 2, 3]]
        model_variants:
          bf16: {model: org/M, served_name: m}
        dimensions:
          tp: [1, 2]
          dp: [1, 2]
    """)
    sp = hist / "search_A_space.yaml"
    sp.write_text(space_yaml)
    space = load_space(sp)

    def _doc(evaluated, generated, measurement=None):
        return {
            "kind": "search", "space": "g", "space_file": str(sp),
            "space_hash": space.space_hash(), "generated_at": generated,
            "objective": {"kind": "sla_throughput"},
            "measurement": measurement or {"input_tokens": 512,
                                           "output_tokens": 256,
                                           "ladder": [8, 32]},
            "summary": {"evaluated": len(evaluated)},
            "state": {"space_hash": space.space_hash(),
                      "evaluated": evaluated},
        }

    cell = {"cell_name": "ladder_c0032", "samples": 32, "errors": 0,
            "timeouts": 0, "ttft_p95_ms": 500.0, "tpot_p95_ms": 40.0,
            "throughput_out_tok_s": 5000.0}
    # Run A: dp shapes. Run B (the "addendum"): tp shapes + a repeat
    # of one key with a WORSE score (the better one must survive).
    (hist / "search_A_g.json").write_text(json.dumps(_doc({
        "tp=1|dp=2": {"status": "ok", "score": 9000.0, "iteration": 0,
                      "params": {"tp": 1, "dp": 2}, "config_name": "a1",
                      "cells": [cell]},
    }, "2026-09-15T20:00:00+00:00")))
    (hist / "search_B_g.json").write_text(json.dumps(_doc({
        "tp=2|dp=1": {"status": "ok", "score": 7000.0, "iteration": 0,
                      "params": {"tp": 2, "dp": 1}, "config_name": "b1",
                      "cells": [cell]},
        "tp=1|dp=2": {"status": "ok", "score": 8500.0, "iteration": 0,
                      "params": {"tp": 1, "dp": 2}, "config_name": "b2",
                      "cells": [cell]},
    }, "2026-09-15T21:00:00+00:00")))
    # Different measurement: incomparable scores — must be excluded.
    (hist / "search_C_g.json").write_text(json.dumps(_doc({
        "tp=2|dp=2": {"status": "ok", "score": 99999.0, "iteration": 0,
                      "params": {"tp": 2, "dp": 2}, "config_name": "c1",
                      "cells": [cell]},
    }, "2026-09-15T22:00:00+00:00",
        measurement={"input_tokens": 128, "output_tokens": 64,
                     "ladder": [4]})))

    stub = tmp_path / "opt.py"
    stub.write_text("import json; print(json.dumps("
                    "{'profiles': {}, 'cells': [], 'default_profile': 'x'}))")
    with TestClient(create_app(runs, optimizer_script=stub)) as client:
        entries = client.get("/api/optimizer/history").json()
        keys = {e["group_key"] for e in entries}
        assert len(keys) == 1                     # same model set → one group
        gk = keys.pop()

        doc = client.get(f"/api/optimizer/combined/{gk}").json()
        s = doc["summary"]
        assert s["evaluated"] == 2                # merged + deduped
        assert s["best"]["score"] == 9000.0       # better duplicate won
        assert s["best"]["source"] == "search_A_g.json"
        assert doc["promote_file"] == "search_A_g.json"
        assert doc["excluded"] == ["search_C_g.json"]
        assert "combined view of 2 run(s)" in s["done_reason"]


def test_arena_start_seeds_from_group_history(tmp_path, monkeypatch) -> None:
    """Reopening a (series, size-range) investigation seeds the new
    run from the group's archived evaluations — the start response
    reports the count and the driver receives --seed-results."""
    import json
    import textwrap

    from fastapi.testclient import TestClient

    from simulator import arena as arena_mod
    from simulator.search import load_space
    from simulator.service import create_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [96.0] * 8)
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)

    runs = tmp_path / "runs"
    hist = runs / "engine_optimizer" / "history"
    hist.mkdir(parents=True)

    # Archived run over the same catalog models (same group + same
    # default objective/measurement as a new arena space).
    models = ["Qwen/Qwen3-30B-A3B-Instruct-2507",
              "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"]
    import yaml as _yaml

    from simulator.arena import build_space_doc
    old_doc = build_space_doc({"models": models}, None, 20)
    sp = hist / "search_old_space.yaml"
    sp.write_text(_yaml.safe_dump(old_doc, sort_keys=False))
    space = load_space(sp)
    (hist / "search_old_arena.json").write_text(json.dumps({
        "kind": "search", "space": "arena", "space_file": str(sp),
        "space_hash": space.space_hash(),
        "generated_at": "2026-09-15T20:00:00+00:00",
        "objective": old_doc["objective"],
        "measurement": old_doc["measurement"],
        "summary": {"evaluated": 1},
        "state": {"space_hash": space.space_hash(), "evaluated": {
            "model_variant=qwen3-30b-a3b-bf16|tp=1|dp=8": {
                "status": "ok", "score": 9000.0, "iteration": 0,
                "params": {"model_variant": "qwen3-30b-a3b-bf16",
                           "tp": 1, "dp": 8}, "config_name": "s1",
                "cells": []}}},
    }))

    # Stub optimizer script records its argv.
    stub = tmp_path / "opt.py"
    stub.write_text(textwrap.dedent("""\
        import json, sys
        if "--list-json" in sys.argv:
            print(json.dumps({"profiles": {}, "cells": [],
                              "default_profile": "x"}))
        else:
            open("argv.txt", "w").write("\\n".join(sys.argv[1:]))
    """))
    with TestClient(create_app(runs, optimizer_script=stub)) as client:
        r = client.post("/api/optimizer/start", json={
            "mode": "arena", "new_run": True, "budget": 24,
            "arena": {"models": models},
        })
        assert r.status_code == 202, r.text
        assert r.json()["seeded"] == 1
        import time as _t
        deadline = _t.time() + 5
        while _t.time() < deadline and not (tmp_path / "argv.txt").exists():
            _t.sleep(0.05)
        argv = (tmp_path / "argv.txt").read_text()
        assert "--seed-results" in argv
        seed = json.loads(
            (runs / "engine_optimizer" / "seed.json").read_text())
        assert len(seed["evaluated"]) == 1
