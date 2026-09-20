"""Load-generator worker death (improvement plan A2): the coordinator
must notice a dead worker, stop aggregating its frozen counters and
void the window; the worker itself must exit non-zero when one of its
internal loops fails instead of generating at a stale rate."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import simulator.stability
from simulator.open_loop import WorkerPool

MOCK_PORT = 19383

_BASE_CONFIG = {
    "persona_weights": {"quick_lookup": 1.0},
    # A closed port: no request is ever attempted at rate 0, and a
    # stray one fails fast instead of hanging.
    "replica_urls": ["http://127.0.0.1:9/v1"],
    "api_key": "EMPTY",
    "api_model_name": "m",
    "model_id": "mock-model",
    "request_timeout_s": 5,
    "seed": 1,
}


async def _wait_until(pred, timeout_s: float, what: str) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_worker_exits_nonzero_when_a_loop_raises(tmp_path):
    """A command that makes the stdin loop raise must terminate the
    worker with a non-zero exit code (FIRST_EXCEPTION), not leave it
    running deaf with its stat loop still heartbeating."""
    cfg_path = tmp_path / "w.json"
    cfg_path.write_text(json.dumps({**_BASE_CONFIG, "worker_index": 0}))

    async def main():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "simulator.loadgen_worker", str(cfg_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), 60.0)
                assert line, "worker exited before ready"
                if json.loads(line).get("t") == "ready":
                    break
            proc.stdin.write(b'{"cmd":"rate","per_s":"not-a-number"}\n')
            await proc.stdin.drain()
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                raise AssertionError(
                    "worker kept running after its stdin loop raised",
                ) from None
            stderr = (await proc.stderr.read()).decode()
            return rc, stderr
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    rc, stderr = asyncio.run(main())
    assert rc != 0
    assert "failed" in stderr and "ValueError" in stderr


def test_pool_notices_a_killed_worker(tmp_path):
    """Killing one of two workers: the pool drops it from the live set
    within a second, queues a death record, stops aggregating its
    stats, and a later scale_to replaces it under a fresh index."""
    async def main():
        pool = WorkerPool(base_config=_BASE_CONFIG, log_dir=tmp_path,
                          max_workers=3)
        try:
            await pool.scale_to(2)
            await pool.set_rate(0.0)
            await _wait_until(
                lambda: all(w.last_stat for w in pool._workers), 5.0,
                "a stat from every worker")
            assert pool.aggregate()["workers"] == 2
            victim = pool._workers[1]
            victim.proc.kill()
            await _wait_until(lambda: pool.size == 1, 3.0, "death noticed")
            deaths = pool.take_deaths()
            assert [d["index"] for d in deaths] == [1]
            assert deaths[0]["returncode"] != 0
            assert pool.take_deaths() == []          # consumed
            agg = pool.aggregate()
            assert agg["workers"] == 1 and agg["dead_workers"] == 1
            await pool.scale_to(2)
            assert pool.size == 2
            assert [w.index for w in pool._workers] == [0, 2]
            assert (tmp_path / "loadgen_worker_2.log").exists()
        finally:
            await pool.stop()

    asyncio.run(main())


def test_open_loop_run_reports_an_injected_worker_death(tmp_path, monkeypatch):
    """End to end on the mock engine: a worker killed mid-window voids
    that window (superseded, with the death in its detail JSON), the
    rate is re-measured with a replacement worker, and the run still
    finishes ok with a valid export."""
    monkeypatch.setattr(simulator.stability, "MIN_SAMPLES", 8)
    from simulator.export import export_dir, validate_export
    from simulator.open_loop import OpenLoopRunner, run_cohort_open_loop
    from simulator.personas import cohort_from_persona
    from tests.test_open_loop import _open_loop_config

    cfg = _open_loop_config(tmp_path)
    cfg.engine.port = MOCK_PORT
    cfg.simulation.open_loop_max_rate_per_s = 4.0   # one rate only
    cfg.output.db_directory = str(tmp_path / "runs")

    orig = OpenLoopRunner._measure_window
    killed = {"done": False}

    async def patched(self, rate, window_s):
        if not killed["done"]:
            killed["done"] = True

            async def _kill():
                await asyncio.sleep(4.0)   # past the 2 s warmup
                self.pool._workers[0].proc.kill()
            asyncio.get_running_loop().create_task(_kill())
        return await orig(self, rate, window_s)

    monkeypatch.setattr(OpenLoopRunner, "_measure_window", patched)
    db_path = asyncio.run(run_cohort_open_loop(
        cfg, cohort_from_persona("quick_lookup"), new_run=True,
    ))

    from simulator.database import Database
    db = Database(db_path)
    assert db.fetchone("SELECT final_status FROM cohort_run")["final_status"] == "ok"
    ms = db.fetchall(
        "SELECT stability, stability_detail, load_workers "
        "FROM cohort_measurements ORDER BY step_index")
    db.close()
    assert [m["stability"] for m in ms] == ["superseded", "stable"]
    detail = json.loads(ms[0]["stability_detail"])
    assert [d["index"] for d in detail["worker_deaths"]] == [0]
    assert "worker_deaths" not in json.loads(ms[1]["stability_detail"])

    doc, _ = export_dir(cfg.output.db_directory)
    assert not validate_export(doc)
    assert doc["cohorts"][0]["open_loop"]["coverage"] == "capped"
