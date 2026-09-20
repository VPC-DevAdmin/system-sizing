"""Service robustness (improvement plan D2/D4): the loopback-only
serve guard, the runs-dir lock, request validation that used to 500,
and the WebSocket handler noticing a client leaving."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from simulator import service as service_mod
from simulator.service import RunsDirBusy, create_app

REPO = Path(__file__).resolve().parent.parent


# ── D2: capsim serve refuses non-loopback binds ──────────────────────

def test_serve_host_guard_accepts_loopback_and_refuses_others() -> None:
    from simulator.cli import check_serve_host
    for h in ("127.0.0.1", "localhost", "::1", " 127.0.0.1 "):
        check_serve_host(h, insecure=False)      # no raise
    import typer
    for h in ("0.0.0.0", "192.168.1.20", "::", "myhost"):
        with pytest.raises(typer.BadParameter) as ei:
            check_serve_host(h, insecure=False)
        assert "container images" in str(ei.value)
        check_serve_host(h, insecure=True)       # explicit opt-in


def test_serve_cli_refuses_0000_without_insecure(monkeypatch) -> None:
    """The CLI exits non-zero, explains why, and never reaches uvicorn."""
    from simulator import cli as cli_mod
    called = []
    monkeypatch.setattr(service_mod, "serve",
                        lambda **kw: called.append(kw))
    r = CliRunner().invoke(cli_mod.app, ["serve", "--host", "0.0.0.0"])
    assert r.exit_code != 0
    assert "arbitrary container images" in r.output
    assert "--insecure" in r.output
    assert called == []


def test_serve_cli_insecure_reaches_serve(monkeypatch, tmp_path) -> None:
    from simulator import cli as cli_mod
    called = []
    monkeypatch.setattr(service_mod, "serve",
                        lambda **kw: called.append(kw))
    r = CliRunner().invoke(cli_mod.app, [
        "serve", "--host", "0.0.0.0", "--insecure", "--port", "9",
        "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert called and called[0]["host"] == "0.0.0.0"
    assert "WARNING" in r.output


# ── D3/D4: a second serve on the same runs dir refuses to start ─────

@pytest.fixture()
def foreign_lock_holder():
    """A separate PROCESS holding the runs-dir lock — what a second
    ``capsim serve`` looks like from this one."""
    import subprocess
    import sys
    procs = []

    def _hold(runs: Path):
        runs.mkdir(parents=True, exist_ok=True)
        code = (
            "import fcntl, sys, time\n"
            f"f = open({str(runs / service_mod.SERVE_LOCK_NAME)!r}, 'a+')\n"
            "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('locked', flush=True)\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE, text=True)
        assert proc.stdout.readline().strip() == "locked"
        procs.append(proc)
        return proc

    yield _hold
    for proc in procs:
        proc.kill()
        proc.wait()


def test_second_process_on_same_runs_dir_is_refused(
        tmp_path, foreign_lock_holder) -> None:
    runs = tmp_path / "runs"
    holder = foreign_lock_holder(runs)
    with pytest.raises(RunsDirBusy) as ei:
        create_app(runs)
    assert "another capsim serve" in str(ei.value)
    # A different dir is fine.
    other = create_app(tmp_path / "other")
    service_mod._release_runs_lock(other.state.runs_lock)
    # The moment the other process lets go, this one can start.
    holder.kill()
    holder.wait()
    app = create_app(runs)
    assert (runs / service_mod.SERVE_LOCK_NAME).read_text().strip().isdigit()
    with TestClient(app):
        pass                       # lifespan shutdown releases it
    assert str(runs.resolve()) not in service_mod._HELD_RUNS_LOCKS


def test_same_process_reuses_the_lock(tmp_path) -> None:
    """Several apps on one dir inside one process (the test suite's
    own pattern) share the lock and release it on the last one."""
    runs = tmp_path / "runs"
    a = create_app(runs)
    b = create_app(runs)
    key = str(runs.resolve())
    assert service_mod._HELD_RUNS_LOCKS[key][1] == 2
    service_mod._release_runs_lock(a.state.runs_lock)
    assert key in service_mod._HELD_RUNS_LOCKS
    service_mod._release_runs_lock(b.state.runs_lock)
    assert key not in service_mod._HELD_RUNS_LOCKS


def test_lock_protects_live_run_from_second_startup(
        tmp_path, foreign_lock_holder) -> None:
    """The failure D3 describes: service B starting on A's dir used to
    stamp A's unfinalised (live) cohort_run 'interrupted'. Now B
    cannot get past the lock, so the row is untouched."""
    import sqlite3

    from simulator.database import Database
    runs = tmp_path / "runs"
    (runs / "run_01").mkdir(parents=True)
    db = Database(runs / "run_01" / "run.db")
    db.insert_run(cohort_run_id="live", started_at="2026-09-19T00:00:00Z",
                  engine_type="mock", model_id="m", cohort_id="c",
                  cohort_definition={"name": "c", "persona_weights": {}},
                  config={})
    db.close()
    foreign_lock_holder(runs)      # "service A" owns the dir
    with pytest.raises(RunsDirBusy):
        create_app(runs)
    conn = sqlite3.connect(runs / "run_01" / "run.db")
    assert conn.execute("SELECT final_status FROM cohort_run").fetchone()[0] \
        is None
    conn.close()


# ── D4: validation that used to be a 500 ────────────────────────────

def test_custom_int_levers_are_validated(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(REPO)
    with TestClient(create_app(tmp_path / "runs")) as client:
        for bad in ("eight", 0, -1, True, 2.5):
            r = client.post("/api/runs", json={
                "custom": {"model_id": "org/M", "replicas": bad, "tp": 1},
                "workload": {"kind": "cohort", "id": "chat_heavy"},
            })
            assert r.status_code == 422, (bad, r.text)
            assert "custom.replicas" in r.json()["detail"]
        r = client.post("/api/runs", json={
            "custom": {"model_id": "org/M", "replicas": 1, "tp": 1,
                       "max_num_seqs": "lots"},
            "workload": {"kind": "cohort", "id": "chat_heavy"},
        })
        assert r.status_code == 422
        assert "custom.max_num_seqs" in r.json()["detail"]


def test_config_must_be_yaml_and_parse(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(REPO)
    not_yaml = tmp_path / "engine.txt"
    not_yaml.write_text("engine: {type: mock}")
    broken = tmp_path / "broken.yaml"
    broken.write_text("engine: [unclosed\n")
    wrong_shape = tmp_path / "list.yaml"
    wrong_shape.write_text("- just\n- a list\n")
    with TestClient(create_app(tmp_path / "runs")) as client:
        wl = {"kind": "cohort", "id": "chat_heavy"}
        r = client.post("/api/runs", json={"config": str(not_yaml),
                                           "workload": wl})
        assert r.status_code == 422 and ".yaml" in r.json()["detail"]
        r = client.post("/api/runs", json={"config": str(broken),
                                           "workload": wl})
        assert r.status_code == 422, r.text
        assert "not a valid capsim config" in r.json()["detail"]
        r = client.post("/api/runs", json={"config": str(wrong_shape),
                                           "workload": wl})
        assert r.status_code == 422, r.text
        r = client.post("/api/runs", json={"config": str(tmp_path / "no.yaml"),
                                           "workload": wl})
        assert r.status_code == 404
        assert client.get("/api/status").json()["active_run"] is None


def test_headline_doc_rejects_path_tricks(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "runs")) as client:
        assert client.get("/api/runs/../x/headline").status_code in (404, 422)
        r = client.get("/api/runs/.hidden/headline")
        assert r.status_code == 422
        r = client.get("/api/runs/run_99/headline")
        assert r.status_code == 404


def test_hardware_endpoint_reports_detected_and_source(
        tmp_path, monkeypatch) -> None:
    from simulator import arena as arena_mod
    cfg = tmp_path / "arena.yaml"
    cfg.write_text("device_groups: [[0, 1, 2, 3], [4, 5, 6, 7]]\n")
    monkeypatch.setattr(arena_mod, "ARENA_CONFIG", cfg)
    monkeypatch.setattr(arena_mod, "detect_gpus", lambda: [])
    with TestClient(create_app(tmp_path / "runs")) as client:
        hw = client.get("/api/hardware").json()
        assert hw["gpus"] == 8 and hw["detected_gpus"] == 0
        assert hw["source"] == "config (unverified: detected 0)"
        arena = client.get("/api/arena").json()
        assert arena["hardware"]["source"].startswith("config (unverified")


def test_tail_text_reads_only_the_end(tmp_path) -> None:
    p = tmp_path / "dl.log"
    p.write_text("x" * 10_000 + "END")
    t = service_mod._tail_text(p, 4096)
    assert len(t) == 4096 and t.endswith("END")
    assert service_mod._tail_text(tmp_path / "missing.log") == ""


def test_models_endpoint_tails_download_log(tmp_path) -> None:
    """A running download's log_tail comes from the last 4 KB, not a
    whole-file read on every poll."""
    import subprocess
    log = tmp_path / "m.log"
    log.write_text("a" * 20_000 + "\nlast line 42%\n")
    with TestClient(create_app(tmp_path / "runs")) as client:
        proc = subprocess.Popen(["sleep", "5"])
        try:
            client.app.state.model_downloads["org/M"] = {
                "proc": proc, "log": str(log), "started_at": 0.0}
            d = client.get("/api/models").json()["downloads"]["org/M"]
            assert d["running"] is True
            assert d["log_tail"].endswith("last line 42%\n")
            assert len(d["log_tail"]) <= 400
        finally:
            proc.kill()
            proc.wait()


def test_ws_client_disconnect_releases_subscriber(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "runs")) as client:
        assert client.get("/api/status").json()["bus_subscribers"] == 0
        with client.websocket_connect("/ws/telemetry"):
            assert client.get("/api/status").json()["bus_subscribers"] == 1
        # The handler must have seen websocket.disconnect and unsubscribed.
        import time
        for _ in range(50):
            if client.get("/api/status").json()["bus_subscribers"] == 0:
                break
            time.sleep(0.05)
        assert client.get("/api/status").json()["bus_subscribers"] == 0


def test_stop_returns_202_when_teardown_outlasts_timeout(
        tmp_path, monkeypatch) -> None:
    """A run whose cancellation takes longer than STOP_TIMEOUT_S gets
    a 202 'stopping' rather than a request held open."""
    import asyncio

    monkeypatch.setattr(service_mod, "STOP_TIMEOUT_S", 0.2)
    with TestClient(create_app(tmp_path / "runs")) as client:
        app = client.app
        loop_holder: dict = {}

        async def _slow_teardown():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(1.0)       # slow teardown
                raise

        def _install():
            task = asyncio.get_running_loop().create_task(_slow_teardown())
            app.state.active = service_mod.ActiveRun(
                task=task, workload={"kind": "cohort", "id": "x"},
                config_path="c", started_at=0.0)
            loop_holder["task"] = task
        client.portal.call(_install)
        r = client.post("/api/runs/stop")
        assert r.status_code == 202, r.text
        assert r.json()["stopping"] is True
        # Teardown finishes on its own; status reflects it.
        import time
        for _ in range(60):
            if not client.get("/api/status").json()["active_run"]["running"]:
                break
            time.sleep(0.05)
        assert not client.get("/api/status").json()["active_run"]["running"]
