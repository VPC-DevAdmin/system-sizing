"""simulation.mode / --mode (improvement plan B3): the methodology is
a config field and a CLI option, unknown config keys are reported
instead of silently dropped, and the shipped YAMLs are clean."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from simulator.config import (
    Config,
    _merge_dataclass,
    apply_cli_overrides,
    load_config,
)

REPO = Path(__file__).parent.parent


def test_unknown_config_keys_warn_with_dotted_path(caplog):
    cfg = Config()
    with caplog.at_level(logging.WARNING, logger="simulator.config"):
        _merge_dataclass(cfg, {
            "simulation": {
                "open_loop_windows_s": 5,          # typo
                "stabilization_cv_threshold": 0.15,  # removed knob
                "open_loop_window_s": 42,           # real
            },
            "nonsense": {"x": 1},
        })
    messages = [r.getMessage() for r in caplog.records]
    assert any("'simulation.open_loop_windows_s'" in m for m in messages)
    assert any("'simulation.stabilization_cv_threshold'" in m for m in messages)
    assert any("'nonsense'" in m for m in messages)
    assert cfg.simulation.open_loop_window_s == 42     # known keys still apply


def test_mode_defaults_open_and_parses_from_yaml(tmp_path):
    assert Config().simulation.mode == "open"
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"simulation": {"mode": "closed"}}))
    assert load_config(p).simulation.mode == "closed"
    p.write_text(yaml.safe_dump({"simulation": {"mode": "sideways"}}))
    with pytest.raises(ValueError, match="simulation.mode"):
        load_config(p)
    cfg = Config()
    apply_cli_overrides(cfg, mode="closed")
    assert cfg.simulation.mode == "closed"
    with pytest.raises(ValueError):
        apply_cli_overrides(cfg, mode="bogus")


def test_shipped_configs_load_without_unknown_keys(caplog):
    """Every YAML under config/ (profiles included) is a live contract
    with the dataclasses; a stale key means the docs or the config
    are lying about a knob."""
    paths = sorted((REPO / "config").glob("*.yaml")) + sorted(
        (REPO / "config" / "profiles").glob("*.yaml"))
    # arena*.yaml is the optimizer's search-space document (read by
    # simulator.arena), not a run config.
    paths = [p for p in paths if not p.name.startswith("arena")]
    assert paths
    with caplog.at_level(logging.WARNING, logger="simulator.config"):
        for p in paths:
            load_config(p)
    unknown = [r.getMessage() for r in caplog.records if "unknown key" in r.getMessage()]
    assert not unknown, unknown


def test_mock_profile_has_short_open_loop_windows():
    cfg = load_config(REPO / "config" / "profiles" / "mock.yaml")
    sim = cfg.simulation
    assert sim.mode == "open"
    assert sim.open_loop_window_s <= 30
    assert sim.open_loop_refine_window_s <= 60
    assert sim.open_loop_warmup_s <= 15
    assert sim.open_loop_settle_max_s is not None and sim.open_loop_settle_max_s <= 30


# ── CLI dispatch ──────────────────────────────────────────────────────


@pytest.fixture()
def dispatch(monkeypatch, tmp_path):
    """Stub both runners; record which one each command picked."""
    import simulator.open_loop as open_loop
    import simulator.runner as runner
    calls: list[tuple[str, dict]] = []

    async def fake_open(cfg, cohort, **kw):
        calls.append(("open", {"cohort": getattr(cohort, "id", cohort),
                               "mode": cfg.simulation.mode, **kw}))
        return tmp_path / "open.db"

    async def fake_closed(cfg, cohort, **kw):
        calls.append(("closed", {"cohort": getattr(cohort, "id", cohort),
                                 "mode": cfg.simulation.mode, **kw}))
        return tmp_path / "closed.db"

    monkeypatch.setattr(open_loop, "run_cohort_open_loop", fake_open)
    monkeypatch.setattr(runner, "run_cohort", fake_closed)
    monkeypatch.chdir(REPO)
    return calls


def _invoke(args):
    from simulator.cli import app
    return CliRunner().invoke(app, args)


def test_run_defaults_to_open_loop(dispatch):
    r = _invoke(["run", "--cohort", "chat_heavy", "--profile", "mock"])
    assert r.exit_code == 0, r.output
    assert [c[0] for c in dispatch] == ["open"]
    assert dispatch[0][1]["mode"] == "open"


def test_run_mode_closed_and_closed_knobs_imply_closed(dispatch):
    r = _invoke(["run", "--cohort", "chat_heavy", "--profile", "mock",
                 "--mode", "closed"])
    assert r.exit_code == 0, r.output
    r = _invoke(["run", "--cohort", "chat_heavy", "--profile", "mock",
                 "--adaptive"])
    assert r.exit_code == 0, r.output
    r = _invoke(["run", "--cohort", "chat_heavy", "--profile", "mock",
                 "--pool-sizes", "8,16"])
    assert r.exit_code == 0, r.output
    assert [c[0] for c in dispatch] == ["closed"] * 3
    assert dispatch[1][1]["adaptive"] is True
    assert dispatch[2][1]["fixed_grid_pool_sizes"] == [8, 16]
    assert all(c[1]["mode"] == "closed" for c in dispatch)


def test_run_mode_open_rejects_closed_loop_knobs(dispatch):
    r = _invoke(["run", "--cohort", "chat_heavy", "--profile", "mock",
                 "--mode", "open", "--adaptive"])
    assert r.exit_code != 0
    assert "closed-loop knobs" in r.output
    r = _invoke(["run", "--cohort", "chat_heavy", "--profile", "mock",
                 "--mode", "sideways"])
    assert r.exit_code != 0
    assert not dispatch


def test_run_persona_honours_mode(dispatch):
    r = _invoke(["run-persona", "--persona", "quick_lookup", "--profile", "mock"])
    assert r.exit_code == 0, r.output
    r = _invoke(["run-persona", "--persona", "quick_lookup", "--profile", "mock",
                 "--mode", "closed"])
    assert r.exit_code == 0, r.output
    assert [c[0] for c in dispatch] == ["open", "closed"]
    assert dispatch[0][1]["cohort"] == "quick_lookup"


def test_sweep_dispatches_each_workload_by_mode(monkeypatch, dispatch, tmp_path):
    """run_sweep launches the engine once and runs every workload with
    the selected methodology, passing the shared engine + run dir."""
    import simulator.runner as runner

    class FakeEngine:
        def launch(self, log_dir=None):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(runner, "make_engine", lambda *_a, **_k: FakeEngine())
    monkeypatch.setattr(runner, "preflight_check", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "resolve_run_dir", lambda *_a, **_k: tmp_path / "run_01")

    r = _invoke(["sweep", "--profile", "mock", "--type", "quick_lookup,chat_heavy",
                 "--new-run"])
    assert r.exit_code == 0, r.output
    assert [c[0] for c in dispatch] == ["open", "open"]
    assert [c[1]["cohort"] for c in dispatch] == ["quick_lookup", "chat_heavy"]
    assert all(c[1]["run_dir"] == tmp_path / "run_01" for c in dispatch)
    assert all(isinstance(c[1]["engine"], FakeEngine) for c in dispatch)
    dispatch.clear()
    r = _invoke(["sweep", "--profile", "mock", "--type", "quick_lookup",
                 "--new-run", "--mode", "closed"])
    assert r.exit_code == 0, r.output
    assert [c[0] for c in dispatch] == ["closed"]
