"""Personas-as-data (roadmap Phase 4): catalog loading, overlays,
validation errors, round-trip serialization, and the editor API."""

from __future__ import annotations

import pytest

from simulator.distributions import Constant, Discrete, LogNormal
from simulator.persona_loader import (
    DEFAULT_CATALOG,
    PersonaSpecError,
    load_catalog,
    load_catalog_file,
    parse_distribution,
    serialize_distribution,
    serialize_persona,
)
from simulator.personas import COHORTS, PERSONAS, reload_personas


@pytest.fixture(autouse=True)
def restore_registry():
    """Any test that reloads with a custom overlay dir must leave the
    process-global registries as the packaged defaults."""
    yield
    reload_personas()


def test_packaged_catalog_loads() -> None:
    personas, cohorts = load_catalog_file(DEFAULT_CATALOG)
    assert set(personas) == {
        "quick_lookup", "conversational", "writer", "document_qa",
        "code_assist", "long_form_generator",
        "headline_generation", "headline_ingest",
    }
    assert len(cohorts) == 5
    ql = personas["quick_lookup"]
    assert isinstance(ql.input_tokens, LogNormal)
    assert ql.ttft_failure_seconds == 15.0
    assert isinstance(ql.turns_per_session, Discrete)
    # Discrete keys parse as ints so sample_int users round-trip.
    assert 1 in ql.turns_per_session.weights


def test_distribution_parse_and_roundtrip() -> None:
    for spec in (
        {"lognormal": {"median": 350, "sigma": 0.5}},
        {"discrete": {1: 0.7, 2: 0.3}},
        {"constant": 42},
    ):
        d = parse_distribution(spec, where="t")
        again = parse_distribution(serialize_distribution(d), where="t")
        assert type(d) is type(again)
    # Bare number is constant shorthand.
    assert isinstance(parse_distribution(5, where="t"), Constant)
    with pytest.raises(PersonaSpecError, match="unknown distribution"):
        parse_distribution({"gaussian": {}}, where="t")


def test_persona_serialize_roundtrip() -> None:
    from simulator.persona_loader import parse_persona
    p = PERSONAS["code_assist"]
    again = parse_persona("code_assist", serialize_persona(p))
    assert again.ttft_target_seconds == p.ttft_target_seconds
    assert abs(again.input_tokens.mu - p.input_tokens.mu) < 1e-9


def test_overlay_merges_and_overrides(tmp_path) -> None:
    (tmp_path / "custom.yaml").write_text("""
personas:
  support_bot:
    description: custom archetype
    input_tokens: {lognormal: {median: 100, sigma: 0.4}}
    output_tokens: {constant: 50}
    turns_per_session: {discrete: {1: 1.0}}
    sessions_before_leaving: {constant: 5}
    inter_session_gap_seconds: {constant: 60}
    read_time_seconds: {constant: 5}
    active_think_seconds: {constant: 5}
    sla:
      ttft_target_seconds: 5
      ttft_failure_seconds: 10
      tpot_target_ms: 100
      tpot_failure_ms: 200
cohorts:
  support_team:
    name: Support
    description: mixes packaged + custom personas
    persona_weights: {support_bot: 0.5, quick_lookup: 0.5}
""")
    personas, cohorts = load_catalog(user_dir=tmp_path)
    assert "support_bot" in personas and "quick_lookup" in personas
    assert cohorts["support_team"].persona_weights["support_bot"] == 0.5

    # In-place registry reload: existing references observe the update.
    ref = PERSONAS
    reload_personas(user_dir=tmp_path)
    assert "support_bot" in ref


def test_validation_errors_are_specific(tmp_path) -> None:
    (tmp_path / "bad.yaml").write_text("""
cohorts:
  broken:
    name: b
    description: b
    persona_weights: {nope: 1.0}
""")
    with pytest.raises(PersonaSpecError, match="unknown persona 'nope'"):
        load_catalog(user_dir=tmp_path)

    (tmp_path / "bad.yaml").write_text("""
personas:
  half:
    description: missing everything
""")
    with pytest.raises(PersonaSpecError, match="persona half: missing"):
        load_catalog(user_dir=tmp_path)


def test_headline_stress_personas_shape() -> None:
    """The marketing-number generators: single turn, zero think —
    open-loop arrivals degenerate to a request firehose, so the
    stability boundary IS max sustained throughput."""
    from simulator.personas import PERSONAS
    for pid in ("headline_generation", "headline_ingest"):
        p = PERSONAS[pid]
        assert p.turns_per_session.sample_int(__import__("random").Random(0)) == 1
        assert p.read_time_seconds.sample(__import__("random").Random(0)) == 0
        assert p.active_think_seconds.sample(__import__("random").Random(0)) == 0
        assert "marketing" in p.description.lower() \
            or "stress" in p.description.lower()
    g = PERSONAS["headline_generation"]
    assert g.input_tokens.sample_int(__import__("random").Random(0)) == 32
    assert g.output_tokens.sample_int(__import__("random").Random(0)) == 1024
    i = PERSONAS["headline_ingest"]
    assert i.input_tokens.sample_int(__import__("random").Random(0)) == 4096


def test_editor_api_create_edit_reject(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from simulator.service import create_app

    catalog_dir = tmp_path / "personas"
    with TestClient(create_app(tmp_path / "runs", catalog_dir=catalog_dir)) as client:
        # Read an existing persona as YAML.
        detail = client.get("/api/personas/quick_lookup").json()
        assert "lognormal" in detail["yaml"]

        # Create a new persona through the editor.
        r = client.put("/api/personas/support_bot", json={"yaml": """
description: created via API
input_tokens: {lognormal: {median: 100, sigma: 0.4}}
output_tokens: {constant: 50}
turns_per_session: {discrete: {1: 1.0}}
sessions_before_leaving: {constant: 5}
inter_session_gap_seconds: {constant: 60}
read_time_seconds: {constant: 5}
active_think_seconds: {constant: 5}
sla:
  ttft_target_seconds: 5
  ttft_failure_seconds: 10
  tpot_target_ms: 100
  tpot_failure_ms: 200
"""})
        assert r.status_code == 200, r.text
        assert (catalog_dir / "support_bot.yaml").exists()
        assert "support_bot" in PERSONAS
        ids = [p["id"] for p in client.get("/api/personas").json()]
        assert "support_bot" in ids

        # Cohort referencing it.
        r = client.put("/api/cohorts/support_team", json={"yaml": """
name: Support
description: via API
persona_weights: {support_bot: 1.0}
"""})
        assert r.status_code == 200, r.text
        assert "support_team" in COHORTS

        # Invalid save: rejected with a specific error, file not kept,
        # registry untouched.
        r = client.put("/api/cohorts/broken", json={"yaml": """
name: broken
description: bad
persona_weights: {ghost: 1.0}
"""})
        assert r.status_code == 422
        assert "unknown persona 'ghost'" in r.json()["detail"]
        assert not (catalog_dir / "broken.yaml").exists()
        assert "broken" not in COHORTS

        # Bad edit of an EXISTING entry: previous file content restored.
        r = client.put("/api/personas/support_bot", json={"yaml": "description: no"})
        assert r.status_code == 422
        assert "missing" in r.json()["detail"]
        assert "support_bot" in PERSONAS   # still intact

        # Structured-JSON save — what the graphical designer sends
        # (the UI has no YAML anywhere). Same validation path.
        detail = client.get("/api/personas/support_bot").json()
        spec = detail["spec"]
        spec["input_tokens"] = {"lognormal": {"median": 500, "sigma": 0.4}}
        r = client.put("/api/personas/support_bot", json={"spec": spec})
        assert r.status_code == 200, r.text
        got = client.get("/api/personas/support_bot").json()["spec"]
        assert got["input_tokens"]["lognormal"]["median"] == 500

        r = client.put("/api/cohorts/support_team", json={"spec": {
            "name": "Support", "description": "via spec JSON",
            "persona_weights": {"support_bot": 1.0},
        }})
        assert r.status_code == 200, r.text

        # Structured save with bad weights rejected the same way.
        r = client.put("/api/cohorts/support_team", json={"spec": {
            "name": "Support", "description": "bad",
            "persona_weights": {"support_bot": 0.4},
        }})
        assert r.status_code == 422
        assert "sum" in r.json()["detail"]

        # Neither yaml nor spec → explicit 422.
        r = client.put("/api/personas/support_bot", json={})
        assert r.status_code == 422
