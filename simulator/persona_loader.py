"""Persona/cohort YAML loading and serialization (roadmap Phase 4).

The catalog is data, not code: the packaged
``simulator/personas_data/default.yaml`` carries the canonical
definitions, and site-local ``config/personas/*.yaml`` files (same
format, resolved against the working directory) merge over them — an
id collision replaces the packaged entry, new ids extend the catalog.
The UI's persona editor round-trips through this module: parse to
validate, serialize to prefill the editor.

Every parse error names the persona/field it came from — a catalog
typo should read like a config error, not a stack trace.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .distributions import Constant, Discrete, Distribution, LogNormal

DEFAULT_CATALOG = Path(__file__).parent / "personas_data" / "default.yaml"
USER_CATALOG_DIR = Path("config/personas")

_DISTRIBUTION_FIELDS = (
    "input_tokens",
    "output_tokens",
    "turns_per_session",
    "sessions_before_leaving",
    "inter_session_gap_seconds",
    "read_time_seconds",
    "active_think_seconds",
)
_SLA_FIELDS = (
    "ttft_target_seconds",
    "ttft_failure_seconds",
    "tpot_target_ms",
    "tpot_failure_ms",
)
_TIMEOUT_FIELDS = ("pre_ttft_factor", "inter_token_factor", "hard_timeout_s")


class PersonaSpecError(ValueError):
    """A persona/cohort YAML spec that doesn't parse or validate."""


def parse_distribution(spec, *, where: str) -> Distribution:
    """``{lognormal: {median, sigma}} | {discrete: {v: w}} |
    {constant: x}`` → Distribution. A bare number is shorthand for
    constant."""
    if isinstance(spec, (int, float)):
        return Constant(float(spec))
    if not isinstance(spec, dict) or len(spec) != 1:
        raise PersonaSpecError(
            f"{where}: expected one of lognormal/discrete/constant, got {spec!r}"
        )
    kind, params = next(iter(spec.items()))
    if kind == "lognormal":
        if not isinstance(params, dict) or "sigma" not in params:
            raise PersonaSpecError(f"{where}: lognormal needs median (or mu) + sigma")
        kwargs = {}
        if "min" in params:
            kwargs["min_value"] = float(params["min"])
        if "max" in params:
            kwargs["max_value"] = float(params["max"])
        if "median" in params:
            return LogNormal.from_median(
                float(params["median"]), float(params["sigma"]), **kwargs
            )
        if "mu" in params:
            return LogNormal(mu=float(params["mu"]), sigma=float(params["sigma"]), **kwargs)
        raise PersonaSpecError(f"{where}: lognormal needs median or mu")
    if kind == "discrete":
        if not isinstance(params, dict) or not params:
            raise PersonaSpecError(f"{where}: discrete needs a {{value: weight}} map")
        try:
            weights = {float(v): float(w) for v, w in params.items()}
        except (TypeError, ValueError) as e:
            raise PersonaSpecError(f"{where}: discrete values/weights must be numbers ({e})") from e
        # Integer-valued keys come back as ints so sample_int users and
        # the serializer round-trip cleanly.
        weights = {int(v) if float(v).is_integer() else v: w for v, w in weights.items()}
        return Discrete(weights)
    if kind == "constant":
        try:
            return Constant(float(params))
        except (TypeError, ValueError) as e:
            raise PersonaSpecError(f"{where}: constant needs a number ({e})") from e
    raise PersonaSpecError(f"{where}: unknown distribution kind {kind!r}")


def serialize_distribution(d: Distribution) -> dict:
    import math
    if isinstance(d, LogNormal):
        out: dict = {"median": round(math.exp(d.mu), 6), "sigma": d.sigma}
        if d.min_value:
            out["min"] = d.min_value
        if d.max_value != float("inf"):
            out["max"] = d.max_value
        return {"lognormal": out}
    if isinstance(d, Discrete):
        return {"discrete": dict(d.weights)}
    if isinstance(d, Constant):
        return {"constant": d.value}
    raise TypeError(f"unserializable distribution {type(d).__name__}")


def parse_persona(persona_id: str, spec: dict):
    from .personas import Persona  # deferred: personas imports us at module load

    if not isinstance(spec, dict):
        raise PersonaSpecError(f"persona {persona_id}: spec must be a mapping")
    unknown = set(spec) - {"name", "description", "sla", "timeouts",
                           "ignore_eos", *_DISTRIBUTION_FIELDS}
    if unknown:
        raise PersonaSpecError(
            f"persona {persona_id}: unknown fields {sorted(unknown)}"
        )
    missing = [f for f in _DISTRIBUTION_FIELDS if f not in spec]
    if missing:
        raise PersonaSpecError(f"persona {persona_id}: missing {missing}")
    sla = spec.get("sla")
    if not isinstance(sla, dict) or [f for f in _SLA_FIELDS if f not in sla]:
        raise PersonaSpecError(
            f"persona {persona_id}: sla block must carry {list(_SLA_FIELDS)}"
        )
    if not (0 < sla["ttft_target_seconds"] <= sla["ttft_failure_seconds"]):
        raise PersonaSpecError(
            f"persona {persona_id}: need 0 < ttft_target <= ttft_failure"
        )
    if not (0 < sla["tpot_target_ms"] <= sla["tpot_failure_ms"]):
        raise PersonaSpecError(
            f"persona {persona_id}: need 0 < tpot_target <= tpot_failure"
        )
    timeouts = spec.get("timeouts") or {}
    bad = set(timeouts) - set(_TIMEOUT_FIELDS)
    if bad:
        raise PersonaSpecError(f"persona {persona_id}: unknown timeouts {sorted(bad)}")

    dists = {
        f: parse_distribution(spec[f], where=f"persona {persona_id}.{f}")
        for f in _DISTRIBUTION_FIELDS
    }
    return Persona(
        id=persona_id,
        # Display name: explicit, or a humanized id — the UI never
        # shows raw underscored ids.
        name=str(spec.get("name") or "")
        or persona_id.replace("_", " ").capitalize(),
        description=str(spec.get("description", "")),
        ignore_eos=bool(spec.get("ignore_eos", False)),
        **dists,
        **{f: float(sla[f]) for f in _SLA_FIELDS},
        **{f: float(timeouts[f]) for f in _TIMEOUT_FIELDS if f in timeouts},
    )


def serialize_persona(p) -> dict:
    """Persona → YAML-able spec (the editor's round-trip)."""
    out: dict = {"name": getattr(p, "name", "") or p.id,
                 "description": p.description}
    for f in _DISTRIBUTION_FIELDS:
        out[f] = serialize_distribution(getattr(p, f))
    out["sla"] = {f: getattr(p, f) for f in _SLA_FIELDS}
    if getattr(p, "ignore_eos", False):
        out["ignore_eos"] = True
    timeouts = {
        f: getattr(p, f) for f in _TIMEOUT_FIELDS
        if getattr(p, f) != type(p).__dataclass_fields__[f].default
    }
    if timeouts:
        out["timeouts"] = timeouts
    return out


def parse_cohort(cohort_id: str, spec: dict):
    from .personas import Cohort

    if not isinstance(spec, dict):
        raise PersonaSpecError(f"cohort {cohort_id}: spec must be a mapping")
    unknown = set(spec) - {"name", "description", "persona_weights"}
    if unknown:
        raise PersonaSpecError(f"cohort {cohort_id}: unknown fields {sorted(unknown)}")
    weights = spec.get("persona_weights")
    if not isinstance(weights, dict) or not weights:
        raise PersonaSpecError(f"cohort {cohort_id}: persona_weights map required")
    return Cohort(
        id=cohort_id,
        name=str(spec.get("name", cohort_id)),
        description=str(spec.get("description", "")),
        persona_weights={str(k): float(v) for k, v in weights.items()},
    )


def serialize_cohort(c) -> dict:
    return {
        "name": c.name,
        "description": c.description,
        "persona_weights": dict(c.persona_weights),
    }


def load_catalog_file(path: Path) -> tuple[dict, dict]:
    """One YAML file → ({id: Persona}, {id: Cohort}). Either top-level
    block may be absent."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise PersonaSpecError(f"{path}: invalid YAML — {e}") from e
    if not isinstance(raw, dict):
        raise PersonaSpecError(f"{path}: top level must be a mapping")
    unknown = set(raw) - {"personas", "cohorts"}
    if unknown:
        raise PersonaSpecError(f"{path}: unknown top-level keys {sorted(unknown)}")
    personas = {
        pid: parse_persona(pid, spec)
        for pid, spec in (raw.get("personas") or {}).items()
    }
    cohorts = {
        cid: parse_cohort(cid, spec)
        for cid, spec in (raw.get("cohorts") or {}).items()
    }
    return personas, cohorts


def load_catalog(
    user_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Packaged defaults + user overlay files, merged in name order.

    Cohort weight/reference validation runs at the end, against the
    fully merged persona set — a user cohort may legitimately mix
    packaged and user personas.
    """
    personas, cohorts = load_catalog_file(DEFAULT_CATALOG)
    user_dir = USER_CATALOG_DIR if user_dir is None else user_dir
    if user_dir.exists():
        for f in sorted(user_dir.glob("*.yaml")):
            p, c = load_catalog_file(f)
            personas.update(p)
            cohorts.update(c)
    for cid, cohort in cohorts.items():
        total = sum(cohort.persona_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise PersonaSpecError(
                f"cohort {cid}: weights sum to {total}, must sum to 1.0"
            )
        for pid in cohort.persona_weights:
            if pid not in personas:
                raise PersonaSpecError(
                    f"cohort {cid}: references unknown persona {pid!r}"
                )
    return personas, cohorts
