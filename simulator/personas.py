"""Persona and cohort dataclasses + the loaded catalog.

The definitions themselves are DATA (roadmap Phase 4): the packaged
``simulator/personas_data/default.yaml`` is canonical, with site-local
``config/personas/*.yaml`` overlays merged on top at import (and on
``reload_personas()``, which the service's persona editor calls after
writing an overlay). This module keeps the dataclasses, the registry
dicts, and the lookup helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions import Distribution


@dataclass
class Persona:
    id: str
    description: str

    input_tokens: Distribution
    output_tokens: Distribution
    turns_per_session: Distribution
    sessions_before_leaving: Distribution
    inter_session_gap_seconds: Distribution

    # ── Post-response delay ──────────────────────────────────────────
    # Recorded as two distributions so the buyer page can show the
    # distinction, summed for the actual virtual-user sleep.
    #
    #   read_time_seconds:    residual reading after stream ends.
    #                         Users read concurrently with streaming;
    #                         this is what's left when the model
    #                         stops talking. Sized as
    #                         ``output_tokens × max(0, 200ms -
    #                         typical_TPOT) / 1000`` assuming a
    #                         healthy ~75 ms TPOT and 5 tok/s
    #                         reading speed (~230 wpm).
    #   active_think_seconds: deliberation, response composition.
    #                         Persona-specific — long for code/doc
    #                         work, short for quick lookups.
    read_time_seconds: Distribution
    active_think_seconds: Distribution

    # ── SLA thresholds ───────────────────────────────────────────────
    # Two-tier SLA per metric:
    #
    #   target_*:  ideal experience. User happy, content with the
    #              response time, would describe service as "fast."
    #   failure_*: hard SLA boundary. Past this, the user abandons
    #              or escalates. Capacity is gated on FAILURE rate;
    #              the target rate is informational quality signal.
    #
    # Cohort-level capacity_status uses the failure thresholds
    # (existing semantics, just with looser bar). target_status is
    # a parallel reading using the target thresholds — so the
    # buyer page can report "supports N users at SLA, M users at
    # premium quality" with both numbers honest.
    ttft_target_seconds: float
    ttft_failure_seconds: float
    tpot_target_ms: float
    tpot_failure_ms: float

    # ── Three-tier abort policy (per-request timeouts) ───────────────
    # Per-request timeouts that fire when a stream is clearly broken,
    # so a stuck request gets aborted (and counted as an SLA violation
    # via the synthetic-TurnEvent path) rather than tying up an engine
    # slot for 15 minutes. Each tier corresponds to a different
    # failure mode:
    #
    #   Tier 1 (pre-TTFT):    no first token by N × ttft_failure →
    #                          scheduler / admission deadlock
    #   Tier 2 (inter-token): no new token for M × tpot_failure →
    #                          mid-stream worker hang
    #   Tier 3 (hard ceiling):total wall time exceeds hard_timeout_s →
    #                          slow-but-progressing past the SLA
    #                          (informative — engine is functional
    #                           but overloaded, not stuck)
    #
    # Sized off the FAILURE thresholds so a request that's slow but
    # within the failure budget still gets to complete. Aborting a
    # request that's just brushing target would be over-aggressive.
    pre_ttft_factor: float = 5.0
    inter_token_factor: float = 20.0
    hard_timeout_s: float = 900.0

    @property
    def pre_ttft_timeout_s(self) -> float:
        return self.ttft_failure_seconds * self.pre_ttft_factor

    @property
    def inter_token_timeout_s(self) -> float:
        return (self.tpot_failure_ms / 1000.0) * self.inter_token_factor


@dataclass
class Cohort:
    id: str
    name: str
    description: str
    persona_weights: dict  # {persona_id: weight}, must sum to 1.0
    # ``"cohort"`` for the curated team-mix entries in COHORTS.
    # ``"persona"`` for ephemeral single-persona Cohorts built by
    # ``cohort_from_persona`` — used internally to plumb persona runs
    # through the same runner code as cohort runs. The category flows
    # into the export JSON so the buyer page can render personas and
    # cohorts in separate sections.
    category: str = "cohort"

    def validate(self) -> None:
        total = sum(self.persona_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Cohort {self.id} weights sum to {total}, must sum to 1.0"
            )
        for pid in self.persona_weights:
            if pid not in PERSONAS:
                raise ValueError(f"Cohort {self.id} references unknown persona {pid}")


# ── Registry ─────────────────────────────────────────────────────────
# Loaded from the packaged default catalog + config/personas/ overlays.
# Mutated IN PLACE by reload_personas(): other modules hold references
# to these dicts (pool_manager indexes PERSONAS directly), so the
# objects must stay identical across reloads.

from .persona_loader import load_catalog  # noqa: E402  (needs dataclasses above)

PERSONAS: dict[str, Persona] = {}
COHORTS: dict[str, Cohort] = {}


def reload_personas(user_dir=None) -> None:
    """(Re)build the registries from the catalog files. In-place so
    existing references observe the update; raises PersonaSpecError
    (leaving the registries untouched) when any file is invalid."""
    personas, cohorts = load_catalog(user_dir=user_dir)
    PERSONAS.clear()
    PERSONAS.update(personas)
    COHORTS.clear()
    COHORTS.update(cohorts)


reload_personas()


def cohort_from_persona(persona_id: str) -> Cohort:
    """Build an ephemeral Cohort wrapping one persona at 100% weight.

    Lets the runner — which works in terms of Cohort objects — drive
    a persona-only run without polluting the COHORTS dict. Stored runs
    will have ``category="persona"`` and the persona's id as the
    cohort_id, so the export and buyer page can render personas in a
    separate section from cohorts.
    """
    persona = get_persona(persona_id)
    return Cohort(
        id=persona_id,
        name=f"Persona: {persona.id}",
        description=persona.description,
        persona_weights={persona_id: 1.0},
        category="persona",
    )


def resolve_workload_group(arg: str) -> tuple[list[str], list[str]]:
    """Resolve the ``--type`` argument for the sweep CLI.

    Returns ``(persona_ids, cohort_ids)`` to run, in the order they
    should be executed. Accepts:

      * ``"all"`` (default) — every persona, then every cohort
      * ``"personas"``      — every persona, no cohorts
      * ``"cohorts"``       — every cohort, no personas
      * comma-list mixing both, with persona/cohort ids resolved by
        membership in PERSONAS / COHORTS respectively
    """
    arg = (arg or "all").strip()
    if arg == "all":
        return list(PERSONAS.keys()), list(COHORTS.keys())
    if arg == "personas":
        return list(PERSONAS.keys()), []
    if arg == "cohorts":
        return [], list(COHORTS.keys())
    persona_ids: list[str] = []
    cohort_ids: list[str] = []
    for item in (s.strip() for s in arg.split(",") if s.strip()):
        if item in PERSONAS:
            persona_ids.append(item)
        elif item in COHORTS:
            cohort_ids.append(item)
        else:
            raise KeyError(
                f"Unknown persona/cohort id {item!r}. "
                f"Personas: {sorted(PERSONAS)}, cohorts: {sorted(COHORTS)}"
            )
    return persona_ids, cohort_ids


def get_cohort(cohort_id: str) -> Cohort:
    if cohort_id not in COHORTS:
        raise KeyError(
            f"Unknown cohort '{cohort_id}'. Known: {sorted(COHORTS)}"
        )
    cohort = COHORTS[cohort_id]
    cohort.validate()
    return cohort


def get_persona(persona_id: str) -> Persona:
    if persona_id not in PERSONAS:
        raise KeyError(
            f"Unknown persona '{persona_id}'. Known: {sorted(PERSONAS)}"
        )
    return PERSONAS[persona_id]
