"""Persona and cohort definitions.

Distribution parameters here are starting points; tune empirically during
Phase 1 sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions import Discrete, Distribution, LogNormal


@dataclass
class Persona:
    id: str
    description: str

    input_tokens: Distribution
    output_tokens: Distribution
    turns_per_session: Distribution
    think_time_seconds: Distribution
    sessions_before_leaving: Distribution
    inter_session_gap_seconds: Distribution

    ttft_floor_seconds: float
    tpot_floor_ms: float

    # ── Three-tier termination policy ─────────────────────────────────
    # Per-request timeouts that fire when a stream is clearly broken,
    # so a stuck request gets aborted (and counted as an SLA violation
    # via the synthetic-TurnEvent path) rather than tying up an engine
    # slot for 15 minutes. Each tier corresponds to a different
    # failure mode:
    #
    #   Tier 1 (pre-TTFT):    no first token by N × ttft_floor →
    #                          scheduler / admission deadlock
    #   Tier 2 (inter-token): no new token for M × tpot_floor →
    #                          mid-stream worker hang
    #   Tier 3 (hard ceiling):total wall time exceeds hard_timeout_s →
    #                          slow-but-progressing past the SLA
    #                          (informative — engine is functional
    #                           but overloaded, not stuck)
    #
    # Defaults are scaled by ``ttft_floor_seconds`` / ``tpot_floor_ms``
    # so tighter SLAs get tighter timeouts automatically.
    pre_ttft_factor: float = 5.0
    inter_token_factor: float = 20.0
    hard_timeout_s: float = 900.0

    @property
    def pre_ttft_timeout_s(self) -> float:
        return self.ttft_floor_seconds * self.pre_ttft_factor

    @property
    def inter_token_timeout_s(self) -> float:
        return (self.tpot_floor_ms / 1000.0) * self.inter_token_factor


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


PERSONAS: dict[str, Persona] = {
    "quick_lookup": Persona(
        id="quick_lookup",
        description="Frontline support, sales reps, inventory checks",
        input_tokens=LogNormal.from_median(350, 0.35),
        output_tokens=LogNormal.from_median(60, 0.4),
        turns_per_session=Discrete({1: 0.7, 2: 0.2, 3: 0.07, 4: 0.03}),
        think_time_seconds=LogNormal.from_median(15, 0.6),
        sessions_before_leaving=Discrete({3: 0.2, 5: 0.3, 10: 0.3, 20: 0.2}),
        inter_session_gap_seconds=LogNormal.from_median(180, 1.0),
        ttft_floor_seconds=2.0,
        tpot_floor_ms=200.0,
    ),
    "conversational": Persona(
        id="conversational",
        description="Back-and-forth chat: tutoring, coaching, customer dialogue",
        input_tokens=LogNormal.from_median(180, 0.5),
        output_tokens=LogNormal.from_median(220, 0.45),
        turns_per_session=Discrete({3: 0.25, 5: 0.35, 8: 0.25, 12: 0.15}),
        think_time_seconds=LogNormal.from_median(25, 0.5),
        sessions_before_leaving=Discrete({2: 0.3, 4: 0.4, 8: 0.3}),
        inter_session_gap_seconds=LogNormal.from_median(600, 1.0),
        ttft_floor_seconds=3.0,
        tpot_floor_ms=150.0,
    ),
    "drafter": Persona(
        id="drafter",
        description="Email/marketing/short-form writing assistant",
        input_tokens=LogNormal.from_median(500, 0.5),
        output_tokens=LogNormal.from_median(350, 0.4),
        turns_per_session=Discrete({1: 0.5, 2: 0.3, 3: 0.15, 5: 0.05}),
        think_time_seconds=LogNormal.from_median(45, 0.7),
        sessions_before_leaving=Discrete({3: 0.3, 6: 0.4, 12: 0.3}),
        inter_session_gap_seconds=LogNormal.from_median(900, 1.0),
        ttft_floor_seconds=4.0,
        tpot_floor_ms=180.0,
    ),
    "document_qa": Persona(
        id="document_qa",
        description="Legal/finance/research analyst working over long docs",
        input_tokens=LogNormal.from_median(3500, 0.5),
        output_tokens=LogNormal.from_median(280, 0.4),
        turns_per_session=Discrete({1: 0.4, 2: 0.3, 4: 0.2, 7: 0.1}),
        think_time_seconds=LogNormal.from_median(60, 0.6),
        sessions_before_leaving=Discrete({2: 0.4, 4: 0.4, 8: 0.2}),
        inter_session_gap_seconds=LogNormal.from_median(1200, 1.0),
        ttft_floor_seconds=8.0,
        tpot_floor_ms=200.0,
    ),
    "code_assist": Persona(
        id="code_assist",
        description="Engineer pair-programming with the model",
        input_tokens=LogNormal.from_median(1200, 0.6),
        output_tokens=LogNormal.from_median(450, 0.5),
        turns_per_session=Discrete({2: 0.3, 4: 0.35, 8: 0.25, 15: 0.1}),
        think_time_seconds=LogNormal.from_median(30, 0.7),
        sessions_before_leaving=Discrete({4: 0.3, 8: 0.4, 16: 0.3}),
        inter_session_gap_seconds=LogNormal.from_median(420, 1.0),
        ttft_floor_seconds=5.0,
        tpot_floor_ms=160.0,
    ),
    "summarizer": Persona(
        id="summarizer",
        description="Single-shot summarisation of large inputs",
        input_tokens=LogNormal.from_median(5000, 0.4),
        output_tokens=LogNormal.from_median(180, 0.35),
        turns_per_session=Discrete({1: 0.85, 2: 0.15}),
        think_time_seconds=LogNormal.from_median(20, 0.5),
        sessions_before_leaving=Discrete({2: 0.4, 4: 0.4, 8: 0.2}),
        inter_session_gap_seconds=LogNormal.from_median(900, 1.0),
        ttft_floor_seconds=10.0,
        tpot_floor_ms=220.0,
    ),
}


COHORTS: dict[str, Cohort] = {
    # Cohorts are *team mixes* — blended workloads representing realistic
    # business types. To run a single persona on its own, use the
    # ``run-persona`` CLI verb which builds an ephemeral one-persona
    # Cohort via ``cohort_from_persona``.
    "chat_heavy": Cohort(
        id="chat_heavy",
        name="Customer support team",
        description="Frontline support: short queries, fast SLAs, high turnover",
        persona_weights={
            "quick_lookup": 0.6,
            "conversational": 0.3,
            "drafter": 0.1,
        },
    ),
    "document_heavy": Cohort(
        id="document_heavy",
        name="Legal / finance research team",
        description="Long-context document Q&A and summarisation",
        persona_weights={
            "document_qa": 0.55,
            "summarizer": 0.30,
            "drafter": 0.15,
        },
    ),
    "balanced_knowledge": Cohort(
        id="balanced_knowledge",
        name="Balanced knowledge work",
        description="Mixed office workload across personas",
        persona_weights={
            "quick_lookup": 0.20,
            "conversational": 0.20,
            "drafter": 0.20,
            "document_qa": 0.15,
            "code_assist": 0.15,
            "summarizer": 0.10,
        },
    ),
    "engineering_heavy": Cohort(
        id="engineering_heavy",
        name="Engineering team",
        description="Code-assist dominant with light docs and chat",
        persona_weights={
            "code_assist": 0.65,
            "document_qa": 0.15,
            "conversational": 0.15,
            "summarizer": 0.05,
        },
    ),
    "drafter_dominant": Cohort(
        id="drafter_dominant",
        name="Marketing / content team",
        description="Drafting-heavy workload with bursts of summarisation",
        persona_weights={
            "drafter": 0.65,
            "summarizer": 0.20,
            "conversational": 0.15,
        },
    ),
}


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
