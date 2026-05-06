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


PERSONAS: dict[str, Persona] = {
    "quick_lookup": Persona(
        id="quick_lookup",
        description="Short factual queries with brief responses; FAQ-style",
        input_tokens=LogNormal.from_median(350, 0.35),
        output_tokens=LogNormal.from_median(60, 0.4),
        turns_per_session=Discrete({1: 0.7, 2: 0.2, 3: 0.07, 4: 0.03}),
        sessions_before_leaving=Discrete({3: 0.2, 5: 0.3, 10: 0.3, 20: 0.2}),
        inter_session_gap_seconds=LogNormal.from_median(180, 1.0),
        # Short response (60 tokens) — at typical fast TPOT (~75 ms)
        # the user falls behind by ~125 ms × 60 = ~8 s. Active think
        # for a quick lookup is short (~10 s).
        read_time_seconds=LogNormal.from_median(8, 0.5),
        active_think_seconds=LogNormal.from_median(10, 0.6),
        ttft_target_seconds=5.0,
        ttft_failure_seconds=15.0,
        tpot_target_ms=150.0,
        tpot_failure_ms=225.0,
    ),
    "conversational": Persona(
        id="conversational",
        description="Multi-turn back-and-forth with growing chat history",
        input_tokens=LogNormal.from_median(180, 0.5),
        output_tokens=LogNormal.from_median(220, 0.45),
        turns_per_session=Discrete({3: 0.25, 5: 0.35, 8: 0.25, 12: 0.15}),
        sessions_before_leaving=Discrete({2: 0.3, 4: 0.4, 8: 0.3}),
        inter_session_gap_seconds=LogNormal.from_median(600, 1.0),
        # 220-token response: residual ~28 s; conversational pace
        # composing the next turn ~25 s.
        read_time_seconds=LogNormal.from_median(28, 0.5),
        active_think_seconds=LogNormal.from_median(25, 0.6),
        ttft_target_seconds=9.0,
        ttft_failure_seconds=30.0,
        tpot_target_ms=150.0,
        tpot_failure_ms=225.0,
    ),
    "writer": Persona(
        id="writer",
        description="Drafting emails, content, and other prose; no long source documents",
        input_tokens=LogNormal.from_median(500, 0.5),
        output_tokens=LogNormal.from_median(350, 0.4),
        turns_per_session=Discrete({1: 0.5, 2: 0.3, 3: 0.15, 5: 0.05}),
        sessions_before_leaving=Discrete({3: 0.3, 6: 0.4, 12: 0.3}),
        inter_session_gap_seconds=LogNormal.from_median(900, 1.0),
        # 350-token draft: residual ~44 s; users review + edit ~70 s.
        read_time_seconds=LogNormal.from_median(44, 0.5),
        active_think_seconds=LogNormal.from_median(70, 0.7),
        ttft_target_seconds=15.0,
        ttft_failure_seconds=45.0,
        tpot_target_ms=180.0,
        tpot_failure_ms=270.0,
    ),
    "document_qa": Persona(
        id="document_qa",
        description="Question answering over 4K+ token source documents",
        input_tokens=LogNormal.from_median(3500, 0.5),
        output_tokens=LogNormal.from_median(280, 0.4),
        turns_per_session=Discrete({1: 0.4, 2: 0.3, 4: 0.2, 7: 0.1}),
        sessions_before_leaving=Discrete({2: 0.4, 4: 0.4, 8: 0.2}),
        inter_session_gap_seconds=LogNormal.from_median(1200, 1.0),
        # 280-token answer: residual ~35 s; analysts cross-reference
        # and verify before continuing — ~115 s active.
        read_time_seconds=LogNormal.from_median(35, 0.5),
        active_think_seconds=LogNormal.from_median(115, 0.6),
        ttft_target_seconds=30.0,
        ttft_failure_seconds=90.0,
        tpot_target_ms=200.0,
        tpot_failure_ms=300.0,
    ),
    "code_assist": Persona(
        id="code_assist",
        description="Pair-programming with code context in the prompt",
        input_tokens=LogNormal.from_median(1200, 0.6),
        output_tokens=LogNormal.from_median(450, 0.5),
        turns_per_session=Discrete({2: 0.3, 4: 0.35, 8: 0.25, 15: 0.1}),
        sessions_before_leaving=Discrete({4: 0.3, 8: 0.4, 16: 0.3}),
        inter_session_gap_seconds=LogNormal.from_median(420, 1.0),
        # 450-token code/explanation: residual ~56 s (and code reading
        # is genuinely slower than prose — some of that "active think"
        # is really careful re-reading). Active deliberation +
        # composing next prompt for sustained pair programming ~180 s.
        read_time_seconds=LogNormal.from_median(56, 0.5),
        active_think_seconds=LogNormal.from_median(180, 0.7),
        ttft_target_seconds=20.0,
        ttft_failure_seconds=60.0,
        tpot_target_ms=220.0,
        tpot_failure_ms=330.0,
    ),
    "summarizer": Persona(
        id="summarizer",
        description="Reducing long inputs to short summaries",
        input_tokens=LogNormal.from_median(5000, 0.4),
        output_tokens=LogNormal.from_median(180, 0.35),
        turns_per_session=Discrete({1: 0.85, 2: 0.15}),
        sessions_before_leaving=Discrete({2: 0.4, 4: 0.4, 8: 0.2}),
        inter_session_gap_seconds=LogNormal.from_median(900, 1.0),
        # 180-token summary: residual ~22 s; user reviews + uses
        # output to inform next action — ~45 s active.
        read_time_seconds=LogNormal.from_median(22, 0.5),
        active_think_seconds=LogNormal.from_median(45, 0.6),
        ttft_target_seconds=30.0,
        ttft_failure_seconds=120.0,
        tpot_target_ms=200.0,
        tpot_failure_ms=300.0,
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
        description="Frontline support: customer service reps, sales, inventory checks",
        persona_weights={
            "quick_lookup": 0.60,
            "conversational": 0.30,
            "writer": 0.10,
        },
    ),
    "general_knowledge": Cohort(
        id="general_knowledge",
        name="General knowledge work",
        description="Typical knowledge work: mixed chat, writing, light research",
        persona_weights={
            "quick_lookup": 0.30,
            "conversational": 0.30,
            "writer": 0.25,
            "summarizer": 0.10,
            "document_qa": 0.05,
        },
    ),
    "writer_dominant": Cohort(
        id="writer_dominant",
        name="Marketing / content team",
        description="Marketing, communications, content team",
        persona_weights={
            "quick_lookup": 0.10,
            "conversational": 0.20,
            "writer": 0.60,
            "summarizer": 0.10,
        },
    ),
    "software_engineering": Cohort(
        id="software_engineering",
        name="Software engineering team",
        description="Software engineering team using AI assistance for code work",
        persona_weights={
            "quick_lookup": 0.15,
            "conversational": 0.15,
            "writer": 0.05,
            "code_assist": 0.60,
            "document_qa": 0.05,
        },
    ),
    "analyst_team": Cohort(
        id="analyst_team",
        name="Analyst team",
        description="Legal, finance, research analysts working over long documents",
        persona_weights={
            "writer": 0.15,
            "summarizer": 0.25,
            "document_qa": 0.60,
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
