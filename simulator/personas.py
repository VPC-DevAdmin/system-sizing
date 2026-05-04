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


@dataclass
class Cohort:
    id: str
    name: str
    description: str
    persona_weights: dict  # {persona_id: weight}, must sum to 1.0
    # ``"single"`` for cohorts that exercise one persona at 100%,
    # ``"mix"`` for blended workloads. Lets the CLI sweep filter by
    # group ("singles" / "mixes" / "all") and the buyer page group
    # cards visually.
    category: str = "mix"

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
    # ── Single-persona cohorts ────────────────────────────────────────
    # Each runs one persona at 100% weight. Used to characterise the
    # individual capacity of each user archetype on a given engine, so
    # multi-persona results can be decomposed: if engineering_heavy
    # underperforms its components, you know the cohort mix is itself
    # producing interference rather than any single persona dragging
    # the rest down.
    "quick_lookup_only": Cohort(
        id="quick_lookup_only",
        name="Quick lookup only",
        description="100% frontline-support workload — short queries, fast SLAs",
        persona_weights={"quick_lookup": 1.0},
        category="single",
    ),
    "conversational_only": Cohort(
        id="conversational_only",
        name="Conversational only",
        description="100% back-and-forth chat — multi-turn dominates",
        persona_weights={"conversational": 1.0},
        category="single",
    ),
    "drafter_only": Cohort(
        id="drafter_only",
        name="Drafter only",
        description="100% short-form writing — moderate input, longer output",
        persona_weights={"drafter": 1.0},
        category="single",
    ),
    "document_qa_only": Cohort(
        id="document_qa_only",
        name="Document Q&A only",
        description="100% long-context document Q&A — heavy prefill, moderate decode",
        persona_weights={"document_qa": 1.0},
        category="single",
    ),
    "code_assist_only": Cohort(
        id="code_assist_only",
        name="Code-assist only",
        description="100% engineer pair-programming — long multi-turn sessions",
        persona_weights={"code_assist": 1.0},
        category="single",
    ),
    "summarizer_only": Cohort(
        id="summarizer_only",
        name="Summarizer only",
        description="100% single-shot summarisation — heaviest prefill, light decode",
        persona_weights={"summarizer": 1.0},
        category="single",
    ),

    # ── Multi-persona cohorts ─────────────────────────────────────────
    # Blended workloads representing realistic business types. The
    # weight numbers came from the original spec; tune empirically
    # against deployment telemetry if you have it.
    "chat_heavy": Cohort(
        id="chat_heavy",
        name="Customer support team",
        description="Frontline support: short queries, fast SLAs, high turnover",
        persona_weights={
            "quick_lookup": 0.6,
            "conversational": 0.3,
            "drafter": 0.1,
        },
        category="mix",
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
        category="mix",
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
        category="mix",
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
        category="mix",
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
        category="mix",
    ),
}


def resolve_cohort_group(arg: str) -> list[str]:
    """Map a CLI argument to a list of cohort ids.

    Accepts:
      * ``"all"``      → every cohort
      * ``"singles"``  → cohorts with ``category == "single"``
      * ``"mixes"``    → cohorts with ``category == "mix"``
      * any comma-separated list of cohort ids (``"chat_heavy,quick_lookup_only"``)
    """
    arg = (arg or "all").strip()
    if arg == "all":
        return list(COHORTS.keys())
    if arg == "singles":
        return [cid for cid, c in COHORTS.items() if c.category == "single"]
    if arg == "mixes":
        return [cid for cid, c in COHORTS.items() if c.category == "mix"]
    return [c.strip() for c in arg.split(",") if c.strip()]


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
