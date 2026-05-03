"""Prefix-cache hit analysis from captured turn events.

The multi-turn premise of every cohort assumes the engine reuses the
prefix it saw on turn 0 when turn 1 arrives with byte-identical history.
If that's not happening, conversational and code-assist personas will
look much heavier on TTFT than they should — and the capacity curves
will be pessimistic in ways that won't be obvious from the violation
rate alone.

This module is pure post-hoc analysis: it reads ``turn_events`` from a
finished cohort run's SQLite file, groups by ``(user_id, session_id)``,
and compares turn-0 TTFT against later-turn TTFT in the same session.
The signal of a working prefix cache: turn>=1 TTFT is meaningfully
lower than turn-0 TTFT for the same session, even though the input
length grew.

Output shape is pickled into the buyer-page JSON via
:func:`simulator.export.export_dir` and surfaced on the dashboard.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Tunables — these are heuristics for the validation verdict, not hard SLAs.
# A "hit" session is one whose later-turn TTFT is at most this fraction
# of its turn-0 TTFT. 0.5 is a deliberate floor: a 50% TTFT reduction is
# the smallest cache effect worth reporting; smaller deltas are within
# normal jitter on a busy host.
_DEFAULT_HIT_RATIO = 0.5


@dataclass
class TurnRow:
    user_id: str
    session_id: str
    turn_index: int
    ttft_ms: float
    persona_id: str
    input_tokens: int
    history_tokens: int


@dataclass
class SessionResult:
    user_id: str
    session_id: str
    persona_id: str
    turn_count: int
    turn0_ttft_ms: float
    later_ttft_ms_median: float
    ratio: float                    # later_median / turn0
    hit: bool                       # ratio <= _DEFAULT_HIT_RATIO


@dataclass
class PrefixCacheReport:
    sessions_analysed: int = 0
    sessions_with_hits: int = 0
    sessions_without_hits: int = 0
    overall_hit_rate: float = 0.0
    median_ttft_ratio: float | None = None
    per_persona: dict[str, dict] = field(default_factory=dict)
    per_cohort_step: dict[int, dict] = field(default_factory=dict)
    sample_sessions: list[dict] = field(default_factory=list)
    verdict: str = "insufficient_data"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sessions_analysed": self.sessions_analysed,
            "sessions_with_hits": self.sessions_with_hits,
            "sessions_without_hits": self.sessions_without_hits,
            "overall_hit_rate": round(self.overall_hit_rate, 3),
            "median_ttft_ratio": (
                round(self.median_ttft_ratio, 3)
                if self.median_ttft_ratio is not None else None
            ),
            "per_persona": self.per_persona,
            "per_cohort_step": self.per_cohort_step,
            "sample_sessions": self.sample_sessions,
            "verdict": self.verdict,
            "notes": self.notes,
        }


def _group_sessions(rows: Iterable[TurnRow]) -> dict[tuple[str, str], list[TurnRow]]:
    sessions: dict[tuple[str, str], list[TurnRow]] = {}
    for r in rows:
        sessions.setdefault((r.user_id, r.session_id), []).append(r)
    for k in sessions:
        sessions[k].sort(key=lambda r: r.turn_index)
    return sessions


def analyse_sessions(
    rows: list[TurnRow],
    *,
    hit_ratio: float = _DEFAULT_HIT_RATIO,
) -> tuple[list[SessionResult], list[str]]:
    """Reduce per-session turn lists to one verdict per session.

    Returns (session_results, notes). Notes carry skip reasons for
    sessions that didn't have enough turns to evaluate.
    """
    notes: list[str] = []
    out: list[SessionResult] = []
    grouped = _group_sessions(rows)
    skipped_short = 0
    for (uid, sid), turns in grouped.items():
        if len(turns) < 2:
            skipped_short += 1
            continue
        turn0 = turns[0]
        if turn0.turn_index != 0:
            # Session-start event missing (engine warm-up turn dropped, etc.).
            skipped_short += 1
            continue
        if turn0.ttft_ms <= 0:
            continue
        later = [t.ttft_ms for t in turns[1:] if t.ttft_ms > 0]
        if not later:
            continue
        median_later = statistics.median(later)
        ratio = median_later / turn0.ttft_ms
        out.append(SessionResult(
            user_id=uid, session_id=sid, persona_id=turn0.persona_id,
            turn_count=len(turns),
            turn0_ttft_ms=turn0.ttft_ms,
            later_ttft_ms_median=median_later,
            ratio=ratio,
            hit=ratio <= hit_ratio,
        ))
    if skipped_short:
        notes.append(f"skipped {skipped_short} single-turn or malformed sessions")
    return out, notes


def analyse_db(
    db_path: str | Path,
    *,
    hit_ratio: float = _DEFAULT_HIT_RATIO,
    sample_session_count: int = 5,
) -> PrefixCacheReport:
    """Walk a finished cohort run's SQLite and emit a PrefixCacheReport."""
    rows = _read_turn_rows(Path(db_path))
    return analyse_rows(
        rows, hit_ratio=hit_ratio,
        sample_session_count=sample_session_count,
    )


def analyse_rows(
    rows: list[TurnRow],
    *,
    hit_ratio: float = _DEFAULT_HIT_RATIO,
    sample_session_count: int = 5,
    rows_with_step: list[tuple[int, TurnRow]] | None = None,
) -> PrefixCacheReport:
    """Pure analysis — exposed for unit tests.

    ``rows_with_step``: optional parallel list of ``(step_index, row)``
    so the report can carry per-cohort-step breakdowns. When None, the
    per-step section is omitted.
    """
    sessions, notes = analyse_sessions(rows, hit_ratio=hit_ratio)
    report = PrefixCacheReport(notes=list(notes))
    if not sessions:
        report.verdict = "insufficient_data"
        report.notes.append("no multi-turn sessions found")
        return report

    report.sessions_analysed = len(sessions)
    report.sessions_with_hits = sum(1 for s in sessions if s.hit)
    report.sessions_without_hits = report.sessions_analysed - report.sessions_with_hits
    report.overall_hit_rate = report.sessions_with_hits / report.sessions_analysed
    report.median_ttft_ratio = statistics.median(s.ratio for s in sessions)

    # Per-persona breakdown
    by_persona: dict[str, list[SessionResult]] = {}
    for s in sessions:
        by_persona.setdefault(s.persona_id, []).append(s)
    for pid, group in by_persona.items():
        report.per_persona[pid] = {
            "sessions": len(group),
            "hit_rate": round(sum(1 for s in group if s.hit) / len(group), 3),
            "median_ratio": round(statistics.median(s.ratio for s in group), 3),
        }

    # Per cohort-step breakdown if we were given step labels
    if rows_with_step is not None:
        by_step_sessions: dict[int, dict[tuple[str, str], list[TurnRow]]] = {}
        for step, r in rows_with_step:
            by_step_sessions.setdefault(step, {}).setdefault(
                (r.user_id, r.session_id), []
            ).append(r)
        for step, sess_dict in by_step_sessions.items():
            step_rows = [r for ts in sess_dict.values() for r in ts]
            step_sessions, _ = analyse_sessions(step_rows, hit_ratio=hit_ratio)
            if not step_sessions:
                continue
            report.per_cohort_step[step] = {
                "sessions": len(step_sessions),
                "hit_rate": round(
                    sum(1 for s in step_sessions if s.hit) / len(step_sessions), 3
                ),
                "median_ratio": round(
                    statistics.median(s.ratio for s in step_sessions), 3
                ),
            }

    # A few illustrative sessions for the dashboard / debugging
    sample_sorted = sorted(sessions, key=lambda s: s.ratio)[:sample_session_count]
    report.sample_sessions = [
        {
            "persona_id": s.persona_id,
            "turn_count": s.turn_count,
            "turn0_ttft_ms": round(s.turn0_ttft_ms, 1),
            "later_ttft_ms_median": round(s.later_ttft_ms_median, 1),
            "ratio": round(s.ratio, 3),
            "hit": s.hit,
        }
        for s in sample_sorted
    ]

    # Verdict — load-bearing on the buyer page; be conservative.
    if report.sessions_analysed < 10:
        report.verdict = "insufficient_data"
    elif report.overall_hit_rate >= 0.7:
        report.verdict = "prefix_cache_effective"
    elif report.overall_hit_rate >= 0.3:
        report.verdict = "prefix_cache_partial"
    else:
        report.verdict = "prefix_cache_ineffective"

    return report


def _read_turn_rows(db_path: Path) -> list[TurnRow]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """SELECT user_id, session_id, turn_index, ttft_ms, persona_id,
                      input_tokens, history_tokens
               FROM turn_events"""
        )
        return [
            TurnRow(
                user_id=r["user_id"],
                session_id=r["session_id"],
                turn_index=r["turn_index"],
                ttft_ms=r["ttft_ms"],
                persona_id=r["persona_id"],
                input_tokens=r["input_tokens"],
                history_tokens=r["history_tokens"],
            )
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


def read_turn_rows_with_step(db_path: Path) -> list[tuple[int, TurnRow]]:
    """Return [(step_index, TurnRow), ...] joining turn_events ↔ measurements."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """SELECT m.step_index AS step_index,
                      e.user_id, e.session_id, e.turn_index, e.ttft_ms,
                      e.persona_id, e.input_tokens, e.history_tokens
               FROM turn_events e
               JOIN cohort_measurements m USING (measurement_id)"""
        )
        return [
            (
                int(r["step_index"]),
                TurnRow(
                    user_id=r["user_id"],
                    session_id=r["session_id"],
                    turn_index=r["turn_index"],
                    ttft_ms=r["ttft_ms"],
                    persona_id=r["persona_id"],
                    input_tokens=r["input_tokens"],
                    history_tokens=r["history_tokens"],
                ),
            )
            for r in cur.fetchall()
        ]
    finally:
        conn.close()
