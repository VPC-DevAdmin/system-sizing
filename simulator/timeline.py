"""Per-measurement phase-distribution timeline.

For each measurement window (one cohort × one pool size), reconstruct
second-by-second how many users in the pool were in each lifecycle
phase. Useful for diagnosing rate oscillation / query storms in
closed-loop simulation: if the system is producing storms, ``prefill``
will spike periodically as users finish their read/think interval in
lockstep and submit simultaneously.

Phases:

    idle      — virtual user has been spawned but hasn't fired its
                first request yet. Initial phase-offset (per-user
                think-time fraction sleep at spawn) lives here.
    prefill   — request submitted, awaiting first token. Span =
                ``[submitted_at_ms, submitted_at_ms + ttft_ms)``.
    decode    — first token received, streaming continuing. Span =
                ``[submitted_at_ms + ttft_ms, completed_at_ms)``.
    think     — user is reading the response / thinking up the next
                turn. Span = ``[prev_completed_at_ms, this_submitted_at_ms)``
                for consecutive turns within a session.

At any given timestamp, ``prefill + decode + think + idle`` equals the
number of users alive in the pool — modulo measurement-window edge
effects (a user spawned outside the window contributes a partial
phase trail). The sum should hover near the target pool size after
warmup.

The sweep-line algorithm builds a flat list of phase-boundary events
and replays them in chronological order, snapshotting counters at
each sample step. O(events + samples) — fast at simulator scale even
for the longest measurement windows we run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Phase counter keys — the schema fields downstream consumers expect.
PHASES: tuple[str, ...] = ("prefill", "decode", "think", "idle")


@dataclass
class TimelinePoint:
    """One sample. ``t_offset_s`` is seconds relative to the
    measurement window's earliest event."""

    t_offset_s: int
    prefill: int
    decode: int
    think: int
    idle: int

    @property
    def total(self) -> int:
        return self.prefill + self.decode + self.think + self.idle

    def as_row(self) -> list[int]:
        """Ordered ``[t_offset_s, prefill, decode, think, idle]`` —
        matches the compact array-of-arrays JSON shape."""
        return [self.t_offset_s, self.prefill, self.decode, self.think, self.idle]


def compute_timeline(
    conn: sqlite3.Connection,
    measurement_id: int,
    *,
    resolution_ms: int = 1000,
) -> list[TimelinePoint]:
    """Build phase counts at ``resolution_ms`` cadence for one
    measurement window.

    Returns an empty list when no turn_events were recorded (e.g. a
    measurement that timed out before any user completed). The caller
    decides whether that surfaces in the export.
    """
    if resolution_ms <= 0:
        raise ValueError(f"resolution_ms must be positive (got {resolution_ms})")

    # Pull every turn for this measurement, ordered for per-user
    # iteration so we can derive inter-turn ``think`` gaps in one pass.
    turn_rows = conn.execute(
        """SELECT user_id, session_id, turn_index,
                  submitted_at_ms, ttft_ms, completed_at_ms
           FROM turn_events
           WHERE measurement_id = ?
           ORDER BY user_id, turn_index""",
        (measurement_id,),
    ).fetchall()
    if not turn_rows:
        return []

    # Window bounds — earliest submit and latest completion across all
    # users in this measurement. Not the cohort_measurements window
    # itself: that bounds the measurement_started_at + duration, but
    # turn data inside that window is what we actually have phase
    # information for. Going wider would just pad zeros.
    t_start = min(r[3] for r in turn_rows)
    t_end = max(r[5] for r in turn_rows)

    # Group turns by user. We need per-user iteration to derive the
    # ``think`` interval between consecutive turns. ``idle`` (spawn →
    # first submit) is sourced from ``virtual_users``.
    user_turns: dict[str, list[tuple]] = {}
    for row in turn_rows:
        user_turns.setdefault(row[0], []).append(row)

    # Spawn timestamps for the idle window. Some users in turn_events
    # may not have a virtual_users row if the run was forklifted from
    # a pre-table-existence DB; skip the idle phase for those users.
    spawn_times: dict[str, int] = {}
    if user_turns:
        placeholders = ",".join("?" * len(user_turns))
        spawn_rows = conn.execute(
            f"SELECT user_id, spawned_at_ms FROM virtual_users "
            f"WHERE user_id IN ({placeholders})",
            list(user_turns.keys()),
        ).fetchall()
        spawn_times = {r[0]: r[1] for r in spawn_rows}

    # Sweep-line event stream: each phase interval emits a +1 at start
    # and -1 at end. We rebuild counters by walking events in order.
    # Tuples are (timestamp_ms, phase_index, delta) — phase_index maps
    # into a parallel counter array keyed by PHASES.
    events: list[tuple[int, int, int]] = []
    phase_index = {name: i for i, name in enumerate(PHASES)}

    def add(start: int, end: int, phase: str) -> None:
        if end <= start:
            return  # zero-or-negative duration: skip rather than emit
                    # a malformed boundary pair
        idx = phase_index[phase]
        events.append((start, idx, +1))
        events.append((end, idx, -1))

    for user_id, turns in user_turns.items():
        # Idle: spawn → first submit. Clamped to window start so a
        # user spawned long before the measurement window doesn't
        # contribute idle time outside the window.
        first_submit = turns[0][3]
        spawned = spawn_times.get(user_id)
        if spawned is not None and spawned < first_submit:
            add(max(spawned, t_start), first_submit, "idle")

        prev_complete: int | None = None
        for (_, _sid, _tidx, submit, ttft, complete) in turns:
            # Think: previous completion → this submit. Captures both
            # in-session read+active-think gaps. (Cross-session gaps
            # don't exist under per-session-respawn — users terminate
            # after one session.)
            if prev_complete is not None:
                add(prev_complete, submit, "think")
            # Prefill: submit → submit + ttft. ttft is REAL ms.
            ttft_int = int(ttft) if ttft else 0
            add(submit, submit + ttft_int, "prefill")
            # Decode: submit + ttft → complete.
            add(submit + ttft_int, complete, "decode")
            prev_complete = complete

    # Sort events. Ties: process -1 (close) BEFORE +1 (open) so that
    # back-to-back phases (e.g. decode ending exactly when think
    # begins) don't double-count the boundary instant.
    events.sort(key=lambda e: (e[0], e[2]))

    # Walk events; emit a counter snapshot at each resolution_ms tick.
    timeline: list[TimelinePoint] = []
    counters = [0] * len(PHASES)
    ev_idx = 0
    t = t_start
    while t <= t_end:
        # Advance events up to (and including) the current tick.
        while ev_idx < len(events) and events[ev_idx][0] <= t:
            _, p_idx, delta = events[ev_idx]
            counters[p_idx] += delta
            ev_idx += 1
        timeline.append(
            TimelinePoint(
                t_offset_s=(t - t_start) // 1000,
                prefill=counters[phase_index["prefill"]],
                decode=counters[phase_index["decode"]],
                think=counters[phase_index["think"]],
                idle=counters[phase_index["idle"]],
            )
        )
        t += resolution_ms

    return timeline


def timeline_to_export_dict(
    timeline: list[TimelinePoint],
    *,
    resolution_ms: int = 1000,
) -> dict:
    """Pack a timeline list into the compact JSON shape used by the
    full export.

    Schema:
        {
          "resolution_ms": 1000,
          "schema": ["t_offset_s", "prefill", "decode", "think", "idle"],
          "rows": [[t, p, d, th, i], ...]
        }

    Array-of-arrays is ~3× smaller than array-of-objects on this
    column count, and pandas / spreadsheet importers handle both.
    """
    return {
        "resolution_ms": resolution_ms,
        "schema": ["t_offset_s", *PHASES],
        "rows": [pt.as_row() for pt in timeline],
    }
