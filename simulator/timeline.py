"""Per-measurement phase-distribution timeline.

For each measurement window (one cohort × one pool size), reconstruct
second-by-second how many users in the pool were in each lifecycle
phase. Useful for diagnosing rate oscillation / query storms in
closed-loop simulation: if the system is producing storms, ``prefill``
will spike periodically as users finish their read/think interval in
lockstep and submit simultaneously.

Phases:

    prefill   — request submitted, awaiting first token. Span =
                ``[submitted_at_ms, submitted_at_ms + ttft_ms)``.
    decode    — first token received, streaming continuing. Span =
                ``[submitted_at_ms + ttft_ms, completed_at_ms)``.
    think     — user is reading the response / thinking up the next
                turn. Span = ``[prev_completed_at_ms, this_submitted_at_ms)``
                for consecutive turns within a session. Also covers the
                pre-first-turn window: a freshly-spawned user sleeps for
                a random fraction of one (read_time + active_think)
                sample as its "initial phase offset" before submitting
                turn 0 — physically the same state as a post-turn
                think gap (waiting to fire next request), so it folds
                into ``think`` rather than getting its own bucket.

At any given timestamp, ``prefill + decode + think`` equals the
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
# Pre-first-turn "idle" (initial phase offset) folds into ``think``
# since it's physically the same state — see module docstring.
PHASES: tuple[str, ...] = ("prefill", "decode", "think")


@dataclass
class TimelinePoint:
    """One sample. ``t_offset_s`` is seconds relative to the
    measurement window's earliest event."""

    t_offset_s: int
    prefill: int
    decode: int
    think: int

    @property
    def total(self) -> int:
        return self.prefill + self.decode + self.think

    def as_row(self) -> list[int]:
        """Ordered ``[t_offset_s, prefill, decode, think]`` —
        matches the compact array-of-arrays JSON shape."""
        return [self.t_offset_s, self.prefill, self.decode, self.think]


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
    # ``think`` interval between consecutive turns.
    user_turns: dict[str, list[tuple]] = {}
    for row in turn_rows:
        user_turns.setdefault(row[0], []).append(row)

    # Look up the cohort_run_id for this measurement so we can
    # enumerate ALL users alive during the window (not just users that
    # completed turns in it). Under per-session-respawn at high
    # concurrency on slow-cohorts, many users alive in the pool never
    # complete a turn within the measurement window — they were either
    # spawned late or are mid-turn at window-end. Without enumerating
    # them, the timeline drains toward window-end (chart shows a
    # vanishing pool when reality is steady-state at target size).
    cohort_run_id_row = conn.execute(
        "SELECT cohort_run_id FROM cohort_measurements WHERE measurement_id = ?",
        (measurement_id,),
    ).fetchone()

    # ``user_lifetimes`` maps user_id -> (spawned_at_ms, terminated_at_ms
    # or None). Covers both users with turns AND users alive but turn-
    # less in this window. Missing rows (e.g. forklifted DBs without a
    # virtual_users table) degrade gracefully — those users get the
    # legacy "pre-first-turn from min(submitted) is the only think we
    # can infer" treatment.
    user_lifetimes: dict[str, tuple[int, int | None]] = {}
    if cohort_run_id_row is not None:
        cohort_run_id = cohort_run_id_row[0]
        # Alive-during-window predicate:
        #   spawn <= t_end                         (born before window ends)
        # AND (terminated IS NULL OR terminated >= t_start)
        #                                          (still alive or
        #                                           died after window
        #                                           opened)
        for r in conn.execute(
            """SELECT user_id, spawned_at_ms, terminated_at_ms
               FROM virtual_users
               WHERE cohort_run_id = ?
                 AND spawned_at_ms <= ?
                 AND (terminated_at_ms IS NULL OR terminated_at_ms >= ?)""",
            (cohort_run_id, t_end, t_start),
        ):
            user_lifetimes[r[0]] = (r[1], r[2])

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

    def alive_window_end(user_id: str) -> int:
        """The upper bound for a user's presence in the chart window.
        Capped at the chart's t_end so a user that survives the window
        doesn't contribute phase samples past it; capped at their
        terminate timestamp so a user that died mid-window doesn't
        contribute think past their death."""
        lifetime = user_lifetimes.get(user_id)
        if lifetime is None or lifetime[1] is None:
            return t_end
        return min(lifetime[1], t_end)

    for user_id, turns in user_turns.items():
        # Pre-first-turn read+think: spawn → first submit. Clamped
        # to window start so a user spawned long before the window
        # doesn't contribute time outside the window. Goes into
        # ``think`` rather than a separate ``idle`` bucket because
        # physically it IS the same state — the initial phase offset
        # is a random fraction of one (read_time + active_think)
        # sample, so the user is "sleeping until time to fire next
        # request," indistinguishable from a between-turn gap.
        first_submit = turns[0][3]
        lifetime = user_lifetimes.get(user_id)
        spawned = lifetime[0] if lifetime else None
        if spawned is not None and spawned < first_submit:
            add(max(spawned, t_start), first_submit, "think")

        prev_complete: int | None = None
        for (_, _sid, _tidx, submit, ttft, complete) in turns:
            # Think: previous completion → this submit. Captures the
            # in-session read+active-think gap. (Cross-session gaps
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

        # Trailing think: from the user's last turn completion to
        # either their termination or the window end, whichever is
        # earlier. Under per-session-respawn ``terminate ≈
        # last-turn-complete`` so this is usually a no-op for users
        # whose session ended within the window — but for users still
        # alive at t_end (because their next session was scheduled to
        # fire after window-close), this captures the post-last-turn
        # think that would otherwise drop off the chart.
        if prev_complete is not None:
            end_t = alive_window_end(user_id)
            if end_t > prev_complete:
                add(prev_complete, end_t, "think")

    # Users alive during the window but with NO turns in turn_events.
    # Most common cause: a replacement user spawned mid-window whose
    # first turn hasn't completed by window-end (especially on slow
    # cohorts where one turn takes 20-60s). These users are in
    # initial-phase-offset / pre-first-turn think for their entire
    # alive-in-window interval. Without this branch the chart's pool
    # count drains toward window-end as old users terminate and new
    # arrivals stay invisible.
    for user_id, (spawned, terminated) in user_lifetimes.items():
        if user_id in user_turns:
            continue  # already handled above
        alive_start = max(spawned, t_start)
        alive_end = min(terminated if terminated is not None else t_end, t_end)
        if alive_end > alive_start:
            add(alive_start, alive_end, "think")

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
          "schema": ["t_offset_s", "prefill", "decode", "think"],
          "rows": [[t, p, d, th], ...]
        }

    Array-of-arrays is ~3× smaller than array-of-objects on this
    column count, and pandas / spreadsheet importers handle both.
    """
    return {
        "resolution_ms": resolution_ms,
        "schema": ["t_offset_s", *PHASES],
        "rows": [pt.as_row() for pt in timeline],
    }
