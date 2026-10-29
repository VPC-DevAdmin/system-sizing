"""Service levels applied after the run.

Planner campaign runs search for the queue knee with the SLA gate off
(``simulation.open_loop_sla_gate: false``): no latency target stops or
fails a window. This module reads the stored turns of every window and
asks, per service tier, which arrival rates met it:

  * a turn meets the first-token limit when it did not error and its
    first answer token (``ttfct_ms``, which equals ``ttft_ms`` for
    non-reasoning models) arrived within the tier's base limit plus an
    allowance per 4K prompt tokens (prompt = new input + sent history)
    above 2K;
  * a turn meets the speed limit when it did not error and its time
    per output token is within the tier's limit;
  * a window passes when it is stable and at least ``ttft_share`` of
    its turns meet the first-token limit and ``speed_share`` meet the
    speed limit.

A tier's capacity is the arrival rate where the pass margin crosses
zero, interpolated in log rate between the last passing window and the
first failing one above it. A tier that still passes at the knee gets
the knee; one that fails at the lowest measured rate is unsuitable
(``None``). The Batch tier has no latency limits: its capacity is the
knee. Everything carried at a capacity point (sessions, turns, tokens,
power) is interpolated the same way.

    python -m simulator.rescore runs_planner/g0/run_03 [--tiers tiers.yaml] [--out x.json]
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIERS = {
    "responsive": {"ttft_s": 2.0, "ttft_per_4k_s": 1.0, "tpot_ms": 1000 / 35},
    "standard": {"ttft_s": 5.0, "ttft_per_4k_s": 2.0, "tpot_ms": 1000 / 15},
    "relaxed": {"ttft_s": 10.0, "ttft_per_4k_s": 4.0, "tpot_ms": 1000 / 7},
    "batch": {"batch": True},
}
# Persona-specific limits that replace a tier's (voice is held to a
# spoken-reply budget and only scored at Responsive).
DEFAULT_OVERRIDES = {
    "voice_assistant": {
        "responsive": {"ttft_s": 0.8, "ttft_per_4k_s": 0.0, "tpot_ms": 1000 / 35},
        "standard": None, "relaxed": None, "batch": None,
    },
}
TTFT_SHARE = 0.95
SPEED_SHARE = 0.90
PROMPT_FREE_TOKENS = 2000
# Window values carried to each capacity point.
CARRIED = ("rate_per_min", "sessions", "in_flight", "turns_per_s",
           "prompt_tok_per_s", "output_tok_per_s", "power_w", "gpu_power_w",
           "wall_power_w", "kv_cache_pct", "error_rate", "prefix_hit_rate")


def load_wall_power(path: str | None) -> list[tuple[float, float]]:
    """``epoch,watts[,...]`` rows from scripts/idrac_power.py or
    scripts/ipmi_power.sh, sorted by time."""
    if not path or not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split(",")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    rows.sort()
    return rows


def _wall_mean(rows: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    vals = [w for t, w in rows if t0 <= t <= t1]
    return sum(vals) / len(vals) if vals else None


@dataclass
class Window:
    rate_per_min: float
    stable: bool
    values: dict
    turns: list = field(default_factory=list)


def _load_windows(conn: sqlite3.Connection, cohort_run_id: str,
                  wall: list[tuple[float, float]] | None = None) -> list[Window]:
    import datetime as _dt
    rows = conn.execute(
        """SELECT measurement_id, arrival_rate_per_min, stability, stability_detail,
                  measurement_started_at,
                  measured_avg_pool_size, measured_avg_in_flight,
                  measurement_duration_s, power_w_avg, gpu_power_w_avg,
                  avg_kv_cache_pct
             FROM cohort_measurements
            WHERE cohort_run_id = ? AND arrival_rate_per_min IS NOT NULL
              AND stability IN ('stable', 'divergent')""",
        (cohort_run_id,),
    ).fetchall()
    out = []
    for mid, rate, stab, detail, started, sessions, inflight, dur, pw, gpw, kv in rows:
        try:
            phit = json.loads(detail or "{}").get("prefix_hit_rate")
        except ValueError:
            phit = None
        turns = conn.execute(
            """SELECT ttft_ms, ttfct_ms, tpot_ms, input_tokens, history_tokens,
                      output_tokens, error
                 FROM turn_events WHERE measurement_id = ?""", (mid,),
        ).fetchall()
        dur = max(1, int(dur or 0))
        n = len(turns)
        wall_w = None
        if wall and started:
            try:
                t0 = _dt.datetime.fromisoformat(started).timestamp()
                wall_w = _wall_mean(wall, t0, t0 + dur)
            except ValueError:
                pass
        values = {
            "rate_per_min": rate,
            "sessions": sessions,
            "in_flight": inflight,
            "turns_per_s": n / dur,
            "prompt_tok_per_s": sum((t[3] or 0) + (t[4] or 0) for t in turns) / dur,
            "output_tok_per_s": sum(t[5] or 0 for t in turns if not t[6]) / dur,
            "power_w": pw,
            "gpu_power_w": gpw,
            "wall_power_w": wall_w,
            "prefix_hit_rate": phit,
            "kv_cache_pct": kv,
            "error_rate": (sum(1 for t in turns if t[6]) / n) if n else 0.0,
            "measurement_id": mid,
        }
        out.append(Window(rate, stab == "stable", values, turns))
    out.sort(key=lambda w: w.rate_per_min)
    return out


def _margin(w: Window, limits: dict) -> float:
    """min(ttft share − 0.95, speed share − 0.90); −1 for a divergent
    or empty window."""
    if not w.stable or not w.turns:
        return -1.0
    n = len(w.turns)
    ttft_ok = speed_ok = 0
    for ttft, ttfct, tpot, inp, hist, out, err in w.turns:
        if err:
            continue
        first = ttfct if ttfct else ttft
        prompt = (inp or 0) + (hist or 0)
        limit_s = limits["ttft_s"] + limits.get("ttft_per_4k_s", 0.0) * max(
            0.0, (prompt - PROMPT_FREE_TOKENS) / 4000.0)
        if first is not None and first <= limit_s * 1000.0:
            ttft_ok += 1
        if (out or 0) <= 1 or (tpot is not None and 0 < tpot <= limits["tpot_ms"]):
            speed_ok += 1
    return min(ttft_ok / n - TTFT_SHARE, speed_ok / n - SPEED_SHARE)


def _interp(a: Window, b: Window, frac: float) -> dict:
    out = {}
    for k in CARRIED:
        va, vb = a.values.get(k), b.values.get(k)
        if va is None or vb is None:
            out[k] = va if vb is None else vb
            continue
        out[k] = va + (vb - va) * frac
    return out


def _point(w: Window) -> dict:
    return {k: w.values.get(k) for k in CARRIED}


def tier_capacity(windows: list[Window], limits: dict | None) -> dict | None:
    stable = [w for w in windows if w.stable and w.values["error_rate"] < 0.05]
    if not stable:
        return None
    knee = max(stable, key=lambda w: w.rate_per_min)
    if limits is None or limits.get("batch"):
        return {"basis": "knee", **_point(knee)}
    margins = [(w, _margin(w, limits)) for w in windows]
    prev = None
    for w, m in margins:
        if m >= 0:
            prev = (w, m)
            if w is knee:
                return {"basis": "passes_at_knee", **_point(knee)}
            continue
        if prev is None:
            return {"basis": "fails_at_lowest_rate", "rate_per_min": None}
        a, ma = prev
        la, lb = math.log(a.rate_per_min), math.log(w.rate_per_min)
        frac = ma / (ma - m) if ma - m > 0 else 0.0
        frac = min(1.0, max(0.0, frac))
        rate = math.exp(la + (lb - la) * frac)
        if rate >= knee.rate_per_min:
            return {"basis": "passes_at_knee", **_point(knee)}
        point = _interp(a, w, frac)
        point["rate_per_min"] = rate
        return {"basis": "interpolated",
                "bracket_per_min": [a.rate_per_min, w.rate_per_min], **point}
    if prev is not None:
        return {"basis": "passes_at_knee", **_point(knee)}
    return None


def rescore_run(run_dir: Path, tiers: dict | None = None,
                overrides: dict | None = None, wall_power: str | None = None) -> dict:
    tiers = tiers or DEFAULT_TIERS
    overrides = DEFAULT_OVERRIDES if overrides is None else overrides
    wall = load_wall_power(wall_power or os.environ.get("CAPSIM_WALL_POWER_CSV"))
    conn = sqlite3.connect(str(Path(run_dir) / "run.db"))
    try:
        runs = conn.execute(
            """SELECT cohort_run_id, cohort_id, model_id, engine_type,
                      final_status, config_json, started_at
                 FROM cohort_run WHERE mode = 'open_loop'
                ORDER BY started_at""").fetchall()
        result = {"run_dir": str(run_dir), "tiers": tiers, "workloads": {}}
        for crid, cohort, model, engine, status, cfg_json, started in runs:
            windows = _load_windows(conn, crid, wall)
            stable = [w for w in windows if w.stable]
            divergent = [w for w in windows if not w.stable]
            knee = max((w.rate_per_min for w in stable), default=None)
            above = [w.rate_per_min for w in divergent
                     if knee is not None and w.rate_per_min > knee]
            entry = {
                "model_id": model, "engine": engine, "final_status": status,
                "started_at": started, "windows": len(windows),
                "knee_per_min": knee,
                "ceiling_per_min": min(above) if above else None,
                "knee_point": _point(max(stable, key=lambda w: w.rate_per_min))
                if stable else None,
                "tiers": {},
            }
            for tier, limits in tiers.items():
                lim = limits
                o = (overrides.get(cohort) or {})
                if tier in o:
                    lim = o[tier]
                    if lim is None:
                        entry["tiers"][tier] = {"basis": "not_scored"}
                        continue
                entry["tiers"][tier] = tier_capacity(windows, lim)
            try:
                cfg = json.loads(cfg_json or "{}")
                sim = cfg.get("simulation") or {}
                entry["history_window_turns"] = sim.get("history_window_turns")
                entry["context_limit_tokens"] = sim.get("context_limit_tokens")
            except ValueError:
                pass
            # A workload re-run in the same run dir: keep the latest.
            result["workloads"][cohort] = entry
        return result
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="simulator.rescore")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--tiers", help="YAML {tier: {ttft_s, ttft_per_4k_s, tpot_ms} | {batch: true}}")
    ap.add_argument("--out", help="write JSON here (one run dir) or to <run_dir>/tiers.json")
    ap.add_argument("--wall-power", help="epoch,watts CSV (default $CAPSIM_WALL_POWER_CSV)")
    a = ap.parse_args(argv)
    tiers = None
    if a.tiers:
        import yaml
        tiers = yaml.safe_load(Path(a.tiers).read_text())
    for rd in a.run_dirs:
        res = rescore_run(Path(rd), tiers, wall_power=a.wall_power)
        out = Path(a.out) if a.out and len(a.run_dirs) == 1 else Path(rd) / "tiers.json"
        out.write_text(json.dumps(res, indent=1, default=str))
        for wl, e in res["workloads"].items():
            caps = {t: (round(v["rate_per_min"], 1) if v and v.get("rate_per_min") else None)
                    for t, v in e["tiers"].items()}
            print(f"{rd} {wl}: knee {e['knee_per_min']}/min  {caps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
