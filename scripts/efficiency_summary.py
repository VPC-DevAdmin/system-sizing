#!/usr/bin/env python3
"""Summarize the XE7740 efficiency sweep against the Dell/Metrum baseline.

    efficiency_summary.py RESULTS_DIR [--scores docs/efficiency-intelligence-scores.json]
                          [--target 35] [--baseline 7.03] [--out docs/efficiency-summary.md]
                          [--chart docs/efficiency-chart-{wl}.svg]

RESULTS_DIR holds <workload>/<config>/result.json from efficiency_sweep.py (rungs carry
gen_tok_s, wall_w from the iDRAC PSU sum, intvty_p90_tok_s). For every workload the
script emits one table with two readings per model: the peak tokens per wall watt over all
rungs (how the Dell post reads), and the best rung whose p90 per-user speed meets the target.
Both are divided by the baseline figure (Dell / Metrum AI, 3 Nov 2025: Llama 3.3 70B FP4 on
TensorRT-LLM, 8x RTX PRO 6000, 7.03 tokens/s per system watt).

The chart is plain SVG (no plotting dependency): Artificial Analysis Intelligence Index on
the y axis, tokens/s per wall kW (log) on the x axis, one point per model per reading.
"""

import argparse
import json
import math
from pathlib import Path

BASELINE_NOTE = ("Dell / Metrum AI, Llama 3.3 70B FP4, TensorRT-LLM, 8x RTX PRO 6000, "
                 "45,527 tok/s at 7.03 tok/s per system W")
LABELS = {
    "llama-3.3-70b-nvfp4": "Llama 3.3 70B NVFP4 (vLLM)",
    "llama-3.3-70b-nvfp4-trtllm": "Llama 3.3 70B NVFP4 (TensorRT-LLM)",
    "gemma-4-31b-nvfp4": "Gemma 4 31B dense NVFP4",
    "qwen3-32b-fp8": "Qwen3 32B dense FP8",
    "mistral-small-3.2-24b": "Mistral Small 3.2 24B",
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gemma-4-26b-a4b-nvfp4": "Gemma 4 26B-A4B NVFP4",
    "qwen3-30b-a3b-fp8": "Qwen3 30B-A3B FP8",
    "qwen3.6-35b-a3b-nvfp4": "Qwen3.6 35B-A3B NVFP4",
    "glm-4.7-flash": "GLM-4.7 Flash",
    "nemotron-3-super-120b-nvfp4": "Nemotron 3 Super 120B NVFP4",
    "qwen3-235b-a22b-fp8": "Qwen3 235B-A22B FP8",
    "minimax-m2.7-fp8": "MiniMax M2.7",
    "qwen3.5-397b-a17b-nvfp4-sglang": "Qwen3.5 397B NVFP4 (SGLang)",
    "qwen3.5-397b-a17b-nvfp4-sglang-mtp": "Qwen3.5 397B NVFP4 (SGLang, MTP)",
    "beam-1.0-501b-nvfp4": "Beam 1.0 501B NVFP4",
    "nemotron-3-ultra-550b-nvfp4": "Nemotron 3 Ultra 550B NVFP4",
}
WL_TITLE = {"1k1k": "1k in / 1k out", "8k1k": "8k in / 1k out", "128x128": "128 in / 128 out (roofline)"}


def load(results_dir):
    out = {}
    for f in sorted(results_dir.glob("*/*/result.json")):
        r = json.loads(f.read_text())
        wl, cfg = f.parent.parent.name, f.parent.name
        if cfg.startswith("_"):
            continue
        rungs = [x for x in r.get("rungs", []) if isinstance(x, dict) and not x.get("error")
                 and x.get("gen_tok_s") and x.get("wall_w")]
        if rungs:
            out[(wl, cfg)] = rungs
    return out


def tps_w(r):
    return r["gen_tok_s"] / r["wall_w"]


def readings(rungs, target):
    peak = max(rungs, key=tps_w)
    ok = [r for r in rungs if (r.get("intvty_p90_tok_s") or 0) >= target]
    at = max(ok, key=lambda r: r["gen_tok_s"]) if ok else None
    return peak, at


def fmt(v, nd=0):
    return "" if v is None else f"{v:,.{nd}f}"


def table(wl, rows, scores, target, baseline):
    lines = [f"### {WL_TITLE.get(wl, wl)}", "",
             f"| Model | Index | Peak tok/s | Users | Wall W | Tok/s per W | vs {baseline:g} | "
             f"At {target:g} tok/s/user: tok/s | Users | Tok/s per W | vs {baseline:g} |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, peak, at in sorted(rows, key=lambda x: -tps_w(x[1])):
        s = scores.get(cfg, {})
        idx = ("~" if s.get("estimate") else "") + str(s.get("index", ""))
        line = (f"| {LABELS.get(cfg, cfg)} | {idx} | {fmt(peak['gen_tok_s'])} | {peak['concurrency']} | "
                f"{fmt(peak['wall_w'])} | {tps_w(peak):.2f} | {tps_w(peak) / baseline:.2f}x | ")
        if at:
            line += f"{fmt(at['gen_tok_s'])} | {at['concurrency']} | {tps_w(at):.2f} | {tps_w(at) / baseline:.2f}x |"
        else:
            line += "misses target | | | |"
        lines.append(line)
    return lines


def svg_chart(wl, rows, scores, target, baseline, path):
    """Index (y) against tokens/s per wall kW (x, log10). Filled = peak, hollow = at target."""
    W, H, L, R, T, B = 900, 560, 70, 30, 40, 70
    pts = []
    for cfg, peak, at in rows:
        idx = scores.get(cfg, {}).get("index")
        if idx is None:
            continue
        pts.append((cfg, tps_w(peak) * 1000, idx, True))
        if at:
            pts.append((cfg, tps_w(at) * 1000, idx, False))
    if not pts:
        return
    xs = [p[1] for p in pts] + [baseline * 1000]
    x0, x1 = math.floor(math.log10(min(xs)) * 2) / 2, math.ceil(math.log10(max(xs)) * 2) / 2
    y1 = max(p[2] for p in pts) + 3

    def X(v):
        return L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)

    def Y(v):
        return T + (1 - v / y1) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold">XE7740, 8x RTX PRO 6000: intelligence against energy efficiency, {WL_TITLE.get(wl, wl)}</text>']
    d = x0
    while d <= x1 + 1e-9:
        for m in (1, 2, 5):
            v = m * 10 ** d
            if 10 ** x0 <= v <= 10 ** x1:
                o.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{H - B}" stroke="#e5e5e5"/>')
                o.append(f'<text x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle" fill="#555">{v:,.0f}</text>')
        d += 1
    for yv in range(0, int(y1) + 1, 5):
        o.append(f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{W - R}" y2="{Y(yv):.1f}" stroke="#e5e5e5"/>')
        o.append(f'<text x="{L - 8}" y="{Y(yv) + 4:.1f}" text-anchor="end" fill="#555">{yv}</text>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 28}" text-anchor="middle" fill="#333">Output tokens/s per wall kW (iDRAC PSU input), log scale</text>')
    o.append(f'<text transform="translate(18 {(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle" fill="#333">Artificial Analysis Intelligence Index</text>')
    bx = X(baseline * 1000)
    o.append(f'<line x1="{bx:.1f}" y1="{T}" x2="{bx:.1f}" y2="{H - B}" stroke="#c0392b" stroke-dasharray="6 4"/>')
    o.append(f'<text x="{bx + 6:.1f}" y="{T + 14}" fill="#c0392b">published baseline {baseline:g} tok/s/W</text>')
    for cfg, xv, idx, is_peak in pts:
        fill = "#2a6fdb" if is_peak else "white"
        o.append(f'<circle cx="{X(xv):.1f}" cy="{Y(idx):.1f}" r="6" fill="{fill}" stroke="#2a6fdb" stroke-width="2"/>')
        if is_peak:
            o.append(f'<text x="{X(xv) + 9:.1f}" y="{Y(idx) + 4:.1f}" fill="#222">{LABELS.get(cfg, cfg)}</text>')
    o.append(f'<circle cx="{W - 260}" cy="{H - 50}" r="6" fill="#2a6fdb" stroke="#2a6fdb" stroke-width="2"/>'
             f'<text x="{W - 248}" y="{H - 46}">peak tok/s per W</text>'
             f'<circle cx="{W - 130}" cy="{H - 50}" r="6" fill="white" stroke="#2a6fdb" stroke-width="2"/>'
             f'<text x="{W - 118}" y="{H - 46}">at {target:g} tok/s/user p90</text>')
    o.append("</svg>")
    Path(path).write_text("\n".join(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--scores", type=Path, default=Path("docs/efficiency-intelligence-scores.json"))
    ap.add_argument("--target", type=float, default=35.0)
    ap.add_argument("--baseline", type=float, default=7.03)
    ap.add_argument("--out", type=Path, default=Path("docs/efficiency-summary.md"))
    ap.add_argument("--chart", default="docs/efficiency-chart-{wl}.svg", help="SVG path pattern, {wl} substituted; '' to skip")
    a = ap.parse_args()
    scores = json.loads(a.scores.read_text())["scores"] if a.scores.is_file() else {}
    data = load(a.results_dir)
    by_wl = {}
    for (wl, cfg), rungs in data.items():
        peak, at = readings(rungs, a.target)
        by_wl.setdefault(wl, []).append((cfg, peak, at))
    md = ["# Efficiency sweep summary (generated by scripts/efficiency_summary.py)", "",
          f"Baseline: {BASELINE_NOTE}. Peak = rung with the best tokens per wall watt; at-target = fastest "
          f"rung with p90 per-user speed >= {a.target:g} tok/s. Wall power is the iDRAC PSU input sum.", ""]
    for wl in ("1k1k", "8k1k", "128x128"):
        if wl not in by_wl:
            continue
        md += table(wl, by_wl[wl], scores, a.target, a.baseline) + [""]
        if a.chart:
            path = a.chart.format(wl=wl)
            svg_chart(wl, by_wl[wl], scores, a.target, a.baseline, path)
            md += [f"![{WL_TITLE.get(wl, wl)}]({Path(path).name})", ""]
    a.out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
