#!/usr/bin/env python3
"""Summarize P11 US model survey results (scripts/p11_us_survey.py) into one
JSON file and a markdown table. Run on the box next to the results directory.

Adds an engine-side cross-check: the generation-token counter rate scraped from
each replica's /metrics during the rung. AIPerf counts only requests that finish
inside the window, so for slow models (answers longer than the window) its
throughput reads low; the engine counter does not. vLLM counts tokens as they
are generated; SGLang counts at request finish, so treat its rate as a floor.
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/capsim/p11/results")


def engine_gen_rate(rung_dir):
    rates = {}
    for f in rung_dir.glob("s*/server_metrics_export.json"):
        metrics = json.loads(f.read_text()).get("metrics", {})
        for key in ("vllm:generation_tokens", "vllm:generation_tokens_total", "sglang:generation_tokens"):
            per_url = {}
            for s in metrics.get(key, {}).get("series", []):
                rate = s.get("stats", {}).get("rate")
                if rate is not None:
                    url = s.get("endpoint_url")
                    per_url[url] = per_url.get(url, 0) + rate
            for url, rate in per_url.items():
                rates[url] = rate  # shards scraping the same replica see the same counter
    return sum(rates.values()) if rates else None


def main():
    rows = []
    for f in sorted((ROOT / "sweep").glob("*/result.json")):
        cid = f.parent.name
        if any(tag in cid for tag in ("failed", "no-moe-flag")):
            continue
        d = json.loads(f.read_text())
        for r in d.get("rungs", []):
            r["engine_gen_tok_s"] = engine_gen_rate(f.parent / f"c{r['concurrency']}")
        q = d.get("quality") or {}
        ok_rungs = [r for r in d.get("rungs", []) if not r.get("error")]
        peak = max(ok_rungs, key=lambda r: r.get("gen_tok_s") or 0, default=None)
        rows.append({
            "config": cid,
            "model": d["model"],
            "engine": d["engine"],
            "placement": f"tp{d['tp']} x {d['replicas']}",
            "status": d.get("status"),
            "error": (d.get("error") or "")[-300:] or None,
            "startup_s": d.get("startup_s"),
            "peak_gen_tok_s": peak["gen_tok_s"] if peak else None,
            "peak_users": peak["concurrency"] if peak else None,
            "quality": {k: f"{v['correct']}/{v['total']}" for k, v in q.items()},
            "quality_prompt_tokens": {k: v.get("mean_prompt_tokens") for k, v in q.items()},
            "provenance": d.get("provenance"),
            "rungs": d.get("rungs", []),
        })
    screens = {}
    for f in sorted((ROOT / "screen").glob("*/result.json")):
        d = json.loads(f.read_text())
        screens[f.parent.name] = {"status": d.get("status"), "gen_tok_s": d.get("peak_gen_tok_s"),
                                  "error": (d.get("error") or "")[-200:] or None}
    (ROOT / "summary.json").write_text(json.dumps({"models": rows, "screens": screens}, indent=1))

    k = lambda v: f"{v / 1000:.1f}k" if v and v >= 1000 else (f"{v:.0f}" if v else "-")  # noqa: E731
    ms = lambda v: "-" if v is None else (f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms")  # noqa: E731
    lines = [
        "| Model | Engine | Placement | Peak gen tok/s (users) | Quality ~6.5K / ~26K | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -(r["peak_gen_tok_s"] or 0)):
        q = r["quality"]
        lines.append(f"| {r['model']} | {r['engine']} | {r['placement']} | "
                     f"{k(r['peak_gen_tok_s'])} ({r['peak_users']}) | "
                     f"{q.get('8k', '-')} / {q.get('32k', '-')} | {r['status']} |")
    for r in sorted(rows, key=lambda r: -(r["peak_gen_tok_s"] or 0)):
        lines += ["", f"### {r['config']}: {r['model']} on {r['engine']} ({r['placement']})", "",
                  "| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |",
                  "|---|---|---|---|---|---|---|"]
        for x in r["rungs"]:
            if x.get("error"):
                lines.append(f"| {x['concurrency']} | error: {x['error']} | | | | | |")
                continue
            lines.append(
                f"| {x['concurrency']} | {k(x['gen_tok_s'])} | {k(x.get('engine_gen_tok_s'))} | "
                f"{ms(x['ttft_p50_ms'])} / {ms(x['ttft_p95_ms'])} | "
                f"{ms(x['itl_p50_ms'])} / {ms(x['itl_p95_ms'])} | "
                f"{x.get('engine_running_p50') or '-'} | {x['error_fraction']:.2%} |")
    lines += ["", "| Screen (1,024 users) | Status | Gen tok/s |", "|---|---|---|"]
    for cid, s in screens.items():
        lines.append(f"| {cid} | {s['status']} | {k(s['gen_tok_s'])} |")
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
