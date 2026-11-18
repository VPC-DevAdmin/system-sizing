#!/usr/bin/env python3
"""Summarize the XE7740 InferenceX-style sweeps beside InferenceX's published rows.

    inferencex_summary.py RESULTS_DIR [--ix IX_API_DIR] [--server-kw 7.0]

RESULTS_DIR holds <workload>/<config>/result.json from inferencex_xe7740.py.
IX_API_DIR holds `b_<model>.json` files saved from
https://inferencex.semianalysis.com/api/v1/benchmarks?model=<model> (gzip:
use curl --compressed) and `options.json` from /api/v1/views/options.

Per-GPU output tokens/s is the comparable quantity: InferenceX divides by the
chips an instance uses; ours is the server total over its eight GPUs. Costs
use InferenceX's own $/GPU-hour for each hardware key. Energy per token is GPU
board power on our side and InferenceX's measured avg_power_w on theirs.
"""

import argparse
import collections
import json
from pathlib import Path

OUR_HW = "rtx6000pro"
# InferenceX model key -> our config name prefixes
MODEL_MAP = {
    "gptoss120b": ("gpt-oss-120b", "b_gpt-oss-120b.json"),
    "llama70b": ("llama-70b-fp8", "b_Llama-3.3-70B-Instruct-FP8.json"),
    "qwen3.5": ("qwen3.5-397b", "b_Qwen-3.5-397B-A17B.json"),
}


def load_options(ix_dir):
    o = json.load(open(ix_dir / "options.json"))
    return {h["key"]: h for h in o["hardware"]}


def dollars_per_m_output(out_tok_s_per_gpu, cost_per_hour):
    return cost_per_hour / (out_tok_s_per_gpu * 3600 / 1e6) if out_tok_s_per_gpu else None


def our_rows(results_dir):
    out = collections.defaultdict(list)
    for f in sorted(results_dir.glob("*/*/result.json")):
        r = json.loads(f.read_text())
        wl = f.parent.parent.name
        gpus = r["tp"] * r["replicas"]
        for row in r.get("rungs", []):
            if row.get("error"):
                continue
            out[(wl, r["config"])].append({
                "conc_per_gpu": row["concurrency"] / gpus,
                "conc": row["concurrency"],
                "out_tok_s_per_gpu": row["gen_tok_s"] / gpus,
                "intvty_p90": row.get("intvty_p90_tok_s"),
                "tpot_med_ms": row.get("itl_p50_ms"),
                "ttft_med_ms": row.get("ttft_p50_ms"),
                "w_per_gpu": row.get("avg_w_per_gpu"),
                "j_per_out": row.get("j_per_output_token"),
                "running": row.get("engine_running_p50"),
                "errors": row.get("error_fraction"),
            })
    return out


def ix_rows(ix_dir, model_file, isl, osl):
    rows = json.load(open(ix_dir / model_file))
    groups = collections.defaultdict(list)
    for r in rows:
        if r["isl"] == isl and r["osl"] == osl:
            groups[(r["hardware"], r["framework"], r["precision"], r["spec_method"])].append(r)
    out = {}
    for g, lst in groups.items():
        d = max(x["date"] for x in lst)
        out[g] = sorted([x for x in lst if x["date"] == d], key=lambda x: x["conc"]), d
    return out


def fmt(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--ix", type=Path, help="directory with saved InferenceX API JSON")
    ap.add_argument("--server-kw", type=float, default=7.0,
                    help="XE7740 provisioned (nameplate) kW for tokens per provisioned MW")
    ap.add_argument("--target", type=float, default=35.0, help="p90 interactivity target, tok/s/user")
    a = ap.parse_args()
    hw = load_options(a.ix) if a.ix else {}
    our_cost = hw.get(OUR_HW, {}).get("costPerHour", {}).get("h")
    tdp = hw.get(OUR_HW, {}).get("tdpW", 600)

    ours = our_rows(a.results_dir)
    for (wl, cfg), rows in sorted(ours.items()):
        print(f"\n## XE7740 {cfg} {wl}  (GPU board power; ${our_cost}/GPU-h InferenceX 'h' basis)\n")
        print("| conc/GPU | out tok/s/GPU | p90 intvty | TPOT med ms | TTFT med ms | W/GPU | J/out tok | $/M out tok | tok/s per prov. MW (TDP) | tok/s per prov. MW (server) | running | err |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["conc"]):
            o = r["out_tok_s_per_gpu"]
            print(f"| {r['conc_per_gpu']:g} | {o:.0f} | {fmt(r['intvty_p90'], 1)} | {fmt(r['tpot_med_ms'], 1)} | "
                  f"{fmt(r['ttft_med_ms'])} | {fmt(r['w_per_gpu'])} | {fmt(r['j_per_out'], 2)} | "
                  f"{fmt(dollars_per_m_output(o, our_cost), 3) if our_cost else '-'} | "
                  f"{o / tdp * 1e6:,.0f} | {o * 8 / (a.server_kw * 1000) * 1e6:,.0f} | {fmt(r['running'])} | {fmt(r['errors'], 3)} |")
        best = [r for r in rows if r["intvty_p90"] and r["intvty_p90"] >= a.target]
        if best:
            b = max(best, key=lambda r: r["out_tok_s_per_gpu"])
            print(f"\nAt >= {a.target:g} tok/s/user p90: {b['out_tok_s_per_gpu']:.0f} out tok/s/GPU at conc/GPU {b['conc_per_gpu']:g}, "
                  f"{fmt(b['w_per_gpu'])} W/GPU, {fmt(b['j_per_out'], 2)} J/out tok.")

    if not a.ix:
        return
    for key, (prefix, model_file) in MODEL_MAP.items():
        if not (a.ix / model_file).is_file():
            continue
        for wl, (isl, osl) in (("8k1k", (8192, 1024)), ("1k1k", (1024, 1024))):
            groups = ix_rows(a.ix, model_file, isl, osl)
            if not groups:
                continue
            print(f"\n## InferenceX {key} {isl}/{osl}: latest curve per platform\n")
            print(f"| hardware | framework | precision | spec | date | chips/instance | peak out tok/s/chip | at >= {a.target:g} tok/s/user p90: out tok/s/chip (conc) | W/chip there | J/out tok there | $/M out tok there (h) |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for g, (lst, d) in sorted(groups.items()):
                m = [x["metrics"] for x in lst]
                peak = max(x["output_tput_per_gpu"] for x in m)
                # Older rows lack p90_intvty; fall back to 1 / p90 TPOT, then median.
                def intvty(x):
                    m = x["metrics"]
                    if m.get("p90_intvty"):
                        return m["p90_intvty"]
                    if m.get("p90_tpot"):
                        return 1 / m["p90_tpot"]
                    return 1 / m["median_tpot"] if m.get("median_tpot") else 0
                ok = [x for x in lst if intvty(x) >= a.target]
                if ok:
                    b = max(ok, key=lambda x: x["metrics"]["output_tput_per_gpu"])
                    bm = b["metrics"]
                    cost = hw.get(g[0], {}).get("costPerHour", {}).get("h")
                    at = (f"{bm['output_tput_per_gpu']:.0f} ({b['conc']})", fmt(bm.get("avg_power_w")),
                          fmt(bm.get("joules_per_output_token"), 2),
                          fmt(dollars_per_m_output(bm["output_tput_per_gpu"], cost), 3) if cost else "-")
                else:
                    at = ("none meets target", "-", "-", "-")
                print(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {d} | {lst[0]['num_decode_gpu']} | {peak:.0f} | {at[0]} | {at[1]} | {at[2]} | {at[3]} |")


if __name__ == "__main__":
    main()
