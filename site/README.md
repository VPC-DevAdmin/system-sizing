# Capacity Site

Self-contained static site presenting the persona capacity benchmark for the
Intel (PowerEdge R470 / Xeon 6) and AMD (PowerEdge R7625 / dual EPYC 9374F) runs.
No build step, no framework.

## Pages

| File | What it is |
|---|---|
| `index.html` | Landing page — pick a platform |
| `intel.html` | Intel Xeon 6 / R470 benchmark narrative |
| `amd.html`   | AMD EPYC / R7625 benchmark narrative |

Shared support files: `assets/style.css` (design system), `assets/app.js`
(charts, replay engine, calculators), `assets/*` (logos and server imagery),
`data/*.js` (slim benchmark data).

## Viewing

Open `index.html` directly in a browser — everything works from `file://`
because the data ships as `.js` files rather than fetched JSON.
Chart.js and the Inter font load from CDNs, so charts need network access.

Or serve the folder:

```bash
cd site && python3 -m http.server 8080
```

## Refreshing the data

`data/intel-data.js` and `data/amd-data.js` are generated from the full
simulator exports (`artifacts/Intel_sizing_qwen3.json`, `artifacts/AMD_sizing_qwen3.json`,
~20 MB each). The script keeps the capacity curves, knees, bottleneck evidence,
and — for each team — the measured turns and downsampled telemetry of the
knee-pool measurement window (this powers the live replay section).

```bash
python3 site/build_data.py                          # default artifact paths
python3 site/build_data.py --intel X.json --amd Y.json
```

Re-run it whenever a new sizing export lands; the pages pick the data up on reload.

## Editing the narrative

All page copy lives in the HTML files. Numbers that come from measurement are
injected by `assets/app.js` from the data files; static copy (hero claims,
funnel percentages, profile tables in `window.PAGE`) is per-page and should be
updated when a new run changes the headline numbers.
