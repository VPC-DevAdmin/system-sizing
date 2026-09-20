/* Shared helpers: the fetch wrapper every module talks to the
 * service through, the DOM shorthand and the small formatters. */

export const $ = (sel) => document.querySelector(sel);

export const STATUS_CLASS = { pass: "status-pass", marginal: "status-marginal",
                       fail: "status-fail", ok: "status-pass" };

export async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* text */ }
    throw new Error(detail);
  }
  return r.json();
}

export const fmt = {
  ms: (v) => v == null ? "—" : v >= 10000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`,
  pct: (v) => v == null ? "—" : `${(v * 100).toFixed(1)}%`,
  ts: (iso) => iso ? iso.replace("T", " ").slice(0, 19) : "—",
  clock: (ms) => new Date(ms).toLocaleTimeString("en-GB"),
};

export function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[idx];
}

/* Enter/Space on a clickable non-button (run rows, filesystem chips,
 * persona list items): the keyboard path buttons get for free. Keys
 * pressed on a focusable CHILD (a row's checkbox) are left alone. */
export function keyActivate(el) {
  el.setAttribute("role", "button");
  el.tabIndex = 0;
  el.addEventListener("keydown", (e) => {
    if (e.target !== el) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
  });
}

/* The Workload picker's value for the mock profile: a self-test
 * engine that needs no download and no GPU. */
export const MOCK_MODEL = "__mock__";

/* Log-scale slider mapping: range inputs run 0..1000, values are
 * ratio-scaled (tokens, seconds) so linear sliders would waste 90% of
 * their travel on the top decade. */
export const logTo = (pos, min, max) =>
  min * Math.exp((pos / 1000) * Math.log(max / min));
export const logFrom = (v, min, max) =>
  1000 * Math.log(Math.max(min, Math.min(max, v)) / min) / Math.log(max / min);
export const fmtNum = (v, dec) =>
  dec === 0 ? String(Math.round(v)) : String(+(+v).toFixed(dec));

export function fmtCompact(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B`
    : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M`
    : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${Math.round(n)}`;
}
