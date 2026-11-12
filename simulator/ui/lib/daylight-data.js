/* Pure helpers for the Daylight UI: what to say about a result and how
 * to draw it. No DOM access, so `node --test` covers them
 * (tests/ui/daylight-data.test.mjs). */

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 0) => v == null || Number.isNaN(+v) ? "—"
  : Number(v).toLocaleString("en-US", { maximumFractionDigits: d });

/* The launch shape "Automatic" means: the fewest GPUs per copy that hold
 * the model, and as many copies as the server then fits. */
export function autoShape(model, hardware) {
  const gpus = Math.max(1, hardware?.gpus || 1);
  const vram = hardware?.vram_per_gpu_gb || 0;
  const need = model?.min_vram_gb || 0;
  let tp = 1;
  while (vram && need > tp * vram && tp < gpus) tp *= 2;
  tp = Math.min(tp, gpus);
  return { tp, replicas: Math.max(1, Math.floor(gpus / tp)) };
}

/* Tokens per second one user receives: the inverse of the median time
 * between tokens. */
export function perUser(rung) {
  return rung?.tpot_p50_ms ? 1000 / rung.tpot_p50_ms : null;
}

/* True when the peak is the highest load measured and that level was
 * served without the engine falling behind: more load may give more. */
export function notAtCeiling(doc) {
  const rungs = (doc?.rungs || []).filter((r) => r.out_tok_s);
  const peak = doc?.peak;
  if (!peak || rungs.length < 2) return false;
  const top = Math.max(...rungs.map((r) => r.concurrency));
  return peak.concurrency === top && peak.held !== false;
}

/* The short notices under a result's headline, most important first.
 * kind: "bad" (do not quote the number), "warn" (quote with care). */
export function resultNotes(doc) {
  const notes = [];
  const peak = doc?.peak;
  if (!peak) {
    return [{ kind: "bad", title: "No result.", text: "The run ended before any load level was measured." }];
  }
  const done = (peak.samples || 0) + (peak.errors || 0) + (peak.no_content || 0);
  if (done > 0 && (peak.errors || 0) / done > 0.1) {
    notes.push({ kind: "bad", title: "Many requests failed.",
                 text: `${num(peak.errors)} of ${num(done)} requests at the peak did not complete, so this is not a rate anyone was served at.` });
  }
  if (peak.prefix_reuse === "above_template") {
    notes.push({ kind: "bad", title: "Too much cache reuse.",
                 text: "Prompts were reused beyond the designed level, so this figure overstates throughput." });
  }
  if (doc.status && doc.status !== "ok") {
    notes.push({ kind: "warn", title: "The run did not finish cleanly.", text: doc.stop_reason || "" });
  }
  if (peak.steady_state === false) {
    notes.push({ kind: "warn", title: "Still settling.", text: "The peak was read before the rate stopped moving. Treat it as provisional." });
  }
  if (notAtCeiling(doc)) {
    notes.push({ kind: "warn", title: "Not yet the ceiling.",
                 text: "Output was still rising at the highest load measured, so the true peak is probably higher." });
  }
  return notes;
}

/* Horizontal bars, one row per item {name, sub?, value, text, href?, kind?}. */
export function barRows(items) {
  const max = Math.max(0, ...items.map((i) => +i.value || 0));
  const rows = items.map((i) => {
    const width = max ? Math.max(1, Math.round(78 * (+i.value || 0) / max)) : 0;
    const tag = i.href ? "a" : "div";
    const name = `<span class="name">${esc(i.name)}${i.sub ? `<span class="muted small" style="display:block;font-weight:400">${esc(i.sub)}</span>` : ""}</span>`;
    return `<${tag} class="bar-row"${i.href ? ` href="${esc(i.href)}"` : ""}>${name}`
      + `<span class="track"><span class="fill${i.kind === "power" ? " power" : ""}" style="width:${width}%" aria-hidden="true"></span><b>${esc(i.text)}</b></span></${tag}>`;
  });
  return `<div class="bars">${rows.join("")}</div>`;
}

/* Output against load, one point per load level, evenly spaced (the
 * ladder doubles each step). The peak is marked and labelled. */
export function lineChart(rungs, peak) {
  if (!rungs.length) return '<p class="muted">No load level was measured.</p>';
  const W = 560, H = 270, x0 = 60, x1 = 520, y0 = 225, y1 = 40;
  const max = Math.max(...rungs.map((r) => r.out_tok_s));
  const x = (i) => rungs.length === 1 ? (x0 + x1) / 2 : x0 + (x1 - x0) * i / (rungs.length - 1);
  const y = (v) => y0 - (y0 - y1) * v / max;
  const pts = rungs.map((r, i) => [x(i), y(r.out_tok_s)]);
  const path = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
  const area = `${path} L${pts[pts.length - 1][0].toFixed(1)} ${y0} L${pts[0][0].toFixed(1)} ${y0} Z`;
  const label = rungs.map((r) => `${num(r.out_tok_s)} at ${num(r.concurrency)}`).join(", ");
  const every = Math.ceil(rungs.length / 8);
  const marks = rungs.map((r, i) => {
    const isPeak = peak && r.concurrency === peak.concurrency;
    const [px, py] = pts[i];
    const anchor = i === rungs.length - 1 && rungs.length > 1 ? "end" : "middle";
    return `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${isPeak ? 8 : 5}" fill="#0b7a75"${isPeak ? ' stroke="#fff" stroke-width="3"' : ""}/>`
      + (isPeak ? `<text class="v" x="${px.toFixed(1)}" y="${(py - 16).toFixed(1)}" text-anchor="${anchor}">${num(r.out_tok_s)} peak</text>` : "")
      + (i % every === 0 || isPeak ? `<text x="${px.toFixed(1)}" y="${y0 + 22}" text-anchor="middle">${num(r.concurrency)}</text>` : "");
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Tokens per second by simultaneous requests: ${esc(label)}">
    <line x1="${x0}" y1="${y0}" x2="${W - 20}" y2="${y0}" stroke="#d3dee6" stroke-width="2"/>
    <text x="${x0 - 8}" y="${y0 + 4}" text-anchor="end">0</text>
    <text x="${x0 - 8}" y="${y1 + 4}" text-anchor="end">${num(max / 1000, max < 10000 ? 1 : 0)}k</text>
    <path d="${area}" fill="#d5efed"/><path d="${path}" fill="none" stroke="#0b7a75" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
    ${marks}
    <text x="${(x0 + x1) / 2}" y="${H - 4}" text-anchor="middle">Simultaneous requests</text></svg>`;
}

/* The ladder while a run is in flight: solid bars for measured levels,
 * a dashed outline for the one being measured, empty for the rest. */
export function ladderChart(ladder, done, current) {
  const levels = ladder.length ? ladder : done.map((r) => r.concurrency);
  if (!levels.length) return '<p class="muted">Waiting for the engine to load.</p>';
  const W = 560, H = 250, y0 = 210, top = 30;
  const slot = (W - 60) / levels.length, bw = Math.min(80, slot * 0.7);
  const by = new Map(done.map((r) => [r.concurrency, r.out_tok_s || 0]));
  const max = Math.max(1, ...by.values());
  const bars = levels.map((n, i) => {
    const bx = 40 + slot * i + (slot - bw) / 2, cx = bx + bw / 2;
    const tick = `<text x="${cx.toFixed(1)}" y="${y0 + 24}" text-anchor="middle">${num(n)}</text>`;
    if (by.has(n)) {
      const h = Math.max(4, (y0 - top - 30) * by.get(n) / max);
      return `<rect x="${bx.toFixed(1)}" y="${(y0 - h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="8" fill="#0b7a75"/>`
        + `<text class="v" x="${cx.toFixed(1)}" y="${(y0 - h - 8).toFixed(1)}" text-anchor="middle">${num(by.get(n))}</text>${tick}`;
    }
    const now = n === current;
    return `<rect x="${bx.toFixed(1)}" y="${top}" width="${bw.toFixed(1)}" height="${y0 - top}" rx="8" fill="none" stroke="${now ? "#0b7a75" : "#d3dee6"}" stroke-width="${now ? 2.5 : 2}"${now ? ' stroke-dasharray="7 6"' : ""}/>`
      + (now && bw >= 60 ? `<text x="${cx.toFixed(1)}" y="${(y0 + top) / 2}" text-anchor="middle" style="fill:#075a56">measuring</text>` : "") + tick;
  }).join("");
  const label = levels.map((n) => by.has(n) ? `${num(n)} requests: ${num(by.get(n))} tokens per second`
    : n === current ? `${num(n)} requests: being measured` : `${num(n)} requests: not started`).join("; ");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">
    <line x1="40" y1="${y0}" x2="${W - 20}" y2="${y0}" stroke="#d3dee6" stroke-width="2"/>${bars}</svg>`;
}

/* ── Capacity runs ("how many users can it support?") ───────────── */

/* How one tested arrival rate went: "ok" (kept up, within response-time
 * targets), "slow" (kept up, some turns over target), "over" (fell
 * behind: the queue kept growing), "pending" (not judged yet). */
export function stepVerdict(step) {
  const status = step?.status ?? step?.capacity_status;
  if (step?.stability === "divergent") return "over";
  if (step?.stability === "stable") return status === "pass" ? "ok" : "slow";
  if (!step?.stability && status && status !== "pending" && step?.sample_size) {
    return status === "pass" ? "ok" : status === "fail" ? "over" : "slow";
  }
  return "pending";
}
export const VERDICT = {
  ok: { label: "kept up", color: "#0b7a75" },
  slow: { label: "slow turns", color: "#c2650f" },
  over: { label: "fell behind", color: "#b3261e" },
  pending: { label: "measuring", color: "#9db5b3" },
};

/* The answer a capacity run gives, from one cohort of the export. */
export function capacitySummary(cohort) {
  const ol = cohort?.open_loop || {};
  const atTarget = ol.rate_sla_per_min != null;
  const rate = atTarget ? ol.rate_sla_per_min : ol.rate_max_per_min ?? null;
  const users = atTarget
    ? ol.derived_concurrent_sessions_at_sla ?? ol.measured_concurrent_sessions_at_max
    : ol.measured_concurrent_sessions_at_max;
  const notes = [];
  if (rate == null) {
    notes.push({ kind: "bad", title: "No capacity found.",
                 text: "The engine did not keep up at any rate tested, or the run ended before one was judged." });
    return { rate: null, users: null, atTarget: false, notes };
  }
  if (!atTarget) {
    notes.push({ kind: "warn", title: "Response-time targets were missed.",
                 text: "This is the highest rate the engine kept up with, but some turns were slower than the workload's target." });
  }
  if (cohort.final_status && !["ok", "completed"].includes(cohort.final_status)) {
    notes.push({ kind: "warn", title: "The run did not finish.",
                 text: `It ended as "${cohort.final_status}", so higher rates were not tested.` });
  }
  if (ol.rate_ceiling_per_min == null || ol.rates_are_lower_bounds) {
    notes.push({ kind: "warn", title: "Not yet the ceiling.",
                 text: "The engine never fell behind in the range tested, so it can probably take more." });
  }
  return { rate, users: users ?? null, atTarget, notes };
}

/* Wait for the first token at each arrival rate tested, one marker per
 * rate, coloured and labelled by how that rate went. */
export function rateChart(points) {
  // A rate still being measured has no response time yet: leave it off.
  const pts = points.filter((p) => p.arrival_rate_per_min != null && p.ttft_p95_ms != null)
    .sort((a, b) => a.arrival_rate_per_min - b.arrival_rate_per_min);
  if (!pts.length) return '<p class="muted">No arrival rate has been judged yet.</p>';
  const W = 560, H = 270, x0 = 70, x1 = 510, y0 = 205, y1 = 40;
  const max = Math.max(1, ...pts.map((p) => p.ttft_p95_ms || 0));
  const x = (i) => pts.length === 1 ? (x0 + x1) / 2 : x0 + (x1 - x0) * i / (pts.length - 1);
  const y = (v) => y0 - (y0 - y1) * (v || 0) / max;
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p.ttft_p95_ms).toFixed(1)}`).join(" ");
  const every = Math.ceil(pts.length / 7);
  const marks = pts.map((p, i) => {
    const v = VERDICT[stepVerdict(p)];
    const cx = x(i).toFixed(1);
    return `<circle cx="${cx}" cy="${y(p.ttft_p95_ms).toFixed(1)}" r="7" fill="${v.color}" stroke="#fff" stroke-width="2"/>`
      + (i % every === 0 ? `<text x="${cx}" y="${y0 + 20}" text-anchor="middle">${num(p.arrival_rate_per_min, 1)}</text>`
        + `<text x="${cx}" y="${y0 + 36}" text-anchor="middle" style="font-size:11px;fill:${v.color}">${v.label}</text>` : "");
  }).join("");
  const label = pts.map((p) => `${num(p.arrival_rate_per_min, 1)} per minute: ${VERDICT[stepVerdict(p)].label}, first token in ${num(p.ttft_p95_ms)} ms`).join("; ");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">
    <line x1="${x0 - 20}" y1="${y0}" x2="${W - 20}" y2="${y0}" stroke="#d3dee6" stroke-width="2"/>
    <text x="${x0 - 26}" y="${y0 + 4}" text-anchor="end">0</text>
    <text x="${x0 - 26}" y="${y1 + 4}" text-anchor="end">${num(max)} ms</text>
    <path d="${line}" fill="none" stroke="#9db5b3" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    ${marks}
    <text x="${(x0 + x1) / 2}" y="${H - 6}" text-anchor="middle">New sessions per minute</text></svg>`;
}

/* The best rate a running capacity search has confirmed so far. */
export function bestRate(steps) {
  const held = steps.filter((s) => ["ok", "slow"].includes(stepVerdict(s)) && s.arrival_rate_per_min != null);
  return held.length ? Math.max(...held.map((s) => s.arrival_rate_per_min)) : null;
}
