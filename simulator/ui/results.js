import { $, api, fmt, STATUS_CLASS, keyActivate } from "./lib/api.js";
import { C, fill, PALETTE } from "./lib/theme.js";
import { on } from "./lib/events.js";
import { onShow } from "./lib/tabs.js";

/* ══ Results ══════════════════════════════════════════════════── */

export const Results = {
  runs: [],
  doc: null,            // loaded export for the selected run
  cohort: null,
  charts: {},
  compare: [],          // {label, curve}
  _shown: false,
  running: false,       // mirrors Control's status poll

  onShow() {
    // Refresh EVERY visit — a once-only guard here meant a Results
    // tab opened before the first run finished cached an empty
    // picker forever.
    this.refreshRuns();
  },

  flat: [],             // flattened (run, cohort) rows, newest first
  openId: null,         // cohort_run_id currently displayed
  checked: new Set(),   // cohort_run_ids ticked for comparison

  init() {
    onShow("results", () => this.onShow());
    on("status", ({ running }) => { this.running = running; });
    on("run:finished", () => {
      // Any export cached mid-run is now stale (the server rebuilds
      // when run.db is newer, but only if we actually refetch).
      this.exportCache = {};
      this.openId = null;   // re-open the freshest run on refresh
      this.refreshRuns();
    });
    $("#result-export-dl").addEventListener("click", () => this.download());
    $("#compare-btn").addEventListener("click", () => this.runCompare());
    $("#compare-clear").addEventListener("click", () => {
      this.compare = [];
      this.checked.clear();
      $("#compare-panel").hidden = true;
      this.renderList();
    });
  },

  async refreshRuns() {
    let runs;
    try { runs = await api("/api/runs"); } catch { return; }
    this.setRuns(runs);
  },

  /* One flat list of every cohort run across every run_NN dir,
   * newest first — what "my runs" actually means to an operator.
   * Each row is self-describing (workload, model, methodology, date,
   * headline verdict) so nothing needs a Load button to make sense. */
  setRuns(runs) {
    this.runs = runs.filter(r => r.cohorts.length);
    this.flat = this.runs.flatMap(r =>
      r.cohorts.map(c => ({ run: r.name, ...c })));
    this.flat.sort((a, b) =>
      (b.started_at || "").localeCompare(a.started_at || ""));
    this.renderList();
    // Auto-open the newest run with data — the page should never sit
    // blank waiting for the user to find a picker. Prefer the newest
    // COMPLETED run; fall back to anything with measurements.
    if (!this.openId) {
      const first = this.flat.find(e => e.final_status === "ok" && e.steps > 0)
        ?? this.flat.find(e => e.steps > 0);
      if (first) this.openEntry(first);
    }
  },

  headline(e) {
    // A saturation sweep has no arrival rate and no pool — its
    // headline is peak output throughput, carried on the entry.
    if (e.mode === "headline_sweep") {
      return e.peak_out_tok_s != null
        ? { n: `${Math.round(e.peak_out_tok_s).toLocaleString()} tok/s`,
            d: `peak output · ${Math.round(e.peak_streams || 0)} streams` }
        : { n: "—", d: "saturation sweep" };
    }
    if (e.rate_max_per_min != null) {
      const rate = e.rate_sla_per_min ?? e.rate_max_per_min;
      const sess = e.sessions_at_max != null
        ? ` · ~${Math.round(e.sessions_at_max)} sessions` : "";
      return { n: `${rate}/min${sess}`,
               d: "stable arrival rate (open-loop)" };
    }
    if (e.capacity_pool != null) {
      return { n: `≤${e.capacity_pool} users`, d: "pool capacity" };
    }
    return { n: "—", d: e.steps > 0 ? "no verdict yet" : "no data" };
  },

  renderList() {
    const box = $("#run-list");
    box.innerHTML = "";
    if (!this.flat.length) {
      box.innerHTML = `<span class="msg">no runs yet — start one on the
        Benchmark tab</span>`;
      return;
    }
    // Only ONE run can be active — the newest unfinalised row. Any
    // other row without a final status is an orphan from a hard stop
    // (the server also stamps these 'interrupted' at startup).
    const newestOpen = this.flat.find(x => !x.final_status);
    for (const e of this.flat) {
      const status = e.final_status
        ?? (this.running && e === newestOpen ? "running" : "interrupted");
      const cls = STATUS_CLASS[status]
        ?? (status === "running" ? "status-marginal" : "status-error");
      const mode = (e.mode === "open_loop") ? "open-loop"
        : e.mode === "headline_sweep" ? "saturation"
        : "pool ramp";
      const h = this.headline(e);
      const model = (e.model_id || "").split("/").pop();
      const row = document.createElement("div");
      row.className = "run-row"
        + (e.cohort_run_id === this.openId ? " active" : "");
      row.innerHTML = `
        <input type="checkbox" ${this.checked.has(e.cohort_run_id) ? "checked" : ""}>
        <span class="r-title">${e.cohort_name || e.cohort_id}
          <span class="hint">${model} · ${e.engine_type} · ${e.run}</span></span>
        <span class="r-mode">${mode}</span>
        <span class="r-headline">${h.n}<span class="hint">${h.d}</span></span>
        <span class="${cls}">${status}</span>
        <span class="r-date">${fmt.ts(e.started_at)}</span>
        <button class="remove" title="delete this run's data">×</button>`;
      row.querySelector("input").addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (ev.target.checked) this.checked.add(e.cohort_run_id);
        else this.checked.delete(e.cohort_run_id);
        $("#compare-btn").disabled = this.checked.size < 2;
      });
      row.querySelector(".remove").addEventListener("click", (ev) => {
        ev.stopPropagation();
        this.deleteEntry(e);
      });
      row.addEventListener("click", () => this.openEntry(e));
      keyActivate(row);
      row.setAttribute("aria-label",
        `${e.cohort_name || e.cohort_id}, ${model}, ${mode}, ${status}`);
      box.append(row);
    }
    $("#compare-btn").disabled = this.checked.size < 2;
  },

  msg(text, cls = "") {
    const el = $("#result-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async deleteEntry(e) {
    const label = `${e.cohort_name || e.cohort_id} (${e.run}, `
      + `${fmt.ts(e.started_at)})`;
    if (!window.confirm(
      `Delete "${label}"?\n\nThis permanently removes its measurements, `
      + `turns and telemetry and cannot be undone.`)) return;
    try {
      await api(`/api/runs/${e.run}/cohorts/${e.cohort_run_id}`,
                { method: "DELETE" });
    } catch (err) {
      this.msg(err.message, "error");
      return;
    }
    delete this.exportCache[e.run];
    this.checked.delete(e.cohort_run_id);
    if (this.openId === e.cohort_run_id) {
      this.openId = null;
      this.cohort = null;
      $("#result-summary").hidden = true;
      $("#report").hidden = true;
    }
    this.msg(`deleted ${label}`, "ok");
    this.refreshRuns();
  },

  async loadExport(runName) {
    if (!this.exportCache[runName]) {
      this.exportCache[runName] = await api(`/api/runs/${runName}/export`);
    }
    return this.exportCache[runName];
  },

  async openEntry(entry) {
    this.msg("loading…");
    // A headline sweep measured a different thing and gets a
    // different report — never force it through the capacity view.
    if (entry.mode === "headline_sweep") return this.openHeadline(entry);
    let doc;
    try {
      doc = await this.loadExport(entry.run);
    } catch (e) {
      this.msg(e.message, "error");
      return;
    }
    const c = doc.cohorts.find(x => x.cohort_run_id === entry.cohort_run_id);
    if (!c) { this.msg("run has no export data yet", "error"); return; }
    this.doc = doc;
    this.openId = entry.cohort_run_id;
    $("#result-export-dl").disabled = false;
    this.msg("");
    this.renderList();
    this.render(c);
  },

  /* ── Headline sweep report ──────────────────────────────────────
   * The saturation curve and what it costs. No SLA verdicts, no
   * arrival rate, no "concurrent users" — none of that means
   * anything for a firehose benchmark. */
  async openHeadline(entry) {
    let doc;
    try {
      doc = await api(`/api/runs/${entry.run}/headline`);
    } catch (e) {
      this.msg(e.message, "error");
      return;
    }
    this.openId = entry.cohort_run_id;
    this.msg("");
    this.renderList();
    $("#result-summary").hidden = true;
    $("#report").hidden = true;
    $("#step-detail-panel").hidden = true;
    $("#headline-report").hidden = false;
    this.renderHeadlineSweep(doc);
  },

  renderHeadlineSweep(doc) {
    const rungs = (doc.rungs || []).filter(r => r.out_tok_s);
    const peak = doc.peak;
    const labels = rungs.map(r => String(r.concurrency));
    const num = (v, d = 0) => v == null ? "—"
      : Number(v).toLocaleString(undefined, { maximumFractionDigits: d });

    $("#hl-title").textContent =
      `${doc.cohort_name || doc.cohort_id} — `
      + `${(doc.model || "").split("/").pop()} · saturation benchmark`;

    const effAt = r => (r.gpu_power_w && r.out_tok_s)
      ? r.out_tok_s / (r.gpu_power_w / 1000) : null;
    const stat = (n, label, hint) => `<div class="stat"><span
      class="k">${label}</span><span class="v">${n}</span>
      <span class="hint">${hint}</span></div>`;
    $("#hl-stats").innerHTML =
      stat(num(peak?.out_tok_s), "Peak output tokens/sec",
           "generation only — the headline number")
      + stat(num(peak?.in_flight), "Concurrent streams",
             "running in the engine's batch at that peak")
      + stat(num(peak?.total_tok_s), "Total tokens/sec",
             "prefill + decode through the box")
      + stat(num(effAt(peak || {})), "Tokens/sec per kW",
             "generation throughput per kilowatt of GPU draw")
      + stat(peak ? `${num(peak.ttft_p95_ms)} ms` : "—", "TTFT p95",
             "what the headline costs — not a gate")
      + stat(peak ? `${num(peak.tpot_p95_ms, 1)} ms` : "—", "TPOT p95",
             "per-token pacing at the peak");

    const stopped = doc.stop_reason || "the ladder was exhausted";
    const notSteady = rungs.filter(r => r.steady_state === false).length;
    const sh = doc.shape;
    const shapeTxt = sh
      ? `${num(sh.input_tokens)} tokens in → ${num(sh.output_tokens)} out`
        + (sh.ignore_eos ? ", EOS ignored" : "")
      : "the workload's shape";
    $("#hl-verdict").innerHTML =
      `<b>${num(peak?.out_tok_s)} output tokens/sec</b> sustained at
       <b>${num(peak?.in_flight)}</b> concurrent streams, at
       <b>${shapeTxt}</b>. The sweep stopped because ${stopped}.
       ${notSteady ? ` <span class="status-fail">${notSteady} rung(s)
         hit the measurement cap before settling — treat those as
         provisional.</span>` : ""}
       <div class="msg" style="margin-top:8px">${doc.note || ""}</div>`;

    this.xy("hl-chart-tput", labels, [
      { label: "output tok/s", data: rungs.map(r => r.out_tok_s) },
      { label: "total tok/s", data: rungs.map(r => r.total_tok_s) },
    ], { ytitle: "tokens/sec" });
    this.xy("hl-chart-lat", labels, [
      { label: "TTFT p95 (ms)", data: rungs.map(r => r.ttft_p95_ms) },
      { label: "TPOT p95 (ms)", data: rungs.map(r => r.tpot_p95_ms),
        yAxisID: "y2" },
    ], { ytitle: "TTFT ms", y2title: "TPOT ms" });
    this.xy("hl-chart-batch", labels, [
      { label: "offered", data: rungs.map(r => r.concurrency) },
      { label: "running in engine", data: rungs.map(r => r.in_flight) },
    ], { ytitle: "streams" });
    this.xy("hl-chart-split", labels, [
      { label: "decode tok/s", data: rungs.map(r => r.out_tok_s) },
      { label: "prefill tok/s", data: rungs.map(r => r.prompt_tok_s) },
    ], { ytitle: "tokens/sec" });
    this.xy("hl-chart-kv", labels, [
      { label: "KV cache %", data: rungs.map(r => r.kv_cache_pct) },
    ], { ytitle: "percent" });
    this.xy("hl-chart-power", labels, [
      { label: "GPU watts", data: rungs.map(r => r.gpu_power_w) },
    ], { ytitle: "watts" });
    this.xy("hl-chart-eff", labels, [
      { label: "tok/s per kW", data: rungs.map(effAt) },
    ], { ytitle: "tokens/sec/kW" });
    this.xy("hl-chart-queue", labels, [
      { label: "queued", data: rungs.map(r => r.queue_depth) },
    ], { ytitle: "requests waiting" });

    const bestEff = rungs.reduce((a, b) =>
      (effAt(b) ?? 0) > (effAt(a) ?? 0) ? b : a, rungs[0] || {});
    $("#hl-take-curve").textContent = peak
      ? `peaks at ${num(peak.out_tok_s)} tok/s on ${num(peak.in_flight)} streams`
      : "no rung produced tokens";
    $("#hl-narr-curve").innerHTML =
      `<p>Throughput climbs with concurrency until the engine runs out of
       batch or KV room, then flattens. The peak here is
       <b>${num(peak?.out_tok_s)} output tokens/sec</b> with
       <b>${num(peak?.in_flight)}</b> streams actually running.</p>
       <p>The <b>running batch vs offered</b> chart is the honest one: while
       the two lines track each other the engine is serving everything
       offered. Where they separate, the extra streams are queued, not
       served — that is the engine's own ceiling, and past it latency
       grows without throughput following.</p>
       <p>Prefill and decode are split out because a generation headline
       should be overwhelmingly decode. A large prefill share means the
       shape is spending your GPUs on reading rather than writing.</p>`;
    $("#hl-take-cost").textContent = bestEff?.concurrency
      ? `most efficient at ${bestEff.concurrency} streams`
      : "";
    $("#hl-narr-cost").innerHTML =
      `<p>Efficiency usually peaks <i>before</i> throughput does:
       ${bestEff?.concurrency ? `here the best tokens/sec per kW is at
       <b>${num(bestEff.concurrency)}</b> streams, while peak throughput
       needs <b>${num(peak?.concurrency)}</b>.` : ""}
       The gap between those two points is what the last few percent of
       headline throughput costs in power.</p>
       <p>KV occupancy shows what caps concurrency. If it reaches the
       high nineties before throughput plateaus, KV capacity is the
       binding constraint and fp8 KV or a shorter shape buys more
       streams. If it stays low while the batch stops growing,
       <code>max_num_seqs</code> is the constraint instead.</p>`;

    const cells = rungs.map(r => `<tr>
      <td>${num(r.concurrency)}</td><td>${num(r.in_flight)}</td>
      <td>${num(r.out_tok_s)}</td><td>${num(r.prompt_tok_s)}</td>
      <td>${num(r.ttft_p95_ms)}</td><td>${num(r.tpot_p95_ms, 1)}</td>
      <td>${num(r.kv_cache_pct, 1)}</td><td>${num(r.gpu_power_w)}</td>
      <td>${r.steady_state ? "yes" : "capped"}</td>
      <td>${r.measure_s}s</td></tr>`).join("");
    $("#hl-table").innerHTML = `<table><thead><tr>
      <th>Offered</th><th>Running</th><th>Out tok/s</th>
      <th>Prefill tok/s</th><th>TTFT p95</th><th>TPOT p95</th>
      <th>KV %</th><th>GPU W</th><th>Settled</th><th>Measured</th>
      </tr></thead><tbody>${cells}</tbody></table>`;
    $("#hl-take-table").textContent = `${rungs.length} rungs measured`;
  },

  render(c) {
    c = c ?? this.cohort;
    if (!c) return;
    this.cohort = c;
    $("#headline-report").hidden = true;
    $("#result-summary").hidden = false;
    $("#report").hidden = false;
    $("#step-detail-panel").hidden = true;
    const ctx = this.analyze(c);
    $("#result-title").textContent =
      `${c.name || c.id} — ${(c.model || "").split("/").pop()}` +
      (ctx.isOpen ? " · open-loop" : " · pool ramp");
    if (!ctx.pts.length) { this.renderEmpty(c, ctx); return; }
    this.renderHeadline(c, ctx);
    this.renderUX(c, ctx);
    this.renderGPU(c, ctx);
    this.renderCPU(c, ctx);
    this.renderPower(c, ctx);
  },

  /* One pass over the curve that every section shares: clean points
   * (superseded / client-limited windows excluded — they measured
   * the generator, not the engine), the LAST STABLE operating point
   * (the number the report is anchored on) and the knee. */
  analyze(c) {
    const ax = this.xAxis(c);
    const pts = [...c.curve]
      .filter(p => p.stability !== "superseded"
                && p.stability !== "client_limited")
      .sort((a, b) => (a[ax.key] ?? 0) - (b[ax.key] ?? 0));
    // The export SAYS which methodology ran; inferring it from the
    // points sent a cancelled open-loop run (one empty window) down
    // the pool-ramp template as "stable up to 0 users".
    const isOpen = c.methodology === "open_loop"
      || (c.open_loop != null && pts.some(p => p.arrival_rate_per_min != null));
    const stable = pts.filter(p =>
      isOpen ? p.stability === "stable" : p.status === "pass");
    const last = stable.length ? stable[stable.length - 1]
      : (pts.length ? pts[pts.length - 1] : null);
    const knee = pts.find(p =>
      isOpen ? p.stability === "divergent" : p.status === "fail") ?? null;
    const xOf = p => p ? `${p[ax.key]}${isOpen ? "/min" : " users"}` : "—";
    return { ax, pts, isOpen, last, knee, xOf, ol: c.open_loop };
  },

  /* Nothing measured: a run cancelled during warmup, or one whose
   * only windows were superseded / client-limited. Say that, rather
   * than driving the report template to "stable up to 0 users". */
  renderEmpty(c, ctx) {
    $("#report").hidden = true;
    const status = c.final_status;
    const cancelled = status === "interrupted" || status === "cancelled";
    const why = cancelled
      ? "the run was cancelled before its first verdict"
      : (c.curve || []).length
        ? "every window was superseded or client-limited — the load "
          + "generator, not the engine, was what got measured"
        : "no measurement window completed";
    $("#headline-stats").innerHTML = "";
    $("#headline-verdict").innerHTML =
      `<b>No measured windows</b> — ${why}.`
      + (status ? ` <span class="msg">Run status: ${status}.</span>` : "")
      + ` <span class="msg">Start it again from the Workload tab; ${
          ctx.isOpen
            ? "the first verdict arrives after warmup plus one measuring window"
            : "the first pool step has to complete"}.</span>`;
  },

  /* Bottleneck evidence → a human phrase ("KV cache at 97%, GPU DRAM
   * controllers 82% busy") so the headline says WHY, not just what. */
  bottleneckWhy(c) {
    const ev = c.bottleneck_evidence || {};
    const parts = [];
    const p = (cond, s) => { if (cond != null) parts.push(s); };
    p(ev.kv_cache_used_pct, `KV cache at ${Math.round(ev.kv_cache_used_pct)}%`);
    p(ev.gpu_sm_util_pct_avg, `GPU SM ${Math.round(ev.gpu_sm_util_pct_avg)}%`);
    if (ev.gpu_throttle_fraction > 0.05) {
      parts.push(`${Math.round(ev.gpu_throttle_fraction * 100)}% of samples throttled`);
    }
    p(ev.memory_bw_total_gb_s, `${Math.round(ev.memory_bw_total_gb_s)} GB/s DRAM`);
    if (ev.ttft_violation_rate != null && ev.tpot_violation_rate != null) {
      parts.push(ev.ttft_violation_rate > ev.tpot_violation_rate * 1.5
        ? "first-token waits break before streaming pace"
        : "streaming pace breaks alongside first-token waits");
    }
    if (ev.note) parts.push(ev.note);
    return parts.slice(0, 3).join(" · ") || "no evidence recorded";
  },

  prettyBottleneck(b) {
    return {
      kv_cache: "KV cache capacity", gpu_compute: "GPU compute",
      gpu_throttled: "GPU thermal/power throttling",
      memory_bandwidth: "memory bandwidth",
      prefill_throughput: "prefill throughput",
      decode_throughput: "decode throughput",
      frequency_droop: "CPU frequency droop",
      amx_underutilised: "AMX under-utilisation",
      none_observed: "none observed", unknown: "unknown",
    }[b] ?? b;
  },

  renderHeadline(c, ctx) {
    const { last, knee, isOpen, ol, xOf } = ctx;
    const n = v => v == null ? "—"
      : v >= 1000 ? Math.round(v).toLocaleString() : `${Math.round(v)}`;
    const stat = (k, v, hint) => `<div class="stat"><span class="k">${k}</span>
      <span class="v" style="font-size:1.3rem">${v}</span>
      <span class="c">${hint}</span></div>`;
    const sess = last
      ? (last.active_sessions_mean ?? last.pool_size) : null;
    const gen = last?.avg_in_flight;
    const box = $("#headline-stats");
    // Total sessions vs sessions actively generating: the gap is the
    // read/think population whose warm KV sits in the cache between
    // turns — the very thing that drives KV pressure on this box.
    box.innerHTML =
      stat("Active sessions", n(sess),
           "in a session at the last stable load") +
      stat("Generating now", n(gen),
           gen != null && sess
             ? `streaming from the engine — the other
                ${n(sess - gen)} are reading/thinking, their KV held warm`
             : "requests actively streaming from the engine") +
      stat("Throughput",
           last?.visible_output_tok_per_s != null
             ? `${n(last.visible_output_tok_per_s)} tok/s` : "—",
           `answers out · ${n(last?.prompt_tok_per_s)} tok/s prompts in`) +
      stat("Experience",
           last ? `${fmt.ms(last.ttft_p95_ms)} / ${fmt.ms(last.tpot_p95_ms)}`
                : "—",
           "TTFT p95 / per-token p95 at that load") +
      stat("Bottleneck", this.prettyBottleneck(c.bottleneck),
           this.bottleneckWhy(c));
    const cap = c.capacity_is_lower_bound
      ? ` These figures are <b>lower bounds</b> — the true limit was not
         reached (${c.measurement_coverage.replaceAll("_", " ")}).`
      : "";
    $("#headline-verdict").innerHTML = last == null
      ? "No usable measurements in this run."
      : (knee
        ? `Held <b>${xOf(last)}</b> in steady state; pushed to
           <b>${xOf(knee)}</b> the queue grew without bound
           (${knee.queue_depth_slope_per_min ?? "?"} requests/min) and
           ${fmt.pct(knee.violation_rate)} of turns broke SLA — that
           collapse is the capacity boundary.` + cap
        : `Stable at every load tested, up to <b>${xOf(last)}</b> —
           no collapse point observed.` + cap);
  },

  /* Shared small-chart helper for the report quads. */
  xy(id, labels, datasets, { ytitle, y2title, stacked, type = "line" } = {}) {
    this.charts[id]?.destroy();
    const el = $("#" + id);
    if (!el) return;
    const scales = {
      x: { ticks: { font: { size: 10 } } },
      y: { beginAtZero: true, stacked: !!stacked,
           ticks: { font: { size: 10 } },
           title: { display: !!ytitle, text: ytitle, font: { size: 10 } } },
    };
    if (y2title) {
      scales.y2 = { beginAtZero: true, position: "right",
        grid: { drawOnChartArea: false }, ticks: { font: { size: 10 } },
        title: { display: true, text: y2title, font: { size: 10 } } };
    }
    this.charts[id] = new Chart(el, {
      type, data: { labels, datasets },
      options: {
        maintainAspectRatio: false, animation: { duration: 300 },
        scales,
        plugins: { legend: { position: "bottom",
          labels: { boxWidth: 9, font: { size: 10 } } } },
      },
    });
  },

  ds(label, data, color, extra = {}) {
    return { label, data, borderColor: color, backgroundColor: color,
             pointRadius: 2, borderWidth: 2, spanGaps: true, ...extra };
  },

  renderUX(c, ctx) {
    const { pts, ax, last, knee, isOpen, xOf } = ctx;
    const x = pts.map(p => p[ax.key]);
    this.renderKnee(c, ctx);
    this.renderLatency(c, ctx);
    this.xy("chart-queue", x, [
      this.ds("queue waiting", pts.map(p => p.queue_depth_mean), C.fail),
      this.ds("requests in flight", pts.map(p => p.avg_in_flight), C.gold),
    ], { ytitle: "requests" });
    this.xy("chart-throughput", x, [
      this.ds("prompts in tok/s", pts.map(p => p.prompt_tok_per_s), C.blue,
        { fill: true, backgroundColor: fill(C.blue, "1c") }),
      this.ds("answers out tok/s",
        pts.map(p => p.visible_output_tok_per_s), C.teal,
        { fill: true, backgroundColor: fill(C.teal, "1c") }),
    ], { ytitle: "tokens / s" });

    $("#take-ux").textContent = last == null ? "no data" :
      `SLA-clean to ${xOf(last)} · ` + (knee
        ? `collapse at ${xOf(knee)} (TTFT p95 ${fmt.ms(knee.ttft_p95_ms)})`
        : "no collapse observed");
    $("#narr-ux").innerHTML = last == null ? "" : `
      <p>At the last stable load (<b>${xOf(last)}</b>,
      <b>${Math.round(last.active_sessions_mean ?? last.pool_size)}</b>
      concurrent users) a user waited <b>${fmt.ms(last.ttft_p50_ms)}</b>
      for the answer to start (p95 <b>${fmt.ms(last.ttft_p95_ms)}</b>)
      and tokens streamed every <b>${fmt.ms(last.tpot_p50_ms)}</b>
      (p95 ${fmt.ms(last.tpot_p95_ms)}); <b>${fmt.pct(last.violation_rate)}</b>
      of ${last.sample_size.toLocaleString()} turns broke SLA.</p>
      ${knee ? `<p>At <b>${xOf(knee)}</b> the system tipped over:
        the waiting queue grew <b>${knee.queue_depth_slope_per_min}</b>
        requests/min without bound, first-token waits stretched to
        <b>${fmt.ms(knee.ttft_p95_ms)}</b> p95 and
        <b>${fmt.pct(knee.violation_rate)}</b> of turns violated SLA.
        ${isOpen ? `Sessions arrive faster than the box completes
        them — that is the capacity boundary.` : ""}</p>`
      : `<p>No overload point was observed in the tested range — the
        capacity figures are lower bounds.</p>`}
      ${c.open_loop ? `<p class="no-data">${c.capacity_landing_zones.fast}</p>` : ""}`;
  },

  renderGPU(c, ctx) {
    const { pts, ax, last, knee, xOf } = ctx;
    const hw = p => p.hw || {};
    const kvOf = p => p ? (p.kv_cache_used_pct ?? hw(p).kv_cache_pct) : null;
    const x = pts.map(p => p[ax.key]);
    const has = pts.some(p => hw(p).gpu_sm_pct != null);
    $("#sec-gpu .quad").style.display = has ? "" : "none";
    if (!has) {
      $("#sec-gpu").open = false;
      $("#take-gpu").textContent = "no GPU telemetry on this run";
      $("#narr-gpu").innerHTML =
        `<p class="no-data">The GPU collector produced no data for this
         run (older export or CPU-only host).</p>`;
      return;
    }
    this.xy("chart-gpu-sm", x, [
      this.ds("SM util %", pts.map(p => hw(p).gpu_sm_pct), C.purple,
        { fill: true, backgroundColor: fill(C.purple, "1c") }),
      this.ds("throttled samples %",
        pts.map(p => hw(p).gpu_throttle_fraction != null
          ? hw(p).gpu_throttle_fraction * 100 : null),
        C.fail, { borderDash: [5, 4] }),
    ], { ytitle: "%" });
    this.xy("chart-gpu-mem", x, [
      this.ds("KV cache used %", pts.map(p => kvOf(p)), C.gold),
      this.ds("DRAM controllers busy %",
        pts.map(p => hw(p).gpu_mem_busy_pct), C.teal),
    ], { ytitle: "%" });
    const vramTotal = hw(last ?? pts[0]).gpu_vram_total_gb;
    this.xy("chart-gpu-vram", x, [
      this.ds("VRAM used GB", pts.map(p => hw(p).gpu_vram_gb), C.blue,
        { fill: true, backgroundColor: fill(C.blue, "1c") }),
      ...(vramTotal ? [this.ds("total", pts.map(() => vramTotal), C.muted,
        { borderDash: [4, 4], pointRadius: 0 })] : []),
    ], { ytitle: "GB" });
    this.xy("chart-gpu-power", x, [
      this.ds("power W (all GPUs)", pts.map(p => hw(p).gpu_power_w), C.gold),
      this.ds("SM clock MHz", pts.map(p => hw(p).gpu_clock_mhz), C.muted,
        { yAxisID: "y2", borderDash: [4, 4] }),
    ], { ytitle: "W", y2title: "MHz" });

    const L = hw(last ?? {});
    const K = hw(knee ?? {});
    $("#take-gpu").textContent = last == null ? "no data" :
      `SM ${Math.round(L.gpu_sm_pct ?? 0)}% · DRAM busy
       ${Math.round(L.gpu_mem_busy_pct ?? 0)}% · KV
       ${Math.round(kvOf(last) ?? 0)}% at the last stable load`;
    $("#narr-gpu").innerHTML = last == null ? "" : `
      <p>At <b>${xOf(last)}</b> the GPUs averaged
      <b>${Math.round(L.gpu_sm_pct ?? 0)}%</b> SM utilization with DRAM
      controllers <b>${Math.round(L.gpu_mem_busy_pct ?? 0)}%</b> busy —
      decode is memory-bandwidth-bound, so the DRAM line is the truer
      "how full is the box" signal. The KV cache held
      <b>${Math.round(kvOf(last) ?? 0)}%</b> of its pool and
      VRAM sat at <b>${Math.round(L.gpu_vram_gb ?? 0)}</b>${vramTotal
        ? ` of ${Math.round(vramTotal)}` : ""} GB (vLLM pre-allocates —
      capacity pressure shows in KV%, not raw VRAM).</p>
      ${knee ? `<p>At the collapse point the same gauges read SM
        <b>${Math.round(K.gpu_sm_pct ?? 0)}%</b>, DRAM
        <b>${Math.round(K.gpu_mem_busy_pct ?? 0)}%</b>, KV
        <b>${Math.round(kvOf(knee) ?? 0)}%</b> —
        whichever moved hardest with load is the resource that ran
        out.</p>` : ""}
      ${L.gpu_temp_c_max != null ? `<p>Hottest device:
        <b>${Math.round(L.gpu_temp_c_max)}°C</b>${
        (L.gpu_throttle_fraction ?? 0) > 0.05
          ? ` with <b>${Math.round(L.gpu_throttle_fraction * 100)}%</b>
             of samples throttled — cooling is shaping these numbers`
          : " — no thermal throttling observed"}.</p>` : ""}`;
  },

  renderCPU(c, ctx) {
    const { pts, ax, last, xOf } = ctx;
    const hw = p => p.hw || {};
    const x = pts.map(p => p[ax.key]);
    const has = pts.some(p => hw(p).cpu_util_pct != null);
    $("#sec-cpu .quad").style.display = has ? "" : "none";
    if (!has) {
      $("#sec-cpu").open = false;
      $("#take-cpu").textContent = "no CPU telemetry on this run";
      $("#narr-cpu").innerHTML =
        `<p class="no-data">No host telemetry in this export (older run
         or remote target).</p>`;
      return;
    }
    this.xy("chart-cpu-util", x, [
      this.ds("host CPU %", pts.map(p => hw(p).cpu_util_pct), C.teal,
        { fill: true, backgroundColor: fill(C.teal, "1c") }),
      this.ds("engine bound-set %", pts.map(p => hw(p).cpu_bound_pct),
        C.gold),
    ], { ytitle: "%" });
    const bd = k => pts.map(p => hw(p).cpu_breakdown_pct?.[k]);
    this.xy("chart-cpu-breakdown", x, [
      this.ds("user", bd("user"), C.blue,
        { fill: true, backgroundColor: fill(C.blue, "44") }),
      this.ds("system", bd("system"), C.gold,
        { fill: true, backgroundColor: fill(C.gold, "44") }),
      this.ds("iowait", bd("iowait"), C.fail,
        { fill: true, backgroundColor: fill(C.fail, "44") }),
    ], { ytitle: "% of all cycles", stacked: true });
    const cores = hw(last ?? {}).cores_util_pct || [];
    this.xy("chart-cores", cores.map((_, i) => i), [
      this.ds("core util %", cores, C.teal, { borderWidth: 0 }),
    ], { ytitle: "%", type: "bar" });
    const hasBw = pts.some(p => hw(p).mem_bw_read_gb_s != null);
    this.xy("chart-mem", x, [
      this.ds("host memory GB", pts.map(p => hw(p).mem_used_gb), C.blue),
      this.ds("engine RSS GB", pts.map(p => hw(p).engine_rss_gb), C.purple),
      ...(hasBw ? [
        this.ds("DDR read GB/s", pts.map(p => hw(p).mem_bw_read_gb_s),
          C.gold, { yAxisID: "y2", borderDash: [4, 4] }),
        this.ds("DDR write GB/s", pts.map(p => hw(p).mem_bw_write_gb_s),
          C.fail, { yAxisID: "y2", borderDash: [4, 4] }),
      ] : []),
    ], { ytitle: "GB", ...(hasBw ? { y2title: "GB/s" } : {}) });

    const L = hw(last ?? {});
    const busy = cores.filter(v => v != null && v > 50).length;
    $("#take-cpu").textContent = last == null ? "no data" :
      `${Math.round(L.cpu_util_pct ?? 0)}% host CPU ·
       ${busy}/${cores.length || "?"} cores busy ·
       ${Math.round(L.mem_used_gb ?? 0)} GB memory`;
    const bdl = L.cpu_breakdown_pct || {};
    $("#narr-cpu").innerHTML = last == null ? "" : `
      <p>At <b>${xOf(last)}</b> the host ran at
      <b>${Math.round(L.cpu_util_pct ?? 0)}%</b> CPU overall —
      <b>${busy}</b> of ${cores.length || "?"} cores above 50%. The
      cycles went <b>${bdl.user ?? "?"}%</b> to user space (the engine
      and the load generator), <b>${bdl.system ?? "?"}%</b> to the
      kernel and <b>${bdl.iowait ?? "?"}%</b> to iowait${
        (bdl.iowait ?? 0) > 5
          ? " — storage is in the request path, worth investigating"
          : " — storage is not a factor"}.</p>
      <p>Memory: <b>${Math.round(L.mem_used_gb ?? 0)} GB</b> used on the
      host, of which the engine held
      <b>${Math.round(L.engine_rss_gb ?? 0)} GB</b> RSS.
      ${hasBw ? `DDR traffic peaked at
        <b>${Math.round(Math.max(...pts.map(p =>
          (hw(p).mem_bw_read_gb_s ?? 0) + (hw(p).mem_bw_write_gb_s ?? 0))))}
        GB/s</b>.`
      : `<span class="no-data">DDR bandwidth counters were not available
        on this run (perf uncore access).</span>`}</p>`;
  },

  renderPower(c, ctx) {
    const { pts, ax, last, xOf } = ctx;
    const hw = p => p.hw || {};
    const x = pts.map(p => p[ax.key]);
    const total = p => {
      const g = hw(p).gpu_power_w, cpu = hw(p).cpu_power_w,
            sys = hw(p).system_power_w;
      if (sys != null) return sys;
      if (g == null && cpu == null) return null;
      return (g ?? 0) + (cpu ?? 0);
    };
    const hasAny = pts.some(p => total(p) != null);
    $("#sec-power .quad").style.display = hasAny ? "" : "none";
    if (!hasAny) {
      $("#sec-power").open = false;
      $("#take-power").textContent = "no power telemetry on this run";
      $("#narr-power").innerHTML = `<p class="no-data">No power data in
        this export — GPU power needs NVML, CPU package power needs
        RAPL, chassis power needs ipmitool.</p>`;
      return;
    }
    const hasSys = pts.some(p => hw(p).system_power_w != null);
    this.xy("chart-power", x, [
      this.ds("GPUs W", pts.map(p => hw(p).gpu_power_w), C.gold,
        { fill: true, backgroundColor: fill(C.gold, "1c") }),
      this.ds("CPU package W", pts.map(p => hw(p).cpu_power_w), C.teal),
      ...(hasSys ? [this.ds("chassis W (IPMI)",
        pts.map(p => hw(p).system_power_w), C.text)] : []),
      this.ds(hasSys ? "total (chassis)" : "measured total W",
        pts.map(total), C.muted, { borderDash: [5, 4] }),
    ], { ytitle: "W" });
    this.xy("chart-efficiency", x, [
      this.ds("output tokens / joule", pts.map(p => {
        const w = total(p);
        return (w && p.visible_output_tok_per_s != null)
          ? p.visible_output_tok_per_s / w : null;
      }), C.ok),
    ], { ytitle: "tok/J" });

    const lastW = total(last ?? {});
    const eff = (lastW && last?.visible_output_tok_per_s)
      ? (last.visible_output_tok_per_s / lastW).toFixed(2) : null;
    $("#take-power").textContent = lastW == null ? "no data" :
      `${Math.round(lastW)} W ${hasSys ? "chassis" : "measured"} at the
       last stable load${eff ? ` · ${eff} tok/J` : ""}`;
    $("#narr-power").innerHTML = `
      <p>At <b>${xOf(last)}</b> the ${hasSys ? "chassis drew" :
      "measured components drew"} <b>${Math.round(lastW ?? 0)} W</b>
      (GPUs <b>${Math.round(hw(last ?? {}).gpu_power_w ?? 0)} W</b>${
        hw(last ?? {}).cpu_power_w != null
          ? `, CPU package ${Math.round(hw(last ?? {}).cpu_power_w)} W`
          : ""}) — <b>${eff ?? "?"}</b> output tokens per joule.</p>
      ${hasSys ? "" : `<p class="no-data">Chassis wall power is not
        measured on this host (needs ipmitool + /dev/ipmi0 access) —
        the total shown is the sum of measured components and excludes
        fans, DIMMs, NICs and PSU losses.</p>`}`;
  },

  // Open-loop curves are indexed by arrival rate (bisection makes
  // step order non-monotonic in every other field); closed-loop by
  // pool size. One accessor keeps both chart renderers honest.
  xAxis(c) {
    const open = c.open_loop != null
      && c.curve.some(p => p.arrival_rate_per_min != null);
    return open
      ? { key: "arrival_rate_per_min", label: "session arrivals / min" }
      : { key: "pool_size", label: "pool size (concurrent sessions)" };
  },

  renderKnee(c, ctx) {
    const ax = ctx?.ax ?? this.xAxis(c);
    const curve = ctx?.pts
      ?? [...c.curve].sort((a, b) => (a[ax.key] ?? 0) - (b[ax.key] ?? 0));
    const x = curve.map(p => p[ax.key]);
    const mk = (label, key, color, extra = {}) => ({
      label, data: curve.map(p => p[key] == null ? null : p[key] * 100),
      borderColor: color, pointRadius: 3, pointBackgroundColor: color, ...extra,
    });
    this.charts.knee?.destroy();
    this.charts.knee = new Chart($("#chart-knee"), {
      type: "line",
      data: {
        labels: x,
        datasets: [
          { label: "CI upper", data: curve.map(p => p.ci_upper * 100),
            borderColor: "transparent", pointRadius: 0 },
          { label: "CI lower", data: curve.map(p => p.ci_lower * 100),
            borderColor: "transparent", pointRadius: 0,
            fill: "-1", backgroundColor: C.fail + "1f" },
          mk("SLA violation %", "violation_rate", C.fail),
          mk("target miss %", "target_miss_rate", C.warn, { borderDash: [6, 4] }),
        ],
      },
      options: {
        maintainAspectRatio: false,
        animation: { duration: 450, easing: "easeOutQuart" },
        zoneLines: c.open_loop
          ? [
              { value: c.open_loop.rate_sla_per_min, color: C.ok, label: "SLA rate" },
              { value: c.open_loop.rate_max_per_min, color: C.warn, label: "stable max" },
              { value: c.open_loop.rate_ceiling_per_min, color: C.fail, label: "collapse" },
            ]
          : [
              { value: c.capacity_pool_size, color: C.ok, label: "capacity" },
              { value: c.soft_capacity_pool_size, color: C.warn, label: "soft cap" },
              { value: c.fail_pool_size, color: C.fail, label: "fail" },
            ],
        scales: {
          x: { title: { display: true, text: ax.label } },
          y: { beginAtZero: true, title: { display: true, text: "% of turns" } },
        },
        plugins: {
          legend: { position: "bottom",
            labels: { filter: (item) => !item.text.startsWith("CI") } },
        },
        onClick: (_e, els) => {
          const el = els.find(e => e.datasetIndex >= 2);
          if (el) this.showStepDetail(curve[el.index]);
        },
      },
    });
  },

  renderLatency(c, ctx) {
    const ax = ctx?.ax ?? this.xAxis(c);
    const curve = ctx?.pts
      ?? [...c.curve].sort((a, b) => (a[ax.key] ?? 0) - (b[ax.key] ?? 0));
    this.charts.latency?.destroy();
    this.charts.latency = new Chart($("#chart-latency"), {
      type: "line",
      data: {
        labels: curve.map(p => p[ax.key]),
        datasets: [
          { label: "TTFT p50 (ms)", data: curve.map(p => p.ttft_p50_ms),
            borderColor: C.blue, pointRadius: 3, fill: true,
            backgroundColor: fill(C.blue, "1c") },
          { label: "TTFT p95 (ms)", data: curve.map(p => p.ttft_p95_ms),
            borderColor: C.blue, borderDash: [6, 4], pointRadius: 3 },
          { label: "TPOT p95 (ms)", data: curve.map(p => p.tpot_p95_ms),
            borderColor: C.warn, yAxisID: "y2", pointRadius: 3 },
        ],
      },
      options: {
        maintainAspectRatio: false,
        animation: { duration: 450, easing: "easeOutQuart" },
        scales: {
          x: { title: { display: true, text: ax.label } },
          y: { beginAtZero: true, title: { display: true, text: "TTFT ms" } },
          y2: { beginAtZero: true, position: "right",
                grid: { drawOnChartArea: false },
                title: { display: true, text: "TPOT ms" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  },

  showStepDetail(p) {
    $("#step-detail-panel").hidden = false;
    $("#step-detail-title").textContent = p.arrival_rate_per_min != null
      ? `Step ${p.step_index} — ${p.arrival_rate_per_min}/min ` +
        `(${p.stability ?? p.status})`
      : `Step ${p.step_index} — pool ${p.pool_size} (${p.status})`;
    const fields = {
      samples: p.sample_size,
      "violation rate": fmt.pct(p.violation_rate),
      "target miss": fmt.pct(p.target_miss_rate),
      "CI": `${fmt.pct(p.ci_lower)} – ${fmt.pct(p.ci_upper)}`,
      "TTFT p50/p95": `${fmt.ms(p.ttft_p50_ms)} / ${fmt.ms(p.ttft_p95_ms)}`,
      "TPOT p50/p95": `${fmt.ms(p.tpot_p50_ms)} / ${fmt.ms(p.tpot_p95_ms)}`,
      "KV cache": p.kv_cache_used_pct == null ? "—" : `${p.kv_cache_used_pct.toFixed(1)}%`,
      "output tok/s": p.visible_output_tok_per_s ?? "—",
      "prompt tok/s": p.prompt_tok_per_s ?? "—",
      "duration": `${p.measurement_duration_s ?? "—"}s`,
      "turns captured": p.turns ? p.turns.length : "(slim export)",
    };
    if (p.arrival_rate_per_min != null) {
      fields["sessions (mean)"] = p.active_sessions_mean ?? "—";
      fields["queue depth (mean)"] = p.queue_depth_mean ?? "—";
      fields["queue slope"] = p.queue_depth_slope_per_min != null
        ? `${p.queue_depth_slope_per_min}/min` : "—";
      fields["arrival tardiness p99"] = p.arrival_tardiness_p99_ms != null
        ? fmt.ms(p.arrival_tardiness_p99_ms) : "—";
      fields["load workers"] = p.load_workers ?? "—";
    }
    $("#step-detail-out").innerHTML = `<div class="kv-grid">${
      Object.entries(fields).map(([k, v]) =>
        `<div><span class="k">${k}</span><span>${v}</span></div>`).join("")
    }</div>`;
  },

  download() {
    if (!this.doc) return;
    const blob = new Blob([JSON.stringify(this.doc, null, 2)],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const entry = this.flat.find(e => e.cohort_run_id === this.openId);
    a.download = `capsim_export_${entry?.run ?? "run"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  },

  /* ── comparison (spans runs: Intel vs AMD vs GPU, config A vs B) ── */

  exportCache: {},      // run name -> export doc

  async runCompare() {
    this.compare = [];
    for (const crid of this.checked) {
      const entry = this.flat.find(e => e.cohort_run_id === crid);
      if (!entry) continue;
      let doc;
      try {
        doc = await this.loadExport(entry.run);
      } catch (e) {
        this.msg(`compare load failed: ${e.message}`, "error");
        return;
      }
      const c = doc.cohorts.find(x => x.cohort_run_id === crid);
      if (!c) continue;
      const open = c.open_loop != null;
      // Superseded / client-limited windows carry no engine verdict —
      // they'd draw misleading dips on a comparison.
      const curve = c.curve.filter(p =>
        p.stability !== "superseded" && p.stability !== "client_limited");
      this.compare.push({
        label: `${entry.run} · ${c.name || c.id} (${c.model.split("/").pop()})`,
        open, curve,
      });
    }
    if (this.compare.length < 2) return;
    $("#compare-panel").hidden = false;
    this.renderCompare();
    $("#compare-panel").scrollIntoView({ behavior: "smooth" });
  },

  renderCompare() {
    this.charts.compare?.destroy();
    if (!this.compare.length) return;
    // Shared x-axis: arrival rate when every compared run is
    // open-loop, pool size otherwise (mixing the two on one axis
    // would compare unlike quantities).
    const allOpen = this.compare.every(c => c.open);
    const key = allOpen ? "arrival_rate_per_min" : "pool_size";
    const xs = [...new Set(
      this.compare.flatMap(c => c.curve.map(p => p[key]).filter(v => v != null))
    )].sort((a, b) => a - b);
    this.charts.compare = new Chart($("#chart-compare"), {
      type: "line",
      data: {
        labels: xs,
        datasets: this.compare.map((c, i) => ({
          label: c.label,
          data: xs.map(x => {
            const p = c.curve.find(q => q[key] === x);
            return p ? p.violation_rate * 100 : null;
          }),
          borderColor: PALETTE[i % PALETTE.length],
          pointRadius: 3, spanGaps: true,
        })),
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true,
                        text: allOpen ? "session arrivals / min"
                                      : "pool size" } },
          y: { beginAtZero: true, title: { display: true, text: "SLA violation %" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  },
};
