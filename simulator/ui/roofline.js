import { $, api } from "./lib/api.js";
import { C, fill, makeChart } from "./lib/theme.js";
import { onShow } from "./lib/tabs.js";
import { Engines } from "./engines.js";

/* ── Roofline autopilot ────────────────────────────────────────────
 * The Workload tab asks how many users a deployment holds. This asks
 * what the largest sustained token rate this hardware can produce is,
 * across every model, engine and shape worth trying -- and, since a
 * box is not only its fastest model, what the LARGEST model it can
 * hold does, and what only KTransformers (experts in host RAM) can
 * serve at all. The picks come in three tiers and the results are a
 * spectrum: speed across size, with the fastest and the largest named.
 *
 * It runs for hours and nobody watches it. So this view is driven
 * ENTIRELY by polling a state document the server writes to disk after
 * every step -- no event replay, no WebSocket dependency, nothing that
 * a closed laptop or a dropped VPN can desynchronise. Reconnecting is
 * a GET. */

const TIER = {
  fast: { label: "fast", color: () => C.gold,
    title: "fastest per vendor — the vendor round-robin" },
  large: { label: "largest on GPU", color: () => C.teal,
    title: "the largest weights the GPUs hold" },
  beyond_vram: { label: "beyond VRAM", color: () => C.purple,
    title: "only KTransformers can serve it — experts in host RAM" },
};
const tierTag = (t) => TIER[t]
  ? `<span class="tier tier-${t}" title="${TIER[t].title}">${TIER[t].label}</span>`
  : "";
const short = (id) => (id || "").split("/").pop();
const num = (v, d = 0) => (v == null || !Number.isFinite(+v)) ? "—"
  : (+v).toLocaleString(undefined, { maximumFractionDigits: d });

/* Which engines a candidate gets: GPU engines when the weights fit the
 * cards, the GGUF engines (KTransformers, llama.cpp) when a GGUF
 * companion is catalogued and its allow-list names them -- the same
 * rule as roofline.engines_for on the server. */
const GGUF_ENGINES = ["ktransformers", "llamacpp"];
const enginesFor = (c, engines) => engines.filter(e =>
  GGUF_ENGINES.includes(e)
    ? c.kt_eligible && (c.gguf_engines || GGUF_ENGINES).includes(e)
    : c.fits_gpu !== false);

export const Roofline = {
  doc: null,
  candidates: [],
  chosen: new Set(),
  engines: new Set(),
  shapes: { max_num_seqs: [1024, 2048], output_tokens: [128, 256] },
  charts: {},

  init() {
    $("#rf-start").addEventListener("click", () => this.start());
    $("#rf-stop").addEventListener("click", () => this.stop());
    // Mode and tier-size changes refetch: the pick is made on the
    // server (round-robin, then largest, then beyond VRAM), so the
    // list for eight models is not the first eight of a longer one.
    for (const id of ["#rf-model-mode", "#rf-model-limit",
                      "#rf-large-limit", "#rf-beyond-limit"]) {
      $(id).addEventListener("change", () => this.refresh(true));
    }
    $("#rf-confirm").addEventListener("change", () => this.renderPlan());
    $("#rf-cross-domain").addEventListener("change", () => this.refresh(true));
    onShow("roofline", () => this.refresh(true));
    // Poll regardless of which tab is showing: a run started here keeps
    // going, and the page must be right the moment it is looked at.
    setInterval(() => this.refresh(), 5000);
    this.refresh(true);
  },

  msg(text, cls = "") {
    const el = $("#rf-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  limits() {
    return {
      model_limit: +$("#rf-model-limit").value || 8,
      large_limit: Math.max(0, +$("#rf-large-limit").value || 0),
      beyond_limit: Math.max(0, +$("#rf-beyond-limit").value || 0),
    };
  },

  async refresh(withPlan = false) {
    if (withPlan || !this.candidates.length) {
      try {
        // Pick order, not raw rank: the server fills the tiers, and
        // auto mode plans exactly what comes back tagged with a tier,
        // so this must be the list the run would use. A few unpicked
        // models follow (tier "") so the table can say why they are out.
        const mode = $("#rf-model-mode").value;
        const l = this.limits();
        const d = await api(`/api/roofline/candidates?limit=${l.model_limit}`
          + `&large_limit=${l.large_limit}&beyond_limit=${l.beyond_limit}`
          + `&extra=${mode === "manual" ? 40 : 6}&cached_only=${mode === "cached"}`
          + `&allow_cross_domain_tp=${$("#rf-cross-domain").checked}`);
        this.candidates = d.candidates || [];
        this.hardware = d.hardware;
      } catch { /* keep whatever we had */ }
      try {
        const e = await api("/api/engines");
        this.allEngines = e.available || [];
        if (!this.engines.size) this.allEngines.forEach(x => this.engines.add(x));
      } catch { /* keep whatever we had */ }
      this.renderPlan();
    }
    let doc;
    try { doc = await api("/api/roofline"); } catch { return; }
    this.doc = doc;
    this.render();
  },

  /* ── plan ─────────────────────────────────────────────────────── */

  plannedCandidates() {
    const mode = $("#rf-model-mode").value;
    if (mode === "manual") {
      // Every chosen id, whether or not the shortlist fetched it (a
      // live run's plan can name models the table does not show);
      // an unknown one is assumed to be a plain GPU model.
      const known = new Map(this.candidates.map(c => [c.id, c]));
      return [...this.chosen].map(id => known.get(id)
        || { id, fits: true, fits_gpu: true, kt_eligible: false });
    }
    return this.candidates.filter(c => c.tier
      && (mode !== "cached" || c.cached));
  },

  plannedModels() {
    return this.plannedCandidates().map(c => c.id);
  },

  /* Cells per model: the GPU engines take the whole shape grid, while
   * the GGUF engines clamp to one batch width (KTransformers' documented
   * four, llama.cpp's 32 slots) so only the output lengths vary. */
  cellCount(cands, engines) {
    const mns = this.shapes.max_num_seqs.length;
    const outs = this.shapes.output_tokens.length;
    let n = 0;
    for (const c of cands) {
      for (const e of enginesFor(c, engines)) {
        n += GGUF_ENGINES.includes(e) ? outs : mns * outs;
      }
    }
    return n;
  },

  renderPlan() {
    const cands = this.plannedCandidates();
    const models = cands.map(c => c.id);
    const engines = [...this.engines];
    const nCells = this.cellCount(cands, engines);
    const confirm = $("#rf-confirm").checked ? models.length : 0;
    const mins = Math.round(nCells * (6 + 4 * 1.5) + confirm * 18);

    $("#rf-engines").innerHTML = (this.allEngines || []).map(e =>
      `<label class="chip ${this.engines.has(e) ? "selected" : ""}"
        data-rf-engine="${e}"><input type="checkbox"
        ${this.engines.has(e) ? "checked" : ""}> ${Engines.label(e)}</label>`
    ).join("") || '<span class="msg">no engine staged — pull one in Prepare</span>';
    $("#rf-engines").querySelectorAll("[data-rf-engine]").forEach(el =>
      el.addEventListener("change", () => {
        const k = el.dataset.rfEngine;
        this.engines.has(k) ? this.engines.delete(k) : this.engines.add(k);
        this.renderPlan();
      }));

    $("#rf-shapes").innerHTML = ["max_num_seqs", "output_tokens"].map(dim =>
      `<label>${dim === "max_num_seqs" ? "Batch width" : "Output tokens"}
        <input data-rf-shape="${dim}" value="${this.shapes[dim].join(", ")}"
          style="width:130px"></label>`).join("");
    $("#rf-shapes").querySelectorAll("[data-rf-shape]").forEach(el =>
      el.addEventListener("change", () => {
        const vals = el.value.split(/[,\s]+/).map(Number)
          .filter(n => Number.isFinite(n) && n > 0);
        if (vals.length) this.shapes[el.dataset.rfShape] = vals;
        this.renderPlan();
      }));

    const hw = this.hardware || {};
    const box = hw.count
      ? `${hw.count} × ${num(hw.vram_per_gpu_gb)} GB VRAM, ${
          hw.host_ram_gb ? num(hw.host_ram_gb) + " GB host RAM" : "host RAM unknown"}`
      : "no GPUs detected";
    const tiers = { fast: 0, large: 0, beyond_vram: 0 };
    for (const c of cands) if (c.tier in tiers) tiers[c.tier]++;
    const tierLine = models.length && $("#rf-model-mode").value !== "manual"
      ? ` <span class="msg">(${tiers.fast} fast, ${tiers.large} largest on GPU,
          ${tiers.beyond_vram} beyond VRAM${
          !this.engines.has("ktransformers") && tiers.beyond_vram === 0
            ? " — beyond-VRAM models need the KTransformers engine ticked" : ""})</span>`
      : "";

    const live = !!this.doc?.live;
    const hrs = mins >= 90 ? ` (about ${(mins / 60).toFixed(1)} hours)` : "";
    if (live) {
      const planned = (this.doc.plan?.cells ?? []).length;
      const done = this.doc.summary?.attempted ?? 0;
      $("#rf-cost").innerHTML = `<b>Running now:</b> ${planned} cells,
        ${done} measured. <span class="msg">The controls above describe
        the NEXT run — changing them does not affect the one in
        flight.</span>`;
      return;
    }
    $("#rf-cost").innerHTML = nCells
      ? `<b>${nCells}</b> cells &mdash; ${models.length} model${
          models.length === 1 ? "" : "s"}${tierLine} &times; the engine${
          engines.length === 1 ? "" : "s"} each can run &times; ${
          this.shapes.max_num_seqs.length * this.shapes.output_tokens.length
        } shapes${confirm ? `, plus ${confirm} confirmation sweep${
          confirm === 1 ? "" : "s"}` : ""}. Roughly <b>${mins} minutes</b>${hrs}.
        <span class="msg">Box: ${box}. Every cell is an engine launch,
        which is where the time goes. Nothing is lost if this is
        interrupted.</span>`
      : '<span class="status-fail">nothing to run — pick at least one model and engine</span>';

    const manual = $("#rf-model-mode").value === "manual";
    $("#rf-candidates").innerHTML = `<table><thead><tr>
      ${manual ? "<th></th>" : ""}<th>Model</th><th>Tier</th><th>Series</th>
      <th>Precision</th><th>Weights</th><th>Fits</th><th>Engines</th>
      <th>KV / token</th><th>Staged</th>
      <th>Why it ranks here</th></tr></thead><tbody>
      ${this.candidates.map(c => {
        const planned = models.includes(c.id);
        const eng = enginesFor(c, engines);
        const fitsHow = !hw.vram_per_gpu_gb ? '<span class="msg">unknown here</span>'
          : c.fits_gpu !== false
            ? (c.tp > 1 ? `GPU · tp${c.tp} × ${c.replicas}` : "GPU · tp1 × 8")
            : c.fits ? "host RAM" : "no";
        return `<tr class="${planned ? "peak" : ""}" style="${
          c.fits ? "" : "opacity:.45"}">
          ${manual ? `<td><input type="checkbox" data-rf-model="${c.id}"
            ${this.chosen.has(c.id) ? "checked" : ""}
            ${c.fits ? "" : "disabled"}></td>` : ""}
          <td>${c.id}</td>
          <td>${tierTag(c.tier)}</td>
          <td>${c.series || "—"}</td><td>${c.quant || "—"}</td>
          <td>${c.size_gb ? c.size_gb + " GB" : "—"}</td>
          <td>${fitsHow}</td>
          <td>${eng.length ? eng.map(e => Engines.label(e)).join(", ")
            : '<span class="status-fail">none</span>'}</td>
          <td>${c.kv_bytes ? (c.kv_bytes / 1024).toFixed(0) + " KiB" : "—"}</td>
          <td>${c.cached ? '<span class="status-pass">yes</span>'
            : '<span class="msg">will download</span>'}</td>
          <td class="msg">${c.why}</td></tr>`;
      }).join("")}</tbody></table>`;
    $("#rf-candidates").querySelectorAll("[data-rf-model]").forEach(el =>
      el.addEventListener("change", () => {
        const id = el.dataset.rfModel;
        this.chosen.has(id) ? this.chosen.delete(id) : this.chosen.add(id);
        this.renderPlan();
      }));
  },

  async start() {
    const models = this.plannedModels();
    if (!models.length || !this.engines.size) {
      this.msg("pick at least one model and one engine", "error"); return;
    }
    const l = this.limits();
    const body = {
      workload: { kind: "roofline", spec: {
        models: $("#rf-model-mode").value === "manual" ? models : null,
        ...l,
        cached_only: $("#rf-model-mode").value === "cached",
        engines: [...this.engines],
        max_num_seqs: this.shapes.max_num_seqs,
        output_tokens: this.shapes.output_tokens,
        input_tokens: +$("#rf-input-tokens").value || 128,
        confirm_winners: $("#rf-confirm").checked,
        allow_cross_domain_tp: $("#rf-cross-domain").checked,
      } },
      new_run: false,
    };
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      this.msg("running — you can close this page; it keeps going", "ok");
      setTimeout(() => this.refresh(), 1500);
    } catch (e) { this.msg(e.message, "error"); }
  },

  async stop() {
    try {
      await api("/api/runs/stop", { method: "POST" });
      this.msg("stopping — measured cells are kept and will resume", "ok");
    } catch (e) { this.msg(e.message, "error"); }
  },

  /* ── results ──────────────────────────────────────────────────── */

  render() {
    const d = this.doc;
    if (!d || d.status === "none") return;
    const running = !!d.live;
    // While a run is live the plan panel must describe THAT run, not
    // the form's defaults. Showing "48 cells" above a progress bar
    // counting to 18 is the kind of small contradiction that makes an
    // operator distrust the whole page.
    if (d.plan?.cells && this._syncedFor !== d.started_at) {
      this._syncedFor = d.started_at;
      this.engines = new Set(d.plan.engines || []);
      this.chosen = new Set(d.plan.models || []);
      if (d.plan.shapes?.max_num_seqs?.length) {
        this.shapes = {
          max_num_seqs: d.plan.shapes.max_num_seqs,
          output_tokens: d.plan.shapes.output_tokens,
        };
      }
      $("#rf-model-mode").value = "manual";
      $("#rf-input-tokens").value = String(d.input_tokens || 128);
      this.renderPlan();
    }
    $("#rf-stop").disabled = !running;
    $("#rf-start").disabled = running;

    const sum = d.summary || {};
    const cellsTotal = (d.plan?.cells ?? []).length;
    const donePanel = $("#rf-progress-panel");
    donePanel.hidden = false;
    const cur = d.current;
    /* Settled planned cells against the plan; attempts (retries,
     * escalations, confirmations) are a separate count, so the bar
     * never reads 157%. */
    const prog = d.progress || {};
    const settled = prog.settled ?? sum.attempted ?? 0;
    const pct = cellsTotal ? Math.round(100 * settled / cellsTotal) : 0;
    $("#rf-progress").innerHTML =
      `<b>${d.status}</b>${d.note ? ` · ${d.note}` : ""}
       · <b>${settled}</b> of ${cellsTotal} cells settled
       (${pct}%)${prog.attempts != null ? ` · ${prog.attempts} attempts` : ""}${
         prog.confirmations ? ` · ${prog.confirmations} confirmed` : ""}${sum.failed?.length
         ? ` · <span class="status-marginal">${sum.failed.length} failed</span>`
         : ""}
       ${cur ? `<br><span class="msg">now: ${cur.model || ""}
         ${cur.engine ? "· " + Engines.label(cur.engine) : ""}
         ${cur.tp > 1 ? `· tp${cur.tp} × ${cur.replicas}` : ""}
         ${cur.max_num_seqs ? "· mns " + cur.max_num_seqs : ""}
         ${cur.output_tokens ? "· " + d.input_tokens + "→" + cur.output_tokens : ""}
         ${cur.phase ? "· " + cur.phase : ""}</span>` : ""}
       ${!running && !d.done
         ? `<br><span class="status-marginal">not running — this service
            is not driving it. Start again to resume from cell
            ${(sum.attempted || 0) + 1}.</span>` : ""}
       ${(d.plan?.notes || []).length
         ? `<details class="rf-notes"><summary class="msg">${
             d.plan.notes.length} plan note${d.plan.notes.length === 1 ? "" : "s"}
             — engines a model does not get</summary>
             <ul class="msg">${d.plan.notes.map(n => `<li>${n}</li>`).join("")}</ul>
           </details>` : ""}`;

    const info = d.plan?.model_info || {};
    $("#rf-staging").innerHTML = (d.models || []).map(m => `<div class="e">
      <div class="n">${m.status} ${tierTag(info[m.id]?.tier)}</div>
      <div class="s">${short(m.id)}</div>
      ${m.error ? `<div class="s status-fail">${m.error}</div>` : ""}
      </div>`).join("");

    this.renderBest(d, sum);
    this.renderMatrix(d, sum);
    this.renderSpectrum(d, sum);
    this.renderTable(d);
    this.renderCharts(sum);
  },

  /* Two winners, not one. The fastest cell is what to publish as a
   * rate; the largest model that actually served is what to publish
   * as a capability. On a box with 2 TB of RAM beside the GPUs they
   * are rarely the same model. */
  renderBest(d, sum) {
    const fast = sum.fastest || sum.best;
    const large = sum.largest_served;
    $("#rf-best-panel").hidden = !(fast || large);
    if ($("#rf-best-panel").hidden) return;
    const info = d.plan?.model_info || {};
    const fi = fast ? info[fast.model] || {} : {};
    $("#rf-fastest").innerHTML = fast ? `
      <div class="hl-label">fastest · generation tokens / sec</div>
      <div class="hl-now">${num(fast.out_tok_s)}</div>
      <div class="hl-sub"><b>${short(fast.model)}</b> ${tierTag(fi.tier)}
        on <b>${Engines.label(fast.engine)}</b>
        ${fast.tp > 1 ? `· tp${fast.tp} × ${fast.replicas}` : ""}
        · mns ${fast.max_num_seqs} · ${d.input_tokens}→${fast.output_tokens}
        ${fast.confirmed ? '· <span class="status-pass">confirmed</span>'
          : '· <span class="msg">search rung — not yet confirmed</span>'}
        ${fast.confirmed && fast.search_out_tok_s
          ? `· search peak ${num(fast.search_out_tok_s)}` : ""}</div>
      <div class="rf-kvs">
        <div class="hl-kv"><span>total tok/s (with prompt)</span><b>${
          num(fast.total_tok_s)}</b></div>
        <div class="hl-kv"><span>completed · success</span><b>${
          fast.samples != null ? `${num(fast.samples)} · ${
            fast.success_rate != null ? (fast.success_rate * 100).toFixed(1) + "%" : "—"}${
            fast.no_content ? ` <small>(+${num(fast.no_content)} reasoning-only)</small>` : ""}`
          : "—"}</b></div>
        <div class="hl-kv"><span>concurrency</span><b>${
          num(fast.concurrency ?? fast.in_flight)}</b></div>
        <div class="hl-kv"><span>tok / W</span><b>${fast.tokens_per_watt ?? "—"}</b></div>
        <div class="hl-kv"><span>KV cache</span><b>${
          fast.kv_cache_pct != null ? fast.kv_cache_pct.toFixed(0) + "%" : "—"}</b></div>
      </div>`
      : '<div class="msg">no cell has settled yet</div>';
    $("#rf-largest").innerHTML = large ? `
      <div class="hl-label">largest served · weights</div>
      <div class="hl-now">${num(large.approx_size_gb)}<small> GB</small></div>
      <div class="hl-sub"><b>${short(large.model)}</b> ${tierTag(large.tier)}
        ${large.params_b ? `· ${num(large.params_b)}B params` : ""}
        on <b>${Engines.label(large.best_engine)}</b></div>
      <div class="rf-kvs">
        <div class="hl-kv"><span>generation tok/s</span><b>${num(large.out_tok_s)}</b></div>
        <div class="hl-kv"><span>total tok/s (with prompt)</span><b>${num(large.total_tok_s)}</b></div>
        <div class="hl-kv"><span>concurrency</span><b>${num(large.concurrency)}</b></div>
        <div class="hl-kv"><span>TTFT p95</span><b>${
          large.ttft_p95_ms != null ? num(large.ttft_p95_ms) + " ms" : "—"}</b></div>
      </div>
      ${sum.largest_attempted && sum.largest_attempted.model !== large.model
        ? `<div class="msg" style="margin-top:6px">largest attempted:
            <b>${short(sum.largest_attempted.model)}</b>
            (${num(sum.largest_attempted.approx_size_gb)} GB) —
            ${sum.largest_attempted.status}</div>` : ""}`
      : '<div class="msg">nothing has served yet</div>';
    $("#rf-best-count").textContent = `${sum.measured} of ${sum.attempted} cells settled`;
  },

  /* The matrix is the finding. A single winner tells you what to
   * publish; which engine wins for which model tells you what to do
   * with the next model you try. */
  renderMatrix(d, sum) {
    const models = d.plan?.models || [];
    const engines = d.plan?.engines || [];
    $("#rf-matrix-panel").hidden = !(models.length && engines.length);
    if ($("#rf-matrix-panel").hidden) return;
    const info = d.plan?.model_info || {};
    const bestOf = {};
    for (const r of d.results || []) {
      if (!r.out_tok_s || r.steady_state === false) continue;
      const k = `${r.model}|${r.engine}`;
      if (!bestOf[k] || r.out_tok_s > bestOf[k].out_tok_s) bestOf[k] = r;
    }
    const failed = new Set((d.results || []).filter(r => r.error)
      .map(r => `${r.model}|${r.engine}`));
    // Pairings the plan never made (a GPU engine for a beyond-VRAM
    // model, KTransformers without a GGUF) are not blanks waiting to
    // be measured; they are not on the plan.
    const planned = new Set((d.plan?.cells || []).map(c => `${c.model}|${c.engine}`));
    const top = Math.max(...Object.values(bestOf).map(r => r.out_tok_s), 0);
    const rowBest = {}, colBest = {};
    for (const [k, r] of Object.entries(bestOf)) {
      const [m, e] = k.split("|");
      if (!rowBest[m] || r.out_tok_s > rowBest[m]) rowBest[m] = r.out_tok_s;
      if (!colBest[e] || r.out_tok_s > colBest[e]) colBest[e] = r.out_tok_s;
    }
    $("#rf-heatmap").innerHTML = `<table class="heatmap"><thead><tr><th></th>
      ${engines.map(e => `<th>${Engines.label(e)}</th>`).join("")}
      <th class="best">best</th></tr></thead><tbody>
      ${models.map(m => `<tr><th>${short(m)} ${tierTag(info[m]?.tier)}</th>
        ${engines.map(e => {
          const r = bestOf[`${m}|${e}`];
          if (!r) {
            if (failed.has(`${m}|${e}`)) {
              return '<td class="hm-fail" title="attempted and failed">—</td>';
            }
            return planned.size && !planned.has(`${m}|${e}`)
              ? '<td class="hm-none msg" title="not planned: this engine cannot load this model here">n/a</td>'
              : '<td class="hm-none"></td>';
          }
          const frac = top ? r.out_tok_s / top : 0;
          const win = r.out_tok_s === rowBest[m];
          return `<td class="hm ${win ? "hm-win" : ""}"
            style="--f:${frac.toFixed(3)}"
            title="mns ${r.max_num_seqs} · ${d.input_tokens}→${r.output_tokens}
              · ${Math.round(r.in_flight || 0)} streams${
              r.tp > 1 ? ` · tp${r.tp} × ${r.replicas}` : ""}${
              r.confirmed ? " · confirmed" : ""}">
            ${Math.round(r.out_tok_s).toLocaleString()}</td>`;
        }).join("")}
        <td class="best">${rowBest[m]
          ? Math.round(rowBest[m]).toLocaleString() : "—"}</td></tr>`).join("")}
      <tr><th class="best">best</th>
        ${engines.map(e => `<td class="best">${colBest[e]
          ? Math.round(colBest[e]).toLocaleString() : "—"}</td>`).join("")}
        <td class="best">${top ? Math.round(top).toLocaleString() : "—"}</td></tr>
      </tbody></table>`;
  },

  /* Speed across size: one point per model that served, weights on a
   * log axis because the spectrum runs from 14 GB to 700 GB, coloured
   * by tier. The table beneath carries every planned model, including
   * the ones that failed or never got there, with their status. */
  renderSpectrum(d, sum) {
    const rows = sum.spectrum || [];
    $("#rf-spectrum-panel").hidden = !rows.length;
    if ($("#rf-spectrum-panel").hidden) return;

    const points = rows.filter(r => r.out_tok_s && r.approx_size_gb);
    if (!this.charts.spectrum) {
      this.charts.spectrum = makeChart("#chart-rf-spectrum", {
        type: "bubble",
        datasets: Object.keys(TIER).map(t => ({
          label: TIER[t].label, data: [], tier: t,
          backgroundColor: fill(TIER[t].color(), "99"),
          borderColor: TIER[t].color(), borderWidth: 1.5,
        })),
        x: { type: "logarithmic", title: { display: true, text: "weights (GB)" },
          ticks: { callback: v => [10, 20, 50, 100, 200, 500, 1000, 2000]
            .includes(v) ? num(v) : "" } },
        y: { title: { display: true, text: "generation tokens / sec" } },
        options: { plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: {
            title: items => items.map(i => i.raw.label),
            label: i => [
              `${num(i.raw.y)} generation tok/s (${num(i.raw.total)} with prompt) on ${i.raw.engine}`,
              `${num(i.raw.x)} GB${i.raw.params ? ` · ${num(i.raw.params)}B params` : ""}`
                + ` · ${TIER[i.raw.tier]?.label || i.raw.tier}`,
              `concurrency ${num(i.raw.conc)}${
                i.raw.ttft != null ? ` · TTFT p95 ${num(i.raw.ttft)} ms` : ""}`,
            ],
          } },
        } },
      });
    }
    const c = this.charts.spectrum;
    for (const ds of c.data.datasets) {
      ds.data = points.filter(r => (r.tier || "fast") === ds.tier).map(r => ({
        x: r.approx_size_gb, y: r.out_tok_s, r: 7,
        label: short(r.model), engine: Engines.label(r.best_engine),
        total: r.total_tok_s, params: r.params_b, tier: r.tier || "fast",
        conc: r.concurrency, ttft: r.ttft_p95_ms,
      }));
    }
    c.update("none");

    const status = (r) => ({
      served: '<span class="status-pass">served</span>',
      failed: '<span class="status-fail">failed</span>',
      unavailable: '<span class="status-fail">could not stage</span>',
      pending: '<span class="msg">pending</span>',
    })[r.status] || r.status;
    const fastest = sum.fastest?.model;
    const largest = sum.largest_served?.model;
    $("#rf-spectrum tbody").innerHTML = rows.map(r => `
      <tr class="${r.model === fastest || r.model === largest ? "peak" : ""}">
        <td>${short(r.model)}${r.model === fastest ? " <b>· fastest</b>" : ""}${
          r.model === largest ? " <b>· largest served</b>" : ""}</td>
        <td>${tierTag(r.tier)}</td>
        <td>${r.vendor || "—"}</td>
        <td>${r.params_b ? num(r.params_b, 1) + "B" : "—"}</td>
        <td>${r.approx_size_gb ? num(r.approx_size_gb) + " GB" : "—"}</td>
        <td>${r.best_engine ? Engines.label(r.best_engine) : "—"}</td>
        <td><b>${num(r.out_tok_s)}</b></td>
        <td>${num(r.total_tok_s)}</td>
        <td>${num(r.concurrency)}</td>
        <td>${r.kv_capacity_tokens ? num(r.kv_capacity_tokens) : "—"}</td>
        <td>${r.ttft_p95_ms != null ? num(r.ttft_p95_ms) + " ms" : "—"}</td>
        <td>${status(r)}</td></tr>`).join("");
  },

  renderTable(d) {
    const rows = (d.results || []).slice().sort(
      (a, b) => (b.out_tok_s || 0) - (a.out_tok_s || 0));
    $("#rf-results-panel").hidden = !rows.length;
    $("#rf-results tbody").innerHTML = rows.map(r => {
      const flags = [
        r.confirmed ? '<span class="status-pass">confirmed</span>' : "",
        r.steady_state === false
          ? '<span class="status-fail" title="measured before it settled">unsettled</span>' : "",
        r.error ? `<span class="status-fail" title="${r.error}">failed</span>` : "",
      ].filter(Boolean).join(" ");
      return `<tr class="${r.confirmed ? "peak" : ""}">
        <td>${short(r.model)}</td>
        <td>${Engines.label(r.engine)}${r.tp > 1 ? ` <span class="msg">tp${r.tp} × ${r.replicas}</span>` : ""}</td>
        <td>${r.max_num_seqs}</td>
        <td>${d.input_tokens}→${r.output_tokens}</td>
        <td><b>${r.out_tok_s ? Math.round(r.out_tok_s).toLocaleString() : "—"}</b></td>
        <td>${r.in_flight ? Math.round(r.in_flight).toLocaleString() : "—"}</td>
        <td>${r.ttft_p95_ms != null ? Math.round(r.ttft_p95_ms) + " ms" : "—"}</td>
        <td>${r.tpot_p95_ms != null ? r.tpot_p95_ms.toFixed(1) + " ms" : "—"}</td>
        <td>${r.kv_cache_pct != null ? r.kv_cache_pct.toFixed(0) + "%" : "—"}</td>
        <td>${r.gpu_power_w != null ? Math.round(r.gpu_power_w) : "—"}</td>
        <td>${r.tokens_per_watt ?? "—"}</td>
        <td>${flags}</td></tr>`;
    }).join("");
  },

  renderCharts(sum) {
    const byModel = sum.best_per_model || {};
    const byEngine = sum.best_per_engine || {};
    $("#rf-charts").hidden = !(Object.keys(byModel).length
      || Object.keys(byEngine).length);
    if ($("#rf-charts").hidden) return;
    const bar = (key, canvas, entries, labelOf) => {
      const labels = Object.keys(entries).map(labelOf);
      const data = Object.values(entries).map(r => Math.round(r.out_tok_s));
      if (!this.charts[key]) {
        this.charts[key] = makeChart(canvas, {
          type: "bar",
          datasets: [{ label: "output tok/s", data: [],
            backgroundColor: fill(C.gold, "88"), borderColor: C.gold,
            borderWidth: 1 }],
          animation: false,
          x: { beginAtZero: true },
          legend: { display: false },
          options: { indexAxis: "y" },
        });
      }
      const c = this.charts[key];
      c.data.labels = labels;
      c.data.datasets[0].data = data;
      c.update("none");
    };
    bar("models", "#chart-rf-models", byModel, m => short(m));
    bar("engines", "#chart-rf-engines", byEngine, e => Engines.label(e));
  },
};
