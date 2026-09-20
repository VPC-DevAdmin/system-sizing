import { $, api } from "./lib/api.js";
import { C, fill, makeChart } from "./lib/theme.js";
import { onShow } from "./lib/tabs.js";
import { Engines } from "./engines.js";

/* ── Roofline autopilot ────────────────────────────────────────────
 * The Workload tab asks how many users a deployment holds. This asks
 * what the largest sustained token rate this hardware can produce is,
 * across every model, engine and shape worth trying.
 *
 * It runs for hours and nobody watches it. So this view is driven
 * ENTIRELY by polling a state document the server writes to disk after
 * every step -- no event replay, no WebSocket dependency, nothing that
 * a closed laptop or a dropped VPN can desynchronise. Reconnecting is
 * a GET. */

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
    $("#rf-model-mode").addEventListener("change", () => this.renderPlan());
    $("#rf-model-limit").addEventListener("input", () => this.renderPlan());
    $("#rf-confirm").addEventListener("change", () => this.renderPlan());
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

  async refresh(withPlan = false) {
    if (withPlan || !this.candidates.length) {
      try {
        const d = await api("/api/roofline/candidates?limit=14");
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

  plannedModels() {
    const mode = $("#rf-model-mode").value;
    const limit = +$("#rf-model-limit").value || 3;
    if (mode === "manual") return [...this.chosen];
    const pool = this.candidates.filter(c => c.fits
      && (mode !== "cached" || c.cached));
    return pool.slice(0, limit).map(c => c.id);
  },

  renderPlan() {
    const models = this.plannedModels();
    const engines = [...this.engines];
    const nCells = models.length * engines.length
      * this.shapes.max_num_seqs.length * this.shapes.output_tokens.length;
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
          models.length === 1 ? "" : "s"} &times; ${engines.length} engine${
          engines.length === 1 ? "" : "s"} &times; ${
          this.shapes.max_num_seqs.length * this.shapes.output_tokens.length
        } shapes${confirm ? `, plus ${confirm} confirmation sweep${
          confirm === 1 ? "" : "s"}` : ""}. Roughly <b>${mins} minutes</b>${hrs}.
        <span class="msg">Every cell is an engine launch, which is where
        the time goes. Nothing is lost if this is interrupted.</span>`
      : '<span class="status-fail">nothing to run — pick at least one model and engine</span>';

    const manual = $("#rf-model-mode").value === "manual";
    $("#rf-candidates").innerHTML = `<table><thead><tr>
      ${manual ? "<th></th>" : ""}<th>Model</th><th>Precision</th>
      <th>KV / token</th><th>Weights</th><th>Staged</th>
      <th>Why it ranks here</th></tr></thead><tbody>
      ${this.candidates.map(c => {
        const planned = models.includes(c.id);
        return `<tr class="${planned ? "peak" : ""}" style="${
          c.fits ? "" : "opacity:.45"}">
          ${manual ? `<td><input type="checkbox" data-rf-model="${c.id}"
            ${this.chosen.has(c.id) ? "checked" : ""}></td>` : ""}
          <td>${c.id}</td><td>${c.quant || "—"}</td>
          <td>${c.kv_bytes ? (c.kv_bytes / 1024).toFixed(0) + " KiB" : "—"}</td>
          <td>${c.size_gb ? c.size_gb + " GB" : "—"}</td>
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
    const body = {
      workload: { kind: "roofline", spec: {
        models: $("#rf-model-mode").value === "manual" ? models : null,
        model_limit: +$("#rf-model-limit").value || 3,
        cached_only: $("#rf-model-mode").value === "cached",
        engines: [...this.engines],
        max_num_seqs: this.shapes.max_num_seqs,
        output_tokens: this.shapes.output_tokens,
        input_tokens: +$("#rf-input-tokens").value || 128,
        confirm_winners: $("#rf-confirm").checked,
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
    const pct = cellsTotal
      ? Math.round(100 * (sum.attempted || 0) / cellsTotal) : 0;
    $("#rf-progress").innerHTML =
      `<b>${d.status}</b>${d.note ? ` · ${d.note}` : ""}
       · <b>${sum.attempted || 0}</b> of ${cellsTotal} cells
       (${pct}%)${sum.failed?.length
         ? ` · <span class="status-marginal">${sum.failed.length} failed</span>`
         : ""}
       ${cur ? `<br><span class="msg">now: ${cur.model || ""}
         ${cur.engine ? "· " + Engines.label(cur.engine) : ""}
         ${cur.max_num_seqs ? "· mns " + cur.max_num_seqs : ""}
         ${cur.output_tokens ? "· " + d.input_tokens + "→" + cur.output_tokens : ""}
         ${cur.phase ? "· " + cur.phase : ""}</span>` : ""}
       ${!running && !d.done
         ? `<br><span class="status-marginal">not running — this service
            is not driving it. Start again to resume from cell
            ${(sum.attempted || 0) + 1}.</span>` : ""}`;

    $("#rf-staging").innerHTML = (d.models || []).map(m => `<div class="e">
      <div class="n">${m.status}</div>
      <div class="s">${m.id.split("/").pop()}</div>
      ${m.error ? `<div class="s status-fail">${m.error}</div>` : ""}
      </div>`).join("");

    const best = sum.best;
    $("#rf-best-panel").hidden = !best;
    if (best) {
      $("#rf-best").textContent = Math.round(best.out_tok_s).toLocaleString();
      $("#rf-best-detail").innerHTML =
        `<b>${best.model.split("/").pop()}</b> on
         <b>${Engines.label(best.engine)}</b> · mns ${best.max_num_seqs}
         · ${d.input_tokens}→${best.output_tokens}
         ${best.confirmed ? '· <span class="status-pass">confirmed</span>'
           : '· <span class="msg">search rung — not yet confirmed</span>'}`;
      $("#rf-best-streams").textContent = best.in_flight
        ? Math.round(best.in_flight).toLocaleString() : "—";
      $("#rf-best-eff").textContent = best.tokens_per_watt ?? "—";
      $("#rf-best-kv").textContent = best.kv_cache_pct != null
        ? `${best.kv_cache_pct.toFixed(0)}%` : "—";
      $("#rf-best-count").textContent = `${sum.measured} of ${sum.attempted}`;
    }

    this.renderMatrix(d, sum);
    this.renderTable(d);
    this.renderCharts(sum);
  },

  /* The matrix is the finding. A single winner tells you what to
   * publish; which engine wins for which model tells you what to do
   * with the next model you try. */
  renderMatrix(d, sum) {
    const models = d.plan?.models || [];
    const engines = d.plan?.engines || [];
    $("#rf-matrix-panel").hidden = !(models.length && engines.length);
    if ($("#rf-matrix-panel").hidden) return;
    const bestOf = {};
    for (const r of d.results || []) {
      if (!r.out_tok_s || r.steady_state === false) continue;
      const k = `${r.model}|${r.engine}`;
      if (!bestOf[k] || r.out_tok_s > bestOf[k].out_tok_s) bestOf[k] = r;
    }
    const failed = new Set((d.results || []).filter(r => r.error)
      .map(r => `${r.model}|${r.engine}`));
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
      ${models.map(m => `<tr><th>${m.split("/").pop()}</th>
        ${engines.map(e => {
          const r = bestOf[`${m}|${e}`];
          if (!r) {
            return failed.has(`${m}|${e}`)
              ? '<td class="hm-fail" title="attempted and failed">—</td>'
              : '<td class="hm-none"></td>';
          }
          const frac = top ? r.out_tok_s / top : 0;
          const win = r.out_tok_s === rowBest[m];
          return `<td class="hm ${win ? "hm-win" : ""}"
            style="--f:${frac.toFixed(3)}"
            title="mns ${r.max_num_seqs} · ${d.input_tokens}→${r.output_tokens}
              · ${Math.round(r.in_flight || 0)} streams${
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
        <td>${r.model.split("/").pop()}</td>
        <td>${Engines.label(r.engine)}</td>
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
    bar("models", "#chart-rf-models", byModel, m => m.split("/").pop());
    bar("engines", "#chart-rf-engines", byEngine, e => Engines.label(e));
  },
};
