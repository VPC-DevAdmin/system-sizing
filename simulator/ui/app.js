/* capsim UI — no-build frontend for the control-plane service.
 *
 * Three views over the Phase 2 API:
 *   Run control — start/stop runs, doctor, run history   (/api/*)
 *   Live       — event-bus charts over /ws/telemetry
 *   Results    — knee curves, landing zones, bottleneck evidence,
 *                step drill-down, cross-run comparison
 */

const $ = (sel) => document.querySelector(sel);

const STATUS_CLASS = { pass: "status-pass", marginal: "status-marginal",
                       fail: "status-fail", ok: "status-pass" };

async function api(path, opts = {}) {
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

const fmt = {
  ms: (v) => v == null ? "—" : v >= 10000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`,
  pct: (v) => v == null ? "—" : `${(v * 100).toFixed(1)}%`,
  ts: (iso) => iso ? iso.replace("T", " ").slice(0, 19) : "—",
  clock: (ms) => new Date(ms).toLocaleTimeString("en-GB"),
};

function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[idx];
}

/* ── Chart.js theming ─────────────────────────────────────────── */

const css = getComputedStyle(document.documentElement);
const C = {
  text: css.getPropertyValue("--text").trim(),
  muted: css.getPropertyValue("--muted").trim(),
  line: css.getPropertyValue("--line").trim(),
  accent: css.getPropertyValue("--accent").trim(),
  ok: css.getPropertyValue("--ok").trim(),
  warn: css.getPropertyValue("--warn").trim(),
  fail: css.getPropertyValue("--fail").trim(),
};
Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.line;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.animation = false;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.line.borderWidth = 2;

const PALETTE = [C.accent, C.ok, C.warn, C.fail, "#b07ce8", "#5bc8c8",
                 "#e88a5c", "#8fa8ff"];

/* Vertical landing-zone markers drawn onto the knee chart. */
const zoneLinesPlugin = {
  id: "zoneLines",
  afterDatasetsDraw(chart) {
    const zones = chart.options.zoneLines || [];
    const { ctx, chartArea, scales } = chart;
    if (!scales.x) return;
    for (const z of zones) {
      if (z.value == null) continue;
      const x = scales.x.getPixelForValue(z.value);
      if (x < chartArea.left || x > chartArea.right) continue;
      ctx.save();
      ctx.strokeStyle = z.color;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = z.color;
      ctx.font = "11px sans-serif";
      ctx.fillText(z.label, x + 4, chartArea.top + 12);
      ctx.restore();
    }
  },
};
Chart.register(zoneLinesPlugin);

/* ── Tabs ─────────────────────────────────────────────────────── */

for (const btn of document.querySelectorAll("#tabs button")) {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    btn.classList.add("active");
    $(`#view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "results") Results.onShow();
  });
}

/* ══ Run control ══════════════════════════════════════════════── */

const Control = {
  catalogs: { profiles: {}, personas: [], cohorts: [] },

  async init() {
    try {
      const [profiles, personas, cohorts] = await Promise.all([
        api("/api/profiles"), api("/api/personas"), api("/api/cohorts"),
      ]);
      this.catalogs = { profiles, personas, cohorts };
      const sel = $("#profile-select");
      sel.innerHTML = "";
      for (const name of Object.keys(profiles)) {
        sel.append(new Option(name, name, false, name === "mock"));
      }
      this.fillWorkloads();
      this.refreshRuns();
    } catch (e) {
      this.msg(`catalog load failed: ${e.message}`, "error");
    }
    $("#workload-kind").addEventListener("change", () => this.fillWorkloads());
    $("#start-btn").addEventListener("click", () => this.start());
    $("#stop-btn").addEventListener("click", () => this.stop());
    $("#runs-refresh").addEventListener("click", () => this.refreshRuns());
    $("#doctor-btn").addEventListener("click", () => this.doctor());
    setInterval(() => this.pollStatus(), 2000);
    this.pollStatus();
  },

  msg(text, cls = "") {
    const el = $("#control-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  fillWorkloads() {
    const kind = $("#workload-kind").value;
    $("#workload-id-wrap").hidden = kind === "sweep";
    $("#sweep-type-wrap").hidden = kind !== "sweep";
    const sel = $("#workload-id");
    sel.innerHTML = "";
    const items = kind === "cohort" ? this.catalogs.cohorts : this.catalogs.personas;
    for (const item of items) {
      const label = item.name ? `${item.id} — ${item.name}` : item.id;
      sel.append(new Option(label, item.id));
    }
  },

  async start() {
    const kind = $("#workload-kind").value;
    const workload = kind === "sweep"
      ? { kind, type: $("#sweep-type").value.trim() || "all" }
      : { kind, id: $("#workload-id").value };
    const poolRaw = $("#pool-sizes").value.trim();
    const body = {
      profile: $("#profile-select").value,
      workload,
      new_run: $("#new-run").checked,
      adaptive: $("#adaptive").checked,
      pool_sizes: poolRaw
        ? poolRaw.split(",").map(s => parseInt(s.trim(), 10)).filter(Number.isFinite)
        : null,
    };
    try {
      this.msg("starting…");
      await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      this.msg("run started — see Live telemetry", "ok");
      this.pollStatus();
    } catch (e) {
      this.msg(e.message, "error");
    }
  },

  async stop() {
    try {
      await api("/api/runs/stop", { method: "POST" });
      this.msg("stopped", "ok");
    } catch (e) {
      this.msg(e.message, "error");
    }
    this.pollStatus();
  },

  async pollStatus() {
    let status;
    try { status = await api("/api/status"); } catch { return; }
    const active = status.active_run;
    const pill = $("#status-pill");
    const box = $("#active-run-box");
    const running = !!(active && active.running);
    $("#stop-btn").disabled = !running;
    $("#start-btn").disabled = running;
    if (running) {
      pill.textContent = "run active";
      pill.className = "pill running";
    } else if (active && active.error) {
      pill.textContent = active.error === "cancelled" ? "cancelled" : "run failed";
      pill.className = "pill error";
    } else {
      pill.textContent = "idle";
      pill.className = "pill idle";
    }
    if (active) {
      box.hidden = false;
      const w = active.workload;
      const wtxt = w.kind === "sweep" ? `sweep(${w.type})` : `${w.kind} ${w.id}`;
      const since = fmt.clock(active.started_at * 1000);
      box.innerHTML =
        `<b>${wtxt}</b> · config <b>${active.config}</b> · started ${since}` +
        (active.error ? ` · <span class="status-fail">${active.error}</span>` : "") +
        (active.result ? ` · db: <b>${active.result}</b>` : "");
    } else {
      box.hidden = true;
    }
  },

  async refreshRuns() {
    let runs;
    try { runs = await api("/api/runs"); } catch { return; }
    const tbody = $("#runs-table tbody");
    tbody.innerHTML = "";
    for (const run of runs) {
      if (!run.cohorts.length) {
        tbody.insertAdjacentHTML("beforeend",
          `<tr><td>${run.name}</td><td colspan="6" class="msg">empty</td></tr>`);
      }
      for (const c of run.cohorts) {
        const cls = STATUS_CLASS[c.final_status] ?? "status-error";
        tbody.insertAdjacentHTML("beforeend", `<tr>
          <td>${run.name}</td><td>${c.cohort_id}</td>
          <td>${c.engine_type}</td><td>${c.model_id}</td>
          <td>${c.steps}</td>
          <td class="${cls}">${c.final_status ?? "running?"}</td>
          <td>${fmt.ts(c.started_at)}</td></tr>`);
      }
    }
    Results.setRuns(runs);
  },

  async doctor() {
    const out = $("#doctor-out");
    out.textContent = "probing host…";
    try {
      const report = await api("/api/doctor");
      const rows = report.checks.map(c =>
        `<tr><td>${c.name}</td><td class="d-${c.status}">${c.status}</td>
         <td>${c.detail}</td></tr>`).join("");
      const rec = report.recommended_configs.length
        ? `<p>Recommended profiles: <b>${report.recommended_configs.join(", ")}</b></p>` : "";
      out.innerHTML = `<table><thead><tr><th>Check</th><th>Status</th>
        <th>Detail</th></tr></thead><tbody>${rows}</tbody></table>${rec}`;
    } catch (e) {
      out.innerHTML = `<span class="d-fail">doctor failed: ${e.message}</span>`;
    }
  },
};

/* ══ Live telemetry ═══════════════════════════════════════════── */

const WINDOW = 300;           // chart points (~5 min at 1 Hz)
const TURN_WINDOW = 40;       // rolling-percentile turn window

function makeLiveChart(canvas, datasets, yOpts = {}) {
  return new Chart($(canvas), {
    type: "line",
    data: { labels: [], datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { beginAtZero: true, ...yOpts },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

const Live = {
  charts: {},
  turns: [],           // recent {ttft_ms, tpot_ms, ts}
  stepsSeen: new Set(),

  init() {
    this.charts.pool = makeLiveChart("#chart-pool", [
      { label: "pool size", data: [], borderColor: C.muted, stepped: true },
      { label: "in flight", data: [], borderColor: C.accent, fill: true,
        backgroundColor: C.accent + "22" },
    ]);
    this.charts.ttft = makeLiveChart("#chart-ttft", [
      { label: "p50", data: [], borderColor: C.accent },
      { label: "p95", data: [], borderColor: C.warn },
    ]);
    this.charts.tpot = makeLiveChart("#chart-tpot", [
      { label: "p50", data: [], borderColor: C.accent },
      { label: "p95", data: [], borderColor: C.warn },
    ]);
    this.charts.host = makeLiveChart("#chart-host", [
      { label: "KV cache %", data: [], borderColor: C.accent },
      { label: "CPU bound-set %", data: [], borderColor: C.ok },
      { label: "GPU SM %", data: [], borderColor: C.warn },
    ], { suggestedMax: 100 });
    this.connect();
  },

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
    ws.onopen = () => { $("#ws-state").textContent = "live"; };
    ws.onclose = () => {
      $("#ws-state").textContent = "reconnecting…";
      setTimeout(() => this.connect(), 2000);
    };
    ws.onmessage = (m) => {
      const { topic, ts, data } = JSON.parse(m.data);
      if (topic === "snapshot") this.onSnapshot(ts, data);
      else if (topic === "turn") this.onTurn(ts, data);
      else if (topic === "telemetry") this.onTelemetry(ts, data);
      else if (topic === "step") this.onStep(data);
      else if (topic === "run") this.onRun(data);
    };
  },

  push(chart, label, values) {
    const d = chart.data;
    d.labels.push(label);
    values.forEach((v, i) => d.datasets[i].data.push(v));
    if (d.labels.length > WINDOW) {
      d.labels.shift();
      d.datasets.forEach(ds => ds.data.shift());
    }
    chart.update("none");
  },

  onSnapshot(ts, s) {
    $("#live-phase").textContent = s.phase;
    $("#live-pool").textContent = s.pool_size;
    $("#live-inflight").textContent = s.in_flight;
    $("#live-completed").textContent = s.requests_completed;
    $("#live-errors").textContent = s.errors;
    const target = s.step_target_samples || 0;
    if (target > 0) {
      $("#live-progress").textContent = `${s.step_samples} / ${target} samples`;
      $("#live-progress-bar").style.width =
        `${Math.min(100, 100 * s.step_samples / target)}%`;
    } else {
      $("#live-progress").textContent = "—";
      $("#live-progress-bar").style.width = "0";
    }
    this.push(this.charts.pool, fmt.clock(ts), [s.pool_size, s.in_flight]);
  },

  onTurn(ts, t) {
    if (t.error) return;   // synthetic failure rows skew percentiles
    this.turns.push(t);
    if (this.turns.length > TURN_WINDOW) this.turns.shift();
    const ttft = [...this.turns.map(x => x.ttft_ms)].sort((a, b) => a - b);
    const tpot = [...this.turns.map(x => x.tpot_ms)].sort((a, b) => a - b);
    const label = fmt.clock(ts);
    this.push(this.charts.ttft, label,
      [percentile(ttft, 0.5), percentile(ttft, 0.95)]);
    this.push(this.charts.tpot, label,
      [percentile(tpot, 0.5), percentile(tpot, 0.95)]);
  },

  onTelemetry(ts, t) {
    this.push(this.charts.host, fmt.clock(ts), [
      t.kv_cache_used_pct, t.cpu_util_bound_avg ?? t.cpu_util_avg,
      t.gpu_sm_util_pct,
    ]);
  },

  onStep(s) {
    const key = `${s.step_index}:${s.pool_size}`;
    if (this.stepsSeen.has(key)) return;
    this.stepsSeen.add(key);
    const cls = STATUS_CLASS[s.capacity_status] ?? "";
    $("#steps-table tbody").insertAdjacentHTML("afterbegin", `<tr>
      <td>${s.step_index}</td><td>${s.pool_size}</td><td>${s.sample_size}</td>
      <td>${fmt.pct(s.combined_violation_rate)}</td>
      <td>${fmt.pct(s.combined_target_miss_rate)}</td>
      <td>${fmt.ms(s.ttft_p95_ms)}</td><td>${fmt.ms(s.tpot_p95_ms)}</td>
      <td class="${cls}">${s.capacity_status}</td></tr>`);
  },

  onRun(r) {
    if (r.event === "started") {
      $("#steps-table tbody").innerHTML = "";
      this.stepsSeen.clear();
      this.turns = [];
    }
    if (r.event === "finished") Control.refreshRuns();
    Control.pollStatus();
  },
};

/* ══ Results ══════════════════════════════════════════════════── */

const Results = {
  runs: [],
  doc: null,            // loaded export for the selected run
  cohort: null,
  charts: {},
  compare: [],          // {label, curve}
  _shown: false,

  onShow() {
    if (!this._shown) {
      this._shown = true;
      Control.refreshRuns();
    }
  },

  init() {
    $("#result-load").addEventListener("click", () => this.load());
    $("#result-run").addEventListener("change", () => this.load());
    $("#result-cohort").addEventListener("change", () => this.render());
    $("#result-export-dl").addEventListener("click", () => this.download());
    $("#compare-add").addEventListener("click", () => this.addCompare());
    $("#compare-clear").addEventListener("click", () => {
      this.compare = [];
      this.renderCompare();
    });
  },

  setRuns(runs) {
    this.runs = runs.filter(r => r.cohorts.length);
    const sel = $("#result-run");
    const prev = sel.value;
    sel.innerHTML = "";
    for (const r of this.runs) sel.append(new Option(r.name, r.name));
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
    this.fillComparePicker();
  },

  msg(text, cls = "") {
    const el = $("#result-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async load() {
    const run = $("#result-run").value;
    if (!run) { this.msg("no runs with data yet"); return; }
    this.msg("loading export… (builds on first request)");
    try {
      this.doc = await api(`/api/runs/${run}/export`);
      this.msg(`${this.doc.meta.cohort_count} cohort(s), schema ${this.doc.schema_version}`, "ok");
    } catch (e) {
      this.msg(e.message, "error");
      return;
    }
    const sel = $("#result-cohort");
    sel.innerHTML = "";
    for (const c of this.doc.cohorts) {
      sel.append(new Option(`${c.id} (${c.engine})`, c.cohort_run_id));
    }
    $("#result-export-dl").disabled = false;
    this.render();
    this.fillComparePicker();
  },

  current() {
    const id = $("#result-cohort").value;
    return this.doc?.cohorts.find(c => c.cohort_run_id === id) ?? null;
  },

  render() {
    const c = this.current();
    if (!c) return;
    this.cohort = c;
    $("#result-summary").hidden = false;
    $("#result-charts").hidden = false;
    $("#bottleneck-panel").hidden = false;
    $("#step-detail-panel").hidden = true;

    $("#result-title").textContent =
      `${c.name || c.id} — ${c.engine} / ${c.model}`;
    const zones = [
      ["fast", c.capacity_pool_size, c.capacity_landing_zones.fast],
      ["acceptable", c.soft_capacity_pool_size, c.capacity_landing_zones.acceptable],
      ["degraded", c.fail_pool_size, c.capacity_landing_zones.degraded],
    ];
    $("#landing-zones").innerHTML = zones.map(([cls, n, t]) =>
      `<div class="zone ${cls}"><div class="n">${n ?? "—"}</div>
       <div class="t">${t}</div></div>`).join("");
    const tp = c.capacity_throughput;
    $("#throughput-line").innerHTML = tp
      ? `At the capacity pool of <b>${tp.pool_size}</b>: ` +
        `<b>${tp.visible_output_tok_per_s ?? "—"}</b> output tok/s, ` +
        `<b>${tp.prompt_tok_per_s ?? "—"}</b> prompt tok/s ` +
        `(${tp.sample_size} turns over ${tp.measurement_duration_s}s). ` +
        `Band shape: <b>${c.deployment_band_shape}</b>, ` +
        `coverage: <b>${c.measurement_coverage}</b>.`
      : `No clean-pass operating point located. Band shape: ` +
        `<b>${c.deployment_band_shape}</b>, coverage: <b>${c.measurement_coverage}</b>.`;

    this.renderKnee(c);
    this.renderLatency(c);
    this.renderBottleneck(c);
  },

  renderKnee(c) {
    const curve = [...c.curve].sort((a, b) => a.pool_size - b.pool_size);
    const x = curve.map(p => p.pool_size);
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
        zoneLines: [
          { value: c.capacity_pool_size, color: C.ok, label: "capacity" },
          { value: c.soft_capacity_pool_size, color: C.warn, label: "soft cap" },
          { value: c.fail_pool_size, color: C.fail, label: "fail" },
        ],
        scales: {
          x: { title: { display: true, text: "pool size (concurrent sessions)" } },
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

  renderLatency(c) {
    const curve = [...c.curve].sort((a, b) => a.pool_size - b.pool_size);
    this.charts.latency?.destroy();
    this.charts.latency = new Chart($("#chart-latency"), {
      type: "line",
      data: {
        labels: curve.map(p => p.pool_size),
        datasets: [
          { label: "TTFT p50 (ms)", data: curve.map(p => p.ttft_p50_ms),
            borderColor: C.accent, pointRadius: 3 },
          { label: "TTFT p95 (ms)", data: curve.map(p => p.ttft_p95_ms),
            borderColor: C.accent, borderDash: [6, 4], pointRadius: 3 },
          { label: "TPOT p95 (ms)", data: curve.map(p => p.tpot_p95_ms),
            borderColor: C.warn, yAxisID: "y2", pointRadius: 3 },
        ],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "pool size" } },
          y: { beginAtZero: true, title: { display: true, text: "TTFT ms" } },
          y2: { beginAtZero: true, position: "right",
                grid: { drawOnChartArea: false },
                title: { display: true, text: "TPOT ms" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  },

  renderBottleneck(c) {
    const kv = (obj) => Object.entries(obj || {}).map(([k, v]) =>
      `<div><span class="k">${k}</span><span>${
        typeof v === "number" ? +v.toFixed(3) : v}</span></div>`).join("");
    const collectors = c.collectors
      ? Object.entries(c.collectors).map(([k, v]) =>
          `<span class="${v === "ok" ? "d-ok" : "d-skip"}">${k}: ${v}</span>`).join("")
      : "not recorded (older run)";
    $("#bottleneck-out").innerHTML = `
      <div class="kv-grid">
        <div><span class="k">SLA-capacity bottleneck</span><b>${c.bottleneck}</b></div>
        <div><span class="k">quality bottleneck</span><b>${c.target_bottleneck}</b></div>
      </div>
      <div class="kv-grid" style="margin-top:10px">${kv(c.bottleneck_evidence)}</div>
      <div class="reco">${c.hardware_recommendation}</div>
      <div class="collectors">Evidence collectors — ${collectors}</div>`;
  },

  showStepDetail(p) {
    $("#step-detail-panel").hidden = false;
    $("#step-detail-title").textContent =
      `Step ${p.step_index} — pool ${p.pool_size} (${p.status})`;
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
    a.download = `capsim_export_${$("#result-run").value}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  },

  /* ── comparison (spans runs: Intel vs AMD vs GPU, config A vs B) ── */

  exportCache: {},      // run name -> export doc

  fillComparePicker() {
    // Every (run, cohort) pair across ALL runs — the run summaries
    // from /api/runs carry cohort ids without needing each export.
    const sel = $("#compare-pick");
    sel.innerHTML = "";
    for (const r of this.runs) {
      for (const c of r.cohorts) {
        if (c.final_status !== "ok") continue;
        sel.append(new Option(
          `${r.name} / ${c.cohort_id} (${c.engine_type})`,
          `${r.name}::${c.cohort_run_id}`,
        ));
      }
    }
  },

  async addCompare() {
    const raw = $("#compare-pick").value;
    if (!raw) return;
    const [run, cohortRunId] = raw.split("::");
    if (!this.exportCache[run]) {
      try {
        this.exportCache[run] = await api(`/api/runs/${run}/export`);
      } catch (e) {
        this.msg(`compare load failed: ${e.message}`, "error");
        return;
      }
    }
    const c = this.exportCache[run].cohorts
      .find(x => x.cohort_run_id === cohortRunId);
    if (!c) return;
    const label = `${run}/${c.id} (${c.engine})`;
    if (this.compare.some(x => x.label === label)) return;
    this.compare.push({ label, curve: [...c.curve].sort((a, b) => a.pool_size - b.pool_size) });
    this.renderCompare();
  },

  renderCompare() {
    this.charts.compare?.destroy();
    const pools = [...new Set(
      this.compare.flatMap(c => c.curve.map(p => p.pool_size))
    )].sort((a, b) => a - b);
    this.charts.compare = new Chart($("#chart-compare"), {
      type: "line",
      data: {
        labels: pools,
        datasets: this.compare.map((c, i) => ({
          label: c.label,
          data: pools.map(pool => {
            const p = c.curve.find(x => x.pool_size === pool);
            return p ? p.violation_rate * 100 : null;
          }),
          borderColor: PALETTE[i % PALETTE.length],
          pointRadius: 3, spanGaps: true,
        })),
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "pool size" } },
          y: { beginAtZero: true, title: { display: true, text: "SLA violation %" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  },
};

/* ── boot ─────────────────────────────────────────────────────── */

Control.init();
Live.init();
Results.init();
