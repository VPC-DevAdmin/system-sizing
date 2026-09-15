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
const cvar = (name) => css.getPropertyValue(name).trim();
const C = {
  text: cvar("--text"),
  muted: cvar("--muted"),
  line: cvar("--stroke"),
  accent: cvar("--accent"),
  gold: cvar("--gold"),
  teal: cvar("--teal"),
  blue: cvar("--blue"),
  purple: cvar("--purple"),
  ok: cvar("--ok"),
  warn: cvar("--warn"),
  fail: cvar("--fail"),
};
/* Soft area fill for a hex series color. */
const fill = (hex, alpha = "2e") => hex + alpha;
Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.line;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.animation = false;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.line.borderWidth = 2;

const PALETTE = [C.gold, C.teal, C.blue, C.purple, C.accent, "#e88a5c",
                 "#8fa8ff", "#5bc8c8"];

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
    await this.loadCatalogs();
    this.refreshRuns();
    $("#workload-kind").addEventListener("change", () => this.fillWorkloads());
    $("#start-btn").addEventListener("click", () => this.start());
    $("#stop-btn").addEventListener("click", () => this.stop());
    $("#runs-refresh").addEventListener("click", () => this.refreshRuns());
    $("#doctor-btn").addEventListener("click", () => this.doctor());
    setInterval(() => this.pollStatus(), 2000);
    this.pollStatus();
  },

  async loadCatalogs() {
    try {
      const [profiles, personas, cohorts] = await Promise.all([
        api("/api/profiles"), api("/api/personas"), api("/api/cohorts"),
      ]);
      this.catalogs = { profiles, personas, cohorts };
      const sel = $("#profile-select");
      const prev = sel.value;
      sel.innerHTML = "";
      for (const name of Object.keys(profiles)) {
        sel.append(new Option(name, name, false, name === (prev || "mock")));
      }
      this.fillWorkloads();
    } catch (e) {
      this.msg(`catalog load failed: ${e.message}`, "error");
    }
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
      { label: "pool size", data: [], borderColor: C.muted, stepped: true,
        borderDash: [5, 4] },
      { label: "in flight", data: [], borderColor: C.gold, fill: true,
        backgroundColor: fill(C.gold) },
    ]);
    this.charts.ttft = makeLiveChart("#chart-ttft", [
      { label: "p50", data: [], borderColor: C.blue, fill: true,
        backgroundColor: fill(C.blue, "24") },
      { label: "p95", data: [], borderColor: C.gold },
    ]);
    this.charts.tpot = makeLiveChart("#chart-tpot", [
      { label: "p50", data: [], borderColor: C.teal, fill: true,
        backgroundColor: fill(C.teal, "24") },
      { label: "p95", data: [], borderColor: C.gold },
    ]);
    this.charts.host = makeLiveChart("#chart-host", [
      { label: "KV cache %", data: [], borderColor: C.gold, fill: true,
        backgroundColor: fill(C.gold, "1f") },
      { label: "CPU bound-set %", data: [], borderColor: C.teal },
      { label: "GPU SM %", data: [], borderColor: C.purple },
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
        animation: { duration: 450, easing: "easeOutQuart" },
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

/* ══ Engine optimizer ═════════════════════════════════════════── */

const Optimizer = {
  polling: null,
  arena: null,                  // /api/arena document
  // Deselections — the arena default is EVERYTHING in play, the
  // operator subtracts. Only what's unchecked is tracked.
  arenaOff: { models: new Set(), dims: {} },
  previewTimer: null,
  budgetTouched: false,

  init() {
    $("#opt-start").addEventListener("click", () => this.start());
    $("#opt-stop").addEventListener("click", () => this.stop());
    $("#opt-budget").addEventListener("input", () => {
      this.budgetTouched = true;      // the operator took the dial
      this.schedulePreview();
    });
    document.querySelector('#tabs button[data-view="optimizer"]')
      .addEventListener("click", () => { this.refresh(); this.loadArena(); });
    this.loadArena().then(() => this.refresh());
  },

  /* ── Arena: dropdowns + cards, everything in play by default ─── */

  filters: { series: "all", size: "all" },

  async loadArena() {
    if (!this.arena) {
      try { this.arena = await api("/api/arena"); }
      catch (e) { this.msg(e.message, "error"); return; }
    }
    this.renderArena();
    this.schedulePreview();
  },

  /* The cards. Filter cards prune MODELS by an attribute; dim cards
   * prune a launch DIMENSION. Each carries a narrative so the page
   * teaches what the knob does instead of assuming vLLM fluency. */
  cards() {
    const dims = this.arena?.dimensions ?? {};
    return [
      { key: "sparsity", kind: "filter", title: "Model sparsity",
        options: [["moe", "MoE"], ["dense", "Dense"]],
        of: m => (m.moe ? "moe" : "dense"),
        text: `MoE models activate a few experts per token — big-model
          quality at small-model compute, and the reason a 30B MoE can
          outrun a dense 32B. Dense models use every parameter every
          token. Testing both answers which architecture wins on this
          box.` },
      { key: "specialty", kind: "filter", title: "Model specialty",
        options: [["instruct", "General instruct"], ["coder", "Coder"]],
        of: m => m.specialty || "instruct",
        text: `Code-tuned models share the base model's launch behavior
          but serve different workloads. Keep coder variants when the
          production traffic includes coding assistants.` },
      { key: "quant", kind: "filter", title: "Weight precision",
        options: (this.arena?.models ?? []).filter(m => m.feasible)
          .map(m => m.quant).filter((v, i, s) => s.indexOf(v) === i)
          .map(q => [q, q]),
        of: m => m.quant,
        text: `Precision is a different artifact, not a flag: FP8 weights
          halve memory and bandwidth for a small quality cost, which can
          double the replicas that fit. bf16 is the reference.` },
      { key: "tp", kind: "dim", title: "Tensor parallelism",
        options: (dims.tp ?? []).map(v => [String(v), String(v)]),
        text: `How many GPUs share ONE replica's weights. Required when a
          model doesn't fit one card; costs an all-reduce per layer over
          PCIe on this box. TP peers never span PCIe/NUMA domains.` },
      { key: "dp", kind: "dim", title: "Data parallelism",
        options: (dims.dp ?? []).map(v => [String(v), String(v)]),
        text: `Independent replicas behind round-robin. No inter-GPU
          chatter — usually the throughput winner when the model fits a
          single card. tp × dp is capped by the ${this.arena?.hardware?.count ?? "?"}
          GPUs.` },
      { key: "max_num_seqs", kind: "dim", title: "Batch width",
        options: (dims.max_num_seqs ?? []).map(v => [String(v), String(v)]),
        text: `Max concurrent sequences per replica. Wider batches raise
          throughput until they poison per-token latency — the SLA caps
          decide where that line is.` },
      { key: "max_num_batched_tokens", kind: "dim", title: "Prefill chunk",
        options: (dims.max_num_batched_tokens ?? []).map(v => [String(v), String(v)]),
        text: `Tokens the scheduler may batch per step. Smaller chunks
          keep decode latency steady while long prompts prefill;
          "default" lets vLLM choose.` },
      { key: "kv_cache_dtype", kind: "dim", title: "KV cache precision",
        options: (dims.kv_cache_dtype ?? []).map(v => [String(v), String(v)]),
        text: `FP8 KV cache halves cache memory and bandwidth — roughly
          double the concurrent context — for a small accuracy cost.
          One of the highest-leverage knobs on Blackwell.` },
      { key: "expert_parallel", kind: "dim", title: "Expert parallelism",
        options: (dims.expert_parallel ?? []).map(v => [String(v), String(v)]),
        text: `Splits an MoE model's experts across the TP group instead
          of sharding every expert. Only meaningful for MoE models at
          tp > 1 — other candidates ignore it automatically.` },
      { key: "placement", kind: "dim", title: "GPU placement",
        options: (dims.placement ?? []).map(v => [String(v), String(v)]),
        text: `pack keeps TP peers inside one PCIe/NUMA domain (fast
          peer transfers); spread deals replicas across domains
          (balanced host bandwidth).` },
    ].filter(c => c.options.length > 1);
  },

  cardOff: {},          // card key -> Set of deselected option values

  eligibleModels() {
    const f = this.filters;
    return (this.arena?.models ?? []).filter(m => m.feasible
      && (f.series === "all" || m.series === f.series)
      && (f.size === "all" || m.size_class === f.size)
      && !(this.cardOff.sparsity?.has(m.moe ? "moe" : "dense"))
      && !(this.cardOff.specialty?.has(m.specialty || "instruct"))
      && !(this.cardOff.quant?.has(m.quant)));
  },

  arenaSelection() {
    const models = this.eligibleModels()
      .filter(m => !this.arenaOff.models.has(m.id)).map(m => m.id);
    const dims = {};
    for (const [dim, vals] of Object.entries(this.arena.dimensions)) {
      const off = this.cardOff[dim];
      if (off?.size) dims[dim] = vals.filter(v => !off.has(String(v)));
    }
    return { models, dims };
  },

  priorDoc: null,        // arena_space of the last search (resume target)

  /* Apply the last search's selection to the cards — ONLY on the
   * operator's explicit click (the banner's button). The default view
   * is always the full arena with everything checked; the UI never
   * unchecks anything on its own. */
  loadPriorSelection() {
    const doc = this.priorDoc;
    if (!doc || !this.arena) return;
    this.filters = { series: "all", size: "all" };
    this.arenaOff.models = new Set();
    this.cardOff = {};
    const wanted = new Set(Object.values(doc.model_variants ?? {})
      .map(v => v.model));
    for (const m of this.arena.models) {
      if (m.feasible && !wanted.has(m.id)) this.arenaOff.models.add(m.id);
    }
    for (const [dim, vals] of Object.entries(this.arena.dimensions)) {
      const kept = (doc.dimensions ?? {})[dim];
      if (!kept) continue;
      const keptSet = new Set(kept.map(String));
      const off = vals.map(String).filter(v => !keptSet.has(v));
      if (off.length) this.cardOff[dim] = new Set(off);
    }
    $("#opt-new-run").checked = false;
    this.renderArena();
    this.schedulePreview();
    this.msg("previous search's selection loaded — Start resumes it", "ok");
  },

  /* Does the current selection match the resumable search's space?
   * A resume with a different selection would immediately end
   * done:space_changed, so Start falls back to a fresh run instead. */
  selectionMatchesPrior() {
    const doc = this.priorDoc;
    if (!doc) return false;
    const sel = this.arenaSelection();
    const prior = new Set(Object.values(doc.model_variants ?? {})
      .map(v => v.model));
    if (sel.models.length !== prior.size
        || !sel.models.every(m => prior.has(m))) return false;
    for (const [dim, vals] of Object.entries(this.arena.dimensions)) {
      const kept = ((doc.dimensions ?? {})[dim] ?? vals).map(String);
      const cur = (sel.dims[dim] ?? vals).map(String);
      if (kept.length !== cur.length
          || !cur.every(v => kept.includes(v))) return false;
    }
    return true;
  },

  renderArena() {
    const a = this.arena;
    const box = $("#opt-arena");
    if (!a) { box.innerHTML = ""; return; }
    const hw = a.hardware;
    if (!hw.count) {
      box.innerHTML = `<div class="callout">No GPUs detected on this host —
        the arena needs a GPU box (or <code>device_groups</code> in
        <code>config/arena.yaml</code> for planning).</div>`;
      return;
    }
    const uniq = arr => [...new Set(arr)];
    const seriesOpts = uniq(a.models.filter(m => m.feasible).map(m => m.series))
      .sort();
    const sizeOpts = uniq(a.models.filter(m => m.feasible).map(m => m.size_class));
    const eligible = this.eligibleModels();
    const inPlay = eligible.filter(m => !this.arenaOff.models.has(m.id));

    const cardHtml = c => {
      const off = this.cardOff[c.key] ?? new Set();
      const on = c.options.filter(([v]) => !off.has(v));
      const summary = on.length === c.options.length
        ? `All: ${on.map(([, l]) => l).join(" · ")}`
        : on.length ? on.map(([, l]) => l).join(" · ")
        : '<span class="status-fail">nothing selected</span>';
      return `<div class="arena-card" data-card="${c.key}">
        <div class="ac-head"><span class="ac-title">${c.title}</span>
          <button class="ac-edit" data-edit="${c.key}" title="edit">✎ edit</button></div>
        <div class="ac-sel">${summary}</div>
        <div class="ac-text msg">${c.text}</div>
        <div class="card-pop" data-pop="${c.key}" hidden>
          ${c.options.map(([v, l]) => `<label class="pop-row">
            <input type="checkbox" data-card-opt="${c.key}" data-val="${v}"
              ${off.has(v) ? "" : "checked"}> ${l}</label>`).join("")}
        </div></div>`;
    };

    const modelsCard = `<div class="arena-card" data-card="models">
      <div class="ac-head"><span class="ac-title">Models in play</span>
        <button class="ac-edit" data-edit="models" title="edit">✎ edit</button></div>
      <div class="ac-sel">${inPlay.length} of ${eligible.length} eligible
        ${eligible.length < a.models.length
          ? `<span class="msg">(${a.models.length - eligible.length} hidden by
             the family/size dropdowns or filter cards, or won't fit)</span>`
          : ""}</div>
      <div class="ac-text msg">${inPlay.map(m =>
        `${m.id.split("/")[1]} <i>(${m.quant})</i>`).join(", ") || "—"}</div>
      <div class="card-pop" data-pop="models" hidden>
        <div class="pop-row" style="gap:10px">
          <button class="small" data-models-all="on">Select all</button>
          <button class="small" data-models-all="off">None</button>
        </div>
        ${eligible.map(m => `<label class="pop-row">
          <input type="checkbox" data-model-opt="${m.id}"
            ${this.arenaOff.models.has(m.id) ? "" : "checked"}>
          ${m.id} <i>(${m.quant} · ~${m.approx_size_gb ?? "?"} GB
          · tp ${m.feasible_tps.join("/")})</i></label>`).join("")}
      </div></div>`;

    box.innerHTML = `
      <div class="row wrap" style="margin-top:14px">
        <label>Model family
          <select id="arena-series">
            <option value="all">All families</option>
            ${seriesOpts.map(s => `<option value="${s}"
              ${this.filters.series === s ? "selected" : ""}>${s}</option>`).join("")}
          </select></label>
        <label>Model size
          <select id="arena-size">
            <option value="all">All sizes</option>
            ${sizeOpts.map(s => `<option value="${s}"
              ${this.filters.size === s ? "selected" : ""}>${s}</option>`).join("")}
          </select></label>
        <span class="msg" style="align-self:end">${hw.count} GPUs ·
          ${hw.vram_per_gpu_gb ?? "?"} GB each · ${hw.device_groups.length}
          PCIe/NUMA domain(s) · gmu fixed ${a.fixed.gpu_memory_utilization}</span>
      </div>
      <div class="arena-cards">${modelsCard}${this.cards().map(cardHtml).join("")}</div>
      <div id="opt-arena-cost" class="callout arena-cost">computing the arena…</div>`;

    // A dropdown change re-baselines the model set: everything the
    // new filter makes eligible starts CHECKED — per-model unchecks
    // belong to the operator and only survive within one baseline.
    $("#arena-series").addEventListener("change", e => {
      this.filters.series = e.target.value;
      this.arenaOff.models = new Set();
      this.renderArena(); this.schedulePreview();
    });
    $("#arena-size").addEventListener("change", e => {
      this.filters.size = e.target.value;
      this.arenaOff.models = new Set();
      this.renderArena(); this.schedulePreview();
    });
    box.querySelectorAll("[data-edit]").forEach(btn =>
      btn.addEventListener("click", e => {
        e.stopPropagation();
        const pop = box.querySelector(`[data-pop="${btn.dataset.edit}"]`);
        const wasHidden = pop.hidden;
        box.querySelectorAll(".card-pop").forEach(p => { p.hidden = true; });
        pop.hidden = !wasHidden;
      }));
    box.querySelectorAll(".card-pop").forEach(p =>
      p.addEventListener("click", e => e.stopPropagation()));
    if (!this.popCloser) {
      this.popCloser = true;
      document.addEventListener("click", () =>
        document.querySelectorAll("#opt-arena .card-pop")
          .forEach(p => { p.hidden = true; }));
    }
    const FILTER_KEYS = new Set(["sparsity", "specialty", "quant"]);
    box.querySelectorAll("input[data-card-opt]").forEach(el =>
      el.addEventListener("change", () => {
        const key = el.dataset.cardOpt;
        const off = this.cardOff[key] ??= new Set();
        el.checked ? off.delete(el.dataset.val) : off.add(el.dataset.val);
        if (FILTER_KEYS.has(key)) {
          // Filters change WHICH models are eligible — rebuild the
          // cards, then reopen this popover where the user left it.
          this.renderArena();
          const pop = box.querySelector(`[data-pop="${key}"]`);
          if (pop) pop.hidden = false;
        } else {
          this.refreshCardSummaries();
        }
        this.schedulePreview();
      }));
    box.querySelectorAll("input[data-model-opt]").forEach(el =>
      el.addEventListener("change", () => {
        el.checked ? this.arenaOff.models.delete(el.dataset.modelOpt)
                   : this.arenaOff.models.add(el.dataset.modelOpt);
        this.refreshCardSummaries();
        this.schedulePreview();
      }));
    box.querySelectorAll("button[data-models-all]").forEach(btn =>
      btn.addEventListener("click", e => {
        e.stopPropagation();
        const on = btn.dataset.modelsAll === "on";
        for (const m of this.eligibleModels()) {
          on ? this.arenaOff.models.delete(m.id)
             : this.arenaOff.models.add(m.id);
        }
        this.renderArena();
        const pop = box.querySelector('[data-pop="models"]');
        if (pop) pop.hidden = false;
        this.schedulePreview();
      }));
  },

  /* Update card summary lines in place so an open popover survives
   * checkbox clicks (a full re-render would close it). */
  refreshCardSummaries() {
    const box = $("#opt-arena");
    for (const c of this.cards()) {
      const off = this.cardOff[c.key] ?? new Set();
      const on = c.options.filter(([v]) => !off.has(v));
      const el = box.querySelector(`[data-card="${c.key}"] .ac-sel`);
      if (el) el.innerHTML = on.length === c.options.length
        ? `All: ${on.map(([, l]) => l).join(" · ")}`
        : on.length ? on.map(([, l]) => l).join(" · ")
        : '<span class="status-fail">nothing selected</span>';
    }
    const eligible = this.eligibleModels();
    const inPlay = eligible.filter(m => !this.arenaOff.models.has(m.id));
    const mEl = box.querySelector('[data-card="models"] .ac-sel');
    if (mEl) mEl.textContent = `${inPlay.length} of ${eligible.length} eligible`;
  },

  schedulePreview() {
    clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => this.preview(), 350);
  },

  async preview() {
    const out = $("#opt-arena-cost");
    if (!out || !this.arena?.hardware?.count) return;
    let p;
    try {
      p = await api("/api/arena/preview", {
        method: "POST",
        body: JSON.stringify({ mode: "arena", arena: this.arenaSelection(),
                               budget: +$("#opt-budget").value || null }),
      });
    } catch (e) {
      out.innerHTML = `<span class="status-fail">${e.message}</span>`;
      return;
    }
    // Recommended coverage for THIS arena, from the design-of-
    // experiments floor (every knob value measured ≥ once, padded,
    // plus a refinement allowance). The input starts at the
    // recommendation and re-tracks it as the arena changes, until
    // the operator dials it themselves.
    const rec = p.recommendation;
    const input = $("#opt-budget");
    if (rec && !this.budgetTouched
        && +input.value !== rec.recommended) {
      input.value = rec.recommended;
      this.schedulePreview();      // re-quote hours at the new budget
      return;
    }
    const presets = $("#opt-budget-presets");
    if (presets && rec) {
      const label = { screening: "Screening", recommended: "Recommended",
                      thorough: "Thorough" };
      presets.innerHTML = rec.tiers.map(t =>
        `<button class="small ${+input.value === t.budget ? "primary" : ""}"
           data-budget-tier="${t.budget}">${label[t.name]}
           ${t.budget} · ~${t.hours}h</button>`).join(" ");
      presets.querySelectorAll("button[data-budget-tier]").forEach(btn =>
        btn.addEventListener("click", () => {
          input.value = btn.dataset.budgetTier;
          this.budgetTouched = true;
          this.schedulePreview();
        }));
    }
    const ctx = $("#opt-budget-context");
    if (ctx) {
      const total = p.total_combinations || 1;
      const pct = (Math.min(p.budget, total) / total) * 100;
      const pctText = pct >= 99.5 ? "the whole arena"
        : `${pct < 1 ? "<1" : pct.toFixed(pct < 10 ? 1 : 0)}% of
           ${total.toLocaleString()} combinations`;
      ctx.innerHTML = `= ${pctText} · ~${p.estimated_hours} h`;
    }
    const budgetHint = rec ? `<span class="msg">Sizing: the floor for this
      arena is ${rec.ofat_min} evaluations (every value of all
      ${rec.dimensions} dimensions measured at least once);
      ${rec.recommended} adds interaction coverage plus
      ${rec.refinement_stage} refinement runs around the leaders.
      <b>Dialing down</b> toward ${rec.screening} screens faster but risks
      missing effects that only appear in combination (e.g. FP8 KV paying
      off only at wide batch). <b>Dialing up</b> past ${rec.thorough} mostly
      buys fine adjustment — the search stops itself once a round improves
      the best by &lt;3%.</span><br>` : "";
    out.innerHTML = `<b>${p.launch_shapes}</b> feasible launch shapes ·
      <b>${p.total_combinations.toLocaleString()}</b> total combinations with
      batch/KV knobs. The search MEASURES <b>${p.budget}</b> of them — each
      one is a real engine launch + ladder climb — covering the space first,
      then refining around the leaders. Rough wall clock:
      <b>~${p.estimated_hours} h</b>.<br>
      ${budgetHint}
      <span class="msg">Scoring: each candidate climbs a concurrency ladder
      (${(p.ladder ?? []).join(" → ")}) at
      ${(p.measurement_tokens ?? []).join("in/")}out tokens, stopping when the
      SLA breaks — scored at its own best SLA-passing point, so wide DP shapes
      aren't judged under-saturated.</span><br>
      <span class="msg">Cost model: every evaluation restarts the engine
      (~${p.estimated_engine_restarts} restarts), but batches are ordered by
      model so cold weight loads stay near ${p.estimated_cold_weight_loads}
      (~one per model per iteration) — the rest relaunch on hot weights.</span>`;
  },

  msg(text, cls = "") {
    const el = $("#opt-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async refresh() {
    let status;
    try {
      status = await api("/api/optimizer");
    } catch (e) {
      this.msg(e.message, "error");
      return;
    }
    $("#opt-start").disabled = status.running;
    $("#opt-stop").disabled = !status.running;
    if (status.running) {
      this.msg(`running (${status.active.profile}${status.active.external
        ? " — attached to in-flight run" : ""}) — log: ${status.active.log}`);
      if (!this.polling) {
        this.polling = setInterval(() => this.refresh(), 4000);
      }
    } else {
      if (this.polling) { clearInterval(this.polling); this.polling = null; }
      if (status.active && status.active.exit_code !== null) {
        this.msg(
          status.active.exit_code === 0
            ? "optimizer finished" : `optimizer exited (${status.active.exit_code})`,
          status.active.exit_code === 0 ? "ok" : "error");
      }
    }
    // Prior search on disk: offer the resume, never rearrange the
    // operator's cards on our own.
    this.priorDoc = status.arena_space ?? null;
    const resume = $("#opt-resume");
    const s = status.search_results?.summary;
    if (s && !status.running) {
      resume.hidden = false;
      const done = s.done_reason
        ? `finished (${s.done_reason})` : "interrupted mid-search";
      const prior = this.priorDoc
        ? `${Object.keys(this.priorDoc.model_variants ?? {}).length} models`
        : "";
      resume.innerHTML = `A prior search is on record — <b>${s.evaluated}
        evaluated</b>, ${done}${s.best
          ? `, best so far <b>${s.best.score.toFixed(0)}</b>` : ""}${prior
          ? ` (${prior})` : ""}.
        <button class="small" id="opt-load-prior" style="margin:0 6px">
          Load its selection &amp; resume</button>
        Starting with a different selection runs fresh automatically.`;
      $("#opt-load-prior")?.addEventListener("click", () =>
        this.loadPriorSelection());
    } else {
      resume.hidden = true;
    }
    // Show the CURRENT work, not everything ever written: while an
    // optimizer runs, only its own mode's panel; otherwise whichever
    // result file is newer wins and the stale one hides.
    const sr = status.search_results;
    const rr = status.results;
    const activeMode = status.running ? status.active?.mode : null;
    const searchNewer = !!sr?.generated_at
      && (!rr?.generated_at || sr.generated_at > rr.generated_at);
    const showRegistry = activeMode
      ? activeMode === "registry" : !searchNewer;
    const showSearch = activeMode
      ? activeMode !== "registry" : searchNewer;
    this.renderResults(showRegistry ? rr : null);
    this.renderSearch(showSearch ? sr : null);
  },

  renderSearch(doc) {
    const panel = $("#opt-search-panel");
    if (!doc || !doc.summary) { panel.hidden = true; return; }
    panel.hidden = false;
    const s = doc.summary;
    $("#opt-search-title").textContent =
      `Guided search — ${doc.space} · ${s.evaluated} evaluated` +
      (s.done_reason ? ` · stopped: ${s.done_reason}` : " · in progress");
    const prog = (s.iterations ?? []).map(it =>
      `<div class="opt-rank-card"><span class="n">iter ${it.iteration}</span>
       <span class="s"> ${it.kind}, ${it.candidates} cand · best ${
         it.best_score_so_far == null ? "—" : it.best_score_so_far.toFixed(0)}</span></div>`
    ).join("");
    // "delivers N tok/s at concurrency C inside SLA" — the rung the
    // score was demonstrated at, when the summary carries it.
    const rungText = r => {
      if (!r) return "";
      const c = (r.cell_name || "").replace("ladder_c", "").replace(/^0+/, "");
      return ` — ${(r.throughput_out_tok_s ?? 0).toFixed(0)} tok/s at
        concurrency ${c || "?"} ${r.sla_ok ? "inside SLA"
          : '<span class="status-marginal">over SLA caps</span>'}`;
    };
    const best = s.best ? `
      <div class="opt-rank-card winner"><span class="n">BEST ${
        s.best.score.toFixed(0)}</span>
       <span class="s"> ${s.best.key.replaceAll("|", " · ")}${
         rungText(s.best.best_rung)}</span>
       <button class="small primary" id="opt-promote-search"
         style="margin-left:12px">Save as optimized launch</button></div>` : "";
    $("#opt-search-best").innerHTML = best + prog;
    $("#opt-promote-search")?.addEventListener("click", () =>
      this.promote("search"));
    const tbody = $("#opt-search-table tbody");
    tbody.innerHTML = "";
    (s.top ?? []).forEach((e, i) => {
      const r = e.best_rung;
      const at = r ? `${(r.throughput_out_tok_s ?? 0).toFixed(0)} tok/s @ c${
        (r.cell_name || "").replace("ladder_c", "").replace(/^0+/, "")}${
        r.sla_ok ? "" : " (over SLA)"}` : "—";
      tbody.insertAdjacentHTML("beforeend", `<tr>
        <td>${i + 1}</td><td>${e.iteration}</td>
        <td>${e.key.replaceAll("|", " · ")}</td>
        <td>${e.score == null ? "—" : e.score.toFixed(1)}</td>
        <td>${at}</td></tr>`);
    });
    const failed = s.failed ?? 0;
    if (failed) {
      tbody.insertAdjacentHTML("beforeend",
        `<tr><td colspan="5" class="msg">${failed} candidate(s) failed to launch
         (see the optimizer log for reasons)</td></tr>`);
    }
  },

  async start() {
    const sel = this.arenaSelection();
    if (!sel.models.length) { this.msg("every model is unchecked", "error"); return; }
    let newRun = $("#opt-new-run").checked;
    // A resume only makes sense against the SAME selection — with a
    // different one the driver would refuse (space_changed). Fall
    // back to a fresh run rather than a dead start.
    if (!newRun && this.priorDoc && !this.selectionMatchesPrior()) {
      newRun = true;
      this.msg("selection differs from the prior search — starting fresh", "ok");
    }
    const body = { mode: "arena", arena: sel,
                   budget: +$("#opt-budget").value || null,
                   new_run: newRun };
    try {
      await api("/api/optimizer/start", {
        method: "POST",
        body: JSON.stringify(body),
      });
      this.msg("started", "ok");
      this.refresh();
    } catch (e) {
      this.msg(e.message, "error");
    }
  },

  async stop() {
    try {
      await api("/api/optimizer/stop", { method: "POST" });
      this.msg("stopped", "ok");
    } catch (e) {
      this.msg(e.message, "error");
    }
    this.refresh();
  },

  /* Composite ranking: for each cell, rank ok-configs by TTFT p95
   * (asc) and by throughput (desc); a config's score is the mean of
   * all its ranks. Lower = better. Explainable, no magic weights. */
  rank(configs) {
    const ok = configs.filter(c => c.status === "ok" && c.cells.length);
    const scores = new Map(ok.map(c => [c.name, []]));
    const cells = [...new Set(ok.flatMap(c => c.cells.map(x => x.cell_name)))];
    for (const cell of cells) {
      const rows = ok
        .map(c => ({ name: c.name, r: c.cells.find(x => x.cell_name === cell) }))
        .filter(x => x.r);
      for (const [key, dir] of [["ttft_p95_ms", 1], ["throughput_out_tok_s", -1]]) {
        const ranked = [...rows]
          .filter(x => x.r[key] != null)
          .sort((a, b) => dir * (a.r[key] - b.r[key]));
        ranked.forEach((x, i) => scores.get(x.name).push(i + 1));
      }
    }
    return ok
      .map(c => ({
        name: c.name,
        score: scores.get(c.name).length
          ? scores.get(c.name).reduce((a, b) => a + b, 0) / scores.get(c.name).length
          : Infinity,
      }))
      .sort((a, b) => a.score - b.score);
  },

  renderResults(results) {
    const panel = $("#opt-results-panel");
    if (!results || !results.configs?.length) { panel.hidden = true; return; }
    panel.hidden = false;
    $("#opt-results-title").textContent =
      `Results — ${results.profile} · ${results.model} · ${results.generated_at?.slice(0, 19) ?? ""}`;

    const ranking = this.rank(results.configs);
    $("#opt-ranking").innerHTML = ranking.map((r, i) => `
      <div class="opt-rank-card ${i === 0 ? "winner" : ""}">
        <span class="n">#${i + 1} ${r.name}</span>
        <span class="s"> mean rank ${r.score === Infinity ? "—" : r.score.toFixed(1)}</span>
        ${i === 0 ? `<button class="small primary" id="opt-promote-registry"
          data-cfg="${r.name}" style="margin-left:12px">Save as optimized launch</button>` : ""}
      </div>`).join("")
      + results.configs.filter(c => c.status !== "ok").map(c => `
      <div class="opt-rank-card">
        <span class="n">${c.name}</span>
        <span class="s status-fail"> ${c.status}${c.failure_reason ? ": " + c.failure_reason.slice(0, 80) : ""}</span>
      </div>`).join("");

    // Per-cell best highlighting on TTFT p95 and throughput.
    const best = {};
    for (const c of results.configs) {
      for (const r of c.cells ?? []) {
        const b = best[r.cell_name] ??= {};
        if (r.ttft_p95_ms != null && (b.ttft == null || r.ttft_p95_ms < b.ttft)) b.ttft = r.ttft_p95_ms;
        if (r.throughput_out_tok_s != null && (b.tps == null || r.throughput_out_tok_s > b.tps)) b.tps = r.throughput_out_tok_s;
      }
    }
    const tbody = $("#opt-results tbody");
    tbody.innerHTML = "";
    for (const c of results.configs) {
      (c.cells ?? []).forEach((r, i) => {
        const b = best[r.cell_name] ?? {};
        tbody.insertAdjacentHTML("beforeend", `<tr class="${i === 0 ? "cfg-first" : ""}">
          <td>${i === 0 ? c.name : ""}</td><td>${r.cell_name}</td>
          <td>${r.samples}</td><td>${r.errors || ""}</td><td>${r.timeouts || ""}</td>
          <td>${fmt.ms(r.ttft_p50_ms)}</td>
          <td class="${r.ttft_p95_ms === b.ttft ? "best" : ""}">${fmt.ms(r.ttft_p95_ms)}</td>
          <td>${fmt.ms(r.tpot_p50_ms)}</td><td>${fmt.ms(r.tpot_p95_ms)}</td>
          <td class="${r.throughput_out_tok_s === b.tps ? "best" : ""}">${
            r.throughput_out_tok_s == null ? "—" : r.throughput_out_tok_s.toFixed(1)}</td>
        </tr>`);
      });
      if (!(c.cells ?? []).length) {
        tbody.insertAdjacentHTML("beforeend",
          `<tr class="cfg-first"><td>${c.name}</td>
           <td colspan="9" class="status-fail">${c.status}${
             c.failure_reason ? " — " + c.failure_reason.slice(0, 120) : ""}</td></tr>`);
      }
    }
    $("#opt-promote-registry")?.addEventListener("click", e =>
      this.promote("registry", e.currentTarget.dataset.cfg));
  },

  /* Winner → benchmark profile, then hand off to the Benchmark tab
   * with the generated profile preselected. */
  async promote(source, configName) {
    let r;
    try {
      r = await api("/api/optimizer/promote", {
        method: "POST",
        body: JSON.stringify({ source, config_name: configName ?? null }),
      });
    } catch (e) {
      this.msg(e.message, "error");
      return;
    }
    const warn = (r.warnings ?? []).length ? ` — NOTE: ${r.warnings[0]}` : "";
    this.msg(`optimized launch saved as profile "${r.profile}" (${r.path})${warn}`, "ok");
    await Control.loadCatalogs();
    const sel = $("#profile-select");
    if ([...sel.options].some(o => o.value === r.profile)) sel.value = r.profile;
    document.querySelector('#tabs button[data-view="control"]').click();
  },
};

/* ══ Storage (choose the disk + location for weights) ═════════── */

const Storage = {
  selected: null,

  init() {
    $("#storage-refresh").addEventListener("click", () => this.refresh());
    $("#storage-apply").addEventListener("click", () => this.apply());
    document.querySelector('#tabs button[data-view="prepare"]')
      .addEventListener("click", () => this.refresh());
    this.refresh();
  },

  msg(text, cls = "") {
    const el = $("#storage-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async refresh() {
    let doc;
    try { doc = await api("/api/storage"); } catch { return; }
    this.doc = doc;
    $("#storage-current").textContent =
      `${doc.hf_cache} (${doc.hf_cache_free_gb ?? "?"} GB free)`;
    $("#storage-source").textContent = {
      env: "· fixed by OPTIMIZER_HF_CACHE in the service environment",
      configured: "· chosen here",
      "data-layout": "· the /data/ml convention",
      default: "· the default — probably the boot disk",
    }[doc.hf_cache_source] ?? "";

    const fsBox = $("#storage-fs");
    fsBox.innerHTML = "";
    for (const fs of doc.filesystems) {
      const el = document.createElement("div");
      el.className = "chip fs-chip"
        + (doc.hf_cache.startsWith(fs.mountpoint) && fs.mountpoint !== "/"
           || this.selected === fs.mountpoint ? " selected" : "");
      el.innerHTML = `<span class="m">${fs.mountpoint}</span>
        <span class="free">${fs.free_gb.toFixed(0)} GB free</span>
        <span class="f">${fs.device} · ${fs.fstype} · ${fs.total_gb.toFixed(0)} GB total</span>`;
      el.addEventListener("click", () => {
        this.selected = fs.mountpoint;
        const base = fs.mountpoint === "/" ? "" : fs.mountpoint;
        $("#storage-path").value = `${base}/capsim/huggingface`;
        fsBox.querySelectorAll(".fs-chip").forEach(c => c.classList.remove("selected"));
        el.classList.add("selected");
      });
      fsBox.append(el);
    }

    const un = $("#storage-unmounted");
    if (doc.unmounted.length) {
      const label = (d) => `${d.name} (${d.size_gb >= 1000
        ? (d.size_gb / 1000).toFixed(1) + " TB" : d.size_gb + " GB"})`
        + (d.has_partitions
           ? ' <span class="status-marginal">— has existing partitions, check contents first</span>'
           : ' <span class="status-pass">— blank</span>');
      // The pasteable example must be the SAFEST candidate: a blank
      // disk when one exists (the API sorts blank-first).
      const example = doc.unmounted.find(d => !d.has_partitions) ?? doc.unmounted[0];
      un.innerHTML = `<div class="unmounted-box callout">
        <b>${doc.unmounted.length} unmounted disk(s) on this box:</b><br>` +
        doc.unmounted.map(label).join("<br>") +
        `<br><br>capsim won't format or mount disks — that needs root and destroys
        whatever is on them. Run these on the host <b>one line at a time</b>
        (this example uses <code>${example.name}</code>${example.has_partitions
          ? " — read the check-first lines carefully" : ", which is blank"}),
        then Refresh:
        <pre>${example.commands.join("\n")}</pre></div>`;
    } else {
      un.innerHTML = "";
    }
  },

  async apply() {
    const path = $("#storage-path").value.trim();
    if (!path) { this.msg("pick a filesystem or type a directory", "error"); return; }
    try {
      const r = await api("/api/storage", {
        method: "POST", body: JSON.stringify({ hf_cache: path }),
      });
      this.msg(`weights will live at ${r.resolved}`, "ok");
      this.selected = null;
      this.refresh();
      Models.refresh();          // cache dir + statuses just changed
    } catch (e) {
      this.msg(e.message, "error");
    }
  },
};

/* ══ Model staging ════════════════════════════════════════════── */

const Models = {
  polling: null,

  init() {
    $("#models-refresh").addEventListener("click", () => this.refresh());
    document.querySelector('#tabs button[data-view="prepare"]')
      .addEventListener("click", () => this.refresh());
    $("#goto-optimize").addEventListener("click", () =>
      document.querySelector('#tabs button[data-view="optimizer"]').click());
    $("#model-add-btn").addEventListener("click", () => this.add());
    $("#model-add-id").addEventListener("keydown", e => {
      if (e.key === "Enter") this.add();
    });
    this.refresh();
  },

  addMsg(text, cls = "") {
    const el = $("#model-add-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async add(id, checkHub = true) {
    const model = (id ?? $("#model-add-id").value).trim();
    if (!model) { this.addMsg("paste an org/name Hub id", "error"); return; }
    this.addMsg("checking the Hub…");
    let r;
    try {
      r = await api("/api/models/add", {
        method: "POST",
        body: JSON.stringify({ model, check_hub: checkHub }),
      });
    } catch (e) {
      this.addMsg(e.message, "error");
      return;
    }
    if (!id) $("#model-add-id").value = "";
    this.addMsg(r.created
      ? `added ${model} (family ${r.entry.family}, ${r.entry.quant})`
      : `${model} is already in the catalog`, "ok");
    // A sibling click passes an id — keep the base model's chip box
    // (minus the consumed chip) instead of replacing it.
    if (!id) this.renderSiblings(model, r.siblings);
    this.refresh();
  },

  renderSiblings(base, siblings) {
    const box = $("#model-siblings");
    const open = (siblings || []).filter(s => !s.in_catalog);
    if (!open.length) { box.innerHTML = ""; return; }
    box.innerHTML = `<div class="callout" style="margin-top:12px">
      Quantized variants of <b>${base}</b> worth testing too — smaller
      weights, more replicas per box:<br>` +
      open.map(s => `<button class="small" data-sib="${s.id}"
        style="margin:6px 6px 0 0">+ ${s.id}
        <span class="msg">(${s.quant}${s.exists === true ? ", verified on Hub"
          : s.exists === null ? ", unverified" : ""})</span></button>`).join("") +
      `</div>`;
    box.querySelectorAll("button[data-sib]").forEach(btn =>
      btn.addEventListener("click", () => {
        btn.disabled = true;
        this.add(btn.dataset.sib, false);
      }));
  },

  async refresh() {
    let doc;
    try { doc = await api("/api/models"); } catch { return; }
    $("#models-cache-dir").textContent = doc.cache_dir;
    const tbody = $("#models-table tbody");
    tbody.innerHTML = "";
    let anyRunning = false;
    for (const m of doc.models) {
      const dl = doc.downloads[m.model];
      const running = !!(dl && dl.running);
      anyRunning ||= running;
      let status, action = "";
      if (running) {
        const tail = (dl.log_tail || "").trim().split("\n").pop() || "";
        status = `<span class="status-marginal">downloading…</span>
                  <div class="msg dl-tail">${tail.slice(-70)}</div>`;
      } else if (dl && dl.exit_code !== 0 && dl.exit_code !== null) {
        status = `<span class="status-fail">download failed (${dl.exit_code})</span>`;
        action = `<button class="small" data-model="${m.model}">Retry</button>`;
      } else if (m.cached) {
        status = `<span class="status-pass">cached</span>`;
      } else if (m.partial) {
        status = `<span class="status-marginal">partial</span>`;
        action = `<button class="small" data-model="${m.model}">Resume download</button>`;
      } else {
        status = `<span class="msg">not downloaded</span>`;
        action = `<button class="small primary" data-model="${m.model}">Download</button>`;
      }
      const size = m.size_gb ? m.size_gb.toFixed(1) + " GB"
        : m.approx_size_gb ? `~${m.approx_size_gb} GB` : "—";
      tbody.insertAdjacentHTML("beforeend", `<tr>
        <td>${m.model}${m.gated
          ? ' <span class="status-marginal" title="accept the license on the Hub and set HF_TOKEN">gated</span>' : ""}
          ${m.notes ? `<div class="msg">${m.notes}</div>` : ""}</td>
        <td class="msg">${m.quant || "—"}</td>
        <td class="msg">${m.referenced_by.join(", ")}</td>
        <td>${status}</td>
        <td>${size}</td>
        <td>${action}</td></tr>`);
    }
    tbody.querySelectorAll("button[data-model]").forEach(btn =>
      btn.addEventListener("click", () => this.download(btn.dataset.model)));
    if (anyRunning && !this.polling) {
      this.polling = setInterval(() => this.refresh(), 3000);
    } else if (!anyRunning && this.polling) {
      clearInterval(this.polling);
      this.polling = null;
    }
  },

  async download(model) {
    try {
      await api("/api/models/download", {
        method: "POST", body: JSON.stringify({ model }),
      });
    } catch (e) {
      Optimizer.msg(e.message, "error");
    }
    this.refresh();
  },
};

/* ══ Persona / cohort editor ══════════════════════════════════── */

const PERSONA_TEMPLATE = `description: "What this archetype does"
input_tokens: {lognormal: {median: 400, sigma: 0.5}}
output_tokens: {lognormal: {median: 200, sigma: 0.4}}
turns_per_session: {discrete: {1: 0.6, 2: 0.3, 4: 0.1}}
sessions_before_leaving: {discrete: {3: 0.5, 6: 0.5}}
inter_session_gap_seconds: {lognormal: {median: 600, sigma: 1.0}}
read_time_seconds: {lognormal: {median: 25, sigma: 0.5}}
active_think_seconds: {lognormal: {median: 30, sigma: 0.6}}
sla:
  ttft_target_seconds: 10.0
  ttft_failure_seconds: 30.0
  tpot_target_ms: 150.0
  tpot_failure_ms: 225.0
`;

const COHORT_TEMPLATE = `name: "My team"
description: "What this team does"
persona_weights:
  quick_lookup: 0.5
  conversational: 0.5
`;

const Editor = {
  kind: "personas",    // "personas" | "cohorts"
  editing: null,       // id being edited, null for new

  init() {
    $("#editor-save").addEventListener("click", () => this.save());
    $("#persona-new").addEventListener("click", () =>
      this.startNew("personas", PERSONA_TEMPLATE));
    $("#cohort-new").addEventListener("click", () =>
      this.startNew("cohorts", COHORT_TEMPLATE));
    document.querySelector('#tabs button[data-view="personas"]')
      .addEventListener("click", () => this.refreshLists());
  },

  msg(text, cls = "") {
    const el = $("#editor-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async refreshLists() {
    const [personas, cohorts] = await Promise.all([
      api("/api/personas"), api("/api/cohorts"),
    ]);
    const fill = (sel, items, kind) => {
      const ul = $(sel);
      ul.innerHTML = "";
      for (const item of items) {
        const li = document.createElement("li");
        li.textContent = item.id;
        li.classList.toggle(
          "active", this.kind === kind && this.editing === item.id);
        li.addEventListener("click", () => this.open(kind, item.id));
        ul.append(li);
      }
    };
    fill("#persona-list", personas, "personas");
    fill("#cohort-list", cohorts, "cohorts");
  },

  async open(kind, id) {
    this.kind = kind;
    this.editing = id;
    const detail = await api(`/api/${kind}/${id}`);
    $("#editor-id").value = id;
    $("#editor-yaml").value = detail.yaml;
    $("#editor-kind-badge").textContent = kind.slice(0, -1);
    this.msg(`editing ${id} — saves to ${detail.editable_file}`);
    this.refreshLists();
  },

  startNew(kind, template) {
    this.kind = kind;
    this.editing = null;
    $("#editor-id").value = "";
    $("#editor-yaml").value = template;
    $("#editor-kind-badge").textContent = kind.slice(0, -1);
    this.msg("set an id and Save");
    this.refreshLists();
  },

  async save() {
    const id = $("#editor-id").value.trim();
    if (!id) { this.msg("id required", "error"); return; }
    try {
      await api(`/api/${this.kind}/${id}`, {
        method: "PUT",
        body: JSON.stringify({ yaml: $("#editor-yaml").value }),
      });
      this.editing = id;
      this.msg(`saved ${id}`, "ok");
      this.refreshLists();
      Control.loadCatalogs();  // refresh workload pickers with the new entry
    } catch (e) {
      this.msg(e.message, "error");
    }
  },
};

/* ── boot ─────────────────────────────────────────────────────── */

Control.init();
Live.init();
Results.init();
Optimizer.init();
Storage.init();
Models.init();
Editor.init();
