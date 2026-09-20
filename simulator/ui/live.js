import { $, api, fmt, percentile, STATUS_CLASS } from "./lib/api.js";
import { C, fill, makeChart } from "./lib/theme.js";
import { on, emit } from "./lib/events.js";
import { Headline } from "./headline.js";

/* ══ Live telemetry ═══════════════════════════════════════════── */

const WINDOW = 300;           // chart points (~5 min at 1 Hz)
const TURN_WINDOW = 40;       // rolling-percentile turn window

function makeLiveChart(canvas, datasets, yOpts = {}) {
  return makeChart(canvas, {
    datasets,
    x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
    y: yOpts,
  });
}

export const Live = {
  charts: {},
  turns: [],           // recent {ttft_ms, tpot_ms, ts}
  stepsSeen: new Set(),
  hasData: false,      // anything plotted (live or replayed)?

  /* Idle = no run and nothing replayed: the five charts and the stat
   * bar hide behind a "no run in progress" panel rather than sitting
   * as empty 0–1 axes. Driven by Control's status event. */
  setIdle(idle) {
    const wasIdle = $("#live-panels").hidden;
    $("#live-idle").hidden = !idle;
    $("#live-panels").hidden = idle;
    if (wasIdle && !idle) {
      // Charts laid out while hidden have no size; let them measure.
      for (const c of Object.values(this.charts)) c.resize();
    }
  },

  /* Turn events only flow once the measuring window opens, so the
   * latency charts stay empty through warmup even as turns complete.
   * Say what is happening on the cards instead of showing bare axes. */
  setWarmNote(s) {
    const warm = /warm/i.test(s.phase || "")
      && !this.charts.ttft.data.labels.length;
    for (const id of ["#ttft-warm", "#tpot-warm"]) {
      const el = $(id);
      if (!el) continue;
      el.hidden = !warm;
      if (warm) {
        el.textContent = `warming up — ${s.requests_completed ?? 0} turns
          completed so far; rolling percentiles start with the first
          measuring window`;
      }
    }
  },

  init() {
    on("status", ({ running }) => this.setIdle(!running && !this.hasData));
    this.charts.pool = makeLiveChart("#chart-pool", [
      { label: "sessions", data: [], borderColor: C.muted, stepped: true,
        borderDash: [5, 4] },
      { label: "in flight", data: [], borderColor: C.gold, fill: true,
        backgroundColor: fill(C.gold) },
      { label: "queue waiting", data: [], borderColor: C.fail },
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
    this.charts.tokens = makeLiveChart("#chart-tokens", [
      { label: "prefill tok/s", data: [], borderColor: C.blue, fill: true,
        backgroundColor: fill(C.blue, "1f") },
      { label: "decode tok/s", data: [], borderColor: C.teal, fill: true,
        backgroundColor: fill(C.teal, "1f") },
    ]);
    this.charts.sessions = makeLiveChart("#chart-sessions", [
      { label: "prefill", data: [], borderColor: C.blue },
      { label: "decode", data: [], borderColor: C.teal },
      { label: "warm think", data: [], borderColor: C.gold },
      { label: "cold", data: [], borderColor: C.muted, borderDash: [4, 4] },
    ]);
    // Backfill BEFORE the live stream: a page opened mid-run replays
    // the run's recent history (snapshots, telemetry, turns, steps)
    // through the same handlers, so it shows where the run IS and
    // what it has done — not just what happens after load.
    this.backfill().finally(() => this.connect());
  },

  async backfill() {
    let doc;
    // 6-hour window: a page opened AFTER a long run finished should
    // still replay the whole run's history, not stare at empty
    // charts because the last snapshot is older than ten minutes.
    try {
      doc = await api("/api/live/backfill?window_s=21600");
    } catch { return; }
    if (!doc.run) return;
    this.hasData = true;
    this.setIdle(false);
    for (const t of doc.turns ?? []) this.onTurn(t.completed_at_ms, t);
    for (const s of doc.snapshots ?? []) this.onSnapshot(s.snapshot_at_ms, s);
    for (const t of doc.telemetry ?? []) this.onTelemetry(t.sampled_at_ms, t);
    for (const s of doc.steps ?? []) this.onStep(s);
    if (doc.run.final_status) {
      $("#live-phase").textContent = `finished (${doc.run.final_status})`;
    }
  },

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
    ws.onclose = () => setTimeout(() => this.connect(), 2000);
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
    this.hasData = true;
    const d = chart.data;
    d.labels.push(label);
    values.forEach((v, i) => d.datasets[i].data.push(v));
    if (d.labels.length > WINDOW) {
      d.labels.shift();
      d.datasets.forEach(ds => ds.data.shift());
    }
    chart.update("none");
  },

  // A new run must not inherit the previous run's lines — stale
  // charts read as "nothing is happening" or "it resumed the old
  // run" when a fresh run is actually launching.
  resetCharts() {
    for (const chart of Object.values(this.charts)) {
      chart.data.labels = [];
      chart.data.datasets.forEach(ds => { ds.data = []; });
      chart.update("none");
    }
    this.turns = [];
    for (const id of ["live-rate", "live-queue", "live-pool",
                      "live-inflight", "live-completed", "live-errors",
                      "live-warmkv"]) {
      $(`#${id}`).textContent = "—";
    }
  },

  onSnapshot(ts, s) {
    // Phase + how long it's been in that phase — "launching engine
    // (3m 40s)" tells the user a big model is still loading; a bare
    // phase name can't distinguish progress from a hang. Elapsed is
    // computed from snapshot timestamps so backfill replays show the
    // true duration too.
    if (s.phase !== this._phaseName) {
      this._phaseName = s.phase;
      this._phaseSinceTs = ts;
    }
    const inPhase = Math.max(0, Math.round((ts - this._phaseSinceTs) / 1000));
    const el = inPhase >= 5
      ? ` · ${inPhase >= 60 ? `${Math.floor(inPhase / 60)}m ${inPhase % 60}s`
                            : `${inPhase}s`}`
      : "";
    $("#live-phase").textContent = s.phase + el;
    this.setWarmNote(s);
    $("#live-pool").textContent = s.pool_size;
    $("#live-inflight").textContent = s.in_flight;
    $("#live-completed").textContent = s.requests_completed;
    $("#live-errors").textContent = s.errors;
    // Open-loop pressure stats (null on closed-loop snapshots).
    $("#live-rate").textContent =
      s.arrival_rate_per_min != null
        ? `${s.arrival_rate_per_min}/min` : "—";
    $("#live-queue").textContent =
      s.queue_depth != null ? `${s.queue_depth}` : "—";
    if (s.warm_kv_tokens != null) {
      const t = s.warm_kv_tokens;
      $("#live-warmkv").textContent =
        t >= 1e6 ? `${(t / 1e6).toFixed(1)}M` :
        t >= 1e3 ? `${(t / 1e3).toFixed(1)}k` : `${t}`;
    }
    Headline.onSnapshot(s);
    // A saturation sweep publishes no per-turn events (tens of
    // thousands a second at full concurrency), so its latency rides
    // on the snapshot instead. Feed the same charts from it.
    if (s.ttft_p50_ms != null || s.tpot_p50_ms != null) {
      const lbl = fmt.clock(ts);
      this.push(this.charts.ttft, lbl, [s.ttft_p50_ms, s.ttft_p95_ms]);
      this.push(this.charts.tpot, lbl, [s.tpot_p50_ms, s.tpot_p95_ms]);
    }
    this.push(this.charts.pool, fmt.clock(ts),
      [s.pool_size, s.in_flight, s.queue_depth ?? null]);
    if (s.prefill_in_flight != null) {
      this.push(this.charts.sessions, fmt.clock(ts), [
        s.prefill_in_flight, s.decode_in_flight,
        s.sessions_warm, s.sessions_cold,
      ]);
    }
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
    Headline.onTelemetry(t);
    this.push(this.charts.host, fmt.clock(ts), [
      t.kv_cache_used_pct, t.cpu_util_bound_avg ?? t.cpu_util_avg,
      t.gpu_sm_util_pct,
    ]);
    if (t.prefill_tok_s != null || t.decode_tok_s != null) {
      // Smoothed in headline mode. A closed-loop sweep holds N streams
      // of IDENTICAL fixed length, so streams launched together finish
      // together and the engine's counters advance in convoys roughly
      // one request-duration apart. Differencing that per sample makes
      // both series spike to ~5.6x their own mean at the same instants
      // -- which reads as instability and is only the sampling.
      const pair = Headline.active
        ? Headline.smoothTokens(t.prefill_tok_s, t.decode_tok_s)
        : [t.prefill_tok_s, t.decode_tok_s];
      this.push(this.charts.tokens, fmt.clock(ts), pair);
    }
    let host = null, gpus = null;
    try { host = t.host_json ? JSON.parse(t.host_json) : null; } catch { /* skip */ }
    try { gpus = t.gpu_devices_json ? JSON.parse(t.gpu_devices_json) : null; } catch { /* skip */ }
    this.renderHostDetail(host, t);
    this.renderGpuGrid(gpus);
    // The saturation view carries the same grid under its own hero.
    if (Headline.active && gpus?.length) this.renderGpuGrid(gpus, "#hl-gpu-grid");
  },

  /* Compact key/value line: what the CPUs are doing, scheduler and
   * I/O pressure, chassis power when IPMI grants it. */
  renderHostDetail(h, t) {
    const el = $("#host-detail");
    if (!h) return;
    const bd = h.cpu_breakdown_pct || {};
    const disks = Object.entries(h.disk || {});
    const busiest = disks.sort((a, b) => b[1].util_pct - a[1].util_pct)[0];
    const io = disks.length ? {
      r: disks.reduce((s, [, d]) => s + d.read_mb_s, 0),
      w: disks.reduce((s, [, d]) => s + d.write_mb_s, 0),
    } : null;
    const parts = [
      bd.user != null && `CPU: <b>${bd.user}%</b> user · <b>${bd.system}%</b> sys
        · <b>${bd.iowait}%</b> iowait · <b>${bd.irq}%</b> irq`,
      h.load1 != null && `load <b>${h.load1}</b>`,
      h.ctx_switches_s != null && `<b>${(h.ctx_switches_s / 1000).toFixed(1)}k</b> ctx/s`,
      h.mem && `mem avail <b>${h.mem.available_gb}</b> GB · cached
        <b>${h.mem.cached_gb}</b> GB${h.mem.swap_used_gb > 0.05
          ? ` · <span class="status-marginal">swap ${h.mem.swap_used_gb} GB</span>` : ""}`,
      io && `disk <b>${io.r.toFixed(0)}</b>R/<b>${io.w.toFixed(0)}</b>W MB/s${busiest
        && busiest[1].util_pct > 50
          ? ` · <span class="status-marginal">${busiest[0]} ${busiest[1].util_pct}% busy</span>` : ""}`,
      h.net && `net <b>${h.net.rx_mb_s}</b>↓/<b>${h.net.tx_mb_s}</b>↑ MB/s`,
      h.system_power_w != null && `system <b>${h.system_power_w}</b> W`,
      t.gpu_power_w != null && `GPUs <b>${t.gpu_power_w.toFixed(0)}</b> W`,
      t.engine_rss_gb != null && `engine RSS <b>${t.engine_rss_gb.toFixed(1)}</b> GB`,
      t.preemptions != null && t.preemptions > 0
        && `<span class="status-marginal">preemptions ${t.preemptions}</span>`,
    ].filter(Boolean);
    el.innerHTML = parts.join(" &nbsp;·&nbsp; ");
  },

  renderGpuGrid(gpus, sel = "#gpu-grid") {
    const el = $(sel);
    if (!el || !gpus || !gpus.length) return;
    el.innerHTML = gpus.map(g => {
      const vramPct = g.vram_total_gb
        ? 100 * g.vram_used_gb / g.vram_total_gb : 0;
      return `<div class="gpu-row ${g.throttled ? "throttled" : ""}">
        <span class="g-id">GPU ${g.index}</span>
        <span class="g-bar" title="SM ${g.sm_util_pct ?? "—"}%">
          <i style="width:${g.sm_util_pct ?? 0}%"></i></span>
        <span class="g-num">${g.sm_util_pct == null ? "—" : g.sm_util_pct + "%"}</span>
        <span class="g-bar vram" title="VRAM ${(g.vram_used_gb ?? 0).toFixed(1)} GB">
          <i style="width:${vramPct}%"></i></span>
        <span class="g-num">${(g.vram_used_gb ?? 0).toFixed(0)}G</span>
        <span class="g-sub">${g.mem_util_pct != null ? `mem ${g.mem_util_pct}%` : ""}
          ${g.power_w != null ? ` · ${g.power_w.toFixed(0)}W` : ""}
          ${g.temperature_c != null ? ` · ${g.temperature_c}°C` : ""}
          ${g.pcie_tx_mb_s != null
            ? ` · pcie ${(g.pcie_tx_mb_s + g.pcie_rx_mb_s).toFixed(0)}MB/s` : ""}
          ${g.throttled ? ' · <span class="status-fail">throttled</span>' : ""}</span>
      </div>`;
    }).join("");
  },

  onStep(s) {
    const cls = STATUS_CLASS[s.capacity_status] ?? "";
    // Open-loop steps: the control variable is the arrival rate; the
    // pool column shows it with the measured concurrency alongside,
    // and the verdict column leads with stability. client_limited and
    // superseded windows carry no SLA verdict (their samples were
    // inflated by the lagging GENERATOR, not the engine) so they
    // never pair with pass/fail.
    const load = s.arrival_rate_per_min != null
      ? `${s.arrival_rate_per_min}/min (${s.pool_size})`
      : `${s.pool_size}`;
    const verdictCls = s.stability === "divergent" ? "status-fail"
      : (s.stability === "client_limited"
         || s.stability === "superseded") ? "status-marginal" : cls;
    const verdict =
      s.stability === "client_limited"
        ? "client limited — generator maxed, not the engine"
      : s.stability === "superseded"
        ? "superseded — rerun with more load workers"
      : s.stability
        ? `${s.stability} · ${s.capacity_status}`
        : s.capacity_status;
    const row = `<tr data-step="${s.step_index}">
      <td>${s.step_index}</td><td>${load}</td><td>${s.sample_size ?? "—"}</td>
      <td>${s.combined_violation_rate != null
            ? fmt.pct(s.combined_violation_rate) : "—"}</td>
      <td>${s.combined_target_miss_rate != null
            ? fmt.pct(s.combined_target_miss_rate) : "—"}</td>
      <td>${s.ttft_p95_ms != null ? fmt.ms(s.ttft_p95_ms) : "—"}</td>
      <td>${s.tpot_p95_ms != null ? fmt.ms(s.tpot_p95_ms) : "—"}</td>
      <td class="${verdictCls}">${verdict}</td></tr>`;
    // A step can be re-reported (a client-saturated attempt gets
    // re-labeled "superseded" once its retry starts) — update the
    // existing row in place rather than duplicating it.
    const existing = $(`#steps-table tbody tr[data-step="${s.step_index}"]`);
    if (existing) existing.outerHTML = row;
    else $("#steps-table tbody").insertAdjacentHTML("afterbegin", row);
  },

  onRun(r) {
    if (r.event === "started") {
      $("#steps-table tbody").innerHTML = "";
      this.stepsSeen.clear();
      this.resetCharts();
      this._phaseName = null;
      $("#live-phase").textContent = "starting…";
      this.hasData = true;
      this.setIdle(false);
      for (const id of ["#ttft-warm", "#tpot-warm"]) {
        const el = $(id);
        if (el) el.hidden = true;
      }
      emit("run:started", r);
    }
    if (r.event === "finished") {
      $("#live-phase").textContent = `finished (${r.final_status})`;
      emit("run:finished", r);
    }
    // Control re-polls the status pill on both; Results drops its
    // export cache on "finished" and refetches the run list.
  },
};
