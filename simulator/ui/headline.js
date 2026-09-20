import { $, fmt, fmtCompact } from "./lib/api.js";
import { C, fill } from "./lib/theme.js";
import { on } from "./lib/events.js";
import { Engines } from "./engines.js";

/* ── Headline (saturation) live view ───────────────────────────────
 * A capacity run and a saturation run answer different questions, so
 * they need different instruments. The capacity statusbar has no
 * arrival rate, no completed sessions and no warm-KV figure to show
 * during a headline sweep -- it renders dashes where the operator is
 * looking for a number. This swaps in the things a saturation run
 * actually produces: the sustained output rate, the ladder that earns
 * it, and what the box is spending to get there. */

const HL_SYS_WINDOW = 240;        // ~4 min of system-graph history

// Engines advance their token counters in BURSTS. A closed-loop sweep
// holds a fixed number of streams of identical length, so streams that
// start together finish together and the counters discharge in convoys
// about one request-duration apart -- measured here, a median 15.3s
// apart against a 1.9s sample interval, with 14% of samples reading
// near zero because nothing had completed yet.
//
// So the window must cover a convoy, and the average across it MUST BE
// THE MEAN. A median looks tempting -- it stops one burst dragging the
// line -- but the bursts ARE the tokens, and discarding them discards
// real work: on this data a rolling median reads 39.9% BELOW the true
// rate while a rolling mean lands within 1.1%. Quietly under-reporting
// throughput by two fifths is far worse than a jagged line.
const HL_RATE_WINDOW = 12;

/* Mean of a short window, ignoring gaps. */
function windowMean(arr) {
  const v = arr.filter(x => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

export const Headline = {
  active: false,
  charts: {},
  _tel: {},
  // Live samples per offered concurrency. The ladder only gains a row
  // when a rung SETTLES, which is minutes apart; without this the two
  // saturation charts sit empty for most of a run. Settled rungs
  // overwrite these the moment they arrive.
  _live: new Map(),
  _rate: [],
  _sys: { labels: [], power: [], syspower: [], eff: [], sm: [], mem: [],
          vram: [], hostmem: [], rss: [], cpu: [], ghz: [] },

  init() {
    on("status", ({ active }) => this.fromStatus(active));
  },

  setActive(isOn) {
    if (isOn === this.active) return;
    this.active = isOn;
    $("#headline-live").hidden = !isOn;
    $("#view-control").classList.toggle("headline-mode", isOn);
  },

  /* Driven by Control's status event — the search's own progress dict. */
  fromStatus(active) {
    const w = active?.workload || {};
    const isHeadline = w.kind === "headline_optimize"
      || (w.kind === "persona" && (w.id || "").startsWith("headline_"));
    // Stays up after the sweep ends. Reverting to the capacity view on
    // completion would drop the operator back to the blank statusbar
    // at the exact moment the result is worth reading.
    this.setActive(Boolean(isHeadline && active));
    if (!this.active) return;
    const p = active.progress || {};

    // What is being measured, right now. A search runs many engines
    // and shapes; without this the numbers below are unattributable.
    const bits = [];
    if (p.pairs) {
      bits.push(`<b>candidate ${p.pair || 1} of ${p.pairs}</b>`);
      const c = p.current || {};
      if (c.engine) bits.push(`engine <b>${Engines.label(c.engine)}</b>`);
      if (c.max_num_seqs) bits.push(`mns <b>${c.max_num_seqs}</b>`);
      if (c.output_tokens) {
        bits.push(`shape <b>${p.input_tokens || 128}&rarr;`
          + `${c.output_tokens}</b>`);
      }
    } else {
      if (p.engine) bits.push(`engine <b>${Engines.label(p.engine)}</b>`);
      // The sweep reports its shape as a structured cohort record, not
      // a string; rendering it raw prints "[object Object]".
      const sh = p.shape;
      if (sh?.input_tokens != null) {
        bits.push(`shape <b>${Math.round(sh.input_tokens)}&rarr;`
          + `${Math.round(sh.output_tokens)}</b>`);
      }
    }
    if (p.model) bits.push(`model <b>${p.model.split("/").pop()}</b>`);
    if (p.rung) {
      bits.push(`rung <b>${p.rung} of ${p.rungs}</b>`
        + (p.concurrency ? ` at <b>${p.concurrency.toLocaleString()}</b>
             streams` : ""));
    }
    if (p.phase) bits.push(p.phase);
    $("#hl-under").innerHTML = bits.join(" &nbsp;·&nbsp; ")
      || "waiting for the engine…";

    this.renderBoard(p.best_per_engine, p.engines);
    this.renderLadder(p.rungs_done || [], p.peak);
    const peak = p.peak;
    if (peak?.out_tok_s) {
      $("#hl-peak").textContent = Math.round(peak.out_tok_s).toLocaleString();
      $("#hl-peak-at").textContent = peak.in_flight
        ? `at ${Math.round(peak.in_flight).toLocaleString()} streams`
        : "";
    }
    if (p.kv_cache_tokens) {
      $("#hl-kvpool").textContent =
        `pool ${fmtCompact(p.kv_cache_tokens)} tokens`;
    }
    // A finished sweep has no live telemetry to drive the hero, and
    // dashes at the moment the result is worth reading is the fault
    // this view exists to fix. Fall back to the peak rung.
    if (!active.running && peak) {
      $("#hl-label").textContent = "peak output tokens / sec";
      if (peak.out_tok_s) {
        $("#hl-now").textContent = Math.round(peak.out_tok_s)
          .toLocaleString();
      }
      if (peak.in_flight != null) {
        $("#hl-held").textContent =
          `${Math.round(peak.in_flight).toLocaleString()} / `
          + `${(peak.concurrency ?? 0).toLocaleString()}`;
        $("#hl-held-note").textContent = "at the peak rung";
      }
      if (peak.queue_depth != null) {
        $("#hl-queue").textContent =
          Math.round(peak.queue_depth).toLocaleString();
      }
      if (peak.kv_cache_pct != null) {
        $("#hl-kv").textContent = `${peak.kv_cache_pct.toFixed(0)}%`;
      }
      if (peak.gpu_power_w != null) {
        $("#hl-power").textContent = `${Math.round(peak.gpu_power_w)} W`;
        if (peak.out_tok_s) {
          $("#hl-eff").textContent =
            `${(peak.out_tok_s / peak.gpu_power_w).toFixed(1)} tokens/watt`;
        }
      }
    } else {
      $("#hl-label").textContent = "output tokens / sec";
    }
  },

  /* Head-to-head. A single "best" hides the loser the moment one
   * engine sweeps the top, which is the comparison the search exists
   * to produce. */
  renderBoard(best, engines) {
    const el = $("#hl-board");
    const names = engines || (best ? Object.keys(best) : []);
    if (!names || names.length < 2) { el.innerHTML = ""; return; }
    const top = Math.max(...names.map(e => best?.[e]?.out_tok_s || 0));
    el.innerHTML = names.map(e => {
      const b = best?.[e];
      const lead = b && b.out_tok_s === top && top > 0;
      return `<div class="e ${lead ? "lead" : ""}">
        <div class="n">${Engines.label(e)}</div>
        <div class="t">${b?.out_tok_s
          ? Math.round(b.out_tok_s).toLocaleString() : "—"}</div>
        <div class="s">${b?.out_tok_s
          ? `mns ${b.max_num_seqs} · ${b.input_tokens}→${b.output_tokens}`
             + (b.in_flight
                ? ` · ${Math.round(b.in_flight).toLocaleString()} streams`
                : "")
          : "not measured yet"}</div></div>`;
    }).join("");
  },

  renderLadder(rungs, peak) {
    this._rungs = rungs;
    const body = $("#hl-ladder tbody");
    if (!rungs.length) {
      body.innerHTML = `<tr><td colspan="11" class="msg">no rung has
        settled yet — the first one takes a few minutes</td></tr>`;
      return;
    }
    const peakC = peak?.concurrency;
    body.innerHTML = rungs.map(r => {
      const held = r.in_flight != null
        ? Math.round(r.in_flight).toLocaleString() : "—";
      const eff = (r.out_tok_s && r.gpu_power_w)
        ? (r.out_tok_s / r.gpu_power_w).toFixed(1) : "—";
      // Two honesty flags, in the row they belong to rather than a
      // footnote: a rung the engine could not hold, and one that had
      // not settled when it was measured.
      const flags = [
        r.held === false ? `<span class="status-marginal"
          title="the engine ran well under what it was offered — the
          extra streams only queued">ceiling</span>` : "",
        r.steady_state === false ? `<span class="status-fail"
          title="measured before the running batch and token rate
          stopped moving — not a sustained number">unsettled</span>` : "",
      ].filter(Boolean).join(" ");
      return `<tr class="${r.concurrency === peakC ? "peak" : ""}">
        <td>${r.concurrency.toLocaleString()}</td>
        <td>${held}</td>
        <td><b>${r.out_tok_s ? Math.round(r.out_tok_s).toLocaleString()
          : "—"}</b></td>
        <td>${r.total_tok_s ? Math.round(r.total_tok_s).toLocaleString()
          : "—"}</td>
        <td>${r.queue_depth != null
          ? Math.round(r.queue_depth).toLocaleString() : "—"}</td>
        <td>${r.ttft_p95_ms != null ? Math.round(r.ttft_p95_ms) + " ms"
          : "—"}</td>
        <td>${r.tpot_p95_ms != null ? r.tpot_p95_ms.toFixed(1) + " ms"
          : "—"}</td>
        <td>${r.kv_cache_pct != null ? r.kv_cache_pct.toFixed(0) + "%"
          : "—"}</td>
        <td>${r.gpu_power_w != null ? Math.round(r.gpu_power_w) : "—"}</td>
        <td>${eff}</td>
        <td>${flags}</td></tr>`;
    }).join("");
  },

  /* Settled rungs are authoritative; live samples fill the gaps so the
   * curve is never blank while a rung is still being measured. */
  series() {
    const byC = new Map();
    for (const [c, v] of this._live) {
      byC.set(c, { concurrency: c, in_flight: v.held,
                   out_tok_s: v.tok, settled: false });
    }
    for (const r of this._rungs || []) {
      byC.set(r.concurrency, { concurrency: r.concurrency,
        in_flight: r.in_flight, out_tok_s: r.out_tok_s,
        ttft_p95_ms: r.ttft_p95_ms, tpot_p95_ms: r.tpot_p95_ms,
        settled: true });
    }
    return [...byC.values()].sort((a, b) => a.concurrency - b.concurrency);
  },

  drawCurves() {
    const rungs = this.series();
    if (!rungs.length) return;
    const labels = rungs.map(r => r.concurrency.toLocaleString());
    if (!this.charts.curve) {
      this.charts.curve = new Chart($("#chart-hl-curve"), {
        type: "line",
        data: { labels: [], datasets: [
          { label: "output tok/s", data: [], borderColor: C.gold,
            backgroundColor: fill(C.gold), tension: .3, fill: true },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: { y: { beginAtZero: true },
                    x: { title: { display: true,
                                  text: "streams offered" } } },
          plugins: { legend: { position: "bottom" } },
        },
      });
      this.charts.lat = new Chart($("#chart-hl-lat"), {
        type: "line",
        data: { labels: [], datasets: [
          { label: "TTFT p95 (ms)", data: [], borderColor: C.purple,
            tension: .3, yAxisID: "y" },
          { label: "TPOT p95 (ms)", data: [], borderColor: C.accent,
            tension: .3, yAxisID: "y1" },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            y: { beginAtZero: true, position: "left",
                 title: { display: true, text: "TTFT ms" } },
            y1: { beginAtZero: true, position: "right",
                  grid: { drawOnChartArea: false },
                  title: { display: true, text: "TPOT ms" } },
            x: { title: { display: true, text: "streams offered" } },
          },
          plugins: { legend: { position: "bottom" } },
        },
      });
      this.charts.held = new Chart($("#chart-hl-held"), {
        type: "line",
        data: { labels: [], datasets: [
          { label: "offered", data: [], borderColor: C.muted || C.blue,
            borderDash: [5, 4], tension: 0 },
          { label: "held by the engine", data: [], borderColor: C.teal,
            backgroundColor: fill(C.teal), tension: .3, fill: true },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: { y: { beginAtZero: true },
                    x: { title: { display: true,
                                  text: "streams offered" } } },
          plugins: { legend: { position: "bottom" } },
        },
      });
    }
    const c = this.charts.curve;
    c.data.labels = labels;
    c.data.datasets[0].data = rungs.map(r => r.out_tok_s ?? null);
    c.update("none");
    const h = this.charts.held;
    h.data.labels = labels;
    h.data.datasets[0].data = rungs.map(r => r.concurrency);
    h.data.datasets[1].data = rungs.map(r => r.in_flight ?? null);
    h.update("none");
    const l = this.charts.lat;
    if (l) {
      l.data.labels = labels;
      l.data.datasets[0].data = rungs.map(r => r.ttft_p95_ms ?? null);
      l.data.datasets[1].data = rungs.map(r => r.tpot_p95_ms ?? null);
      l.update("none");
    }
  },

  /* Rolling MEAN of both token rates -- see HL_RATE_WINDOW for why a
   * median is the wrong average here. Same window as the hero, so the
   * chart and the big number always agree. */
  _tok: { pre: [], dec: [] },

  smoothTokens(pre, dec) {
    const avg = (arr, v) => {
      if (v != null) { arr.push(v); if (arr.length > HL_RATE_WINDOW) arr.shift(); }
      return windowMean(arr);
    };
    return [avg(this._tok.pre, pre), avg(this._tok.dec, dec)];
  },

  sysChart(key, canvas, datasets, yOpts) {
    if (!this.charts[key]) {
      this.charts[key] = new Chart($(canvas), {
        type: "line", data: { labels: [], datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          animation: false,
          scales: { x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
                    ...(yOpts || { y: { beginAtZero: true } }) },
          plugins: { legend: { position: "bottom" } },
        },
      });
    }
    return this.charts[key];
  },

  /* The host summary was one dense line of text -- readable as a
   * status check, useless for seeing a trend. During a saturation
   * sweep the question is always "what moved when concurrency went
   * up", which is a shape over time, not a snapshot. */
  drawSystem() {
    const S = this._sys;
    const power = this.sysChart("power", "#chart-hl-power", [
      { label: "GPU W", data: [], borderColor: C.gold,
        backgroundColor: fill(C.gold), fill: true, tension: .3,
        pointRadius: 0, yAxisID: "y" },
      { label: "chassis W", data: [], borderColor: C.muted,
        borderDash: [5, 4], tension: .3, pointRadius: 0, yAxisID: "y" },
      { label: "tokens/watt", data: [], borderColor: C.teal,
        tension: .3, pointRadius: 0, yAxisID: "y1" },
    ], { y: { beginAtZero: true, position: "left",
              title: { display: true, text: "watts" } },
         y1: { beginAtZero: true, position: "right",
               grid: { drawOnChartArea: false },
               title: { display: true, text: "tok/W" } } });
    power.data.labels = S.labels;
    power.data.datasets[0].data = S.power;
    power.data.datasets[1].data = S.syspower;
    power.data.datasets[2].data = S.eff;
    power.update("none");

    const gpu = this.sysChart("gpu", "#chart-hl-gpu", [
      { label: "SM util %", data: [], borderColor: C.blue,
        tension: .3, pointRadius: 0 },
      { label: "memory-controller busy %", data: [], borderColor: C.accent,
        backgroundColor: fill(C.accent), fill: true, tension: .3,
        pointRadius: 0 },
    ], { y: { beginAtZero: true, max: 100 } });
    gpu.data.labels = S.labels;
    gpu.data.datasets[0].data = S.sm;
    gpu.data.datasets[1].data = S.mem;
    gpu.update("none");

    const mem = this.sysChart("mem", "#chart-hl-mem", [
      { label: "VRAM GB", data: [], borderColor: C.purple,
        backgroundColor: fill(C.purple), fill: true, tension: .3,
        pointRadius: 0 },
      { label: "host RAM GB", data: [], borderColor: C.teal,
        tension: .3, pointRadius: 0 },
      { label: "engine RSS GB", data: [], borderColor: C.muted,
        borderDash: [5, 4], tension: .3, pointRadius: 0 },
    ]);
    mem.data.labels = S.labels;
    mem.data.datasets[0].data = S.vram;
    mem.data.datasets[1].data = S.hostmem;
    mem.data.datasets[2].data = S.rss;
    mem.update("none");

    const cpu = this.sysChart("cpu", "#chart-hl-cpu", [
      { label: "CPU util %", data: [], borderColor: C.blue,
        backgroundColor: fill(C.blue), fill: true, tension: .3,
        pointRadius: 0, yAxisID: "y" },
      { label: "core GHz", data: [], borderColor: C.gold,
        tension: .3, pointRadius: 0, yAxisID: "y1" },
    ], { y: { beginAtZero: true, max: 100, position: "left" },
         y1: { beginAtZero: true, position: "right",
               grid: { drawOnChartArea: false } } });
    cpu.data.labels = S.labels;
    cpu.data.datasets[0].data = S.cpu;
    cpu.data.datasets[1].data = S.ghz;
    cpu.update("none");
  },

  /* Live numbers between rungs, so the hero is never stale. */
  onSnapshot(s) {
    if (!this.active) return;
    const offered = s.pool_size, held = s.in_flight;
    $("#hl-held").textContent = held != null
      ? `${Math.round(held).toLocaleString()} / ${(offered ?? 0).toLocaleString()}`
      : "—";
    $("#hl-held-note").textContent =
      (held != null && offered) ? `${Math.round(100 * held / offered)}% of
        what was offered` : "";
    $("#hl-queue").textContent = s.queue_depth != null
      ? Math.round(s.queue_depth).toLocaleString() : "—";

    // Only count a sample once the engine is actually working the
    // rung: warmup and drain points would drag the curve down and
    // invent a dip that never happened.
    if (offered && held != null && /measur/i.test(s.phase || "")) {
      const cur = this._live.get(offered) || {};
      this._live.set(offered, {
        held, tok: this._tel.decode ?? cur.tok ?? null,
      });
      this.drawCurves();
    }
  },

  onTelemetry(t) {
    if (!this.active) return;
    if (t.decode_tok_s != null) {
      this._rate.push(t.decode_tok_s);
      if (this._rate.length > HL_RATE_WINDOW) this._rate.shift();
      const smoothed = windowMean(this._rate);
      this._tel.decode = smoothed;
      $("#hl-now").textContent = Math.round(smoothed).toLocaleString();
      $("#hl-now-note").textContent = this._rate.length >= 3
        ? `mean of the last ${this._rate.length} samples — engines `
          + `advance their token counters in bursts, so a one-second `
          + `rate is not a measurement`
        : "";
    }
    if (t.kv_cache_used_pct != null) {
      $("#hl-kv").textContent = `${t.kv_cache_used_pct.toFixed(0)}%`;
    }
    if (t.gpu_power_w != null) {
      this._tel.power = t.gpu_power_w;
      $("#hl-power").textContent = `${Math.round(t.gpu_power_w)} W`;
    }
    // Tokens per watt is the number that survives a hardware refresh,
    // and it is the one a power-capped rack is actually bounded by.
    if (this._tel.decode && this._tel.power) {
      $("#hl-eff").textContent =
        `${(this._tel.decode / this._tel.power).toFixed(1)} tokens/watt`;
    }

    let host = null, gpus = null;
    try { host = t.host_json ? JSON.parse(t.host_json) : null; } catch { /* skip */ }
    try { gpus = t.gpu_devices_json ? JSON.parse(t.gpu_devices_json) : null;
    } catch { /* skip */ }

    const S = this._sys;
    const push = (arr, v) => {
      arr.push(v == null ? null : v);
      if (arr.length > HL_SYS_WINDOW) arr.shift();
    };
    S.labels.push(fmt.clock(t.sampled_at_ms ?? Date.now()));
    if (S.labels.length > HL_SYS_WINDOW) S.labels.shift();
    push(S.power, t.gpu_power_w);
    push(S.syspower, host?.system_power_w);
    push(S.eff, (this._tel.decode && this._tel.power)
      ? +(this._tel.decode / this._tel.power).toFixed(2) : null);
    push(S.sm, t.gpu_sm_util_pct);
    // Memory-controller busy is the bandwidth-pressure signal, and at
    // large batch it is the one that saturates first.
    const memBusy = gpus?.length
      ? gpus.reduce((a, g) => a + (g.mem_util_pct ?? 0), 0) / gpus.length
      : null;
    push(S.mem, memBusy);
    push(S.vram, t.gpu_vram_used_gb);
    push(S.hostmem, t.memory_used_gb);
    push(S.rss, t.engine_rss_gb);
    push(S.cpu, t.cpu_util_bound_avg ?? t.cpu_util_avg);
    push(S.ghz, t.freq_mhz_mean ? +(t.freq_mhz_mean / 1000).toFixed(2) : null);
    this.drawSystem();
    // #hl-gpu-grid is drawn by Live.renderGpuGrid, which owns that
    // widget; it reads `active` to know the saturation view is up.
  },
};
