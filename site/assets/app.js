/* ============================================================
   System Sizing site — shared logic for Intel + AMD pages.
   Expects:
     window.SIZING_DATA — built by site/build_data.py
     window.PAGE        — per-page config (copy, profiles, defaults)
   ============================================================ */
(function () {
  "use strict";

  const DATA = window.SIZING_DATA;
  const PAGE = window.PAGE;
  if (!DATA || !PAGE) { console.error("Missing SIZING_DATA or PAGE config"); return; }

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  const cohortById = {};
  DATA.cohorts.forEach(c => { cohortById[c.id] = c; });

  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const COLORS = {
    accent: css("--accent") || "#0672cb",
    navy: css("--navy") || "#0d2840",
    good: css("--good") || "#198a5a",
    warn: css("--warn") || "#c07c14",
    bad: css("--bad") || "#c0392b",
    faint: css("--ink-faint") || "#6b8093",
    line: css("--line-strong") || "#c5d6e4",
  };
  const STATUS_COLOR = { pass: COLORS.good, marginal: COLORS.warn, fail: COLORS.bad };

  // ---------- formatting ----------
  const fmtInt = n => Math.round(n).toLocaleString("en-US");
  const fmtUSD = n => "$" + Math.round(n).toLocaleString("en-US");
  const fmtUSD0 = n => n >= 1000 ? "$" + (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "K" : "$" + Math.round(n);
  const fmtMs = n => n == null ? "—" : (n >= 1000 ? (n / 1000).toFixed(2) + " s" : Math.round(n) + " ms");

  // ---------- usage rates ----------
  const RATES = {
    light:   { adoptionPct: 45, hourlyActivePct: 15, inFlightDensityPct: 8,  label: "light" },
    typical: { adoptionPct: 65, hourlyActivePct: 22, inFlightDensityPct: 11, label: "typical" },
    heavy:   { adoptionPct: 85, hourlyActivePct: 35, inFlightDensityPct: 18, label: "heavy" },
  };
  function demandFor(people, presetKey) {
    const r = RATES[presetKey || "typical"];
    const users = people * r.adoptionPct / 100;
    const hourly = users * r.hourlyActivePct / 100;
    return {
      users: Math.round(users),
      hourly: Math.round(hourly),
      concurrent: Math.max(1, Math.round(hourly * r.inFlightDensityPct / 100)),
      rates: r,
    };
  }
  function kneeOf() {
    const gk = cohortById["general_knowledge"];
    return (gk && gk.soft_capacity_pool_size) || 32;
  }
  function failOf() {
    const gk = cohortById["general_knowledge"];
    return (gk && gk.fail_pool_size) || kneeOf() * 2;
  }

  // ---------- reveal on scroll ----------
  const revealObs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("visible"); revealObs.unobserve(e.target); } });
  }, { threshold: 0.12 });
  $$(".reveal").forEach(el => revealObs.observe(el));

  // ============================================================
  // ACT 2 — DEMAND: the dot grid
  // One dot per person; dots light while a request is running.
  // ============================================================
  (function initDemandGrid() {
    const canvas = $("#demand-grid");
    if (!canvas) return;
    const PEOPLE = PAGE.story.people;
    const r = RATES.typical;
    const SPEED = 24;             // simulated seconds per real second
    const REQ_MEAN = 26;          // avg request duration (s)
    const GAP_MEAN = REQ_MEAN * (100 / r.inFlightDensityPct - 1); // ~11% duty cycle

    // population: who is even in the picture this hour
    const nUsers = Math.round(PEOPLE * r.adoptionPct / 100);
    const nHourly = Math.round(nUsers * r.hourlyActivePct / 100);

    const exp = mean => -Math.log(1 - Math.random()) * mean;
    const duty = REQ_MEAN / (REQ_MEAN + GAP_MEAN);
    const people = [];
    for (let i = 0; i < PEOPLE; i++) {
      const isUser = i < nUsers;
      const isHourly = i < nHourly;
      // start in steady state so the live count hovers at its true level from the first frame
      const startInFlight = isHourly && Math.random() < duty;
      people.push({
        isUser, isHourly,
        inFlight: startInFlight,
        t: !isHourly ? Infinity : (startInFlight ? exp(REQ_MEAN) : exp(GAP_MEAN)),
      });
    }
    // shuffle so categories are scattered across the grid
    for (let i = people.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [people[i], people[j]] = [people[j], people[i]];
    }

    const ctx = canvas.getContext("2d");
    let cols = 0, rows = 0, pitch = 0, dotR = 0, dpr = 1;
    function layout() {
      const w = canvas.parentElement.clientWidth;
      cols = w > 760 ? 60 : (w > 480 ? 42 : 30);
      rows = Math.ceil(PEOPLE / cols);
      pitch = w / cols;
      dotR = Math.max(1.6, pitch * 0.28);
      dpr = window.devicePixelRatio || 1;
      canvas.width = w * dpr;
      canvas.height = rows * pitch * dpr;
      canvas.style.height = (rows * pitch) + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    layout();
    window.addEventListener("resize", () => { layout(); draw(); });

    const C_NON = "#e4edf4", C_USER = "#c6d9e8", C_HOURLY = "#9dbfda";
    let peak = 0, elapsedSim = 0;

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let inFlight = 0;
      for (let i = 0; i < PEOPLE; i++) {
        const p = people[i];
        const x = (i % cols) * pitch + pitch / 2;
        const y = Math.floor(i / cols) * pitch + pitch / 2;
        if (p.inFlight) {
          inFlight++;
          ctx.beginPath();
          ctx.fillStyle = COLORS.accent + "33";
          ctx.arc(x, y, dotR * 2.1, 0, 7);
          ctx.fill();
          ctx.beginPath();
          ctx.fillStyle = COLORS.accent;
          ctx.arc(x, y, dotR * 1.15, 0, 7);
          ctx.fill();
        } else {
          ctx.beginPath();
          ctx.fillStyle = p.isHourly ? C_HOURLY : (p.isUser ? C_USER : C_NON);
          ctx.arc(x, y, dotR, 0, 7);
          ctx.fill();
        }
      }
      return inFlight;
    }

    let last = null, running = false, raf = null;
    function loop(ts) {
      if (!running) return;
      if (last != null) {
        const dt = Math.min(0.1, (ts - last) / 1000) * SPEED;
        elapsedSim += dt;
        for (const p of people) {
          if (!p.isHourly) continue;
          p.t -= dt;
          if (p.t <= 0) {
            p.inFlight = !p.inFlight;
            p.t = p.inFlight ? exp(REQ_MEAN) : exp(GAP_MEAN);
          }
        }
      }
      last = ts;
      const now = draw();
      peak = Math.max(peak, now);
      $("#demand-now").textContent = fmtInt(now);
      $("#demand-peak").textContent = fmtInt(peak);
      const mins = Math.floor(elapsedSim / 60);
      $("#demand-clock").textContent = mins >= 60
        ? "one working hour shown · loops continuously"
        : `minute ${mins} of a typical working hour`;
      raf = requestAnimationFrame(loop);
    }
    const obs = new IntersectionObserver(es => {
      es.forEach(e => {
        if (e.isIntersecting && !running) { running = true; last = null; raf = requestAnimationFrame(loop); }
        else if (!e.isIntersecting && running) { running = false; cancelAnimationFrame(raf); }
      });
    }, { threshold: 0.2 });
    obs.observe(canvas);
    draw();

    // static context numbers
    const d = demandFor(PEOPLE, "typical");
    $$("[data-demand-people]").forEach(el => el.textContent = fmtInt(PEOPLE));
    $$("[data-demand-users]").forEach(el => el.textContent = fmtInt(d.users));
    $$("[data-demand-hourly]").forEach(el => el.textContent = fmtInt(d.hourly));
    $$("[data-demand-concurrent]").forEach(el => el.textContent = fmtInt(d.concurrent));
  })();

  // ---------- compact persona legend ----------
  (function initPersonaLegend() {
    const wrap = $("#persona-legend");
    if (!wrap) return;
    wrap.innerHTML = PAGE.personas.map(p => `
      <span class="persona-chip" title="${p.example.replace(/"/g, "&quot;")}">
        <span class="dot" style="background:${p.color}"></span>
        <b>${p.name}</b>
        <span>${p.description}</span>
        <span class="tok">~${fmtInt(p.inTok)} in / ${fmtInt(p.outTok)} out</span>
      </span>`).join("");
  })();

  // ---------- team cards with persona-mix bars ----------
  (function initCohorts() {
    const grid = $("#cohort-grid");
    if (!grid) return;
    const personaColor = {};
    PAGE.personas.forEach(p => personaColor[p.id] = p.color);
    grid.innerHTML = PAGE.cohorts.map(c => {
      const raw = cohortById[c.id] || {};
      const weights = raw.persona_weights || {};
      const segs = Object.entries(weights)
        .filter(([, w]) => w > 0)
        .map(([pid, w]) => `<span style="width:${(w * 100).toFixed(1)}%;background:${personaColor[pid] || "#999"}" title="${pid} ${(w * 100).toFixed(0)}%"></span>`)
        .join("");
      const top = Object.entries(weights).sort((a, b) => b[1] - a[1]).slice(0, 2)
        .map(([pid, w]) => { const p = PAGE.personas.find(x => x.id === pid); return `${p ? p.name : pid} ${(w * 100).toFixed(0)}%`; }).join(" · ");
      return `
      <div class="card cohort-card reveal">
        <h3>${c.name}</h3>
        <div class="desc">${c.description}</div>
        <div class="mix-bar">${segs}</div>
        <div class="mix-legend">mostly ${top}</div>
      </div>`;
    }).join("");
    $$(".reveal", grid).forEach(el => revealObs.observe(el));
  })();

  // ---------- Chart.js setup ----------
  const hasChart = typeof window.Chart !== "undefined";
  if (hasChart) {
    Chart.defaults.font.family = css("--font") || "Inter, sans-serif";
    Chart.defaults.color = COLORS.faint;
    Chart.defaults.borderColor = "rgba(13,40,64,.08)";
  }

  // plugin: dashed vertical markers at x labels + optional horizontal line
  const markerPlugin = {
    id: "markers",
    afterDatasetsDraw(chart) {
      const cfg = chart.options.plugins.markers;
      if (!cfg) return;
      const { ctx, chartArea, scales } = chart;
      (cfg.vlines || []).forEach(l => {
        const idx = chart.data.labels.indexOf(l.label);
        if (idx < 0) return;
        const x = scales.x.getPixelForValue(idx);
        ctx.save();
        ctx.strokeStyle = l.color; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = l.color; ctx.font = "700 10px " + Chart.defaults.font.family;
        ctx.textAlign = "center";
        ctx.fillText(l.text, x, chartArea.top - 6);
        ctx.restore();
      });
      if (cfg.hline) {
        const y = scales.y.getPixelForValue(cfg.hline.value);
        if (y >= chartArea.top && y <= chartArea.bottom) {
          ctx.save();
          ctx.strokeStyle = cfg.hline.color; ctx.setLineDash([3, 4]); ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.moveTo(chartArea.left, y); ctx.lineTo(chartArea.right, y); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = cfg.hline.color; ctx.font = "600 10px " + Chart.defaults.font.family;
          ctx.textAlign = "right";
          ctx.fillText(cfg.hline.text, chartArea.right - 4, y - 5);
          ctx.restore();
        }
      }
    },
  };
  if (hasChart) Chart.register(markerPlugin);

  // ============================================================
  // ACT 3 — THE TEST: results per team
  // ============================================================
  const proof = { cohortId: null, view: "ontime", chart: null };

  function kneeMarkers(raw) {
    const v = [];
    if (raw.soft_capacity_pool_size != null)
      v.push({ label: String(raw.soft_capacity_pool_size), color: COLORS.navy, text: "capacity · " + raw.soft_capacity_pool_size + " users" });
    return v;
  }

  function ontimeConfig(raw) {
    const curve = raw.curve || [];
    return {
      type: "bar",
      data: {
        labels: curve.map(p => String(p.pool_size)),
        datasets: [{
          label: "Requests answered within target",
          data: curve.map(p => Math.max(0, (1 - p.violation_rate) * 100)),
          backgroundColor: curve.map(p => (STATUS_COLOR[p.status] || COLORS.faint) + "cc"),
          borderColor: curve.map(p => STATUS_COLOR[p.status] || COLORS.faint),
          borderWidth: 1.5, borderRadius: 6, maxBarThickness: 64,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        layout: { padding: { top: 18 } },
        scales: {
          x: { title: { display: true, text: "People using the server at the same time" }, grid: { display: false } },
          y: { min: 0, max: 100, title: { display: true, text: "% of requests answered within target" }, ticks: { callback: v => v + "%" } },
        },
        plugins: {
          legend: { display: false },
          markers: { vlines: kneeMarkers(raw), hline: { value: 95, color: COLORS.faint, text: "target · 95% on time" } },
          tooltip: {
            callbacks: {
              title: items => items[0].label + " simultaneous users",
              label: ctx => ` ${ctx.parsed.y.toFixed(1)}% answered within target`,
              afterBody: items => {
                const p = (raw.curve || [])[items[0].dataIndex];
                return [
                  `Answer starts in ${fmtMs(p.ttft_p95_ms)} or less for 95% of requests`,
                  `Verdict: ${p.status}`,
                ];
              },
            },
          },
        },
      },
    };
  }

  function latencyConfig(raw) {
    const curve = raw.curve || [];
    const pointColors = curve.map(p => STATUS_COLOR[p.status] || COLORS.faint);
    return {
      type: "line",
      data: {
        labels: curve.map(p => String(p.pool_size)),
        datasets: [
          { label: "Wait for answer to start (p95)", data: curve.map(p => p.ttft_p95_ms),
            borderColor: COLORS.accent, backgroundColor: COLORS.accent,
            pointBackgroundColor: pointColors, pointBorderColor: pointColors,
            pointRadius: 5, pointHoverRadius: 7, borderWidth: 2.5, tension: 0.35 },
          { label: "Time per streamed token (p95)", data: curve.map(p => p.tpot_p95_ms),
            borderColor: COLORS.navy, backgroundColor: COLORS.navy, borderDash: [6, 4],
            pointBackgroundColor: pointColors, pointBorderColor: pointColors,
            pointRadius: 5, pointHoverRadius: 7, borderWidth: 2, tension: 0.35 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 18 } },
        scales: {
          x: { title: { display: true, text: "People using the server at the same time" }, grid: { display: false } },
          y: { type: "logarithmic", title: { display: true, text: "Milliseconds (log scale)" },
               ticks: { callback: v => ([10, 50, 100, 500, 1000, 5000, 10000, 50000].includes(v) ? fmtInt(v) : null) } },
        },
        plugins: {
          legend: { labels: { boxWidth: 18, boxHeight: 3, font: { size: 11.5 } } },
          markers: { vlines: kneeMarkers(raw) },
          tooltip: {
            callbacks: {
              title: items => items[0].label + " simultaneous users",
              label: ctx => ` ${ctx.dataset.label}: ${fmtMs(ctx.parsed.y)}`,
            },
          },
        },
      },
    };
  }

  function renderProof() {
    const copy = PAGE.cohorts.find(c => c.id === proof.cohortId);
    const raw = cohortById[proof.cohortId];
    if (!raw || !copy) return;

    if (proof.chart) proof.chart.destroy();
    if (hasChart) {
      const cfg = proof.view === "latency" ? latencyConfig(raw) : ontimeConfig(raw);
      proof.chart = new Chart($("#proof-chart"), cfg);
    }

    const knee = raw.soft_capacity_pool_size, fail = raw.fail_pool_size;
    $("#ro-knee").textContent = knee != null ? fmtInt(knee) : "—";
    $("#ro-fail").textContent = fail != null ? fmtInt(fail) : "—";
    const tp = raw.capacity_throughput || {};
    $("#ro-tput").textContent = tp.visible_output_tok_per_s != null ? tp.visible_output_tok_per_s.toFixed(0) : "—";
    $("#ro-tput-note").textContent = tp.pool_size != null
      ? `combined output across all users at ${tp.pool_size} simultaneous`
      : "";
    const bottleneckText = {
      decode_throughput: "Text-generation speed: the cores can only produce so many tokens per second in total.",
      frequency_droop: "Core clock speed: under full load the CPU settles below its peak frequency, which slows generation.",
      kv_cache: "Memory for active conversations: the cache that holds in-progress context fills up.",
    };
    $("#ro-bottleneck").textContent =
      bottleneckText[raw.bottleneck] || ("Limiting factor: " + String(raw.bottleneck || "—").replace(/_/g, " ") + ".");
    $("#ro-summary").textContent = copy.summaryLine;
    $("#ro-rec").innerHTML = "<b>If this team needs more room:</b> " + copy.recommendation;
  }

  (function initProof() {
    const row = $("#proof-tabs");
    if (!row) return;
    row.innerHTML = PAGE.cohorts.map((c, i) =>
      `<button class="tab-btn${i === 0 ? " active" : ""}" data-cohort="${c.id}">${c.name}</button>`).join("");
    proof.cohortId = PAGE.cohorts[0].id;
    row.addEventListener("click", e => {
      const btn = e.target.closest(".tab-btn");
      if (!btn) return;
      $$(".tab-btn", row).forEach(b => b.classList.toggle("active", b === btn));
      proof.cohortId = btn.dataset.cohort;
      renderProof();
    });
    $$("#proof-view button").forEach(b => b.addEventListener("click", () => {
      proof.view = b.dataset.view;
      $$("#proof-view button").forEach(x => x.classList.toggle("active", x === b));
      renderProof();
    }));
    renderProof();
  })();

  // ============================================================
  // ACT 4 — THE FIT: demand vs measured capacity
  // ============================================================
  const fit = { people: (PAGE.story && PAGE.story.people) || 1500, preset: "typical" };
  const tco = { ...PAGE.tco };
  let tcoChart = null;

  function renderFit() {
    const bar = $("#cap-bar");
    if (!bar) return;
    const knee = kneeOf(), fail = failOf();
    const d = demandFor(fit.people, fit.preset);
    const max = Math.max(fail * 1.15, d.concurrent * 1.1);

    const pct = v => Math.min(99, v / max * 100);
    const marker = d.concurrent <= knee ? "" : (d.concurrent <= fail ? "warn" : "bad");
    bar.innerHTML = `
      <div class="cap-zone ok" style="left:0;width:${pct(knee)}%"><span class="zlabel">responsive</span></div>
      <div class="cap-zone slow" style="left:${pct(knee)}%;width:${pct(fail) - pct(knee)}%"><span class="zlabel">slower at peak</span></div>
      <div class="cap-zone over" style="left:${pct(fail)}%;width:${100 - pct(fail)}%"><span class="zlabel">over capacity</span></div>
      <div class="cap-tick" style="left:${pct(knee)}%"><span class="tlabel">measured capacity · ${knee}</span></div>
      <div class="cap-marker ${marker}" style="left:${pct(d.concurrent)}%"><span class="mlabel">your peak · ${fmtInt(d.concurrent)}</span></div>`;

    const r = RATES[fit.preset];
    const outgrow = Math.floor(knee / (r.adoptionPct / 100 * r.hourlyActivePct / 100 * r.inFlightDensityPct / 100));
    const ratio = knee / d.concurrent;
    let verdict;
    if (d.concurrent <= knee * 0.75) {
      verdict = `A group of <b>${fmtInt(fit.people)}</b> at ${r.label} usage peaks at about <b>${fmtInt(d.concurrent)} simultaneous requests</b> — inside the measured capacity of ${knee}, with ${ratio.toFixed(1)}× headroom. At this usage level, one server holds up to roughly <b>${fmtInt(outgrow)} people</b> before reaching its measured limit.`;
    } else if (d.concurrent <= knee) {
      verdict = `A group of <b>${fmtInt(fit.people)}</b> at ${r.label} usage peaks at about <b>${fmtInt(d.concurrent)} simultaneous requests</b> — within the measured capacity of ${knee}, but with little reserve. Consider the next configuration up if usage may grow.`;
    } else if (d.concurrent <= fail) {
      verdict = `A group of <b>${fmtInt(fit.people)}</b> at ${r.label} usage peaks at about <b>${fmtInt(d.concurrent)} simultaneous requests</b> — past the measured capacity of ${knee}. The server stays up, but people will wait at busy times. A larger configuration from the order sheet below is the better fit.`;
    } else {
      verdict = `A group of <b>${fmtInt(fit.people)}</b> at ${r.label} usage peaks at about <b>${fmtInt(d.concurrent)} simultaneous requests</b> — beyond what one of these servers handles. Plan a larger configuration or multiple servers from the order sheet below.`;
    }
    $("#fit-verdict").innerHTML = verdict;
    renderTco();
  }

  (function initFit() {
    const input = $("#fit-emp");
    if (!input) return;
    input.value = fit.people;
    const out = $('output[for="fit-emp"]');
    const sync = () => {
      fit.people = +input.value;
      out.textContent = fmtInt(+input.value) + " people";
      const pct = (input.value - input.min) / (input.max - input.min) * 100;
      input.style.setProperty("--pct", pct + "%");
      renderFit();
    };
    input.addEventListener("input", sync);
    $$(".preset-btn").forEach(b => b.addEventListener("click", () => {
      fit.preset = b.dataset.preset;
      $$(".preset-btn").forEach(x => x.classList.toggle("active", x === b));
      renderFit();
    }));
    sync();
  })();

  // ============================================================
  // ACT 5 — REPLAY: measurement window playback
  // ============================================================
  const replay = {
    cohortId: PAGE.cohorts[0].id,
    speed: 16, playing: false, clock: 0, lastTs: null,
    emitted: 0, ttfts: [], outTok: 0, violations: 0,
    chart: null, raf: null, done: false,
  };

  function replayData() {
    const raw = cohortById[replay.cohortId];
    return raw && raw.replay ? raw.replay : null;
  }

  function resetReplay() {
    cancelAnimationFrame(replay.raf);
    replay.clock = 0; replay.lastTs = null; replay.emitted = 0;
    replay.ttfts = []; replay.outTok = 0; replay.violations = 0; replay.done = false;
    $("#turn-feed").innerHTML = `<div class="fine" style="padding:20px 4px">Press play. Each row below is one request from the benchmark log, shown at the moment it completed.</div>`;
    updateReplayStats();
    const rp = replayData();
    $("#replay-pool").textContent = rp ? fmtInt(rp.pool_size) : "—";
    $("#replay-duration").textContent = rp ? Math.round(rp.duration_s) + " s window" : "";
    buildTelemetryChart();
    drawClock();
  }

  function updateReplayStats() {
    $("#rs-done").textContent = fmtInt(replay.emitted);
    const med = (() => {
      if (!replay.ttfts.length) return null;
      const s = [...replay.ttfts].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)];
    })();
    $("#rs-ttft").textContent = med != null ? fmtMs(med) : "—";
    $("#rs-tput").textContent = replay.clock > 1 ? (replay.outTok / replay.clock).toFixed(0) : "—";
    $("#rs-sla").textContent = replay.emitted ? (100 - replay.violations / replay.emitted * 100).toFixed(0) + "%" : "—";
  }

  function drawClock() {
    const rp = replayData();
    const total = rp ? rp.duration_s : 0;
    $("#replay-clock").textContent =
      `t = ${replay.clock.toFixed(1)} s / ${Math.round(total)} s · ${replay.speed}× speed`;
  }

  function buildTelemetryChart() {
    const rp = replayData();
    const el = $("#telemetry-chart");
    if (!el || !hasChart) return;
    if (replay.chart) replay.chart.destroy();
    const tel = rp ? rp.telemetry : [];
    replay.chart = new Chart(el, {
      type: "line",
      data: {
        labels: tel.map(s => s.t_s),
        datasets: [
          { label: "CPU clock (GHz)", data: tel.map(() => null), yAxisID: "y", borderColor: COLORS.accent, borderWidth: 2, pointRadius: 0, tension: .3 },
          { label: "Conversation memory used (%)", data: tel.map(() => null), yAxisID: "y2", borderColor: COLORS.warn, borderWidth: 2, pointRadius: 0, tension: .3 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { display: false },
          y: { position: "left", title: { display: true, text: "GHz" }, suggestedMin: 0 },
          y2: { position: "right", title: { display: true, text: "%" }, suggestedMin: 0, suggestedMax: 100, grid: { drawOnChartArea: false } },
        },
        plugins: { legend: { labels: { boxWidth: 16, boxHeight: 3, font: { size: 10.5 } } } },
      },
    });
  }

  function feedTelemetry() {
    const rp = replayData();
    if (!rp || !replay.chart) return;
    const tel = rp.telemetry;
    let changed = false;
    tel.forEach((s, i) => {
      if (s.t_s <= replay.clock && replay.chart.data.datasets[0].data[i] == null) {
        replay.chart.data.datasets[0].data[i] = s.freq_ghz;
        replay.chart.data.datasets[1].data[i] = s.kv_pct;
        changed = true;
      }
    });
    if (changed) replay.chart.update("none");
  }

  function emitTurns() {
    const rp = replayData();
    if (!rp) return;
    const feed = $("#turn-feed");
    const personaMeta = {};
    PAGE.personas.forEach(p => personaMeta[p.id] = p);
    while (replay.emitted < rp.turns.length) {
      const t = rp.turns[replay.emitted];
      const doneAt = t.t_s + (t.end_to_end_ms || 0) / 1000;
      if (doneAt > replay.clock) break;
      replay.emitted++;
      replay.ttfts.push(t.ttft_ms);
      replay.outTok += t.output_tokens || 0;
      const bad = (t.sla_ttft_violation || t.sla_tpot_violation);
      if (bad) replay.violations++;
      const p = personaMeta[t.persona_id] || { name: t.persona_id, color: "#999" };
      if (replay.emitted === 1) feed.innerHTML = "";
      const div = document.createElement("div");
      div.className = "turn-item";
      div.innerHTML = `
        <span class="pdot" style="background:${p.color}"></span>
        <span class="who">${p.name} <span>· ${fmtInt(t.input_tokens)} in / ${fmtInt(t.output_tokens)} out</span></span>
        <span class="nums">answer started in ${fmtMs(t.ttft_ms)}</span>
        <span class="verdict ${bad ? "slow" : "ok"}">${bad ? "SLOW" : "ON TIME"}</span>`;
      feed.prepend(div);
      while (feed.children.length > 9) feed.removeChild(feed.lastChild);
    }
    if (replay.emitted >= rp.turns.length && !replay.done) {
      replay.done = true;
      replay.playing = false;
      $("#replay-play").textContent = "↻ Replay";
      const div = document.createElement("div");
      div.className = "turn-item";
      div.innerHTML = `<span class="pdot" style="background:${COLORS.good}"></span>
        <span class="who">End of measurement window</span>
        <span class="nums">${fmtInt(rp.turns.length)} requests shown</span>
        <span class="verdict ok">DONE</span>`;
      feed.prepend(div);
    }
  }

  function replayLoop(ts) {
    if (!replay.playing) return;
    if (replay.lastTs != null) replay.clock += (ts - replay.lastTs) / 1000 * replay.speed;
    replay.lastTs = ts;
    emitTurns();
    feedTelemetry();
    updateReplayStats();
    drawClock();
    if (replay.playing) replay.raf = requestAnimationFrame(replayLoop);
  }

  (function initReplay() {
    const section = $("#replay");
    if (!section) return;
    const sel = $("#replay-cohort");
    sel.innerHTML = PAGE.cohorts.filter(c => cohortById[c.id] && cohortById[c.id].replay)
      .map(c => `<option value="${c.id}">${c.name}</option>`).join("");
    sel.addEventListener("change", () => {
      replay.cohortId = sel.value; replay.playing = false;
      $("#replay-play").textContent = "⏵ Play";
      resetReplay();
    });
    $("#replay-play").addEventListener("click", () => {
      if (replay.done) resetReplay();
      replay.playing = !replay.playing;
      $("#replay-play").textContent = replay.playing ? "⏸ Pause" : "⏵ Play";
      replay.lastTs = null;
      if (replay.playing) replay.raf = requestAnimationFrame(replayLoop);
    });
    $("#replay-reset").addEventListener("click", () => {
      replay.playing = false; $("#replay-play").textContent = "⏵ Play"; resetReplay();
    });
    $$("[data-speed]").forEach(b => b.addEventListener("click", () => {
      replay.speed = +b.dataset.speed;
      $$("[data-speed]").forEach(x => x.classList.toggle("active", x === b));
      drawClock();
    }));
    resetReplay();
    const obs = new IntersectionObserver(es => {
      es.forEach(e => {
        if (e.isIntersecting && !replay.playing && replay.emitted === 0) {
          obs.unobserve(e.target);
          $("#replay-play").click();
        }
      });
    }, { threshold: 0.35 });
    obs.observe(section);
  })();

  // ============================================================
  // ACT 6 — COST
  // ============================================================
  function blendedTokensPerRequest() {
    let inSum = 0, outSum = 0, wSum = 0;
    PAGE.cohorts.forEach(c => {
      const raw = cohortById[c.id];
      if (!raw) return;
      let ci = 0, co = 0, cw = 0;
      Object.entries(raw.persona_weights || {}).forEach(([pid, w]) => {
        const p = PAGE.personas.find(x => x.id === pid);
        if (!p) return;
        ci += p.inTok * w; co += p.outTok * w; cw += w;
      });
      if (!cw) return;
      inSum += (ci / cw) * c.defaultMixPct;
      outSum += (co / cw) * c.defaultMixPct;
      wSum += c.defaultMixPct;
    });
    return wSum ? { inTok: inSum / wSum, outTok: outSum / wSum } : { inTok: 1000, outTok: 700 };
  }

  function renderTco() {
    const el = $("#tco-chart");
    if (!el) return;
    const d = demandFor(fit.people, fit.preset);
    const toks = blendedTokensPerRequest();
    const monthlyRequests = d.concurrent * tco.requestsPerInflightHour * tco.businessHoursPerMonth;
    const inMtok = monthlyRequests * toks.inTok / 1e6;
    const outMtok = monthlyRequests * toks.outTok / 1e6;

    const basePower = tco.powerW * 24 * 30 * tco.powerKwhCost / 1000;
    const opex = basePower * (1 + tco.coolingOverheadPct / 100) + tco.adminYearlyCost / 12 + tco.rackMonthlyCost;

    const mid = PAGE.comparators[tco.comparatorIndex] || PAGE.comparators[0];
    const apiRate = inMtok * mid.inputPerMtok + outMtok * mid.outputPerMtok;

    const months = [], onPrem = [], api = [];
    let breakeven = null;
    for (let m = 0; m <= 36; m++) {
      months.push(m);
      const op = tco.serverCost + opex * m;
      onPrem.push(op); api.push(apiRate * m);
      if (breakeven == null && m > 0 && op < apiRate * m) breakeven = m;
    }

    $("#tco-api-mo").textContent = fmtUSD(apiRate);
    $("#tco-opex-mo").textContent = fmtUSD(opex);
    $("#tco-breakeven").textContent = breakeven != null ? breakeven + " mo" : "> 36 mo";
    const verdictEl = $("#tco-verdict");
    if (verdictEl) {
      verdictEl.textContent = breakeven != null
        ? `At this volume, total on-premises cost drops below the API bill in month ${breakeven}.`
        : "At this volume, the API subscription remains cheaper over three years. On-premises is justified here by data control and predictable cost rather than savings; the picture changes as the group or its usage grows.";
    }
    $("#tco-req-note").textContent =
      `${fmtInt(monthlyRequests)} requests/mo · ≈${fmtInt(toks.inTok)} input + ${fmtInt(toks.outTok)} output tokens each · compared against ${mid.name}`;

    const vlines = [];
    if (breakeven != null) vlines.push({ label: String(breakeven), color: COLORS.bad, text: "break-even · month " + breakeven });

    const cfg = {
      type: "line",
      data: {
        labels: months.map(String),
        datasets: [
          { label: "On-premises, cumulative (server + power + admin)", data: onPrem, borderColor: COLORS.accent, borderWidth: 3, pointRadius: 0, tension: .15 },
          { label: `Cloud API, cumulative (${mid.name})`, data: api, borderColor: COLORS.bad, borderWidth: 2, pointRadius: 0, tension: .15 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 18 } },
        scales: {
          x: { title: { display: true, text: "Month" }, ticks: { maxTicksLimit: 13 }, grid: { display: false } },
          y: { title: { display: true, text: "Cumulative spend" }, ticks: { callback: v => fmtUSD0(v) } },
        },
        plugins: {
          legend: { labels: { boxWidth: 18, boxHeight: 3, font: { size: 11.5 } } },
          markers: { vlines },
          tooltip: { callbacks: { title: i => "Month " + i[0].label, label: c => ` ${c.dataset.label}: ${fmtUSD(c.parsed.y)}` } },
        },
      },
    };
    if (tcoChart) tcoChart.destroy();
    if (hasChart) tcoChart = new Chart(el, cfg);
  }

  (function initTco() {
    const sel = $("#tco-comparator");
    if (!sel) return;
    sel.innerHTML = PAGE.comparators.map((c, i) =>
      `<option value="${i}"${i === tco.comparatorIndex ? " selected" : ""}>${c.name} — $${c.inputPerMtok.toFixed(2)} in / $${c.outputPerMtok.toFixed(2)} out per Mtok</option>`).join("");
    sel.addEventListener("change", () => { tco.comparatorIndex = +sel.value; renderTco(); });
    const sliders = [
      ["tco-server", "serverCost", v => fmtUSD0(v)],
      ["tco-power", "powerW", v => v + " W"],
      ["tco-kwh", "powerKwhCost", v => "$" + (+v).toFixed(2) + "/kWh"],
      ["tco-admin", "adminYearlyCost", v => fmtUSD0(v) + "/yr"],
      ["tco-reqs", "requestsPerInflightHour", v => v + "/hr"],
    ];
    sliders.forEach(([id, key, fmt]) => {
      const input = $("#" + id);
      if (!input) return;
      input.value = tco[key];
      const out = $(`output[for="${id}"]`);
      const sync = () => {
        tco[key] = +input.value;
        out.textContent = fmt(+input.value);
        const pct = (input.value - input.min) / (input.max - input.min) * 100;
        input.style.setProperty("--pct", pct + "%");
        renderTco();
      };
      input.addEventListener("input", sync);
      sync();
    });
  })();

  // ============================================================
  // ACT 7 — ORDER SHEET: configuration tiers
  // ============================================================
  (function initTiers() {
    const grid = $("#tier-grid");
    if (!grid) return;
    const knee = kneeOf();
    grid.innerHTML = PAGE.profiles.map(p => {
      const cap = Math.round(knee * p.capacityMultiplier);
      const cost = p.costMin === p.costMax ? fmtUSD0(p.costMin) : `${fmtUSD0(p.costMin)}–${fmtUSD0(p.costMax)}`;
      return `
      <div class="card profile-card${p.isTested ? " recommended" : ""}">
        <h3>${p.name}
          ${p.isTested ? '<span class="badge tested">Tested</span>' : '<span class="badge projected">Estimated</span>'}
        </h3>
        <div class="fit">suits ${p.bestFit} people</div>
        <div class="spec">
          <div class="row"><span>CPU</span><b>${p.cpu}</b></div>
          <div class="row"><span>Memory</span><b>${p.memory}</b></div>
          <div class="row"><span>Handles</span><b>≈ ${cap} simultaneous</b></div>
          <div class="row"><span>Price range</span><b>${cost}</b></div>
        </div>
      </div>`;
    }).join("");
  })();

  // ---------- model benchmark table (inside details) ----------
  (function initCapabilities() {
    const tbody = $("#cap-tbody");
    if (!tbody || !PAGE.capabilities) return;
    const rows = [];
    PAGE.capabilities.forEach(g => {
      rows.push(`<tr class="group-row"><td colspan="4">${g.group}</td></tr>`);
      g.items.forEach(it => {
        const localWin = it.local >= it.cloud;
        rows.push(`<tr>
          <td>${it.capability}</td>
          <td class="num ${localWin ? "winner" : ""}">${it.local.toFixed(1)}</td>
          <td class="num ${!localWin ? "winner" : ""}">${it.cloud.toFixed(1)}</td>
          <td><span class="cap-bar" style="width:${it.local}px"></span><br><span class="cap-bar cloud" style="width:${it.cloud}px"></span></td>
        </tr>`);
      });
    });
    tbody.innerHTML = rows.join("");
  })();

  // data stamp
  $$("[data-generated]").forEach(el => {
    if (DATA.generated_at) el.textContent = new Date(DATA.generated_at).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  });
})();
