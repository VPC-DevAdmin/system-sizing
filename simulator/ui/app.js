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
  hw: { gpus: 0 },
  modelList: [],
  deviceMode: "gpu",
  matchedProfile: null,   // {name, ...profile} when an optimization fits
  engineDefaults: {},     // the values Advanced was prefilled with
  _dlPoll: null,

  async init() {
    await this.loadCatalogs();
    this.refreshRuns();
    $("#bench-model").addEventListener("change", () => this.onModelChange());
    $("#workload-select").addEventListener("change", () => this.showDetails());
    document.querySelectorAll("#device-seg button").forEach(b =>
      b.addEventListener("click", () => {
        this.deviceMode = b.dataset.dev;
        this.renderDeviceSeg();
        this.onModelChange();
      }));
    // Any hand edit to the engine settings is STICKY: nothing may
    // silently reset the form after the user shaped it (a reverted
    // KV dropdown once ran "auto" when the user meant fp8).
    for (const id of ["eng-replicas", "eng-tp", "eng-placement",
                      "eng-gmu", "eng-mns", "eng-mbt", "eng-kv",
                      "eng-ep"]) {
      $("#" + id).addEventListener("input", () => {
        this._engineDirty = true;
      });
    }
    $("#start-btn").addEventListener("click", () => this.start());
    $("#stop-btn").addEventListener("click", () => this.stop());
    $("#runs-refresh").addEventListener("click", () => this.refreshRuns());
    $("#doctor-btn").addEventListener("click", () => this.doctor());
    setInterval(() => this.pollStatus(), 2000);
    this.pollStatus();
  },

  async loadCatalogs() {
    try {
      const [profiles, personas, cohorts, models, hw] = await Promise.all([
        api("/api/profiles"), api("/api/personas"), api("/api/cohorts"),
        api("/api/models").catch(() => ({ models: [] })),
        api("/api/hardware").catch(() => ({ gpus: 0 })),
      ]);
      this.catalogs = { profiles, personas, cohorts };
      this.modelList = models.models || [];
      this.hw = hw;
      this.deviceMode = hw.gpus > 0 ? "gpu" : "cpu";
      this.renderDeviceSeg();
      this.fillModels();
      this.fillWorkloadPicker();
      this.onModelChange();
      this.showDetails();
    } catch (e) {
      this.msg(`catalog load failed: ${e.message}`, "error");
    }
  },

  renderDeviceSeg() {
    const seg = $("#device-seg");
    seg.classList.toggle("disabled", !(this.hw.gpus > 0));
    if (!(this.hw.gpus > 0)) this.deviceMode = "cpu";
    seg.querySelectorAll("button").forEach(b =>
      b.classList.toggle("active", b.dataset.dev === this.deviceMode));
  },

  /* The model IS the choice — profiles are an implementation detail.
   * Downloaded models lead; the rest are pickable but need a download
   * first (offered inline). */
  fillModels() {
    const sel = $("#bench-model");
    const prev = sel.value;
    sel.innerHTML = "";
    const cached = this.modelList.filter(m => m.cached);
    const rest = this.modelList.filter(m => !m.cached);
    const mkGroup = (label, items) => {
      if (!items.length) return;
      const g = document.createElement("optgroup");
      g.label = label;
      for (const m of items) {
        g.append(new Option(m.model, m.model, false, m.model === prev));
      }
      sel.append(g);
    };
    mkGroup("Downloaded", cached);
    mkGroup("Not downloaded", rest);
    if (!prev) {
      // Default to the optimized model when one exists, else the
      // first downloaded model.
      const opt = Object.values(this.catalogs.profiles).find(p =>
        p?.optimized && p.fits_hardware
        && cached.some(m => m.model === p.model_id));
      sel.value = opt?.model_id ?? cached[0]?.model ?? rest[0]?.model ?? "";
    }
  },

  modelEntry() {
    return this.modelList.find(m => m.model === $("#bench-model").value);
  },

  /* Conservative fallback when no optimization exists: one replica,
   * enough TP to fit the weights, stock engine settings. */
  conservativeDefaults(entry) {
    let tp = 1;
    const vram = this.hw.vram_per_gpu_gb;
    if (entry?.approx_size_gb && vram) {
      while (tp < Math.max(1, this.hw.gpus)
             && entry.approx_size_gb > 0.85 * vram * tp) tp *= 2;
    }
    return { replicas: 1, tp, placement: "pack",
             gpu_memory_utilization: 0.9, max_num_seqs: "",
             max_num_batched_tokens: "", kv_cache_dtype: "",
             expert_parallel: false };
  },

  setEngineForm(d) {
    $("#eng-replicas").value = d.replicas ?? 1;
    $("#eng-tp").value = d.tp ?? 1;
    $("#eng-placement").value = d.placement ?? "pack";
    $("#eng-gmu").value = d.gpu_memory_utilization ?? 0.9;
    $("#eng-mns").value = d.max_num_seqs ?? "";
    $("#eng-mbt").value = d.max_num_batched_tokens ?? "";
    $("#eng-kv").value =
      (d.kv_cache_dtype && d.kv_cache_dtype !== "auto")
        ? d.kv_cache_dtype : "";
    $("#eng-ep").value = d.expert_parallel ? "on" : "off";
    this.engineDefaults = this.readEngineForm();
    this._engineDirty = false;
  },

  readEngineForm() {
    return {
      replicas: +$("#eng-replicas").value || 1,
      tp: +$("#eng-tp").value || 1,
      placement: $("#eng-placement").value,
      gpu_memory_utilization: +$("#eng-gmu").value || 0.9,
      max_num_seqs: $("#eng-mns").value.trim(),
      max_num_batched_tokens: $("#eng-mbt").value.trim(),
      kv_cache_dtype: $("#eng-kv").value,
      expert_parallel: $("#eng-ep").value === "on",
    };
  },

  /* Human summary of what an engine form means — echoed at start and
   * in the run banner so there is never doubt about what's running. */
  engineSummary(form) {
    return `${form.replicas}×tp${form.tp} ${form.placement}`
      + ` · gmu ${form.gpu_memory_utilization}`
      + (form.max_num_seqs ? ` · mns ${form.max_num_seqs}` : "")
      + (form.max_num_batched_tokens
          ? ` · mbt ${form.max_num_batched_tokens}` : "")
      + ` · KV ${form.kv_cache_dtype || "auto"}`
      + (form.expert_parallel ? " · EP on" : "");
  },

  /* Model or device changed: find a fitting optimization, prefill
   * Advanced from it (else conservative), and set the note line —
   * green check, optimize link, or download link. */
  onModelChange() {
    const model = $("#bench-model").value;
    const entry = this.modelEntry();
    const cpuMode = this.deviceMode === "cpu";
    const gpuEngines = ["vllm_cuda", "vllm_cuda_multi"];
    this.matchedProfile = null;
    for (const [name, p] of Object.entries(this.catalogs.profiles)) {
      if (!p?.optimized || p.model_id !== model || !p.fits_hardware) continue;
      const isGpu = gpuEngines.includes(p.engine_type);
      if (isGpu === !cpuMode) { this.matchedProfile = { name, ...p }; break; }
    }
    $("#engine-form").style.display = cpuMode ? "none" : "";
    $("#engine-note").textContent = cpuMode
      ? "CPU engine — conservative stock settings (no searched dimensions)."
      : this.matchedProfile
        ? "Prefilled from the optimized engine — change anything to run a variant."
        : "No optimization for this model yet — conservative defaults below.";
    // Reset the engine form ONLY when the model/device actually
    // changed, or the user hasn't touched it. A hand-shaped form
    // (say, KV set to fp8) must survive every incidental refresh —
    // silently reverting it once launched the wrong engine.
    const key = `${model}|${this.deviceMode}`;
    const keyChanged = key !== this._formKey;
    this._formKey = key;
    if (!cpuMode && (keyChanged || !this._engineDirty)) {
      this.setEngineForm(this.matchedProfile?.engine
        ?? this.conservativeDefaults(entry));
    }
    this.renderModelNote(model, entry);
    this.fetchHeadlineShape();
  },

  /* Does this model's FAMILY have a stored optimal shape? Cached on
   * the controller; the workload note reads it. */
  async fetchHeadlineShape() {
    const model = $("#bench-model").value;
    if (!model) { this.headlineShape = null; return; }
    try {
      this.headlineShape = await api(
        `/api/headline-shape?model=${encodeURIComponent(model)}`);
    } catch { this.headlineShape = null; }
    this.updateWorkloadNote();
  },

  renderModelNote(model, entry) {
    const note = $("#model-note");
    note.innerHTML = "";
    if (!model) { note.textContent = "no models in the catalog"; return; }
    if (entry && !entry.cached) {
      const size = entry.approx_size_gb
        ? ` (≈${Math.round(entry.approx_size_gb)} GB)` : "";
      note.innerHTML = `<span id="dl-slot"><button type="button"
        class="note-link" id="model-dl">⬇ download this model${size}</button>
        </span>`;
      $("#model-dl").addEventListener("click", () => this.downloadModel(model));
      return;
    }
    if (this.matchedProfile) {
      note.innerHTML = `<span class="ok-note">✓ Using optimized engine
        — ${this.matchedProfile.detail || this.matchedProfile.label}</span>`;
    } else {
      note.innerHTML = `<button type="button" class="note-link"
        id="goto-optimize">No optimized engine for this model yet —
        run the optimizer →</button>`;
      $("#goto-optimize").addEventListener("click", () =>
        document.querySelector('#tabs button[data-view="optimizer"]')?.click());
    }
  },

  async downloadModel(model) {
    try {
      await api("/api/models/download", {
        method: "POST", body: JSON.stringify({ model }),
      });
    } catch (e) {
      if (!`${e.message}`.includes("already running")) {
        this.msg(e.message, "error");
        return;
      }
    }
    const slot = $("#dl-slot");
    if (slot) {
      slot.innerHTML = `downloading<span class="dl-bar"><i
        id="dl-bar-i" style="width:0%"></i></span><span id="dl-pct">…</span>`;
    }
    clearInterval(this._dlPoll);
    this._dlPoll = setInterval(async () => {
      let doc;
      try { doc = await api("/api/models"); } catch { return; }
      this.modelList = doc.models || [];
      const e = this.modelList.find(m => m.model === model);
      const dl = doc.downloads?.[model];
      if (e?.cached) {
        clearInterval(this._dlPoll);
        this.fillModels();
        this.onModelChange();
        return;
      }
      // hf's progress lines carry percentages — show the last one.
      const pcts = [...(dl?.log_tail || "").matchAll(/(\d{1,3})%/g)];
      const pct = pcts.length ? +pcts[pcts.length - 1][1] : null;
      if (pct != null && $("#dl-bar-i")) {
        $("#dl-bar-i").style.width = `${pct}%`;
        $("#dl-pct").textContent = `${pct}%`;
      }
      if (dl && !dl.running && !e?.cached) {
        clearInterval(this._dlPoll);
        if (slot) slot.innerHTML =
          `<span class="status-fail">download failed — see Prepare tab</span>`;
      }
    }, 2500);
  },

  /* One workload picker, in plain language: team mixes first (that's
   * what capacity questions are usually about), single user types
   * for isolation, and "everything" spelled out honestly. */
  fillWorkloadPicker() {
    const sel = $("#workload-select");
    const prev = sel.value;
    sel.innerHTML = "";
    const g1 = document.createElement("optgroup");
    g1.label = "Team workload mixes";
    for (const c of this.catalogs.cohorts) {
      g1.append(new Option(`${c.name}`, `cohort:${c.id}`, false,
        `cohort:${c.id}` === prev));
    }
    sel.append(g1);
    const g2 = document.createElement("optgroup");
    g2.label = "Single user type";
    const headline = [];
    for (const p of this.catalogs.personas) {
      const opt = new Option(p.name || p.id.replaceAll("_", " "),
        `persona:${p.id}`, false, `persona:${p.id}` === prev);
      if (p.id.startsWith("headline_")) headline.push(opt);
      else g2.append(opt);
    }
    sel.append(g2);
    if (headline.length) {
      const g3 = document.createElement("optgroup");
      g3.label = "Headline stress — marketing numbers, not capacity";
      headline.forEach(o => g3.append(o));
      sel.append(g3);
    }
    // The "sweep everything" option is gone with the closed-loop UI —
    // sweeps run the legacy pool ramp and belong to the API/CLI only.
    if (!prev) sel.value = `cohort:${this.catalogs.cohorts[0]?.id ?? ""}`;
  },

  showDetails() {
    const w = $("#workload-select").value || "";
    let text = "";
    if (w.startsWith("cohort:")) {
      const c = this.catalogs.cohorts.find(x => x.id === w.slice(7));
      text = c?.description ?? "";
      if (c?.blended) {
        const b = c.blended;
        text += ` · typical turn ~${b.input_tokens} tok in → `
          + `${b.output_tokens} out, ~${b.think_gap_s}s between turns`;
      }
    } else if (w.startsWith("persona:")) {
      const per = this.catalogs.personas.find(x => x.id === w.slice(8));
      text = per?.description ?? "";
      const s = per?.summary;
      if (s) {
        text += ` · ~${Math.round(s.input_tokens.median)} tok in → `
          + `${Math.round(s.output_tokens.median)} out, `
          + `~${Math.round(s.think_gap_s.median)}s between turns`;
      }
    }
    $("#workload-detail").textContent = text;
    this.updateHeadlineOpts();
    this.updateWorkloadNote();
  },

  isHeadlineWorkload() {
    const w = $("#workload-select").value || "";
    return w.startsWith("persona:headline_");
  },

  /* Headline workloads swap the whole measurement instrument, so the
   * form grows a small block of controls that only make sense there —
   * and capacity workloads never see it. */
  updateHeadlineOpts() {
    const box = $("#headline-opts");
    if (!box) return;
    const on = this.isHeadlineWorkload();
    box.hidden = !on;
    if (!on) return;
    const pid = $("#workload-select").value.slice(8);
    const p = this.catalogs.personas.find(x => x.id === pid);
    const s = p?.summary;
    $("#hl-shape").textContent = s
      ? `${Math.round(s.input_tokens.median)} tok in → `
        + `${Math.round(s.output_tokens.median)} out, EOS ignored`
      : "—";
  },

  msg(text, cls = "") {
    const el = $("#control-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  /* One body builder for anything that launches the engine (a
   * benchmark run or the headline shape search): validates the model,
   * then profile-vs-custom exactly as Start does. Returns null (with
   * a message shown) when the form can't launch. */
  buildRunBody(workload) {
    // Open-loop only — the closed-loop pool ramp is no longer offered
    // from the UI (it can't find the capacity limit; see
    // docs/algorithm.md). The API keeps mode:"closed" for scripts.
    const body = {
      workload,
      new_run: $("#new-run").checked,
      mode: "open",
    };
    const model = $("#bench-model").value;
    const entry = this.modelEntry();
    if (!model) { this.msg("pick a model first", "error"); return null; }
    if (entry && !entry.cached) {
      this.msg("that model isn't downloaded yet — use the download "
        + "link under the picker", "error");
      return null;
    }
    if (this.deviceMode === "cpu") {
      body.custom = { model_id: model, device: "cpu" };
      body.engineDesc = "CPU engine, stock settings";
      return body;
    }
    const form = this.readEngineForm();
    const untouched = this.matchedProfile
      && JSON.stringify(form) === JSON.stringify(this.engineDefaults);
    if (untouched) {
      // Exactly the optimized launch — run the promoted profile
      // itself (it may carry settings beyond the searched knobs).
      body.profile = this.matchedProfile.name;
      body.engineDesc = `the optimized engine (${this.engineSummary(form)})`;
    } else {
      body.custom = {
        model_id: model,
        device: "gpu",
        replicas: form.replicas,
        tp: form.tp,
        placement: form.placement,
        gpu_memory_utilization: form.gpu_memory_utilization,
        max_num_seqs: +form.max_num_seqs || null,
        max_num_batched_tokens: +form.max_num_batched_tokens || null,
        kv_cache_dtype: form.kv_cache_dtype || null,
        expert_parallel: form.expert_parallel,
      };
      body.engineDesc = `a custom variant (${this.engineSummary(form)})`;
    }
    return body;
  },

  async start() {
    const w = $("#workload-select").value || "";
    const [kind, id] = w.split(":", 2);
    const workload = kind === "sweep"
      ? { kind, type: id || "all" } : { kind, id };
    const body = this.buildRunBody(workload);
    if (!body) return;
    const engineDesc = body.engineDesc;
    delete body.engineDesc;
    const headline = this.isHeadlineWorkload();
    if (headline) {
      const cap = $("#hl-max-conc").value;
      if (cap) body.max_concurrency = +cap;
    }
    try {
      this.msg("starting…");
      await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      this.msg(
        headline
          ? `saturation benchmark started with ${engineDesc} — holding `
            + `streams in flight and stepping concurrency up; no SLA is `
            + `enforced`
          : `run started with ${engineDesc} — engine launch can take `
            + `several minutes; the Phase readout below tracks it`, "ok");
      this.pollStatus();
    } catch (e) {
      this.msg(e.message, "error");
    }
  },

  /* The shape affordance under Headline: Generation — mirrors the
   * engine-optimizer note on the model side. Running → completion bar
   * (cells done / budget, current shape); the winner becomes THIS
   * workload's shape and is remembered per model family, so picking a
   * sibling model later offers a one-click "load the optimal shape". */
  updateWorkloadNote() {
    const note = $("#workload-note");
    if (!note) return;
    const w = $("#workload-select").value || "";
    const pid = w.startsWith("persona:") ? w.slice(8) : "";
    if (pid !== "headline_generation") { note.innerHTML = ""; return; }
    const active = this.lastActive;
    const searching = active?.running
      && active.workload?.kind === "headline_search";
    if (searching) {
      const p = active.progress || {};
      if (!p.cell) {
        // Engine launch: minutes of silence before the first cell.
        note.innerHTML = `<span class="msg">launching the engine —
          model load and CUDA-graph capture take a few minutes before
          the first shape is measured</span>`;
        return;
      }
      const pct = p.budget ? Math.round(100 * ((p.cell || 1) - 1) / p.budget) : 0;
      const shape = p.shape ? `${p.shape[0]}→${p.shape[1]}` : "…";
      const best = p.best?.shape
        ? ` · best so far ${p.best.shape[0]}→${p.best.shape[1]}` : "";
      note.innerHTML = `optimizing shape — cell ${p.cell ?? 1} of
        ${p.budget ?? "?"} (${shape})${best}
        <span class="dl-bar"><i style="width:${pct}%"></i></span>${pct}%`;
      return;
    }
    const hs = this.headlineShape;
    const stored = hs?.shape;
    const fam = hs?.family || "this model family";
    let html = "";
    if (stored && hs.active) {
      html = `<span class="ok-note">✓ Using the optimized shape for
        ${fam} (${stored.input_tokens} in → ${stored.output_tokens}
        out)</span> `;
    } else if (stored) {
      html = `<button type="button" class="note-link"
        id="load-best-shape">⚡ Load the optimal shape for ${fam}
        (${stored.input_tokens} in → ${stored.output_tokens}
        out)</button> `;
    }
    note.innerHTML = html
      + `<button type="button" class="note-link" id="optimize-shape">⚙
         Optimize the shape</button>`;
    $("#load-best-shape")?.addEventListener("click", () =>
      this.applyHeadlineShape());
    $("#optimize-shape")?.addEventListener("click", () =>
      this.startShapeSearch());
  },

  async applyHeadlineShape() {
    const model = $("#bench-model").value;
    try {
      await api("/api/headline-shape/apply", {
        method: "POST", body: JSON.stringify({ model }),
      });
      await this.loadCatalogs();   // refreshed medians → green check
      this.msg("optimal shape loaded into Headline: Generation", "ok");
    } catch (e) {
      this.msg(e.message, "error");
    }
  },

  async startShapeSearch() {
    const body = this.buildRunBody({ kind: "headline_search" });
    if (!body) return;
    delete body.engineDesc;
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      this.msg("shape search started — shapes swap on the fly at "
        + "saturation; short shapes score in ~40s, long-output shapes "
        + "measure until steady state (up to ~5 min); the winner "
        + "becomes Headline: Generation's shape", "ok");
      this._shapeSearchWas = true;
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
    this.running = running;
    this.lastActive = active;
    // Shape-search lifecycle: live progress while it climbs; when it
    // completes, refresh the catalog so "Headline: best shape"
    // appears and the note flips to the green check.
    const searching = running && active?.workload?.kind === "headline_search";
    if (searching) this._shapeSearchWas = true;
    if (!running && this._shapeSearchWas) {
      this._shapeSearchWas = false;
      await this.loadCatalogs();
    }
    this.updateWorkloadNote();
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
      // Display names, never raw ids or a kind with no id.
      const nameOf = (kind, id) => {
        const list = kind === "cohort"
          ? this.catalogs?.cohorts : this.catalogs?.personas;
        return list?.find(x => x.id === id)?.name
          || (id ?? "").replaceAll("_", " ") || kind;
      };
      const isShape = w.kind === "headline_search";
      const isSat = w.kind === "persona" && (w.id || "").startsWith("headline_");
      const wtxt = w.kind === "sweep" ? `sweep(${w.type})`
        : isShape ? "Shape search"
        : isSat ? `${nameOf(w.kind, w.id)} · saturation benchmark`
        : nameOf(w.kind, w.id);
      // Rung-by-rung progress, so the minutes between rungs never
      // read as a hang.
      const p = active.progress || {};
      const satProgress = !(isSat && running) ? ""
        : !p.rung
          ? ` · <span class="msg">launching the engine (a few
              minutes)</span>`
          : ` · <b>rung ${p.rung} of ${p.rungs}</b> (${p.concurrency}
              streams)${p.peak?.out_tok_s
                ? ` · best so far ${Math.round(
                    p.peak.out_tok_s).toLocaleString()} tok/s` : ""}`;
      const since = fmt.clock(active.started_at * 1000);
      const finished = !running && !active.error && active.result;
      box.innerHTML =
        `<b>${wtxt}</b>` +
        (active.engine_summary
          ? ` · engine: <b>${active.engine_summary}</b>`
          : ` · config <b>${active.config}</b>`) +
        ` · started ${since}` + satProgress +
        (active.error ? ` · <span class="status-fail">${active.error}</span>` : "") +
        (finished
          ? isShape
            // A shape search leaves no run entry — its result IS the
            // workload's new shape, shown under the picker above.
            ? ` · <span class="status-pass">finished — winning shape
                 saved to Headline: Generation</span>`
            : ` · <span class="status-pass">finished</span>
               <button id="goto-results" class="small"
                 style="margin-left:8px">View results →</button>`
          : "");
      box.querySelector("#goto-results")?.addEventListener("click", () => {
        document.querySelector('#tabs button[data-view="results"]').click();
      });
    } else {
      box.hidden = true;
    }
  },

  async refreshRuns() {
    let runs;
    try { runs = await api("/api/runs"); } catch { return; }
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
    this.push(this.charts.host, fmt.clock(ts), [
      t.kv_cache_used_pct, t.cpu_util_bound_avg ?? t.cpu_util_avg,
      t.gpu_sm_util_pct,
    ]);
    if (t.prefill_tok_s != null || t.decode_tok_s != null) {
      this.push(this.charts.tokens, fmt.clock(ts),
        [t.prefill_tok_s, t.decode_tok_s]);
    }
    let host = null, gpus = null;
    try { host = t.host_json ? JSON.parse(t.host_json) : null; } catch { /* skip */ }
    try { gpus = t.gpu_devices_json ? JSON.parse(t.gpu_devices_json) : null; } catch { /* skip */ }
    this.renderHostDetail(host, t);
    this.renderGpuGrid(gpus);
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

  renderGpuGrid(gpus) {
    const el = $("#gpu-grid");
    if (!gpus || !gpus.length) return;
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
    }
    if (r.event === "finished") {
      $("#live-phase").textContent = `finished (${r.final_status})`;
      // Any export cached mid-run is now stale (the server rebuilds
      // when run.db is newer, but only if we actually refetch).
      Results.exportCache = {};
      Results.openId = null;   // re-open the freshest run on refresh
      Control.refreshRuns();
    }
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
    // Refresh EVERY visit — a once-only guard here meant a Results
    // tab opened before the first run finished cached an empty
    // picker forever.
    Control.refreshRuns();
  },

  flat: [],             // flattened (run, cohort) rows, newest first
  openId: null,         // cohort_run_id currently displayed
  checked: new Set(),   // cohort_run_ids ticked for comparison

  init() {
    $("#result-export-dl").addEventListener("click", () => this.download());
    $("#compare-btn").addEventListener("click", () => this.runCompare());
    $("#compare-clear").addEventListener("click", () => {
      this.compare = [];
      this.checked.clear();
      $("#compare-panel").hidden = true;
      this.renderList();
    });
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
        ?? (Control.running && e === newestOpen ? "running" : "interrupted");
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
    Control.refreshRuns();
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
    const isOpen = c.open_loop != null
      && pts.some(p => p.arrival_rate_per_min != null);
    const stable = pts.filter(p =>
      isOpen ? p.stability === "stable" : p.status === "pass");
    const last = stable.length ? stable[stable.length - 1]
      : (pts.length ? pts[pts.length - 1] : null);
    const knee = pts.find(p =>
      isOpen ? p.stability === "divergent" : p.status === "fail") ?? null;
    const xOf = p => p ? `${p[ax.key]}${isOpen ? "/min" : " users"}` : "—";
    return { ax, pts, isOpen, last, knee, xOf, ol: c.open_loop };
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
    $("#opt-history").addEventListener("change", () => this.showHistory());
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
          <span class="info-dot" tabindex="0" aria-label="what this knob does">i
            <span class="info-pop msg">${c.text}</span></span>
          <button class="ac-edit" data-edit="${c.key}" title="edit">✎ edit</button></div>
        <div class="ac-sel">${summary}</div>
        <div class="card-pop" data-pop="${c.key}" hidden>
          ${c.options.map(([v, l]) => `<label class="pop-row">
            <input type="checkbox" data-card-opt="${c.key}" data-val="${v}"
              ${off.has(v) ? "" : "checked"}> ${l}</label>`).join("")}
        </div></div>`;
    };

    const modelsCard = `<div class="arena-card" data-card="models">
      <div class="ac-head"><span class="ac-title">Models in play</span>
        <span class="info-dot" tabindex="0" aria-label="which models">i
          <span class="info-pop msg">${inPlay.map(m =>
            `${m.id.split("/")[1]} <i>(${m.quant})</i>`).join(", ") || "—"}
          </span></span>
        <button class="ac-edit" data-edit="models" title="edit">✎ edit</button></div>
      <div class="ac-sel">${inPlay.length} of ${eligible.length} eligible
        ${eligible.length < a.models.length
          ? `<span class="msg">(${a.models.length - eligible.length} hidden by
             the family/size dropdowns or filter cards, or won't fit)</span>`
          : ""}</div>
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
    const mPop = box.querySelector('[data-card="models"] .info-pop');
    if (mPop) mPop.innerHTML = inPlay.map(m =>
      `${m.id.split("/")[1]} <i>(${m.quant})</i>`).join(", ") || "—";
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
      this.renderHeartbeat(status.active);
      if (!this.polling) {
        this.polling = setInterval(() => this.refresh(), 4000);
      }
    } else {
      $("#opt-heartbeat").hidden = true;
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
    this.currentGroup = status.search_group ?? null;
    this.refreshHistoryList();
    // Viewing an archived run: pin the panel to it — live polling
    // must not clobber what the operator chose to look at.
    if (this.historyView && this.historyDoc) {
      this.renderResults(null);
      this.renderSearch(this.historyDoc);
      return;
    }
    const sr = status.search_results;
    const rr = status.results;
    const searchNewer = !!sr?.generated_at
      && (!rr?.generated_at || sr.generated_at > rr.generated_at);
    const m = status.running ? status.active?.mode : null;
    let showSearch;
    if (m && m !== "unknown") {
      showSearch = m !== "registry";      // the active run's own panel
    } else {
      // Idle, or an attached run that can't declare its mode: the
      // file being written RIGHT NOW is the newer one — route by
      // freshness rather than guessing.
      showSearch = searchNewer;
    }
    this.renderResults(showSearch ? null : rr);
    this.renderSearch(showSearch ? sr : null);
  },

  /* Live pulse while the optimizer runs. Each candidate is minutes of
   * silent engine launch before seconds of measurement — narrate the
   * phase and its elapsed time (from the server's log-tail heartbeat)
   * so the launch quiet never reads as a hang. */
  renderHeartbeat(active) {
    const box = $("#opt-heartbeat");
    if (!box) return;
    const hb = active?.heartbeat;
    box.hidden = !hb;
    if (!hb) return;
    const mmss = s => (s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`);
    const PHASES = [
      [/^launching/, () => "launching engine containers"],
      [/^waiting for health/, () => `engine replicas loading the model &
        compiling — the quiet minutes are normal (typically 3–5 min)`],
      [/^probing/, () => "probing replicas"],
      [/^warmup/, () => "warming up"],
      [/^cell \S*c0*(\d+)/, m => `measuring at ${m[1]} concurrent streams`],
      [/^cleanup/, () => "tearing down engines"],
    ];
    let phase = hb.phase || "…";
    for (const [re, label] of PHASES) {
      const m = phase.match(re);
      if (m) { phase = label(m); break; }
    }
    const pct = hb.configs_total
      ? Math.round(100 * ((hb.config || 1) - 1) / hb.configs_total) : 0;
    const stale = (hb.last_activity_s ?? 0) > 600;
    box.innerHTML = `
      <span class="pulse${stale ? " stale" : ""}"></span>
      <b>Config ${hb.config ?? "?"} of ${hb.configs_total ?? "?"}</b>
      ${hb.config_name
        ? `<code>${hb.config_name.replace(/^s\d+_/, "")}</code>` : ""}
      <span class="dl-bar"><i style="width:${pct}%"></i></span>${pct}%
      <div class="hb-line">${phase}${hb.phase_elapsed_s != null
        ? ` · ${mmss(hb.phase_elapsed_s)} in this phase` : ""}
        · last log activity ${mmss(hb.last_activity_s ?? 0)} ago${stale
        ? ' — <span class="status-fail">unusually quiet — check the log</span>'
        : ""}</div>
      ${hb.last_measure ? `<div class="hb-line msg">last measurement:
        <code>${hb.last_measure}</code></div>` : ""}`;
  },

  historyView: null,       // archive file name being viewed, or null
  historyDoc: null,

  currentGroup: null,      // model-set group of the current run

  /* Runs group by the MODEL SET they searched — a follow-up over the
   * same models (the TP gap-fill) is an addendum to the same
   * investigation, and each multi-run group gets a Combined entry
   * that merges the evidence. Different model sets never mix. */
  async refreshHistoryList() {
    let entries;
    try { entries = await api("/api/optimizer/history"); } catch { return; }
    const sel = $("#opt-history");
    const prev = sel.value;
    const groups = new Map();      // key -> {label, entries, hasCurrent}
    if (this.currentGroup?.key) {
      groups.set(this.currentGroup.key, {
        label: this.currentGroup.label, entries: [], hasCurrent: true,
      });
    }
    for (const e of entries) {
      const key = e.group_key || "ungrouped";
      if (!groups.has(key)) {
        groups.set(key, { label: e.group_label || "other runs",
                          entries: [], hasCurrent: false });
      }
      groups.get(key).entries.push(e);
    }
    const opt = (value, text) =>
      `<option value="${value}" ${value === prev ? "selected" : ""}>${text}</option>`;
    let html = opt("", "current");
    for (const [key, g] of groups) {
      const runCount = g.entries.length + (g.hasCurrent ? 1 : 0);
      if (!g.entries.length) continue;
      html += `<optgroup label="${g.label}">`;
      if (runCount > 1 && key !== "ungrouped") {
        html += opt(`combined:${key}`,
          `★ Combined — ${runCount} runs, merged ranking`);
      }
      html += g.entries.map(e => opt(e.file,
        `${(e.generated_at || "").slice(0, 16).replace("T", " ")} · ` +
        `${e.space} · ${e.evaluated} evals · best ` +
        `${e.best_score == null ? "—" : Math.round(e.best_score)}`)).join("");
      html += "</optgroup>";
    }
    sel.innerHTML = html;
  },

  async showHistory() {
    const name = $("#opt-history").value;
    if (!name) {
      this.historyView = this.historyDoc = null;
      this.refresh();
      return;
    }
    try {
      this.historyDoc = name.startsWith("combined:")
        ? await api(`/api/optimizer/combined/${name.slice(9)}`)
        : await api(`/api/optimizer/history/${encodeURIComponent(name)}`);
      this.historyView = name;
      this.renderResults(null);
      this.renderSearch(this.historyDoc);
    } catch (e) {
      this.msg(e.message, "error");
    }
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
      const r = await api("/api/optimizer/start", {
        method: "POST",
        body: JSON.stringify(body),
      });
      this.msg(r.seeded
        ? `started — ${r.seeded} prior result(s) from this group seeded;
           budget spends on new candidates`
        : "started", "ok");
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
      // Promote target: a plain archived run promotes from its file;
      // the Combined view promotes the overall winner from ITS source
      // run (promote_file; null = the current live results).
      let file = null;
      if (source === "search" && this.historyView) {
        file = this.historyDoc?.kind === "combined"
          ? this.historyDoc.promote_file
          : this.historyView;
      }
      r = await api("/api/optimizer/promote", {
        method: "POST",
        body: JSON.stringify({ source, config_name: configName ?? null,
                               file }),
      });
    } catch (e) {
      this.msg(e.message, "error");
      return;
    }
    const warn = (r.warnings ?? []).length ? ` — NOTE: ${r.warnings[0]}` : "";
    this.msg(`optimized launch saved as profile "${r.profile}" (${r.path})${warn}`, "ok");
    await Control.loadCatalogs();
    // Hand off: select the promoted profile's model — the benchmark
    // form finds the optimization itself and shows the green check.
    const prof = Control.catalogs.profiles?.[r.profile];
    if (prof?.model_id) {
      $("#bench-model").value = prof.model_id;
      Control.onModelChange();
    }
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
    $("#model-discover-btn").addEventListener("click", () => this.discover());
    $("#discover-sort").addEventListener("change", () => this.renderDiscovered());
    for (const id of ["#mf-family", "#mf-quant", "#mf-status"]) {
      $(id).addEventListener("change", () => this.renderTable());
    }
    $("#mf-search").addEventListener("input", () => this.renderTable());
    this.refresh();
  },

  discovered: null,

  /* Live Hub discovery: recent models from the leading orgs, sized
   * from their own safetensors metadata and validated against this
   * box's GPUs. */
  async discover() {
    const box = $("#model-discover");
    this.addMsg("querying the Hub — one call per candidate, ~20s…");
    $("#model-discover-btn").disabled = true;
    try {
      const doc = await api("/api/models/discover");
      this.discovered = doc.models;
      $("#discover-sort-wrap").hidden = false;
      this.addMsg(`${doc.models.length} candidates from the Hub`, "ok");
      this.renderDiscovered();
    } catch (e) {
      this.addMsg(e.message, "error");
      box.innerHTML = "";
    } finally {
      $("#model-discover-btn").disabled = false;
    }
  },

  renderDiscovered() {
    const box = $("#model-discover");
    if (!this.discovered) return;
    const sort = $("#discover-sort").value;
    const rows = [...this.discovered].sort((a, b) =>
      sort === "size" ? (b.params_b ?? 0) - (a.params_b ?? 0)
      : sort === "downloads" ? (b.downloads ?? 0) - (a.downloads ?? 0)
      : (b.last_modified ?? "").localeCompare(a.last_modified ?? ""));
    const age = iso => {
      if (!iso) return "—";
      const d = Math.round((Date.now() - Date.parse(iso)) / 86400000);
      return d < 1 ? "today" : d < 30 ? `${d}d` : `${Math.round(d / 30)}mo`;
    };
    box.innerHTML = `<div class="callout" style="margin-top:12px">
      <table><thead><tr><th>Model</th><th>Age</th><th>Params</th>
        <th>Capabilities</th><th>Fits here</th><th>Downloads</th><th></th>
      </tr></thead><tbody>${rows.map(m => `<tr>
        <td>${m.id}${m.gated
          ? ' <span class="status-marginal">gated</span>' : ""}</td>
        <td>${age(m.last_modified)}</td>
        <td>${m.params_b}B <span class="msg">~${m.approx_size_gb}GB
          ${m.quant}</span></td>
        <td class="msg">${(m.capabilities ?? []).join(" · ") || "dense"}</td>
        <td>${m.feasible === false
          ? '<span class="status-fail">won’t fit</span>'
          : m.feasible_tps ? `tp ${m.feasible_tps.join("/")}` : "?"}</td>
        <td class="msg">${m.downloads?.toLocaleString?.() ?? "—"}</td>
        <td>${m.in_catalog ? '<span class="msg">in catalog</span>'
          : `<button class="small primary" data-disc="${m.id}">+ Add</button>`}
        </td></tr>`).join("")}</tbody></table></div>`;
    box.querySelectorAll("button[data-disc]").forEach(btn =>
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        const m = this.discovered.find(x => x.id === btn.dataset.disc);
        try {
          await api("/api/models/add", {
            method: "POST",
            body: JSON.stringify({
              model: m.id, check_hub: false, quant: m.quant, moe: m.moe,
              params_b: m.params_b, approx_size_gb: m.approx_size_gb,
              min_vram_gb: m.min_vram_gb,
            }),
          });
          m.in_catalog = true;
          this.addMsg(`added ${m.id}`, "ok");
          this.renderDiscovered();
          this.refresh();
        } catch (e) {
          this.addMsg(e.message, "error");
          btn.disabled = false;
        }
      }));
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
    this.doc = doc;
    $("#models-cache-dir").textContent = doc.cache_dir;
    // (Re)build filter options, preserving the current selection.
    const fill = (sel, values) => {
      const prev = sel.value || "all";
      sel.innerHTML = '<option value="all">All</option>' + values.map(v =>
        `<option value="${v}" ${v === prev ? "selected" : ""}>${v}</option>`
      ).join("");
    };
    const uniq = arr => [...new Set(arr.filter(Boolean))].sort();
    fill($("#mf-family"), uniq(doc.models.map(m => m.series || m.family)));
    fill($("#mf-quant"), uniq(doc.models.map(m => m.quant)));
    this.renderTable();
  },

  renderTable() {
    const doc = this.doc;
    if (!doc) return;
    const fam = $("#mf-family").value, quant = $("#mf-quant").value;
    const status = $("#mf-status").value;
    const q = $("#mf-search").value.trim().toLowerCase();
    const active = m => {
      const dl = doc.downloads[m.model];
      return m.cached || m.partial || !!(dl && dl.running);
    };
    const rows = doc.models.filter(m =>
      (fam === "all" || (m.series || m.family) === fam)
      && (quant === "all" || m.quant === quant)
      && (status === "all" || (status === "cached") === active(m))
      && (!q || m.model.toLowerCase().includes(q)));
    // Working set first: cached / downloading rows above the rest.
    rows.sort((a, b) => (active(b) - active(a))
      || (a.series || "").localeCompare(b.series || "")
      || a.model.localeCompare(b.model));
    $("#mf-count").textContent = rows.length === doc.models.length
      ? `${rows.length} models`
      : `${rows.length} of ${doc.models.length} models`;
    const tbody = $("#models-table tbody");
    tbody.innerHTML = "";
    let anyRunning = false;
    for (const m of rows) {
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

/* Fresh-persona starting point for "New persona" — a moderate
 * conversational user. The deprecated distribution fields
 * (sessions_before_leaving / inter_session_gap) must exist for the
 * schema but no longer drive runtime; they're carried, never shown. */
const PERSONA_DEFAULT_SPEC = {
  name: "",
  description: "",
  input_tokens: { lognormal: { median: 400, sigma: 0.5 } },
  output_tokens: { lognormal: { median: 200, sigma: 0.4 } },
  turns_per_session: { discrete: { 1: 0.6, 2: 0.3, 4: 0.1 } },
  sessions_before_leaving: { constant: 1 },
  inter_session_gap_seconds: { constant: 60 },
  read_time_seconds: { lognormal: { median: 25, sigma: 0.5 } },
  active_think_seconds: { lognormal: { median: 30, sigma: 0.6 } },
  sla: {
    ttft_target_seconds: 10, ttft_failure_seconds: 30,
    tpot_target_ms: 150, tpot_failure_ms: 225,
  },
};

/* Log-scale slider mapping: range inputs run 0..1000, values are
 * ratio-scaled (tokens, seconds) so linear sliders would waste 90% of
 * their travel on the top decade. */
const logTo = (pos, min, max) =>
  min * Math.exp((pos / 1000) * Math.log(max / min));
const logFrom = (v, min, max) =>
  1000 * Math.log(Math.max(min, Math.min(max, v)) / min) / Math.log(max / min);
const fmtNum = (v, dec) =>
  dec === 0 ? String(Math.round(v)) : String(+(+v).toFixed(dec));

const Editor = {
  kind: "personas",    // "personas" | "cohorts"
  editing: null,       // id being edited, null for new
  spec: null,          // working persona spec (mutated by sliders)
  mix: [],             // working cohort rows [{pid, share}]

  init() {
    $("#editor-save").addEventListener("click", () => this.save());
    $("#persona-new").addEventListener("click", () => this.startNewPersona());
    $("#cohort-new").addEventListener("click", () => this.startNewCohort());
    document.querySelector('#tabs button[data-view="personas"]')
      .addEventListener("click", () => this.refreshLists());
    // Benchmark form → designer jump.
    $("#workload-edit-link").addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelector('#tabs button[data-view="personas"]').click();
    });
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
        li.textContent = item.name || item.id;
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
    $("#editor-kind-badge").textContent = kind.slice(0, -1);
    this.msg(`editing ${id} — changes apply to the next run`);
    if (kind === "personas") {
      this.spec = detail.spec;
      this.buildPersonaForm();
    } else {
      this.spec = detail.spec;
      this.mix = Object.entries(detail.spec.persona_weights || {})
        .map(([pid, w]) => ({ pid, share: Math.round(w * 1000) / 10 }));
      this.buildCohortForm();
    }
    this.renderCard(kind, id);
    this.refreshLists();
  },

  /* The readable card: what this workload MEANS — question/answer
   * sizes, session shape, the read/think gap, SLA bars — so the
   * operator understands the run without parsing YAML. */
  renderCard(kind, id) {
    const card = $("#workload-card");
    const n = v => (v == null ? "—"
      : v >= 100 ? Math.round(v).toLocaleString() : (+v).toFixed(1) * 1);
    const stat = (k, v, c) => `<div class="stat"><span class="k">${k}</span>
      <span class="v" style="font-size:1.15rem">${v}</span>
      <span class="c">${c}</span></div>`;
    if (kind === "personas") {
      const p = Control.catalogs.personas.find(x => x.id === id);
      const s = p?.summary;
      if (!p || !s) { card.hidden = true; return; }
      card.hidden = false;
      card.innerHTML = `<b>${p.name || id}</b> —
        <span class="msg">${p.description}</span>
        <div class="statusbar" style="margin-top:10px;background:none;
          border:none;box-shadow:none;padding:0;backdrop-filter:none">
          ${stat("Question", `${n(s.input_tokens.median)} tok`,
                 `typical · heavy ${n(s.input_tokens.p90)}`)}
          ${stat("Answer", `${n(s.output_tokens.median)} tok`,
                 `typical · long ${n(s.output_tokens.p90)}`)}
          ${stat("Session", `${n(s.turns_per_session.mean)} turns`,
                 "back-and-forth before leaving")}
          ${stat("Between turns", `${n(s.think_gap_s.median)}s`,
                 `reading + thinking · slow ${n(s.think_gap_s.p90)}s`)}
          ${stat("Feels slow at", `${p.ttft_target_s}s / ${p.tpot_target_ms}ms`,
                 "first token / per token")}
          ${stat("Gives up at", `${p.ttft_failure_s}s / ${p.tpot_failure_ms}ms`,
                 "the SLA failure bar")}
        </div>
        <span class="msg">In a run, each simulated user of this type asks a
        ~${n(s.input_tokens.median)}-token question, streams a
        ~${n(s.output_tokens.median)}-token answer, then spends
        ~${n(s.think_gap_s.median)}s reading and composing before the next
        turn — the engine only works during the streaming slice, which is
        why pool size can far exceed in-flight requests.</span>`;
    } else {
      const c = Control.catalogs.cohorts.find(x => x.id === id);
      if (!c) { card.hidden = true; return; }
      card.hidden = false;
      const total = Object.values(c.persona_weights)
        .reduce((a, b) => a + b, 0) || 1;
      const pname = pid =>
        Control.catalogs.personas.find(p => p.id === pid)?.name
        || pid.replaceAll("_", " ");
      const rows = Object.entries(c.persona_weights)
        .sort((a, b) => b[1] - a[1])
        .map(([pid, w]) => {
          const pct = Math.round(100 * w / total);
          return `<div class="gpu-row" style="grid-template-columns:
              160px minmax(80px,1fr) 40px">
            <span class="g-id" style="cursor:pointer" title="open persona"
              data-open-persona="${pid}">${pname(pid)}</span>
            <span class="g-bar"><i style="width:${pct}%"></i></span>
            <span class="g-num">${pct}%</span></div>`;
        }).join("");
      const b = c.blended;
      card.innerHTML = `<b>${c.name}</b> — <span class="msg">${c.description}</span>
        <div class="gpu-grid" style="margin-top:10px">${rows}</div>
        ${b ? `<span class="msg">A typical turn of this mix:
          ~${b.input_tokens} tokens in → ~${b.output_tokens} out,
          ~${b.turns_per_session} turns per session,
          ~${b.think_gap_s}s of reading/thinking between turns.
          Click a persona for its full card.</span>` : ""}`;
      card.querySelectorAll("[data-open-persona]").forEach(el =>
        el.addEventListener("click",
          () => this.open("personas", el.dataset.openPersona)));
    }
  },

  startNewPersona() {
    this.kind = "personas";
    this.editing = null;
    this.spec = structuredClone(PERSONA_DEFAULT_SPEC);
    $("#workload-card").hidden = true;
    $("#editor-id").value = "";
    $("#editor-kind-badge").textContent = "persona";
    this.msg("shape the persona, set an id, Save");
    this.buildPersonaForm();
    this.refreshLists();
  },

  startNewCohort() {
    this.kind = "cohorts";
    this.editing = null;
    const ids = (Control.catalogs.personas || []).map(p => p.id);
    this.spec = { name: "", description: "" };
    this.mix = ids.slice(0, 2).map(pid => ({ pid, share: 50 }));
    $("#workload-card").hidden = true;
    $("#editor-id").value = "";
    $("#editor-kind-badge").textContent = "cohort";
    this.msg("mix the personas, set an id, Save");
    this.buildCohortForm();
    this.refreshLists();
  },

  /* ── form primitives ─────────────────────────────────────────── */

  el(html) {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  },

  /* One slider + synced number input. ``get``/``set`` read and write
   * the working spec so every control is live against one object. */
  sliderRow({ label, hint, unit, min, max, log = false, dec = 0,
              get, set }) {
    const v = get();
    const pos = log ? logFrom(v, min, max)
                    : 1000 * (v - min) / (max - min);
    const row = this.el(`<div class="slider-row">
      <span class="sl-label">${label}
        ${hint ? `<span class="hint">${hint}</span>` : ""}</span>
      <input type="range" min="0" max="1000" value="${Math.round(pos)}">
      <span class="sl-num"><input type="text" value="${fmtNum(v, dec)}">
        <span class="unit">${unit ?? ""}</span></span>
    </div>`);
    const range = row.querySelector("input[type=range]");
    const num = row.querySelector(".sl-num input");
    range.addEventListener("input", () => {
      const val = log ? logTo(+range.value, min, max)
                      : min + (+range.value / 1000) * (max - min);
      num.value = fmtNum(val, dec);
      set(+num.value);
    });
    num.addEventListener("change", () => {
      let val = parseFloat(num.value);
      if (!Number.isFinite(val)) { num.value = fmtNum(get(), dec); return; }
      val = Math.max(min, Math.min(max, val));
      num.value = fmtNum(val, dec);
      range.value = Math.round(
        log ? logFrom(val, min, max) : 1000 * (val - min) / (max - min));
      set(val);
    });
    return row;
  },

  /* Distribution accessors: lognormal edits its median, constant its
   * value. Discrete distributions get their own row editor. */
  distMedian(d) {
    if (d.lognormal) return d.lognormal.median;
    if (d.constant != null) return d.constant;
    if (d.discrete) {
      const e = Object.entries(d.discrete);
      const tot = e.reduce((a, [, w]) => a + +w, 0) || 1;
      return e.reduce((a, [v, w]) => a + (+v) * (+w), 0) / tot;
    }
    return 0;
  },
  setDistMedian(d, v) {
    if (d.lognormal) d.lognormal.median = v;
    else if (d.constant != null) d.constant = v;
  },

  discreteEditor(field, label) {
    const d = this.spec[field].discrete;
    const box = this.el(`<div><h4>${label} — value distribution</h4>
      <span class="hint">weighted choice: each row is (value, relative
      weight)</span><div class="disc-rows"></div></div>`);
    const rowsEl = box.querySelector(".disc-rows");
    const render = () => {
      rowsEl.innerHTML = "";
      for (const [val, w] of Object.entries(d)) {
        const r = this.el(`<div class="disc-row">
          <input type="text" value="${val}" title="value">
          <input type="text" value="${w}" title="weight">
          <button class="remove" title="remove">×</button></div>`);
        const [vi, wi] = r.querySelectorAll("input");
        const commit = () => {
          delete d[val];
          const nv = parseFloat(vi.value), nw = parseFloat(wi.value);
          if (Number.isFinite(nv) && Number.isFinite(nw) && nw > 0) {
            d[Number.isInteger(nv) ? nv : nv] = nw;
          }
          render();
        };
        vi.addEventListener("change", commit);
        wi.addEventListener("change", commit);
        r.querySelector(".remove").addEventListener("click", () => {
          if (Object.keys(d).length > 1) { delete d[val]; render(); }
        });
        rowsEl.append(r);
      }
      const add = this.el(
        `<button class="dotted-add" style="max-width:290px">+ add value</button>`);
      add.addEventListener("click", () => {
        const vals = Object.keys(d).map(Number);
        d[Math.round(Math.max(...vals, 0) + 1)] = 0.1;
        render();
      });
      rowsEl.append(add);
    };
    render();
    return box;
  },

  /* ── persona form ────────────────────────────────────────────── */

  /* Human name → stable backend id. Users never type (or see) the
   * underscored id; it's derived once at creation and stays fixed. */
  slug(name) {
    return name.toLowerCase().replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  },

  buildPersonaForm() {
    const form = $("#editor-form");
    form.innerHTML = "";
    const s = this.spec;

    const nameEl = this.el(`<label style="display:block;margin-bottom:6px">
      Name <input id="pf-name" style="width:100%"
      placeholder="Long-form generator"></label>`);
    nameEl.querySelector("input").value = s.name || "";
    nameEl.querySelector("input").addEventListener("input", (e) => {
      s.name = e.target.value;
      if (!this.editing) $("#editor-id").value = this.slug(e.target.value);
    });
    form.append(nameEl);

    const desc = this.el(`<label style="display:block;margin-bottom:6px">
      Description <input id="pf-desc" style="width:100%"
      placeholder="What this kind of user does"></label>`);
    desc.querySelector("input").value = s.description || "";
    desc.querySelector("input").addEventListener("input",
      (e) => { s.description = e.target.value; });
    form.append(desc);

    form.append(this.el(`<h4>What this user does</h4>`));
    const dists = [
      { f: "input_tokens", label: "Question size", unit: "tok",
        min: 8, max: 32768, log: true,
        hint: "tokens sent per turn (with history on top)" },
      { f: "output_tokens", label: "Answer size", unit: "tok",
        min: 8, max: 32768, log: true,
        hint: "tokens the model streams back" },
      { f: "turns_per_session", label: "Turns per session", unit: "",
        min: 1, max: 50, log: false,
        hint: "back-and-forth before the user leaves" },
      { f: "read_time_seconds", label: "Reading time", unit: "s",
        min: 0.5, max: 600, log: true, dec: 1,
        hint: "catching up after the stream ends" },
      { f: "active_think_seconds", label: "Thinking time", unit: "s",
        min: 0.5, max: 900, log: true, dec: 1,
        hint: "composing the next message" },
    ];
    const discreteFields = [];
    for (const cfg of dists) {
      const d = s[cfg.f];
      if (d.discrete) {
        // A shaped (discrete) distribution has no single knob — show
        // its mean read-only here, edit the shape in Advanced.
        const mean = this.distMedian(d);
        form.append(this.el(`<div class="slider-row">
          <span class="sl-label">${cfg.label}
            <span class="hint">${cfg.hint}</span></span>
          <span class="msg">shaped distribution — mean
            ${fmtNum(mean, 1)}${cfg.unit} · edit values under Advanced</span>
          <span></span></div>`));
        discreteFields.push(cfg);
      } else {
        form.append(this.sliderRow({
          ...cfg,
          get: () => this.distMedian(d),
          set: (v) => this.setDistMedian(d, v),
        }));
      }
    }

    form.append(this.el(`<h4>Experience thresholds</h4>`));
    form.append(this.sliderRow({
      label: "Feels slow at", unit: "s", min: 0.2, max: 60, log: true,
      dec: 1, hint: "first-token wait past this misses the target",
      get: () => s.sla.ttft_target_seconds,
      set: (v) => {
        s.sla.ttft_target_seconds = v;
        if (s.sla.ttft_failure_seconds < v) s.sla.ttft_failure_seconds = v;
      },
    }));
    form.append(this.sliderRow({
      label: "Gives up at", unit: "s", min: 0.5, max: 180, log: true,
      dec: 1, hint: "the hard SLA bar — capacity is gated on this",
      get: () => s.sla.ttft_failure_seconds,
      set: (v) => {
        s.sla.ttft_failure_seconds =
          Math.max(v, s.sla.ttft_target_seconds);
      },
    }));

    const adv = this.el(`<details class="adv-block">
      <summary class="msg">Advanced — variability, per-token SLA,
      timeouts</summary><div class="adv-body"></div></details>`);
    const body = adv.querySelector(".adv-body");
    body.append(this.el(`<h4>Variability</h4>`));
    for (const cfg of dists) {
      const d = s[cfg.f];
      if (!d.lognormal) continue;
      body.append(this.sliderRow({
        label: `${cfg.label} spread`, unit: "σ", min: 0.1, max: 1.3,
        dec: 2, hint: "0.35 tight · 0.5 typical · 0.7+ heavy-tailed",
        get: () => d.lognormal.sigma,
        set: (v) => { d.lognormal.sigma = v; },
      }));
    }
    for (const cfg of discreteFields) {
      body.append(this.discreteEditor(cfg.f, cfg.label));
    }
    body.append(this.el(`<h4>Per-token SLA</h4>`));
    body.append(this.sliderRow({
      label: "Feels slow per token", unit: "ms", min: 5, max: 500,
      log: true, hint: "streaming pace target",
      get: () => s.sla.tpot_target_ms,
      set: (v) => {
        s.sla.tpot_target_ms = v;
        if (s.sla.tpot_failure_ms < v) s.sla.tpot_failure_ms = v;
      },
    }));
    body.append(this.sliderRow({
      label: "Gives up per token", unit: "ms", min: 10, max: 1000,
      log: true, hint: "streaming pace failure bar",
      get: () => s.sla.tpot_failure_ms,
      set: (v) => {
        s.sla.tpot_failure_ms = Math.max(v, s.sla.tpot_target_ms);
      },
    }));
    body.append(this.el(`<h4>Abort timeouts</h4>`));
    s.timeouts = s.timeouts || {};
    const t = s.timeouts;
    body.append(this.sliderRow({
      label: "Hard request ceiling", unit: "s", min: 60, max: 3600,
      log: true, hint: "abort any request past this wall time",
      get: () => t.hard_timeout_s ?? 900,
      set: (v) => { t.hard_timeout_s = Math.round(v); },
    }));
    form.append(adv);
  },

  /* ── cohort form ─────────────────────────────────────────────── */

  buildCohortForm() {
    const form = $("#editor-form");
    form.innerHTML = "";
    const s = this.spec;
    const catalog = Control.catalogs.personas || [];
    const personaIds = catalog.map(p => p.id);
    const pname = pid =>
      catalog.find(p => p.id === pid)?.name || pid.replaceAll("_", " ");

    const head = this.el(`<div>
      <label style="display:block;margin-bottom:6px">Name
        <input id="cf-name" style="width:100%" placeholder="Customer support team">
      </label>
      <label style="display:block;margin-bottom:6px">Description
        <input id="cf-desc" style="width:100%"
          placeholder="What this team does all day"></label></div>`);
    head.querySelector("#cf-name").value = s.name || "";
    head.querySelector("#cf-desc").value = s.description || "";
    head.querySelector("#cf-name").addEventListener("input", (e) => {
      s.name = e.target.value;
      if (!this.editing) $("#editor-id").value = this.slug(e.target.value);
    });
    head.querySelector("#cf-desc").addEventListener("input",
      (e) => { s.description = e.target.value; });
    form.append(head);

    form.append(this.el(`<h4>Traffic mix</h4>`));
    const mixBox = this.el(`<div class="mix-box"></div>`);
    form.append(mixBox);
    const foot = this.el(`<span class="hint"></span>`);
    form.append(foot);

    const render = () => {
      mixBox.innerHTML = "";
      const total = this.mix.reduce((a, r) => a + (+r.share || 0), 0);
      this.mix.forEach((row, i) => {
        const pct = total > 0 ? Math.round(100 * (+row.share || 0) / total) : 0;
        const opts = personaIds.map(pid =>
          `<option value="${pid}" ${pid === row.pid ? "selected" : ""}>
             ${pname(pid)}</option>`).join("");
        const r = this.el(`<div class="mix-row">
          <select>${opts}</select>
          <input type="range" min="0" max="100" value="${row.share}">
          <span class="mix-share">${pct}% of traffic</span>
          <button class="remove" title="remove persona">×</button></div>`);
        r.querySelector("select").addEventListener("change", (e) => {
          row.pid = e.target.value;
        });
        r.querySelector("input[type=range]").addEventListener("input", (e) => {
          row.share = +e.target.value;
          renderShares();
        });
        r.querySelector(".remove").addEventListener("click", () => {
          this.mix.splice(i, 1);
          render();
        });
        mixBox.append(r);
      });
      const add = this.el(`<button class="dotted-add">+ add persona
        to the mix</button>`);
      add.addEventListener("click", () => {
        const used = new Set(this.mix.map(r => r.pid));
        const next = personaIds.find(pid => !used.has(pid)) || personaIds[0];
        if (next) this.mix.push({ pid: next, share: 20 });
        render();
      });
      mixBox.append(add);
      renderShares();
    };
    const renderShares = () => {
      const total = this.mix.reduce((a, r) => a + (+r.share || 0), 0);
      mixBox.querySelectorAll(".mix-row").forEach((r, i) => {
        const pct = total > 0
          ? Math.round(100 * (+this.mix[i].share || 0) / total) : 0;
        r.querySelector(".mix-share").textContent = `${pct}% of traffic`;
      });
      foot.textContent = "Shares are relative — they normalize to 100% "
        + "when you save.";
    };
    render();
  },

  async save() {
    const id = $("#editor-id").value.trim();
    if (!id) { this.msg("id required", "error"); return; }
    let spec;
    if (this.kind === "personas") {
      spec = this.spec;
      if (!spec) { this.msg("nothing to save", "error"); return; }
      if (spec.timeouts && !Object.keys(spec.timeouts).length) {
        delete spec.timeouts;
      }
    } else {
      const seen = new Set();
      for (const r of this.mix) {
        if (seen.has(r.pid)) {
          this.msg(`"${r.pid}" appears twice in the mix — remove one`,
                   "error");
          return;
        }
        seen.add(r.pid);
      }
      const rows = this.mix.filter(r => (+r.share || 0) > 0);
      if (!rows.length) {
        this.msg("the mix needs at least one persona with a share",
                 "error");
        return;
      }
      const total = rows.reduce((a, r) => a + +r.share, 0);
      const weights = {};
      // Normalize to exactly 1.0 — the server validates the sum, and
      // rounding dust would bounce the save.
      let acc = 0;
      rows.forEach((r, i) => {
        const w = i === rows.length - 1
          ? +(1 - acc).toFixed(6)
          : +((+r.share) / total).toFixed(6);
        acc += w;
        weights[r.pid] = w;
      });
      spec = {
        name: this.spec.name || id,
        description: this.spec.description || "",
        persona_weights: weights,
      };
    }
    try {
      await api(`/api/${this.kind}/${id}`, {
        method: "PUT", body: JSON.stringify({ spec }),
      });
      this.editing = id;
      this.msg(`saved ${id} — the next run uses it`, "ok");
      await Control.loadCatalogs();  // refresh pickers + card numbers
      this.renderCard(this.kind, id);
      this.refreshLists();
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
