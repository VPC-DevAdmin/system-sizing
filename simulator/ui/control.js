import { $, api, fmt, MOCK_MODEL } from "./lib/api.js";
import { on, emit } from "./lib/events.js";
import { onShow, goTo } from "./lib/tabs.js";
import { Engines } from "./engines.js";
import { Results } from "./results.js";

/* ══ Run control ══════════════════════════════════════════════── */

export const Control = {
  catalogs: { profiles: {}, personas: [], cohorts: [] },
  hw: { gpus: 0 },
  modelList: [],
  deviceMode: "gpu",
  matchedProfile: null,   // {name, ...profile} when an optimization fits
  engineDefaults: {},     // the values Advanced was prefilled with
  _dlPoll: null,

  async init() {
    // Subscribed before the first await: Engines' first refresh can
    // land while the catalogs are still loading. The run events come
    // off the telemetry socket (Live) and re-poll the status pill.
    on("engines:changed", () => this.syncEngineChoices());
    on("run:started", () => this.pollStatus());
    on("run:finished", () => this.pollStatus());
    await this.loadCatalogs();
    Results.refreshRuns();
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
    for (const id of ["eng-engine", "eng-replicas", "eng-tp",
                      "eng-placement", "eng-gmu", "eng-mns", "eng-mbt",
                      "eng-kv", "eng-ep", "eng-trc"]) {
      $("#" + id).addEventListener("input", () => {
        this._engineDirty = true;
        if (id === "eng-trc") this.updateTrcWarning();
        if (id === "eng-engine") this.updateEngineNote();
      });
    }
    $("#start-btn").addEventListener("click", () => this.start());
    $("#stop-btn").addEventListener("click", () => this.stop());
    $("#runs-refresh").addEventListener("click", () => Results.refreshRuns());
    $("#doctor-btn").addEventListener("click", () => this.doctor());
    // Bound ONCE. This used to live in updateWorkloadNote, which runs
    // on every 2 s status poll — the button gained a listener per
    // poll and one click launched the search N times.
    $("#hl-optimize")?.addEventListener("click", () =>
      this.startHeadlineOptimize());
    onShow("prepare", () => this.onPrepareShow());
    setInterval(() => this.pollStatus(), 2000);
    this.pollStatus();
  },

  /* GPUs nvidia-smi actually saw. `gpus` may come from a pinned
   * config/arena.yaml (a planning shape); the CPU/GPU toggle follows
   * reality, not the plan. */
  detectedGpus() {
    const hw = this.hw || {};
    return (hw.detected_gpus ?? hw.gpus ?? 0) || 0;
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
      this.deviceMode = this.detectedGpus() > 0 ? "gpu" : "cpu";
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
    seg.classList.toggle("disabled", !(this.detectedGpus() > 0));
    if (!(this.detectedGpus() > 0)) this.deviceMode = "cpu";
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
    // The self-test engine is offered whenever nothing is staged: a
    // fresh host can exercise the whole pipeline in minutes, and the
    // default must never be a 230 GB model nobody downloaded.
    const mockAvail = !!this.catalogs.profiles?.mock && !cached.length;
    if (mockAvail) {
      const g = document.createElement("optgroup");
      g.label = "No hardware needed";
      g.append(new Option("Mock engine (no hardware) — pipeline self-test",
                          MOCK_MODEL, false, prev === MOCK_MODEL));
      sel.prepend(g);
    }
    const stillThere = [...sel.options].some(o => o.value === prev);
    if (!prev || !stillThere) {
      // Default order: the optimized model when one is cached, else
      // any cached model, else the mock, else the first catalog entry.
      const opt = Object.values(this.catalogs.profiles).find(p =>
        p?.optimized && p.fits_hardware
        && cached.some(m => m.model === p.model_id));
      sel.value = opt?.model_id ?? cached[0]?.model
        ?? (mockAvail ? MOCK_MODEL : rest[0]?.model ?? "");
    }
  },

  /* The mock profile is a pipeline self-test: no engine form, no
   * device choice, no download. */
  onMockSelected() {
    this.matchedProfile = null;
    this._formKey = MOCK_MODEL;
    $("#engine-form").style.display = "none";
    $("#engine-note").textContent =
      "Mock engine — a simulated server; exercises the run pipeline "
      + "without hardware.";
    $("#device-seg").classList.add("disabled");
    $("#model-note").innerHTML = `<span class="ok-note">✓ Self-test
      engine — completes in minutes, needs no GPU and no download</span>`;
    this.headlineShape = null;
    this.updateWorkloadNote();
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

  /* Offer only engines whose runtime is staged. Runs on Engines'
   * engines:changed event, after every status refresh, so pulling one
   * mid-session makes it selectable without a reload — and an engine
   * that vanished cannot stay silently selected. */
  syncEngineChoices() {
    const sel = $("#eng-engine");
    if (!sel) return;
    const opts = Engines.available();
    const keep = sel.value;
    sel.innerHTML = "";
    for (const e of opts) {
      const o = document.createElement("option");
      o.value = e;
      o.textContent = Engines.label(e);
      sel.append(o);
    }
    sel.value = opts.includes(keep) ? keep : opts[0];
    // One staged engine is not a choice; don't imply it is.
    sel.closest("label").style.display = opts.length > 1 ? "" : "none";
    this.updateEngineNote();
    this.renderHeadlineEngines();
  },

  /* Say where this engine's numbers are NOT comparable with the
   * other's, at the moment it is picked. */
  updateEngineNote() {
    const box = $("#eng-engine-note");
    if (!box) return;
    const list = Engines.caveats($("#eng-engine")?.value);
    box.hidden = !list.length;
    box.innerHTML = list.length
      ? "<ul style='margin:0 0 0 16px'>"
        + list.map(c => `<li>${c}</li>`).join("") + "</ul>"
      : "";
  },

  /* Which engines the joint search covers. Both is a deliberate
   * choice: it doubles the grid, so it is never the default. */
  renderHeadlineEngines() {
    const box = $("#hl-engines");
    if (!box) return;
    const opts = Engines.available();
    if (opts.length < 2) { box.innerHTML = ""; box.hidden = true; return; }
    box.hidden = false;
    const chosen = this.headlineEngines ?? [$("#eng-engine")?.value
                                            || opts[0]];
    box.innerHTML = "";
    for (const e of opts) {
      const id = `hl-eng-${e}`;
      const lab = document.createElement("label");
      lab.className = "check";
      lab.innerHTML = `<input type="checkbox" id="${id}"
        ${chosen.includes(e) ? "checked" : ""}> ${Engines.label(e)}`;
      lab.querySelector("input").addEventListener("change", () => {
        this.headlineEngines = opts.filter(
          x => $(`#hl-eng-${x}`)?.checked);
        if (!this.headlineEngines.length) {
          // Never launch a search with nothing to search.
          this.headlineEngines = [e];
          $(`#${id}`).checked = true;
        }
      });
      box.append(lab);
    }
    this.headlineEngines = chosen;
  },

  setEngineForm(d) {
    if (d.engine && Engines.available().includes(d.engine)) {
      $("#eng-engine").value = d.engine;
    }
    this.updateEngineNote();
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
    // Never inherited from a profile: running a repo's own Python is
    // a decision the operator makes per run, not a setting that
    // rides along with an engine shape.
    $("#eng-trc").value = "off";
    this.updateTrcWarning();
    this.engineDefaults = this.readEngineForm();
    this._engineDirty = false;
  },

  readEngineForm() {
    return {
      engine: $("#eng-engine").value || "vllm_cuda_multi",
      replicas: +$("#eng-replicas").value || 1,
      tp: +$("#eng-tp").value || 1,
      placement: $("#eng-placement").value,
      gpu_memory_utilization: +$("#eng-gmu").value || 0.9,
      max_num_seqs: $("#eng-mns").value.trim(),
      max_num_batched_tokens: $("#eng-mbt").value.trim(),
      kv_cache_dtype: $("#eng-kv").value,
      expert_parallel: $("#eng-ep").value === "on",
      trust_remote_code: $("#eng-trc").value === "on",
    };
  },

  /* Human summary of what an engine form means — echoed at start and
   * in the run banner so there is never doubt about what's running. */
  engineSummary(form) {
    const eng = Engines.available().length > 1
      ? `${Engines.label(form.engine)} · ` : "";
    return eng + `${form.replicas}×tp${form.tp} ${form.placement}`
      + ` · gmu ${form.gpu_memory_utilization}`
      + (form.max_num_seqs ? ` · mns ${form.max_num_seqs}` : "")
      + (form.max_num_batched_tokens
          ? ` · mbt ${form.max_num_batched_tokens}` : "")
      + ` · KV ${form.kv_cache_dtype || "auto"}`
      + (form.expert_parallel ? " · EP on" : "")
      + (form.trust_remote_code ? " · trust-remote-code" : "");
  },

  /* Spell out what the flag does at the moment it is switched on —
   * "trust remote code" understates it. */
  updateTrcWarning() {
    const box = $("#eng-trc-warn");
    if (!box) return;
    const on = $("#eng-trc")?.value === "on";
    box.hidden = !on;
    if (on) {
      box.innerHTML = `<span class="status-fail">The model repository's
        own Python will be executed inside the engine container.</span>
        Only enable this for a repo you have reason to trust — some
        architectures (Kimi-Linear, for one) ship their own config and
        model classes and cannot load without it. The run's engine
        summary records that it was enabled.`;
    }
  },

  /* Model or device changed: find a fitting optimization, prefill
   * Advanced from it (else conservative), and set the note line —
   * green check, optimize link, or download link. */
  onModelChange() {
    const model = $("#bench-model").value;
    if (model === MOCK_MODEL) { this.onMockSelected(); return; }
    this.renderDeviceSeg();       // re-enable after the mock disabled it
    const entry = this.modelEntry();
    const cpuMode = this.deviceMode === "cpu";
    const gpuEngines = ["vllm_cuda", "vllm_cuda_multi", "trtllm"];
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
      // Its own id: this used to inject a second #goto-optimize (the
      // Prepare tab owns that one), so $() found Prepare's button and
      // this link did nothing.
      note.innerHTML = `<button type="button" class="note-link"
        id="goto-optimize-workload">No optimized engine for this model
        yet — run the optimizer →</button>`;
      $("#goto-optimize-workload").addEventListener("click", () =>
        goTo("optimizer"));
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
        emit("models:changed");        // a new model is now runnable
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
    if (model === MOCK_MODEL) {
      body.profile = "mock";
      body.engineDesc = "the mock engine (no hardware)";
      return body;
    }
    if (entry && !entry.cached) {
      this.msg("that model isn't downloaded yet — use the download "
        + "link under the picker", "error");
      return null;
    }
    if (this.deviceMode === "cpu") {
      body.custom = {
        model_id: model, device: "cpu",
        trust_remote_code: $("#eng-trc")?.value === "on",
      };
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
        engine: form.engine,
        replicas: form.replicas,
        tp: form.tp,
        placement: form.placement,
        gpu_memory_utilization: form.gpu_memory_utilization,
        max_num_seqs: +form.max_num_seqs || null,
        max_num_batched_tokens: +form.max_num_batched_tokens || null,
        kv_cache_dtype: form.kv_cache_dtype || null,
        expert_parallel: form.expert_parallel,
        trust_remote_code: form.trust_remote_code,
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
    // #hl-optimize is bound once in init(): it is static markup.
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

  /* Joint engine + shape search. Engine shape and request shape are
   * coupled — optimizing them separately walks in circles — so this
   * walks the product and then sweeps the winner at full resolution. */
  async startHeadlineOptimize() {
    if (this._starting) return;          // double-click guard
    const body = this.buildRunBody({
      kind: "headline_optimize", id: "headline_generation" });
    if (!body) return;
    delete body.engineDesc;
    if (!body.custom) {
      this.msg("the joint search varies engine settings, so it needs a "
        + "custom engine — open Advanced and set your baseline", "error");
      return;
    }
    body.preset = $("#hl-preset").value;
    body.input_tokens = +$("#hl-input-tokens")?.value || 128;
    const engines = (Engines.available().length > 1
                     && this.headlineEngines?.length)
      ? this.headlineEngines : null;
    if (engines) body.search_engines = engines;
    const shapes = { quick: 4, standard: 9, thorough: 16 }[body.preset] || 9;
    const pairs = shapes * (engines ? engines.length : 1);
    this._starting = true;
    $("#hl-optimize").disabled = true;
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      this.msg(`searching ${pairs} engine/shape combinations`
        + (engines && engines.length > 1
           ? ` across ${engines.map(e => Engines.label(e)).join(" and ")}`
           : "")
        + `, then `
        + `sweeping the winner at full resolution — expect roughly `
        + `${Math.round(pairs * 10 + 14)} minutes`, "ok");
      this.pollStatus();
    } catch (e) {
      this.msg(e.message, "error");
    } finally {
      this._starting = false;
      $("#hl-optimize").disabled = false;
    }
  },

  async startShapeSearch() {
    const body = this.buildRunBody({ kind: "headline_search" });
    // Prompt length is pinned, not searched — input tokens can only
    // cost this objective, so searching them walks to the smallest
    // prompt on the lattice and yields a number nobody can quote.
    const pinned = +$("#hl-input-tokens")?.value || 0;
    if (body && pinned) body.input_tokens = pinned;
    if (!body) return;
    delete body.engineDesc;
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      this.msg(`shape search started — prompt pinned at `
        + `${pinned || 128} tokens, searching output length only; `
        + `the winner becomes Headline: Generation's shape`, "ok");
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
    try { status = await api("/api/status"); }
    catch (e) { this.setDisconnected(e); return; }
    if (this._offline) this.setReconnected();
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
    // Saturation runs get their own instrument; see the Headline
    // module for why the capacity panels do not fit them. Headline
    // and Live (the idle panel) both listen for this.
    emit("status", { active, running });
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
        : w.kind === "headline_optimize" ? "Engine + shape search"
        : isSat ? `${nameOf(w.kind, w.id)} · saturation benchmark`
        : nameOf(w.kind, w.id);
      // Rung-by-rung progress, so the minutes between rungs never
      // read as a hang.
      const p = active.progress || {};
      const isOpt = w.kind === "headline_optimize";
      const optProgress = !(isOpt && running) ? ""
        : ` · <b>combination ${p.pair ?? 1} of ${p.pairs ?? "?"}</b>`
          + (p.current?.max_num_seqs
            ? ` (mns ${p.current.max_num_seqs}, ${p.current.output_tokens}
               out)` : "")
          + (p.phase ? ` · ${p.phase}` : "")
          + (p.best?.out_tok_s
            ? ` · best ${Math.round(
                p.best.out_tok_s).toLocaleString()} tok/s` : "");
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
        ` · started ${since}` + satProgress + optProgress +
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
      box.querySelector("#goto-results")?.addEventListener("click", () =>
        goTo("results"));
    } else {
      box.hidden = true;
    }
  },

  /* The server stopped answering. Say so on the pill (it used to
   * freeze on "run active" forever after a serve crash) and hold
   * Start until it is back; Stop stays enabled — a stop against a
   * dead server just reports the error. */
  setDisconnected(err) {
    if (this._offline) return;
    this._offline = true;
    const pill = $("#status-pill");
    pill.textContent = "disconnected";
    pill.className = "pill offline";
    pill.title = `no answer from the service: ${err?.message ?? err}`;
    $("#start-btn").disabled = true;
    this.msg("lost contact with the capsim service — is it still "
      + "running? Retrying every 2 s.", "error");
    emit("disconnected", err);
  },

  setReconnected() {
    this._offline = false;
    $("#status-pill").title = "";
    this.msg("reconnected", "ok");
    // The server may have restarted with different state.
    this.loadCatalogs();
    Results.refreshRuns();
    emit("models:changed");
  },

  /* Prepare's first visit runs doctor unless this browser tab has a
   * result already (sessionStorage survives reloads, not new tabs —
   * a host can change between sessions). */
  onPrepareShow() {
    if (this._doctorShown) return;
    this._doctorShown = true;
    let cached = null;
    try {
      cached = JSON.parse(sessionStorage.getItem("capsim.doctor") || "null");
    } catch { /* storage unavailable */ }
    if (cached?.report) this.renderDoctor(cached.report, cached.at);
    else this.doctor();
  },

  async doctor() {
    const out = $("#doctor-out");
    out.textContent = "probing host…";
    try {
      const report = await api("/api/doctor");
      this.renderDoctor(report, null);
      try {
        sessionStorage.setItem("capsim.doctor",
          JSON.stringify({ report, at: Date.now() }));
      } catch { /* storage unavailable */ }
    } catch (e) {
      out.innerHTML = `<span class="d-fail">doctor failed: ${e.message}</span>`;
    }
  },

  renderDoctor(report, cachedAt) {
    const out = $("#doctor-out");
    const rows = report.checks.map(c =>
      `<tr><td>${c.name}</td><td class="d-${c.status}">${c.status}</td>
       <td>${c.detail}</td></tr>`).join("");
    const rec = report.recommended_configs.length
      ? `<p>Recommended profiles: <b>${report.recommended_configs.join(", ")}</b></p>` : "";
    const stale = cachedAt
      ? `<p class="msg">Result from ${fmt.clock(cachedAt)} this session —
         press Run doctor to probe again.</p>` : "";
    out.innerHTML = `<table><thead><tr><th>Check</th><th>Status</th>
      <th>Detail</th></tr></thead><tbody>${rows}</tbody></table>${rec}${stale}`;
  },
};
