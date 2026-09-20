import { $, api, fmt } from "./lib/api.js";
import { on } from "./lib/events.js";
import { onShow, goTo } from "./lib/tabs.js";
import { Engines } from "./engines.js";
import { Control } from "./control.js";

/* ══ Engine optimizer ═════════════════════════════════════════── */

export const Optimizer = {
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
    onShow("optimizer", () => { this.refresh(); this.loadArena(true); });
    on("models:changed", () => this.invalidateArena());
    this.loadArena().then(() => this.refresh());
  },

  /* Models added or downloaded in Prepare change what the arena can
   * run; drop the cache so the next visit refetches. */
  invalidateArena() { this.arena = null; },

  /* ── Arena: dropdowns + cards, everything in play by default ─── */

  filters: { series: "all", size: "all" },

  async loadArena(force = false) {
    if (!this.arena || force) {
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
      { key: "engine", kind: "dim", title: "Engine",
        options: (dims.engine ?? []).map(v => [String(v),
          Engines.label(String(v))]),
        text: `Which server is under test. Held constant across a
          comparison: same model, same shape, same ladder, same
          stopping rule &mdash; only the server changes.
          ${Object.entries(this.arena?.engine_notes ?? {})
            .map(([e, t]) => `<b>${Engines.label(e)}</b>: ${t}`)
            .join("<br><br>")}` },
      // Engine-specific levers, each carrying what it MEASURED here.
      // A knob offered without its evidence invites the same afternoon
      // to be spent discovering its cost a second time.
      ...(this.arena?.levers ?? [])
        .filter(l => l.searchable && (dims[l.key] ?? []).length > 1)
        .map(l => ({
          key: l.key, kind: "dim", title: l.title,
          options: (dims[l.key] ?? []).map(v => [String(v),
            String(v) === l.default ? `${v} (default)` : String(v)]),
          text: `${l.text}${l.measured
            ? `<br><br><b>Measured here${
                l.verdict === "harm" ? " &mdash; slower"
                : l.verdict === "required" ? " &mdash; required" : ""
              }:</b> ${l.measured}` : ""}`,
        })),
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
        <code>config/arena.yaml</code> for planning — copy
        <code>config/arena.example.yaml</code>).</div>`;
      return;
    }
    // config/arena.yaml pinned a shape that nvidia-smi does not back
    // up: the arena is a plan, and a search here fails at launch.
    const unverified = /unverified/.test(hw.source || "");
    const hwWarn = !unverified ? "" : `<div class="callout"
      style="border-left-color:var(--warn);margin-top:14px" role="alert">
      <b>Hardware map is unverified.</b> <code>config/arena.yaml</code>
      declares ${hw.count} GPUs but nvidia-smi found
      ${hw.detected_count ?? 0} on this host. The arena below is sized
      from the file, not the box; fix or delete
      <code>config/arena.yaml</code> (the shipped template is
      <code>config/arena.example.yaml</code>) to size from what is
      detected.</div>`;
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

    box.innerHTML = hwWarn + `
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
        <span class="msg" style="align-self:end">${hw.count} GPUs${
            unverified ? ` <span class="status-marginal">(unverified)</span>` : ""} ·
          ${hw.vram_per_gpu_gb ?? "?"} GB each · ${hw.device_groups.length}
          PCIe/NUMA domain(s) · <abbr title="GPU memory fraction">gmu</abbr>
          fixed ${a.fixed.gpu_memory_utilization}</span>
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
    goTo("control");
  },
};
