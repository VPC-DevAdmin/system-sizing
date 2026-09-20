import { $, api, keyActivate, logTo, logFrom, fmtNum } from "./lib/api.js";
import { onShow, goTo } from "./lib/tabs.js";
import { Control } from "./control.js";

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

export const Editor = {
  kind: "personas",    // "personas" | "cohorts"
  editing: null,       // id being edited, null for new
  spec: null,          // working persona spec (mutated by sliders)
  mix: [],             // working cohort rows [{pid, share}]

  init() {
    $("#editor-save").addEventListener("click", () => this.save());
    $("#persona-new").addEventListener("click", () => this.startNewPersona());
    $("#cohort-new").addEventListener("click", () => this.startNewCohort());
    onShow("personas", () => this.refreshLists());
    // Benchmark form → designer jump.
    $("#workload-edit-link").addEventListener("click", (e) => {
      e.preventDefault();
      goTo("personas");
    });
  },

  msg(text, cls = "") {
    const el = $("#editor-msg");
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  async refreshLists() {
    let personas, cohorts;
    try {
      [personas, cohorts] = await Promise.all([
        api("/api/personas"), api("/api/cohorts"),
      ]);
    } catch (e) {
      this.msg(`could not load the workload lists: ${e.message}`, "error");
      return;
    }
    const fill = (sel, items, kind) => {
      const ul = $(sel);
      ul.innerHTML = "";
      for (const item of items) {
        const li = document.createElement("li");
        li.textContent = item.name || item.id;
        li.classList.toggle(
          "active", this.kind === kind && this.editing === item.id);
        li.addEventListener("click", () => this.open(kind, item.id));
        keyActivate(li);
        ul.append(li);
      }
    };
    fill("#persona-list", personas, "personas");
    fill("#cohort-list", cohorts, "cohorts");
  },

  async open(kind, id) {
    let detail;
    try {
      detail = await api(`/api/${kind}/${id}`);
    } catch (e) {
      this.msg(`could not open ${id}: ${e.message}`, "error");
      return;
    }
    this.kind = kind;
    this.editing = id;
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
