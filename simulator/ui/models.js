import { $, api } from "./lib/api.js";
import { emit } from "./lib/events.js";
import { onShow, goTo } from "./lib/tabs.js";
import { Control } from "./control.js";

/* ══ Model staging ════════════════════════════════════════════── */

export const Models = {
  polling: null,

  init() {
    $("#models-refresh").addEventListener("click", () => this.refresh());
    onShow("prepare", () => this.refresh());
    $("#goto-optimize").addEventListener("click", () => goTo("optimizer"));
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

  /* Staging-side messages (download failures used to land in the
   * Optimize tab's #opt-msg, where nobody on Prepare could see them). */
  msg(text, cls = "") {
    const el = $("#models-msg");
    if (!el) return;
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

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
          emit("models:changed");
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
    if (r.created) emit("models:changed");
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
    try { doc = await api("/api/models"); }
    catch (e) { this.msg(`could not load the model list: ${e.message}`, "error"); return; }
    this.doc = doc;
    $("#models-cache-dir").textContent = doc.cache_dir;
    // A download finishing (or a cache dir change) alters what the
    // arena and the Workload picker can offer.
    const cachedKey = doc.models.filter(m => m.cached).map(m => m.model)
      .sort().join("|");
    if (this._cachedKey != null && cachedKey !== this._cachedKey) {
      emit("models:changed");
      Control.loadCatalogs();
    }
    this._cachedKey = cachedKey;
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
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="msg"
        style="padding:16px 8px;text-align:center">${doc.models.length
          ? "No models match these filters — clear the family, precision "
            + "or status filter, or the search box."
          : "The catalog is empty — add a model by its Hugging Face id "
            + "below, or press <b>Discover new models</b>."}</td></tr>`;
    }
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
      this.msg(`${model}: ${e.message}`, "error");
      this.refresh();
      return;
    }
    this.msg(`downloading ${model} — progress shows in the table`, "ok");
    this.refresh();
  },
};
