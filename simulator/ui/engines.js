import { $, api } from "./lib/api.js";
import { emit } from "./lib/events.js";
import { onShow } from "./lib/tabs.js";

/* ── engine runtimes (Prepare step 4) ──────────────────────────────
 * An engine is a choice only once its image is on the box. This is
 * the one place that gate is opened, and Optimize and Benchmark read
 * their options from what it reports. */

export const Engines = {
  doc: null,
  _timer: null,

  init() {
    $("#engines-refresh").addEventListener("click", () => this.refresh());
    onShow("prepare", () => this.refresh());
    this.refresh();
  },

  msg(text, cls = "") {
    const el = $("#engines-msg");
    if (!el) return;
    el.textContent = text;
    el.className = `msg ${cls}`;
  },

  /* What Optimize and Benchmark may offer. Falls back to vLLM so a
   * failed status fetch degrades to the historical behaviour rather
   * than an empty picker. */
  available() {
    return this.doc?.available?.length ? this.doc.available
                                       : ["vllm_cuda_multi"];
  },

  row(engine) {
    return (this.doc?.runtimes ?? []).find(r => r.engine === engine) ?? null;
  },

  label(engine) {
    return this.row(engine)?.label ?? engine;
  },

  caveats(engine) {
    return this.row(engine)?.caveats ?? [];
  },

  async refresh() {
    let doc;
    try { doc = await api("/api/engines"); } catch { return; }
    this.doc = doc;

    const store = doc.image_store || {};
    $("#engines-store").textContent = store.path
      ? `${store.path}${store.free_gb != null
          ? ` — ${store.free_gb.toFixed(0)} GB free` : ""}`
      : "an unknown location";

    const box = $("#engines-list");
    box.innerHTML = "";
    let pulling = false;
    for (const r of doc.runtimes) {
      const el = document.createElement("div");
      el.className = "callout" + (r.staged ? "" : " muted");
      const pull = r.pull;
      if (pull?.running) pulling = true;
      let action;
      if (r.staged) {
        action = `<span class="status-pass">staged</span>`;
      } else if (pull?.running) {
        action = `<span class="msg">pulling… ${pull.progress || ""}</span>`;
      } else {
        action = `<button class="small primary" data-pull="${r.engine}">
          Pull ~${r.approx_gb} GB</button>`;
      }
      const failed = pull && !pull.running && pull.returncode
        ? `<div class="hint status-fail">the last pull exited
             ${pull.returncode} — see ${pull.log}</div>` : "";
      const caveats = (r.caveats || []).length
        ? `<ul class="hint" style="margin:6px 0 0 16px">`
          + r.caveats.map(c => `<li>${c}</li>`).join("") + `</ul>`
        : "";
      el.innerHTML = `<div class="row" style="align-items:flex-start">
          <div style="flex:1">
            <b>${r.label}</b>
            <div class="hint">${r.blurb}</div>
            <div class="hint"><code>${r.image}</code></div>
            ${caveats}${failed}
          </div>
          <div style="margin-left:auto">${action}</div>
        </div>`;
      box.append(el);
    }
    box.querySelectorAll("[data-pull]").forEach(b =>
      b.addEventListener("click", () => this.pull(b.dataset.pull)));

    // Keep the benchmark picker honest the moment staging changes
    // (Control re-syncs its engine choices on this).
    emit("engines:changed", doc);

    clearTimeout(this._timer);
    if (pulling) this._timer = setTimeout(() => this.refresh(), 3000);
  },

  async pull(engine) {
    this.msg(`starting the ${this.label(engine)} pull…`);
    try {
      await api("/api/engines/pull", {
        method: "POST", body: JSON.stringify({ engine }),
      });
      this.msg(`pulling ${this.label(engine)} — this runs in the `
        + `background and survives a page reload`, "ok");
    } catch (e) {
      this.msg(e.message, "error");
    }
    this.refresh();
  },
};
