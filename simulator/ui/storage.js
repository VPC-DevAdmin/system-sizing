import { $, api, keyActivate } from "./lib/api.js";
import { onShow } from "./lib/tabs.js";
import { Models } from "./models.js";

/* ══ Storage (choose the disk + location for weights) ═════════── */

export const Storage = {
  selected: null,

  init() {
    $("#storage-refresh").addEventListener("click", () => this.refresh());
    $("#storage-apply").addEventListener("click", () => this.apply());
    onShow("prepare", () => this.refresh());
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
      keyActivate(el);
      el.setAttribute("aria-label",
        `use ${fs.mountpoint}, ${fs.free_gb.toFixed(0)} GB free`);
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
      // The recipe wipes a disk, so it is folded away and armed by a
      // checkbox rather than sitting pasteable on the page.
      un.innerHTML = `<div class="unmounted-box callout">
        <b>${doc.unmounted.length} unmounted disk(s) on this box:</b><br>` +
        doc.unmounted.map(label).join("<br>") +
        `<br><br>capsim won't format or mount disks — that needs root and destroys
        whatever is on them.
        <details><summary>Show the format &amp; mount recipe for
          <code>/dev/${example.name}</code> (destructive)</summary>
          <label class="arm"><input type="checkbox" id="unmounted-arm">
            I understand these commands erase <code>/dev/${example.name}</code>${
              example.has_partitions ? " and everything on its partitions" : ""}</label>
          <div id="unmounted-cmds" hidden>Run these on the host <b>one line at
            a time</b>${example.has_partitions
              ? " — read the check-first lines carefully" : " (the disk is blank)"},
            then Refresh:
            <pre>${example.commands.join("\n")}</pre></div></details></div>`;
      $("#unmounted-arm").addEventListener("change", (e) => {
        $("#unmounted-cmds").hidden = !e.target.checked;
      });
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
