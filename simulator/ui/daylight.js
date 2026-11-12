/* Daylight: the simplified capsim UI (docs/ui-daylight.md).
 *
 * Organised by the question being asked, not by tool. Each screen leads
 * with one number and one chart; detail is opt-in through <details>.
 * Talks to the same service as the Classic UI and covers the saturation
 * ("how fast can this server go?") path end to end. Capacity runs,
 * Roofline, the workload editor and system preparation open in Classic.
 */
import { api } from "./lib/api.js";
import {
  autoShape, barRows, bestRate, capacitySummary, lineChart, ladderChart, perUser,
  rateChart, resultNotes, stepVerdict, VERDICT,
} from "./lib/daylight-data.js";

const view = document.querySelector("#view");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 0) => v == null || Number.isNaN(+v) ? "—"
  : Number(v).toLocaleString("en-US", { maximumFractionDigits: d });
const short = (model) => String(model || "").split("/").pop();
const ENGINES = { vllm_cuda_multi: "vLLM", vllm_cuda: "vLLM", vllm: "vLLM (CPU)", sglang_cuda: "SGLang",
                  sglang: "SGLang (CPU)", trtllm: "TensorRT-LLM", ktransformers: "KTransformers",
                  llamacpp: "llama.cpp", mock: "self-test engine", remote: "remote endpoint" };
const engineName = (e) => ENGINES[e] || e || "";
const MOCK = "__mock__";
const HEADLINE = { kind: "persona", id: "headline_generation" };

const PRESETS = {
  short: { label: "Short", input: 128, output: 128 },
  chat: { label: "Chat", input: 1024, output: 1024 },
  document: { label: "Document", input: 8192, output: 1024 },
};
const REUSE = {
  industry: { label: "Industry standard: 54% of each prompt is shared", share: 0.54,
              why: "A system prompt every user shares, then text nobody else sends. Matches published business API traffic." },
  consumer_chat: { label: "Consumer chat: 62% of each prompt is shared", share: 0.62,
                   why: "The highest published figure for chat traffic." },
  conversation: { label: "Multi-turn conversation: 40% of each prompt is shared", share: 0.40,
                  why: "Published figure for multi-turn conversation traces." },
  template: { label: "None: every prompt is unique", share: 0,
              why: "Only the model's chat template repeats. The most conservative setting." },
};

const state = {
  hardware: null, models: [], runs: [], active: null, wasRunning: false,
  personas: [], cohorts: [], live: null,
  form: { goal: "speed", workload: "", model: "", preset: "short", input: 128, output: 128, reuse: "industry",
          engine: "vllm_cuda_multi", replicas: 1, tp: 1, max_num_seqs: 1024,
          kv_cache_dtype: "", gpu_memory_utilization: 0.9, max_concurrency: "" },
  message: "",
};

/* ── helpers ─────────────────────────────────────────────────────── */

function render(html) {
  const open = [...view.querySelectorAll("details[open]")].map((d) => d.id).filter(Boolean);
  const focus = document.activeElement?.id;
  view.innerHTML = html;
  for (const id of open) { const d = document.getElementById(id); if (d) d.open = true; }
  if (focus) document.getElementById(focus)?.focus();
}

function headlineRuns() {
  const out = [];
  for (const r of state.runs) {
    for (const c of r.cohorts || []) {
      if (c.mode === "headline_sweep" && c.peak_out_tok_s) {
        out.push({ run: r.name, model: c.model_id, engine: c.engine_type,
                   peak: c.peak_out_tok_s, streams: c.peak_streams, at: c.started_at });
      }
    }
  }
  return out;
}

function capacityRuns() {
  const out = [];
  for (const r of state.runs) {
    for (const c of r.cohorts || []) {
      const rate = c.rate_sla_per_min ?? c.rate_max_per_min;
      if (c.mode !== "headline_sweep" && rate != null) {
        out.push({ run: r.name, model: c.model_id, workload: c.cohort_name || c.cohort_id,
                   rate, users: c.sessions_at_max, at: c.started_at });
      }
    }
  }
  return out;
}

function isSpeedRun(a) {
  return a?.workload?.kind === "persona" && String(a.workload.id || "").startsWith("headline_");
}
function isCapacityRun(a) {
  return ["persona", "cohort"].includes(a?.workload?.kind) && !isSpeedRun(a);
}

async function refreshRuns() {
  try { state.runs = await api("/api/runs"); } catch { /* keep the old list */ }
}

function isRunning() { return Boolean(state.active?.running); }

function chip() {
  const el = document.querySelector("#server-chip");
  el.classList.toggle("busy", isRunning());
  document.querySelector("#server-state").textContent = isRunning() ? "running" : "idle";
}

/* ── Home ────────────────────────────────────────────────────────── */

const ICONS = {
  speed: '<svg viewBox="0 0 40 40" aria-hidden="true"><path d="M6 32 L14 22 L21 26 L34 8"/><path d="M26 8 H34 V16"/></svg>',
  users: '<svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="14" cy="14" r="5"/><circle cx="28" cy="16" r="4"/><path d="M5 32 C5 25 23 25 23 32"/><path d="M24 31 C24 26 35 26 35 31"/></svg>',
  compare: '<svg viewBox="0 0 40 40" aria-hidden="true"><rect x="5" y="6" width="8" height="28" rx="2"/><rect x="16" y="14" width="8" height="20" rx="2"/><rect x="27" y="10" width="8" height="24" rx="2"/></svg>',
};

function serverCard() {
  const hw = state.hardware || {};
  const cached = state.models.filter((m) => m.cached).length;
  const gpus = hw.gpus || 0;
  const ready = gpus > 0 && cached > 0;
  const line = gpus
    ? `${gpus} GPU${gpus > 1 ? "s" : ""}${hw.vram_per_gpu_gb ? `, ${num(hw.vram_per_gpu_gb)} GB each` : ""}. ${cached} model${cached === 1 ? "" : "s"} downloaded.`
    : "No GPUs detected on this machine. The self-test engine runs without hardware.";
  const status = isRunning()
    ? '<div class="note warn"><b>A run is in progress.</b> <a href="#run">Watch it</a></div>'
    : ready ? '<div class="note ok">Ready to run</div>'
      : `<div class="note">${gpus ? 'Download a model in <a href="index.html#prepare">System</a> to start.' : "Self-test only"}</div>`;
  return `<section class="card"><h2>This server</h2>${status}
    <p class="muted">${esc(line)}</p>
    <details class="inline" id="d-checks"><summary>Show system checks</summary>
      <div id="checks" class="muted small">Open to run the checks.</div></details></section>`;
}

function latestCard() {
  const speed = headlineRuns()[0], cap = capacityRuns()[0];
  if (cap && (!speed || (cap.at || "") > (speed.at || ""))) {
    return `<section class="card"><h2>Latest result</h2>
      <div class="muted small">${esc(short(cap.model))} · ${esc(cap.workload)} · ${esc((cap.at || "").slice(0, 10))}</div>
      <div class="mid">${num(cap.rate, 1)} <span class="small muted" style="font-weight:400">new sessions/min</span></div>
      <p><a href="#result/${esc(cap.run)}">See the result</a></p></section>`;
  }
  const latest = speed;
  if (!latest) {
    return `<section class="card"><h2>Latest result</h2>
      <p class="muted">No finished runs yet. Start one above and the result appears here.</p></section>`;
  }
  return `<section class="card"><h2>Latest result</h2>
    <div class="muted small">${esc(short(latest.model))} · ${esc((latest.at || "").slice(0, 10))}</div>
    <div class="mid">${num(latest.peak)} <span class="small muted" style="font-weight:400">tokens/sec</span></div>
    <p><a href="#result/${esc(latest.run)}">See the result</a></p></section>`;
}

function homeView() {
  render(`
    <header><h1 style="font-size:32px">What do you want to find out?</h1>
      <p class="muted">Pick a question. Capsim chooses sensible settings, and you can change any of them before it starts.</p></header>
    <div class="grid">
      <a class="card goal primary" href="#new">${ICONS.speed}
        <div class="t">How fast can this server go?</div>
        <div class="muted">Peak tokens per second for one model, pushed to saturation.</div>
        <div class="foot">Most used · about 20 minutes</div></a>
      <a class="card goal" href="#new/users">${ICONS.users}
        <div class="t">How many users can it support?</div>
        <div class="muted">Realistic sessions arriving over time, until response times slip.</div>
        <div class="foot">Capacity run · about 30 minutes</div></a>
      <a class="card goal" href="index.html#roofline">${ICONS.compare}
        <div class="t">Which model and engine is best?</div>
        <div class="muted">Compare several models and engines side by side, unattended.</div>
        <div class="foot">Roofline · runs for hours</div></a>
    </div>
    <div class="grid">${serverCard()}${latestCard()}</div>`);
  document.querySelector("#d-checks").addEventListener("toggle", async (e) => {
    if (!e.target.open || e.target.dataset.loaded) return;
    e.target.dataset.loaded = "1";
    const box = document.querySelector("#checks");
    box.textContent = "Checking…";
    try {
      const d = await api("/api/doctor");
      box.innerHTML = `<ul class="plain">${(d.checks || []).map((c) =>
        `<li><b>${esc(c.name)}</b>: ${esc(c.status)}${c.detail ? ` — ${esc(c.detail)}` : ""}</li>`).join("")}</ul>`;
    } catch (err) { box.textContent = `Could not run the checks: ${err.message}`; }
  });
}

/* ── New run ─────────────────────────────────────────────────────── */

function modelEntry(id) { return state.models.find((m) => m.model === id); }

function applyAuto() {
  const f = state.form;
  if (f.model === MOCK || !f.model) return;
  const shape = autoShape(modelEntry(f.model), state.hardware);
  f.replicas = shape.replicas; f.tp = shape.tp;
}

function engineSummary() {
  const f = state.form;
  if (f.model === MOCK) return "Self-test engine, no hardware";
  const eng = f.engine === "sglang_cuda" ? "SGLang" : "vLLM";
  return `${eng}, ${f.replicas} cop${f.replicas === 1 ? "y" : "ies"} on ${f.tp} GPU${f.tp > 1 ? "s" : ""} each, ${num(f.max_num_seqs)} requests per copy`;
}

function workloadInfo() {
  const [kind, id] = (state.form.workload || "").split(":");
  const src = kind === "cohort" ? state.cohorts.find((c) => c.id === id) : state.personas.find((p) => p.id === id);
  if (!src) return { kind, id, name: "the workload", description: "", tiles: [] };
  const b = kind === "cohort" ? src.blended || {} : {
    input_tokens: src.summary?.input_tokens?.median, output_tokens: src.summary?.output_tokens?.median,
    turns_per_session: src.summary?.turns_per_session?.median, think_gap_s: src.summary?.think_gap_s?.median };
  return { kind, id, name: src.name || id, description: src.description || "", tiles: [
    [num(b.input_tokens), "tokens in a typical question"], [num(b.output_tokens), "tokens in a typical answer"],
    [`${num(b.think_gap_s)} s`, "pause between turns"]] };
}

function workloadSection() {
  const f = state.form, w = workloadInfo();
  const singles = state.personas.filter((p) => !String(p.id).startsWith("headline_"));
  const opt = (kind, x) => `<option value="${kind}:${esc(x.id)}"${f.workload === `${kind}:${x.id}` ? " selected" : ""}>${esc(x.name || x.id)}</option>`;
  return `<section class="card">
    <label class="step-title" for="f-workload">2. Who is using it?</label>
    <select id="f-workload">
      <optgroup label="Teams (a mix of user types)">${state.cohorts.map((c) => opt("cohort", c)).join("")}</optgroup>
      <optgroup label="One type of user">${singles.map((p) => opt("persona", p)).join("")}</optgroup>
    </select>
    <p class="muted">${esc(w.description)}</p>
    <div class="tiles" style="margin-top:12px">${w.tiles.map(([v, l]) => `<div class="tile"><b>${v}</b><span>${l}</span></div>`).join("")}</div>
    <p class="small"><a href="index.html#personas">Edit or create workloads</a></p></section>`;
}

function newRunView() {
  const f = state.form;
  const cached = state.models.filter((m) => m.cached);
  if (!f.model) { f.model = cached[0]?.model || MOCK; applyAuto(); }
  const reuse = REUSE[f.reuse];
  const shared = Math.round(200 * reuse.share);
  const busy = isRunning();
  const users = f.goal === "users";
  if (users && !f.workload) f.workload = state.cohorts[0] ? `cohort:${state.cohorts[0].id}` : "";
  const speedSections = `        <section class="card">
          <span class="step-title">2. Request size</span>
          <div class="choices" role="group" aria-label="Request size">
            ${Object.entries(PRESETS).map(([k, p]) => `<button type="button" class="choice" data-preset="${k}" aria-pressed="${f.preset === k}">${p.label}<span>${num(p.input)} in · ${num(p.output)} out</span></button>`).join("")}
          </div>
          <details class="inline" id="d-shape"><summary>Set exact lengths</summary>
            <div class="fields">
              <label class="field">Prompt tokens<input type="number" id="f-input" min="1" value="${f.input}"></label>
              <label class="field">Output tokens<input type="number" id="f-output" min="1" value="${f.output}"></label>
            </div></details>
        </section>
        <section class="card">
          <span class="step-title">3. How realistic should the traffic be?</span>
          <div class="reuse">
            <svg viewBox="0 0 200 22" role="img" aria-label="${Math.round(reuse.share * 100)} percent of each prompt is shared">
              ${shared ? `<rect x="0" y="0" width="${shared}" height="22" rx="6" fill="#0b7a75"/>` : ""}
              <rect x="${shared ? shared + 4 : 0}" y="0" width="${200 - (shared ? shared + 4 : 0)}" height="22" rx="6" fill="#f0a35e"/></svg>
            <div><b>${esc(reuse.label)}</b><div class="muted small">${esc(reuse.why)}</div></div>
          </div>
          <details class="inline" id="d-reuse"><summary>Change the reuse level</summary>
            <label class="field" style="max-width:420px">Prefix-cache reuse
              <select id="f-reuse">${Object.entries(REUSE).map(([k, r]) => `<option value="${k}"${k === f.reuse ? " selected" : ""}>${esc(r.label)}</option>`).join("")}</select></label>
          </details>
        </section>
`;
  render(`
    <header><h1>${users ? "How many users can it support?" : "How fast can this server go?"}</h1>
      <p class="muted">${users ? "Two choices are enough to start." : "Three choices are enough to start."} Everything else has a default you can open and change.</p></header>
    <div class="row">
      <div class="main">
        <section class="card">
          <label class="step-title" for="f-model">1. Model</label>
          <select id="f-model">
            ${cached.map((m) => `<option value="${esc(m.model)}"${m.model === f.model ? " selected" : ""}>${esc(short(m.model))} · downloaded</option>`).join("")}
            <option value="${MOCK}"${f.model === MOCK ? " selected" : ""}>Self-test (no hardware needed)</option>
          </select>
          <p class="muted small">${cached.length ? "Only downloaded models are listed." : "No models are downloaded yet."}
            <a href="index.html#prepare">Download more in System</a></p>
        </section>
        ${users ? workloadSection() : speedSections}
        <details class="more" id="d-engine"${f.model === MOCK ? " hidden" : ""}>
          <summary>Engine settings <span class="hint">Automatic: ${esc(engineSummary())}</span></summary>
          <div class="fields">
            <label class="field">Engine<select id="f-engine">
              <option value="vllm_cuda_multi"${f.engine === "vllm_cuda_multi" ? " selected" : ""}>vLLM</option>
              <option value="sglang_cuda"${f.engine === "sglang_cuda" ? " selected" : ""}>SGLang</option></select></label>
            <label class="field">Copies of the model<input type="number" id="f-replicas" min="1" value="${f.replicas}"></label>
            <label class="field">GPUs per copy<input type="number" id="f-tp" min="1" value="${f.tp}"></label>
            <label class="field">Requests per copy<input type="number" id="f-max_num_seqs" min="1" value="${f.max_num_seqs}"></label>
            <label class="field">Cache precision<select id="f-kv_cache_dtype">
              <option value=""${!f.kv_cache_dtype ? " selected" : ""}>Engine default</option>
              <option value="fp8"${f.kv_cache_dtype === "fp8" ? " selected" : ""}>FP8</option></select></label>
            <label class="field">GPU memory share<input type="number" id="f-gpu_memory_utilization" min="0.1" max="1" step="0.05" value="${f.gpu_memory_utilization}"></label>
            <label class="field"${users ? " hidden" : ""}>Highest load to try<select id="f-max_concurrency">
              ${["", "512", "1024", "2048", "4096"].map((v) => `<option value="${v}"${v === String(f.max_concurrency) ? " selected" : ""}>${v ? `${num(v)} requests` : "Up to 8,192 requests"}</option>`).join("")}</select></label>
          </div>
        </details>
      </div>
      <aside class="card side">
        <h2>This run will</h2>
        <ul class="plain">
          <li>Load ${esc(f.model === MOCK ? "the self-test engine" : short(f.model))}</li>
          ${users ? `<li>Start simulated ${esc(workloadInfo().name)} sessions arriving at a steady rate</li>
          <li>Raise the rate until responses slow down or requests start to queue</li>
          <li>Report how many users that is</li>`
          : `<li>Send ${num(f.input)}-token prompts and ask for ${num(f.output)} tokens back</li>
          <li>Raise the load step by step${f.max_concurrency ? ` to ${num(f.max_concurrency)}` : " to 8,192"} simultaneous requests</li>
          <li>Stop when output stops rising</li>`}
        </ul>
        <div class="note small">${esc(engineSummary())}</div>
        <button type="button" class="btn primary block" id="start"${busy ? " disabled" : ""}>Start run</button>
        <div role="alert" class="note bad" id="start-msg">${esc(state.message)}</div>
        <p class="muted small">${busy ? 'The server is busy with another run. <a href="#run">Watch it</a>' : "The server is busy until it finishes. You can stop it at any time."}</p>
      </aside>
    </div>`);

  const set = (key, cast = (v) => v) => (e) => { state.form[key] = cast(e.target.value); newRunView(); };
  const on = (id, fn) => document.getElementById(id)?.addEventListener("change", fn);
  on("f-model", (e) => { f.model = e.target.value; applyAuto(); newRunView(); });
  on("f-reuse", set("reuse"));
  on("f-workload", set("workload"));
  on("f-engine", set("engine"));
  for (const k of ["replicas", "tp", "max_num_seqs"]) on(`f-${k}`, set(k, (v) => Math.max(1, parseInt(v, 10) || 1)));
  on("f-kv_cache_dtype", set("kv_cache_dtype"));
  on("f-gpu_memory_utilization", set("gpu_memory_utilization", (v) => +v || 0.9));
  on("f-max_concurrency", set("max_concurrency"));
  for (const k of ["input", "output"]) {
    on(`f-${k}`, (e) => { f[k] = Math.max(1, parseInt(e.target.value, 10) || 1); f.preset = ""; newRunView(); });
  }
  for (const b of view.querySelectorAll("[data-preset]")) {
    b.addEventListener("click", () => {
      const p = PRESETS[b.dataset.preset];
      Object.assign(f, { preset: b.dataset.preset, input: p.input, output: p.output });
      newRunView();
    });
  }
  document.getElementById("start").addEventListener("click", startRun);
}

export function runBody(f) {
  const users = f.goal === "users";
  const [kind, id] = (f.workload || "").split(":");
  const body = users
    ? { workload: { kind, id }, new_run: true, mode: "open" }
    : { workload: HEADLINE, new_run: true, mode: "open", prefix_reuse: f.reuse,
        shape: { input_tokens: f.input, output_tokens: f.output } };
  if (!users && f.max_concurrency) body.max_concurrency = +f.max_concurrency;
  if (f.model === MOCK) { body.profile = "mock"; return body; }
  body.custom = {
    model_id: f.model, device: "gpu", engine: f.engine,
    replicas: f.replicas, tp: f.tp,
    gpu_memory_utilization: f.gpu_memory_utilization,
    max_num_seqs: f.max_num_seqs,
    kv_cache_dtype: f.kv_cache_dtype || null,
  };
  return body;
}

async function startRun() {
  state.message = "";
  const btn = document.getElementById("start");
  btn.disabled = true; btn.textContent = "Starting…";
  try {
    await api("/api/runs", { method: "POST", body: JSON.stringify(runBody(state.form)) });
    await poll();
    location.hash = "#run";
  } catch (err) {
    state.message = `Could not start: ${err.message}`;
    newRunView();
  }
}

/* ── Running ─────────────────────────────────────────────────────── */

/* Where a finished run sends you: its own result when it produced one. */
function doneButton() {
  const run = state.runs[0]?.name;
  return run && !state.active?.error
    ? `<a class="btn primary" href="#result/${encodeURIComponent(run)}">See the result</a>`
    : '<a class="btn primary" href="#results">See results</a>';
}

function runningView() {
  const a = state.active;
  if (!a) {
    const latest = headlineRuns()[0];
    render(`<header><h1>Nothing is running</h1>
      <p class="muted">Start a run to watch it here.</p></header>
      <div><a class="btn primary" href="#new">New run</a>
      ${latest ? ` <a class="btn" href="#result/${esc(latest.run)}">Open the latest result</a>` : ""}</div>`);
    return;
  }
  const p = a.progress || {};
  if (isCapacityRun(a)) { capacityRunningView(a); return; }
  if (!isSpeedRun(a)) {
    render(`<header><h1>A run is in progress</h1>
      <p class="muted">Capacity, Roofline and Optimize runs have their own live views.</p></header>
      <div><a class="btn primary" href="index.html#control">Open the live view</a></div>`);
    return;
  }
  const done = p.rungs_done || [];
  const ladder = p.ladder || [];
  const peak = p.peak;
  const sh = p.shape || {};
  const loading = !p.rung;
  const steps = [
    `<li class="${loading ? "now" : "done"}"${loading ? ' aria-current="step"' : ""}>${loading ? "Loading the engine" : "Engine loaded"}</li>`,
    done.length ? `<li class="done">${done.length} load level${done.length > 1 ? "s" : ""} measured</li>` : "",
    !loading && a.running ? `<li class="now" aria-current="step">Measuring ${num(p.concurrency)} requests</li>` : "",
    !loading && a.running && p.rungs > p.rung ? `<li>${p.rungs - p.rung} more to go</li>` : "",
  ].join("");
  const rising = done.length >= 2 && done[done.length - 1].out_tok_s > done[done.length - 2].out_tok_s;
  render(`
    <header class="head">
      <div><h1>${a.running ? "Measuring" : a.error === "cancelled" ? "Stopped" : a.error ? "Failed" : "Finished"} ${esc(short(p.model) || "the model")}</h1>
        <p class="muted">${sh.input_tokens != null ? `${num(sh.input_tokens)} tokens in, ${num(sh.output_tokens)} out` : "Getting ready"}${p.engine ? ` · ${esc(engineName(p.engine))}` : ""}</p></div>
      ${a.running ? '<button type="button" class="btn danger" id="stop">Stop run</button>' : doneButton()}
    </header>
    ${a.error === "cancelled" ? '<div class="note">You stopped this run. Load levels measured before the stop are kept.</div>'
      : a.error ? `<div class="note bad" role="alert"><b>The run failed.</b> ${esc(a.error)}</div>` : ""}
    <ol class="steps" aria-label="Progress">${steps}</ol>
    <div class="row">
      <section class="card narrow">
        <div><div class="muted">Best so far</div>
          <div class="big">${peak?.out_tok_s ? num(peak.out_tok_s) : "—"}</div>
          <div class="muted">${peak?.out_tok_s ? `tokens per second, at ${num(peak.concurrency)} simultaneous requests` : "The first number arrives after the first load level."}</div></div>
        ${done.length >= 2 ? `<div class="note">${rising ? "Output is still rising with load. The next step shows whether it keeps going." : "Output has levelled off, so the run is close to the ceiling."}</div>` : ""}
      </section>
      <section class="card wide"><h2>Output at each load level</h2>
        ${ladderChart(ladder, done, a.running ? p.concurrency : null)}
        <div class="muted small" style="text-align:center">Simultaneous requests</div></section>
    </div>
    <div class="stack">
      <details class="more" id="d-live-times"><summary>Show response times</summary>
        ${done.length ? barRows(done.map((r) => ({ name: `${num(r.concurrency)} requests`, value: r.ttft_p95_ms, text: r.ttft_p95_ms == null ? "—" : `${num(r.ttft_p95_ms)} ms to first token` })))
          : '<p class="muted">Available after the first load level.</p>'}</details>
      <details class="more" id="d-live-detail"><summary>Show run details</summary>
        <ul class="plain"><li>Model: ${esc(p.model || "—")}</li><li>Engine: ${esc(engineName(p.engine) || "—")}</li>
          <li>Load levels: ${esc(ladder.map((n) => num(n)).join(", ") || "—")}</li><li>Phase: ${esc(p.phase || "—")}</li></ul></details>
    </div>`);
  document.getElementById("stop")?.addEventListener("click", async () => {
    if (!window.confirm("Stop this run? Load levels already measured are kept.")) return;
    try { await api("/api/runs/stop", { method: "POST" }); } catch { /* the poll shows the state */ }
    poll();
  });
}

function capacityRunningView(a) {
  const live = state.live || {};
  const steps = (live.steps || []).filter((s) => s.arrival_rate_per_min != null);
  const snap = (live.snapshots || []).reduce((m, x) => !m || x.snapshot_at_ms > m.snapshot_at_ms ? x : m, null) || {};
  const best = bestRate(steps);
  const judged = steps.filter((s) => stepVerdict(s) !== "pending");
  const w = (() => { const [kind, id] = [a.workload.kind, a.workload.id];
    const src = kind === "cohort" ? state.cohorts.find((c) => c.id === id) : state.personas.find((x) => x.id === id);
    return src?.name || id; })();
  const stopped = a.error === "cancelled";
  render(`
    <header class="head">
      <div><h1>${a.running ? "Finding the limit for" : stopped ? "Stopped" : a.error ? "Failed" : "Finished"} ${esc(short(live.run?.model_id) || "the model")}</h1>
        <p class="muted">${esc(w)} sessions${live.run?.engine_type ? ` · ${esc(engineName(live.run.engine_type))}` : ""}</p></div>
      ${a.running ? '<button type="button" class="btn danger" id="stop">Stop run</button>' : doneButton()}
    </header>
    ${stopped ? '<div class="note">You stopped this run. Rates judged before the stop are kept.</div>'
      : a.error ? `<div class="note bad" role="alert"><b>The run failed.</b> ${esc(a.error)}</div>` : ""}
    <ol class="steps" aria-label="Progress">
      <li class="${snap.arrival_rate_per_min != null || judged.length ? "done" : "now"}">${snap.arrival_rate_per_min != null || judged.length ? "Engine loaded" : "Loading the engine"}</li>
      ${judged.length ? `<li class="done">${judged.length} rate${judged.length > 1 ? "s" : ""} judged</li>` : ""}
      ${a.running && snap.arrival_rate_per_min != null ? `<li class="now" aria-current="step">Testing ${num(snap.arrival_rate_per_min, 1)} new sessions per minute</li>` : ""}
    </ol>
    <div class="row">
      <section class="card narrow">
        <div><div class="muted">Keeping up so far</div>
          <div class="big">${best != null ? num(best, 1) : "—"}</div>
          <div class="muted">${best != null ? "new sessions per minute" : "The first verdict arrives after the first rate is measured."}</div></div>
        <div class="tiles">
          <div class="tile"><b>${num(snap.active_sessions)}</b><span>sessions active now</span></div>
          <div class="tile"><b>${num(snap.queue_depth)}</b><span>requests waiting</span></div>
          <div class="tile"><b>${num(snap.requests_completed)}</b><span>turns answered</span></div>
        </div>
        <div class="note small">${esc(snap.phase || "Getting ready")}</div>
      </section>
      <section class="card wide"><h2>Wait for the first token at each rate</h2>${rateChart(steps)}</section>
    </div>
    <div class="stack">
      <details class="more" id="d-cap-rates"><summary>Show each rate tested</summary>
        ${steps.length ? barRows(steps.map((s) => ({ name: `${num(s.arrival_rate_per_min, 1)} per minute`, sub: VERDICT[stepVerdict(s)].label,
          value: s.ttft_p95_ms, text: s.ttft_p95_ms == null ? "—" : `${num(s.ttft_p95_ms)} ms to first token` })))
          : '<p class="muted">Available after the first rate is measured.</p>'}</details>
      <details class="more" id="d-cap-live"><summary>Show the full live view <span class="hint">Charts for sessions, queue, GPU and power</span></summary>
        <p><a href="index.html#control">Open the live charts</a></p></details>
    </div>`);
  document.getElementById("stop")?.addEventListener("click", async () => {
    if (!window.confirm("Stop this run? Rates already judged are kept.")) return;
    try { await api("/api/runs/stop", { method: "POST" }); } catch { /* the poll shows the state */ }
    poll();
  });
}

/* ── Result ──────────────────────────────────────────────────────── */

async function capacityResultView(run) {
  let doc;
  try { doc = await api(`/api/runs/${encodeURIComponent(run)}/export?slim=true`); }
  catch (err) {
    render(`<header><h1>Result not available</h1><p class="muted">${esc(err.message)}</p></header>
      <div><a class="btn" href="#results">Back to results</a></div>`);
    return;
  }
  const c = (doc.cohorts || [])[0];
  if (!c) { render('<header><h1>Result not available</h1><p class="muted">This run recorded no measurements.</p></header>'); return; }
  const sum = capacitySummary(c);
  const ol = c.open_loop || {};
  const curve = (c.curve || []).filter((p) => p.arrival_rate_per_min != null).sort((x, y) => x.arrival_rate_per_min - y.arrival_rate_per_min);
  const at = curve.find((p) => p.arrival_rate_per_min === sum.rate) || {};
  const notes = sum.notes.map((n) => `<div class="note ${n.kind}"><b>${esc(n.title)}</b> ${esc(n.text)}</div>`).join("");
  render(`
    <header class="head">
      <div><h1>${esc(short(c.model?.id || c.model))}</h1>
        <p class="muted">${esc(c.name || c.id)} · ${esc(engineName(c.engine?.type || c.engine))} · ${esc((c.started_at || "").slice(0, 10))}</p></div>
      <a class="btn" href="#new/users">Run again</a>
      <a class="btn primary" href="/api/runs/${encodeURIComponent(run)}/export" download="${esc(run)}-result.json">Export</a>
    </header>
    <div class="row">
      <section class="card narrow">
        <div><div class="muted">Supports about</div><div class="big">${sum.users != null ? num(sum.users) : "—"}</div>
          <div class="muted">people using it at the same time</div></div>
        <div class="tiles">
          <div class="tile"><b>${num(sum.rate, 1)}</b><span>new sessions per minute</span></div>
          <div class="tile"><b>${at.ttft_p95_ms != null ? `${num(at.ttft_p95_ms)} ms` : "—"}</b><span>wait for the first token</span></div>
          <div class="tile"><b>${num(perUser(at), 0)}</b><span>tokens/sec per user</span></div>
        </div>
        ${notes}
      </section>
      <section class="card wide"><h2>Wait for the first token as traffic rises</h2>${rateChart(curve)}</section>
    </div>
    <div class="stack">
      <details class="more" id="d-zones"><summary>What each traffic level feels like</summary>
        <ul class="plain">${Object.entries(ol.zones || {}).map(([k, v]) => `<li><b>${esc(k)}</b>: ${esc(v)}</li>`).join("") || "<li>Not available for this run.</li>"}</ul></details>
      <details class="more" id="d-limit"><summary>What limited it <span class="hint">${esc(String(c.bottleneck || "not identified").replace(/_/g, " "))}</span></summary>
        <p class="muted">Bottleneck from telemetry: ${esc(String(c.bottleneck || "not identified").replace(/_/g, " "))}. The full evidence is in the detailed report.</p></details>
      <details class="more" id="d-cap-data"><summary>See the data</summary>
        <div class="scroll"><table>
          <thead><tr><th>New sessions/min</th><th>Verdict</th><th>Active sessions</th><th>First token, p95</th><th>Per user</th><th>Turns over target</th><th>Turns measured</th></tr></thead>
          <tbody>${curve.map((p) => `<tr${p.arrival_rate_per_min === sum.rate ? ' class="peak"' : ""}>
            <td>${num(p.arrival_rate_per_min, 1)}${p.arrival_rate_per_min === sum.rate ? " · capacity" : ""}</td><td>${VERDICT[stepVerdict(p)].label}</td>
            <td>${num(p.active_sessions_mean, 1)}</td><td>${p.ttft_p95_ms == null ? "—" : `${num(p.ttft_p95_ms)} ms`}</td><td>${num(perUser(p), 0)}</td>
            <td>${p.target_miss_rate == null ? "—" : `${Math.round(p.target_miss_rate * 100)}%`}</td><td>${num(p.sample_size)}</td></tr>`).join("")}</tbody></table></div></details>
      <details class="more" id="d-cap-how"><summary>How this was measured <span class="hint">Settings to quote alongside the number</span></summary>
        <ul class="plain">
          <li>Model: ${esc(c.model?.id || c.model || "—")}</li>
          <li>Workload: ${esc(c.name || c.id)}${c.description ? ` — ${esc(c.description)}` : ""}</li>
          <li>Method: sessions arrive at a steady rate; capacity is the highest rate the engine keeps up with${sum.atTarget ? " while meeting the workload's response-time targets" : ""}</li>
          <li>Users at the same time: ${sum.atTarget ? "derived from the rate and the measured session length" : "measured at the highest stable rate"}</li>
        </ul>
        <p><a href="index.html#results">Open the detailed report in All results</a></p></details>
    </div>`);
}

async function resultView(run) {
  render('<p class="muted">Loading the result…</p>');
  let doc;
  try { doc = await api(`/api/runs/${encodeURIComponent(run)}/headline`); }
  catch { await capacityResultView(run); return; }
  const rungs = (doc.rungs || []).filter((r) => r.out_tok_s).sort((a, b) => a.concurrency - b.concurrency);
  const peak = doc.peak;
  const sh = doc.shape || {};
  const pc = doc.prefix_cache || {};
  const pct = (v) => v == null ? "—" : `${Math.round(v * 1000) / 10}%`;
  const perWatt = peak?.gpu_power_w ? peak.out_tok_s / peak.gpu_power_w : null;
  const notes = resultNotes(doc).map((n) => `<div class="note ${n.kind}"><b>${esc(n.title)}</b> ${esc(n.text)}</div>`).join("");
  const powered = rungs.filter((r) => r.gpu_power_w);
  render(`
    <header class="head">
      <div><h1>${esc(short(doc.model))}</h1>
        <p class="muted">${sh.input_tokens != null ? `${num(sh.input_tokens)} tokens in, ${num(sh.output_tokens)} out` : ""} · ${esc(engineName(doc.engine))} · ${esc((doc.generated_at || "").slice(0, 10))}</p></div>
      <a class="btn" href="#new">Run again</a>
      <a class="btn primary" href="/api/runs/${encodeURIComponent(run)}/headline" download="${esc(run)}-result.json">Export</a>
    </header>
    <div class="row">
      <section class="card narrow">
        <div><div class="muted">Peak output</div><div class="big">${num(peak?.out_tok_s)}</div><div class="muted">tokens per second</div></div>
        <div class="tiles">
          <div class="tile"><b>${num(peak?.concurrency)}</b><span>requests at once</span></div>
          <div class="tile"><b>${num(perUser(peak), 0)}</b><span>tokens/sec per user</span></div>
          <div class="tile"><b>${num(perWatt, 1)}</b><span>tokens/sec per GPU watt</span></div>
        </div>
        ${notes}
      </section>
      <section class="card wide"><h2>Output as load increases</h2>${lineChart(rungs, peak)}</section>
    </div>
    <div class="stack">
      <details class="more" id="d-user"><summary>Speed each user sees <span class="hint">${num(perUser(peak), 0)} tokens/sec at the peak</span></summary>
        ${barRows(rungs.map((r) => ({ name: `${num(r.concurrency)} requests`, value: perUser(r), text: `${num(perUser(r), 0)} tokens/sec` })))}</details>
      <details class="more" id="d-wait"><summary>Wait for the first token <span class="hint">${num(peak?.ttft_p95_ms)} ms at the peak, 95th percentile</span></summary>
        ${barRows(rungs.map((r) => ({ name: `${num(r.concurrency)} requests`, value: r.ttft_p95_ms, text: `${num(r.ttft_p95_ms)} ms` })))}</details>
      ${powered.length ? `<details class="more" id="d-power"><summary>Power <span class="hint">${num(peak?.gpu_power_w)} W across the GPUs at the peak</span></summary>
        ${barRows(powered.map((r) => ({ name: `${num(r.concurrency)} requests`, value: r.gpu_power_w, text: `${num(r.gpu_power_w)} W`, kind: "power" })))}</details>` : ""}
      <details class="more" id="d-data"><summary>See the data</summary>
        <div class="scroll"><table>
          <thead><tr><th>Requests at once</th><th>Tokens/sec</th><th>Per user</th><th>First token, p95</th><th>GPU power</th><th>Cache reuse</th><th>Failed</th></tr></thead>
          <tbody>${rungs.map((r) => `<tr${r.concurrency === peak?.concurrency ? ' class="peak"' : ""}>
            <td>${num(r.concurrency)}${r.concurrency === peak?.concurrency ? " · peak" : ""}</td><td>${num(r.out_tok_s)}</td>
            <td>${num(perUser(r), 0)}</td><td>${num(r.ttft_p95_ms)} ms</td><td>${r.gpu_power_w ? `${num(r.gpu_power_w)} W` : "—"}</td>
            <td>${pct(r.prefix_hit_rate)}</td><td>${num(r.errors)}</td></tr>`).join("")}</tbody></table></div></details>
      <details class="more" id="d-how"><summary>How this was measured <span class="hint">Settings to quote alongside the number</span></summary>
        <ul class="plain">
          <li>Model: ${esc(doc.model || "—")}</li><li>Engine: ${esc(engineName(doc.engine) || "—")}</li>
          <li>Request size: ${num(sh.input_tokens)} tokens in, ${num(sh.output_tokens)} out${sh.ignore_eos ? ", fixed length" : ""}</li>
          <li>Prefix cache: ${pct(peak?.prefix_hit_rate)} of prompt tokens served from cache at the peak${pc.template_floor != null ? `, designed for ${pct(pc.template_floor)}` : ""}${pc.system_prompt_actual_tokens ? ` (${num(pc.system_prompt_actual_tokens)}-token shared system prompt, ${num(pc.user_tokens)} unique user tokens)` : ""}</li>
          ${pc.source ? `<li>Reuse level: ${esc(pc.source)}</li>` : ""}
          <li>Why it stopped: ${esc(doc.stop_reason || "—")}</li>
        </ul>
        <p><a href="index.html#results">Open the full report in All results</a></p></details>
    </div>`);
}

/* ── Results ─────────────────────────────────────────────────────── */

function resultsView() {
  const all = headlineRuns();
  const q = (state.query || "").toLowerCase();
  const shown = all.filter((r) => !q || `${r.model} ${r.engine} ${r.run}`.toLowerCase().includes(q))
    .sort((a, b) => b.peak - a.peak);
  const caps = capacityRuns().filter((r) => !q || `${r.model} ${r.workload} ${r.run}`.toLowerCase().includes(q))
    .sort((a, b) => b.rate - a.rate);
  const listed = new Set([...all, ...capacityRuns()].map((r) => r.run));
  const others = state.runs.filter((r) => !listed.has(r.name));
  render(`
    <header><h1>Results</h1><p class="muted">Speed runs and capacity runs. Select a bar to open that run.</p></header>
    <label class="field" style="max-width:360px">Find a run<input type="search" id="q" placeholder="Model or engine" value="${esc(state.query || "")}"></label>
    <section class="card"><h2>Peak tokens per second</h2>
      ${shown.length ? barRows(shown.map((r) => ({
        name: short(r.model), sub: `${r.run} · ${(r.at || "").slice(0, 10)}`, value: r.peak,
        text: num(r.peak), href: `#result/${encodeURIComponent(r.run)}` })))
        : `<p class="muted">${all.length ? "No run matches that search." : 'No speed runs yet. <a href="#new">Start one</a>'}</p>`}
    </section>
    <section class="card"><h2>New sessions per minute supported</h2>
      ${caps.length ? barRows(caps.map((r) => ({
        name: short(r.model), sub: `${r.workload} · ${r.run} · ${(r.at || "").slice(0, 10)}`, value: r.rate,
        text: num(r.rate, 1), href: `#result/${encodeURIComponent(r.run)}` })))
        : `<p class="muted">${capacityRuns().length ? "No run matches that search." : 'No capacity runs yet. <a href="#new/users">Start one</a>'}</p>`}
    </section>
    <details class="more" id="d-all"><summary>Show every run <span class="hint">${others.length} run${others.length === 1 ? "" : "s"} with no result, such as stopped ones</span></summary>
      ${others.length ? `<div class="scroll"><table><thead><tr><th>Run</th><th>Workload</th><th>Model</th><th>Status</th></tr></thead><tbody>
        ${others.map((r) => { const c = (r.cohorts || [])[0] || {}; return `<tr><td>${esc(r.name)}</td><td>${esc(c.cohort_name || "—")}</td><td>${esc(short(c.model_id) || "—")}</td><td>${esc(c.final_status || "—")}</td></tr>`; }).join("")}
        </tbody></table></div><p><a href="index.html#results">Open these in All results</a></p>` : '<p class="muted">There are no other runs.</p>'}
    </details>`);
  document.getElementById("q").addEventListener("input", (e) => { state.query = e.target.value; resultsView(); });
}

/* ── routing and polling ─────────────────────────────────────────── */

function route() {
  const [name, arg] = (location.hash.replace(/^#/, "") || "home").split("/");
  for (const a of document.querySelectorAll(".top a[data-route]")) {
    const on = a.dataset.route === name || (a.dataset.route === "new" && name === "run")
      || (a.dataset.route === "results" && name === "result");
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  if (name === "new") { state.form.goal = arg === "users" ? "users" : "speed"; state.message = ""; newRunView(); }
  else if (name === "run") runningView();
  else if (name === "result" && arg) resultView(decodeURIComponent(arg));
  else if (name === "results") resultsView();
  else homeView();
}

async function poll() {
  try { state.active = (await api("/api/status")).active_run; } catch { return; }
  chip();
  const name = (location.hash.replace(/^#/, "") || "home").split("/")[0];
  if (state.wasRunning && !isRunning()) {
    // A run just ended: pick up its result.
    await refreshRuns();
    const latest = state.runs[0] && { run: state.runs[0].name };
    if (name === "run" && latest && !state.active?.error) { location.hash = `#result/${encodeURIComponent(latest.run)}`; }
    else if (name !== "result") route();
  } else if (name === "run") {
    if (isCapacityRun(state.active)) {
      try { state.live = await api("/api/live/backfill"); } catch { /* keep the last reading */ }
    }
    runningView();
  }
  state.wasRunning = isRunning();
}

async function init() {
  const [hw, models, personas, cohorts] = await Promise.allSettled([
    api("/api/hardware"), api("/api/models"), api("/api/personas"), api("/api/cohorts")]);
  state.hardware = hw.status === "fulfilled" ? hw.value : {};
  state.models = models.status === "fulfilled" ? models.value.models || [] : [];
  state.personas = personas.status === "fulfilled" ? personas.value : [];
  state.cohorts = cohorts.status === "fulfilled" ? cohorts.value : [];
  await refreshRuns();
  await poll();
  state.wasRunning = isRunning();
  window.addEventListener("hashchange", route);
  route();
  setInterval(poll, 2500);
}

if (typeof document !== "undefined" && view) init();
