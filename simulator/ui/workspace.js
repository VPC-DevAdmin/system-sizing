import { Benchmarks } from "./benchmarks.js";
/* Parallel workspace entry point. Reuse the canonical form markup and tested
 * controllers, but compose their sections into a separate information design.
 * index.html/app.js remain the Classic entry point; no iframe or second worker. */
import { escapeHTML as esc, number as num, pairResult, pairState, metricScale,
  METRICS, methodFromWorkload, rooflineRequest } from './lib/workspace-data.js';
import { api } from './lib/api.js';
import { openEvidence } from './workspace-evidence.js';

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const short = s => String(s || '').split('/').pop();
const iconPaths = {
  system: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/>',
  assets: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  lab: '<path d="M9 3h6m-5 0v7l-6 9q-1 2 2 2h12q3 0 2-2l-6-9V3M8 15h8"/>',
  workloads: '<path d="M4 18V6l16 6-16 6Z"/>',
  history: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">${iconPaths[name] || iconPaths.lab}</svg>`;
const state = { view: 'prepare', method: 'roofline', stage: 'setup', lens: 'matrix',
  asset: 'models', metric: 'out_tok_s', relative: false, tier: 'all', selection: null,
  inspector: 'configuration', historyPage: 0, doctor: null, doctorAt: null, pins: [], search: '' };
let Control, Roofline, Engines, Storage, Models, Optimizer, Results, goTo;
let matrixSignature = '', inspectorSignature = '', optionsSyncedFor = null;
try {
  const saved=JSON.parse(sessionStorage.getItem('capsim.workspace')||'{}');
  for(const [key,allowed] of Object.entries({stage:['setup','workspace'],lens:['matrix','spectrum','hardware'],asset:['models','engines','profiles'],metric:Object.keys(METRICS),tier:['all','fast','large','beyond_vram']})) if(allowed.includes(saved[key]))state[key]=saved[key];
} catch { /* Preference storage is optional. */ }
const LABELS = { prepare: ['System readiness', 'Prepare the host, storage and dependencies for a reliable experiment.'],
  assets: ['Models & runtimes', 'Stage the artifacts and engines your experiments need.'],
  roofline: ['Roofline exploration', 'Find the performance envelope across fast, large and beyond-VRAM models.'],
  optimizer: ['Launch optimization', 'Explore launch configurations and preserve the evidence behind the winner.'],
  control: ['Workload capacity', 'Measure service limits against the experience your users expect.'],
  personas: ['Workload designer', 'Define how people use the system, then combine them into realistic traffic.'],
  infrastructure: ['Infrastructure Tests', 'Repeatable A/B experiments with separate outputs for each claim.'],
  results: ['History & evidence', 'Revisit measurements, compare conditions and export the recorded evidence.'] };
const labViews = ['roofline', 'optimizer', 'control'];

function element(tag, cls, html) {
  const el = document.createElement(tag); if (cls) el.className = cls;
  if (html) el.innerHTML = html; return el;
}
function fold(node, title, open = false) {
  const details = element('details', 'ws-disclosure'); details.open = open;
  details.append(element('summary', '', title));
  node.before(details); details.append(node); return details;
}
function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = element('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function resizeCharts() {
  requestAnimationFrame(() => Object.values(Chart.instances).forEach(chart => chart.resize()));
}

async function boot() {
  const response = await fetch('index.html');
  if (!response.ok) throw new Error(`Interface template unavailable (${response.status})`);
  const template = new DOMParser().parseFromString(await response.text(), 'text/html');
  const shell = element('div', 'ws-shell', `
    <aside class="ws-sidebar" aria-label="Workspace navigation">
      <a class="ws-brand" href="workspace.html"><span class="ws-mark">◇</span> capsim</a>
      <div class="ws-brand-note">INFERENCE WORKBENCH</div>
      <nav class="ws-nav" aria-label="Main navigation">
        ${[['prepare','system','System'],['assets','assets','Assets'],['roofline','lab','Lab'],['personas','workloads','Workloads'],['infrastructure','lab','Infrastructure'],['results','history','History']].map(([view,i,label]) => `<button data-nav="${view}">${icon(i)}<span>${label}</span></button>`).join('')}
      </nav>
      <div class="ws-host"><span class="ws-eyebrow">CONNECTED SERVICE</span><strong id="ws-host-name"></strong><span id="ws-connection" role="status">Connecting…</span></div>
      <a class="ws-classic" href="index.html" id="ws-classic">↗ &nbsp; Classic UI</a>
      <span class="ws-version">New workspace · Preview</span>
    </aside>
    <div class="ws-main" id="ws-content" tabindex="-1">
      <header class="ws-heading"><div><div class="ws-eyebrow" id="ws-context">SYSTEM / READINESS</div><h1 id="ws-title"></h1><p id="ws-description"></p></div><div class="ws-header-actions" id="ws-status-slot"></div></header>
      <div class="ws-job" id="ws-job" hidden role="status"></div>
      <div class="ws-lab-toolbar" id="ws-lab-toolbar" hidden>
        <div class="ws-methods" aria-label="Experiment method">
          ${[['roofline','Roofline','Performance envelope'],['optimizer','Optimize','Launch configuration'],['capacity','Capacity','Workload service limits'],['saturation','Saturation','Maximum sustained rate']].map(([id,title,sub])=>`<button data-method="${id}"><b>${title}</b><span>${sub}</span></button>`).join('')}
        </div>
        <div class="ws-stagebar"><div class="ws-segment" aria-label="Lab view"><button data-stage="setup">Setup</button><button data-stage="workspace">Workspace</button></div><span id="ws-method-note" class="msg"></span><button id="ws-stop" class="danger small" hidden>Stop active experiment</button></div>
      </div>
      <div id="ws-panels"></div>
    </div>`);
  document.body.append(shell);
  $('#ws-host-name').textContent = location.host;
  const header = document.importNode(template.querySelector('header'), true);
  header.hidden = true; header.id = 'ws-native-navigation'; document.body.append(header);
  $('#ws-status-slot').append($('#status-pill'));
  const tab = element('button'); tab.dataset.view = 'assets'; tab.id = 'tab-assets';
  tab.textContent='Assets'; tab.setAttribute('role','tab'); tab.setAttribute('aria-controls','view-assets'); tab.setAttribute('aria-selected','false'); tab.tabIndex = -1;
  $('#tabs').append(tab);
  for (const main of template.querySelectorAll('main.view')) $('#ws-panels').append(document.importNode(main, true));
  const assets = element('main','view'); assets.id = 'view-assets'; assets.setAttribute('role','tabpanel'); assets.setAttribute('aria-labelledby','tab-assets');
  $('#ws-panels').append(assets);
  composeSystem(); composeAssets(); composeLab(); composeRoofline(); composeHistory();
  // These imports must follow template attachment: legacy modules register
  // tabs and read chart colors at module evaluation time.
  const modules = await Promise.all([
    import('./control.js'), import('./roofline.js'), import('./engines.js'), import('./storage.js'),
    import('./models.js'), import('./optimizer.js'), import('./results.js'), import('./lib/tabs.js'), import('./lib/events.js'),
  ]);
  [{Control}, {Roofline}, {Engines}, {Storage}, {Models}, {Optimizer}, {Results}, {goTo}] = modules;
  installAdapters();
  modules[7].onShow('assets', () => { Models.refresh(); Engines.refresh(); renderProfiles(); });
  modules[8].on('status', renderStatus);
  modules[8].on('disconnected', () => { $('#ws-connection').textContent = 'Disconnected · data may be stale'; $('#ws-connection').className = 'status-fail'; });
  bindNavigation();
  await import('./app.js');
  for (const btn of $$('#tabs [data-view]')) btn.addEventListener('click', () => showPage(btn.dataset.view));
  showPage(modules[7].currentView());
  $('#workload-select').addEventListener('change', () => {
    state.method = methodFromWorkload($('#workload-select').value); if(state.view === 'control') renderChrome();
  });
  $('#ws-loading').remove();
  document.body.dataset.ready = 'true';
  renderSystem(); renderProfiles();
}

function composeSystem() {
  const grid = $('#view-prepare .steps-grid');
  const panels = [...grid.children];
  panels[0].classList.add('ws-checks'); panels[1].classList.add('ws-storage');
  panels[2].dataset.asset = 'models'; panels[3].dataset.asset = 'engines';
  $('#view-assets').append(panels[2], panels[3]);
  panels[4].classList.add('ws-next');
  panels[0].querySelector('.t').textContent = 'Dependencies & host checks';
  panels[1].querySelector('.t').textContent = 'Storage & model cache';
  panels[4].querySelector('.t').textContent = 'Ready to explore';
  $('#doctor-btn').textContent = 'Recheck host';
  panels[1].querySelector('.d').firstChild.textContent = 'Choose the filesystem and directory used by model downloads and engine launches. Current: ';
  panels[4].querySelector('.d').textContent = 'Explore the hardware, optimize a launch configuration, or measure workload capacity.';
  $('#view-prepare').prepend(element('section','ws-system-overview panel', '<div class="ws-card-head"><div><span class="ws-eyebrow">HARDWARE CONTEXT</span><h2>Know the machine you are testing</h2></div><span class="ws-badge" id="ws-hardware-source">Loading</span></div><div id="ws-hardware"></div>'));
  const status = element('div','ws-readiness'); status.id='ws-readiness';
  status.innerHTML='<span class="ws-orb"></span><div><strong>Checking system readiness</strong><p>Host checks identify prerequisites; an engine smoke test verifies execution.</p></div>';
  $('#view-prepare').prepend(status);
  fold($('#doctor-out'), 'All checks and remediation');
}
function composeAssets() {
  $('#view-assets').prepend(element('div','ws-segment ws-asset-tabs', '<button data-asset="models">Models</button><button data-asset="engines">Engine runtimes</button><button data-asset="profiles">Launch profiles</button>'));
  const profiles = element('section','panel'); profiles.dataset.asset='profiles'; profiles.id='ws-profiles';
  $('#view-assets').append(profiles);
  $('#view-assets [data-asset="models"] .t').textContent = 'Model library';
  $('#view-assets [data-asset="engines"] .t').textContent = 'Engine runtimes';
}
function composeLab() {
  // The shared optimizer rewrites the heading text when data arrives. Keep
  // its history selector outside that heading so it survives rendering.
  const historyLabel=$('#opt-history').closest('label');
  $('#opt-search-title').before(historyLabel);historyLabel.classList.add('ws-optimizer-history');
  for (const view of labViews) {
    const main=$(`#view-${view}`), children=[...main.children];
    const setup=element('div','ws-setup'), workspace=element('div','ws-workspace');
    setup.append(children[0]); children.slice(1).forEach(el=>workspace.append(el));
    main.append(setup,workspace);
    const description=children[0].querySelector(':scope > p.msg');
    if(description) fold(description, 'How this experiment works');
    const empty=element('div','ws-empty panel'); empty.dataset.empty=view;
    empty.innerHTML=`${icon('lab')}<h2>Your experiment workspace</h2><p>${view==='roofline'?'The model × engine landscape appears after a search is planned.':view==='optimizer'?'Candidate results appear here as the search progresses.':'Live measurements appear here when a run starts.'}</p><button class="primary" data-stage="setup">Configure experiment</button>`;
    workspace.prepend(empty);
  }
  $('#view-roofline .ws-setup h2').textContent='Configure the performance search';
  $('#view-optimizer .ws-setup h2').textContent='Define the launch search';
  const freshLabel=$('#new-run').closest('label');
  for(const node of [...freshLabel.childNodes])if(node!==$('#new-run'))node.remove();
  freshLabel.append(document.createTextNode(' New results directory'));
  $('#bench-advanced summary').textContent='Expert controls · engine, precision & placement';
  const searchOptions=element('details','ws-disclosure',`<summary>Search strategy & recovery</summary><div class="form-grid"><label>Recovery<select id="ws-recovery"><option value="resume">Resume matching results</option><option value="retry">Retry failed engine cells</option><option value="redo">Remeasure an engine</option><option value="fresh">Start a fresh investigation</option></select></label><label id="ws-recovery-engine-wrap" hidden>Engine scope<select id="ws-recovery-engine"><option value="">All selected engines</option></select></label><label>Priority model IDs<input id="ws-priority" placeholder="org/model, org/model"></label></div><div class="row wrap"><label class="check"><input id="ws-kt-scale" type="checkbox">KTransformers scaling · TP × GPU experts × streams</label></div><p id="ws-recovery-note" class="msg"></p>`);
  $('#rf-shapes').after(searchOptions);
  for(const control of searchOptions.querySelectorAll('input,select'))control.addEventListener('change',renderScope);
  const candidateDetails=fold($('#rf-candidates'),'Model selection & fit reasoning'); candidateDetails.id='ws-candidates-detail';
  $('#rf-model-mode').addEventListener('change',()=>{ candidateDetails.open=$('#rf-model-mode').value==='manual'; });
  const preview=element('aside','ws-setup-preview panel', '<span class="ws-eyebrow">EXPERIMENT PREVIEW</span><h2>Scope & readiness</h2><div id="ws-scope"></div><div class="ws-mini-note">Generation throughput is the search objective. Confirmation repeats each winner at full resolution.</div><div id="ws-setup-readiness"></div><button data-nav="prepare" class="ghost small">Review system readiness ↗</button>');
  $('#view-roofline .ws-setup').append(preview);
  preview.insertBefore($('#rf-cost'),$('#ws-setup-readiness'));
  const actions=element('div','ws-setup-actions');
  actions.append($('#rf-start'),$('#rf-stop'),$('#rf-msg'));
  preview.append(actions);
  const tiers=element('div','ws-tier-intro', '<div><span class="ws-tier-dot fast"></span><b>Fast</b><small>Peak rate across vendor lines</small></div><div><span class="ws-tier-dot large"></span><b>Large</b><small>Explore GPU memory limits</small></div><div><span class="ws-tier-dot beyond_vram"></span><b>Beyond VRAM</b><small>Host-memory execution</small></div>');
  $('#view-roofline .ws-setup h2').after(tiers);
  // Keep start/stop ownership in the existing controllers. The workspace stop
  // action forwards to the corresponding original control, never to a new API.
  $('#ws-stop').addEventListener('click',()=>$(state.view==='roofline'?'#rf-stop':state.view==='optimizer'?'#opt-stop':'#stop-btn').click());
}
function composeRoofline() {
  const workspace=$('#view-roofline .ws-workspace');
  const summary=element('div','ws-findings'); summary.id='ws-findings'; workspace.prepend(summary);
  const toolbar=element('div','ws-landscape-toolbar', `<div><span class="ws-eyebrow">PERFORMANCE LANDSCAPE</span><div class="ws-segment"><button data-lens="matrix">Matrix</button><button data-lens="spectrum">Spectrum</button><button data-lens="hardware">Hardware</button></div></div><div class="row wrap"><label>Tier<select id="ws-tier"><option value="all">All tiers</option><option value="fast">Fast</option><option value="large">Large on GPU</option><option value="beyond_vram">Beyond VRAM</option></select></label><label>Metric<select id="ws-metric">${Object.entries(METRICS).map(([key,m])=>`<option value="${key}">${m.label} (${m.unit})</option>`).join('')}</select></label><label>Scale<select id="ws-scale"><option value="absolute">Absolute</option><option value="relative">Within model</option></select></label></div>`);
  summary.after(toolbar);
  const analysis=element('div','ws-analysis'); analysis.id='ws-analysis';
  toolbar.after(analysis);
  const plot=element('div','ws-plot'); analysis.append(plot);
  plot.append($('#rf-matrix-panel'),$('#rf-spectrum-panel'),$('#rf-charts'));
  const hardware=element('section','panel'); hardware.id='ws-recorded-hardware'; plot.append(hardware);
  const inspector=element('aside','ws-inspector panel', `<div class="ws-eyebrow">SELECTED PAIRING</div><h2 id="ws-inspector-title">Select a result</h2><div id="ws-inspector-head"></div><div class="ws-inspector-tabs" aria-label="Result detail">${['configuration','measurement','hardware','evidence'].map(t=>`<button data-inspect="${t}">${t[0].toUpperCase()+t.slice(1)}</button>`).join('')}</div><div id="ws-inspector-body"></div><div id="ws-inspector-actions" class="ws-inspector-actions"></div>`);
  analysis.append(inspector);
  $('#rf-best-panel').classList.add('ws-original-summary');
  const details=fold($('#rf-results-panel'), 'All candidate shapes & attempts'); details.classList.add('ws-attempts');
  fold($('#rf-staging'), 'Model staging');
  workspace.prepend($('#rf-progress-panel'));
  $('#rf-progress-panel h2').textContent='Search progress';
  fold($('#rf-spectrum'), 'Model facts & evidence');
  $('#rf-matrix-panel h2').textContent='Model × engine';
  $('#rf-matrix-panel .chart-note').textContent='The service selects each pairing by completion success, confirmation, then generation rate. Color shows the selected metric; the label shows evidence. Missing values are never zero.';
}
function composeHistory() {
  const exportKind=element('label','ws-export-kind','Export detail<select id="ws-export-kind"><option value="full">Full JSON</option><option value="slim">Slim JSON</option></select>');
  $('#result-export-dl').before(exportKind);
  const controls=element('div','ws-history-controls row wrap', '<label>Find a run<input id="ws-history-search" type="search" placeholder="Model, workload or run name"></label><label>Method<select id="ws-history-method"><option value="all">All methods</option><option value="open_loop">Open-loop capacity</option><option value="headline_sweep">Saturation</option><option value="pool">Pool ramp</option></select></label><span id="ws-history-count" class="msg"></span>');
  $('#run-list').before(controls);
  const pager=element('div','ws-history-pager','<button id="ws-history-prev" class="small">Previous</button><span id="ws-history-page" class="msg"></span><button id="ws-history-next" class="small">Next</button>');
  $('#run-list').after(pager);
  $('#run-list').after(element('div','ws-comparison-note msg', 'Compare results with matching models, hardware, workload and measurement conditions. Different methods do not share a capacity axis.'));
  const pins=element('section','panel'); pins.id='ws-pins'; pins.hidden=true; $('#view-results').prepend(pins);
  const queries=element('div','ws-history-shortcuts', '<button data-nav="roofline">Open Roofline investigation ↗</button><button data-nav="optimizer">Browse optimization history ↗</button>');
  $('#view-results').prepend(queries);
  $('#view-personas > .panel > p.msg').textContent='Personas describe requests, response lengths, read time and think time. Cohorts combine them into a traffic mix. Use Save to apply edits to future runs.';
}

function bindNavigation() {
  document.addEventListener('click', event=>{
    const b=event.target.closest('button'); if(!b) return;
    if(b.dataset.nav) { if(b.dataset.nav==='roofline' && b.closest('.ws-nav')) goTo(state.method==='optimizer'?'optimizer':['capacity','saturation'].includes(state.method)?'control':'roofline'); else goTo(b.dataset.nav); }
    if(b.dataset.method) selectMethod(b.dataset.method);
    if(b.dataset.stage) {state.stage=b.dataset.stage; renderChrome();}
    if(b.dataset.asset) {state.asset=b.dataset.asset; renderChrome();}
    if(b.dataset.lens) {state.lens=b.dataset.lens; renderRoofline(); renderChrome();}
    if(b.dataset.inspect) {state.inspector=b.dataset.inspect; inspectorSignature=''; renderInspector();}
  });
  for(const [id,key] of [['ws-tier','tier'],['ws-metric','metric']]) $('#'+id).addEventListener('change',e=>{state[key]=e.target.value;matrixSignature='';renderRoofline();});
  $('#ws-scale').addEventListener('change',e=>{state.relative=e.target.value==='relative';matrixSignature='';renderRoofline();});
  $('#rf-heatmap').addEventListener('click',e=>{
    const b=e.target.closest('[data-pair]'); if(!b)return;
    state.selection=JSON.parse(b.dataset.pair);matrixSignature='';renderRoofline();
  });
  $('#ws-history-search').addEventListener('input',filterHistory);
  $('#ws-history-method').addEventListener('change',filterHistory);
  $('#ws-history-prev').onclick=()=>{state.historyPage--;filterHistory(false);};
  $('#ws-history-next').onclick=()=>{state.historyPage++;filterHistory(false);};
  $('#run-list').addEventListener('change',()=>{
    const modes=new Set(Results.flat.filter(e=>Results.checked.has(e.cohort_run_id)).map(e=>e.mode));
    if(modes.size>1){ $('#compare-btn').disabled=true; $('.ws-comparison-note').textContent='These runs use different measurement methods. Open their reports separately or select runs using the same method.'; }
    else $('.ws-comparison-note').textContent='Check model, precision, engine version, hardware, workload and measurement conditions before treating this as a like-for-like comparison.';
  });
}
function selectMethod(method) {
  state.method=method;
  if(['capacity','saturation'].includes(method)) {
    const sel=$('#workload-select');
    const options=[...sel.options].filter(o=>methodFromWorkload(o.value)===method);
    if(methodFromWorkload(sel.value)!==method && options.length) {sel.value=options[0].value;sel.dispatchEvent(new Event('change'));}
    goTo('control');
  } else goTo(method);
}
function showPage(view) {
  state.view=view;
  if(view==='roofline'||view==='optimizer') state.method=view;
  if(view==='control' && $('#workload-select').value) state.method=methodFromWorkload($('#workload-select').value);
  $('#ws-classic').href=`index.html#${view==='assets'?'prepare':view}`;
  renderChrome();
}
function renderChrome() {
  try{sessionStorage.setItem('capsim.workspace',JSON.stringify({stage:state.stage,lens:state.lens,asset:state.asset,metric:state.metric,tier:state.tier}));}catch{/* optional preferences */}
  const isLab=labViews.includes(state.view);
  let [title,description]=LABELS[state.view]||LABELS.prepare;
  if(state.view==='control'&&state.method==='saturation') {title='Saturation & shape search'; description='Find the sustained rate, held batch and hardware limits of a deployment.';}
  $('#view-control .ws-setup h2').textContent=state.method==='saturation'?'Configure a saturation benchmark':'Configure a capacity benchmark';
  $('#ws-title').textContent=title; $('#ws-description').textContent=description;
  $('#ws-context').textContent=isLab?`LAB / ${state.stage.toUpperCase()}`:`WORKBENCH / ${state.view==='prepare'?'SYSTEM':state.view==='personas'?'WORKLOADS':state.view.toUpperCase()}`;
  $('#ws-lab-toolbar').hidden=!isLab;
  for(const b of $$('.ws-nav [data-nav]')) {const active=b.dataset.nav===(isLab?'roofline':state.view); b.classList.toggle('active',active); if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');}
  for(const b of $$('[data-method]')) b.setAttribute('aria-pressed',String(b.dataset.method===state.method));
  for(const b of $$('[data-stage]')) b.setAttribute('aria-pressed',String(b.dataset.stage===state.stage));
  for(const view of labViews) { $(`#view-${view} .ws-setup`).hidden=state.stage!=='setup';$(`#view-${view} .ws-workspace`).hidden=state.stage!=='workspace'; }
  for(const b of $$('.ws-asset-tabs button')) b.setAttribute('aria-pressed',String(b.dataset.asset===state.asset));
  for(const p of $$('#view-assets > section')) p.hidden=p.dataset.asset!==state.asset;
  $('#ws-metric').value=state.metric;$('#ws-tier').value=state.tier;$('#ws-scale').value=state.relative?'relative':'absolute';
  $('#ws-metric').closest('label').hidden=state.lens!=='matrix';$('#ws-scale').closest('label').hidden=state.lens!=='matrix';
  for(const b of $$('[data-lens]')) b.setAttribute('aria-pressed',String(b.dataset.lens===state.lens));
  if(Roofline) applyLens();
  if(Control) renderStatus({active:Control.lastActive,running:Control.running});
  resizeCharts();
}
function applyLens() {
  for(const [id,lens] of [['rf-matrix-panel','matrix'],['rf-spectrum-panel','spectrum'],['rf-charts','hardware'],['ws-recorded-hardware','hardware']]) $('#'+id).classList.toggle('ws-lens-hidden',state.lens!==lens);
  const ready=!!Roofline.doc?.plan?.models?.length;
  $('.ws-landscape-toolbar').hidden=!ready;$('#ws-analysis').hidden=!ready;
  $('[data-empty="roofline"]').hidden=ready;
  $('[data-empty="optimizer"]').hidden=!$('#opt-search-panel').hidden||!$('#opt-results-panel').hidden;
  // Live owns the idle/measurement state for capacity and saturation.
  $('[data-empty="control"]').hidden=true;
}
function renderStatus({active,running}) {
  const offline=Control._offline;
  $('#ws-connection').textContent=offline?'Disconnected · data may be stale':'Connected';
  $('#ws-connection').className=offline?'status-fail':'status-pass';
  $('#ws-job').hidden=!active;
  if(active) $('#ws-job').textContent=`${running?'Active experiment':active.error?'Experiment stopped':'Most recent experiment'} · ${active.workload?.kind||'run'}${active.engine_summary?' · '+active.engine_summary:''}${active.error?' · '+active.error:''}${offline?' · connection lost; last known state':''}`;
  const stop=$(state.view==='roofline'?'#rf-stop':state.view==='optimizer'?'#opt-stop':'#stop-btn');
  $('#ws-stop').hidden=!labViews.includes(state.view)||state.stage!=='workspace'||stop.disabled;
  $('#ws-method-note').textContent=state.stage==='setup'?'One setup page · expert controls available':offline?'Connection lost · recorded state retained':'Live state and recorded results';
  if(Roofline) applyLens();
}
function installAdapters() {
  const wrap=(obj,method,after)=>{const original=obj[method];obj[method]=function(...args){const result=original.apply(this,args);if(result?.then)return result.then(value=>{after(...args);return value;});after(...args);return result;};};
  wrap(Control,'renderDoctor',(report,at)=>{state.doctor=report;state.doctorAt=at||Date.now();renderSystem();});
  wrap(Control,'loadCatalogs',()=>{renderSystem();renderProfiles();});
  wrap(Engines,'refresh',()=>{for(const list of $$('#engines-list ul.hint'))if(list.parentElement.tagName!=='DETAILS')fold(list,'Compatibility & measurement notes');renderSystem();});
  wrap(Storage,'refresh',renderSystem);
  wrap(Models,'refresh',renderSystem);
  wrap(Roofline,'render',renderRoofline);
  wrap(Roofline,'renderPlan',renderScope);
  // Only this entry point replaces matrix rendering. The summary service is
  // the single selection authority for confirmations and outcome thresholds.
  Roofline.renderMatrix=renderMatrix;
  wrap(Optimizer,'refresh',()=>{applyLens();renderSystem();});
  wrap(Results,'renderList',groupHistory);
  wrap(Results,'openHeadline',()=>{$('#result-export-dl').disabled=!Results.openId;});
  Results.download=async function(){const entry=this.flat.find(e=>e.cohort_run_id===this.openId);if(!entry)return;try{const slim=$('#ws-export-kind').value==='slim';const data=await api('/api/runs/'+encodeURIComponent(entry.run)+'/export'+(slim?'?slim=true':''));download('capsim-'+entry.run+(slim?'-slim':'')+'.json',data);}catch(e){this.msg(e.message,'error');}};
  Roofline.start=async function(){
    let body;
    try{body=rooflineRequest({models:this.plannedModels(),engines:[...this.engines],shapes:this.shapes,mode:$('#rf-model-mode').value,limits:this.limits(),inputTokens:+$('#rf-input-tokens').value,confirm:$('#rf-confirm').checked,crossDomain:$('#rf-cross-domain').checked},{recovery:$('#ws-recovery').value,engine:$('#ws-recovery-engine').value,priority:$('#ws-priority').value.split(',').map(s=>s.trim()).filter(Boolean),ktScale:$('#ws-kt-scale').checked});}catch(e){this.msg(e.message,'error');return;}
    body.workload.spec.benchmarks=Benchmarks.spec();
    if(['fresh','redo'].includes($('#ws-recovery').value)&&!window.confirm($('#ws-recovery-note').textContent+' Continue?'))return;
    $('#rf-start').disabled=true;
    try{await api('/api/runs',{method:'POST',body:JSON.stringify(body)});this.msg('Search started. Recorded cells and current progress are available in Workspace.','ok');await this.refresh();}catch(e){this.msg(e.message,'error');}finally{$('#rf-start').disabled=!!this.doc?.live;}
  };
  for(const controller of [Roofline,Optimizer]) {const start=controller.start;controller.start=async function(...args){await start.apply(this,args);await Control.pollStatus();if(Control.running){state.stage='workspace';renderChrome();}};}
  const originalStart=Control.start;
  Control.start=async function(...args){await originalStart.apply(this,args);if(this.running){state.stage='workspace';renderChrome();}};
}

function renderSystem() {
  if(!Control)return;
  const d=state.doctor, hw=Control.hw||{};
  const arena=Optimizer.arena?.hardware||{};
  const source=arena.source||hw.source||'unavailable';
  $('#ws-hardware-source').textContent=source;
  const detected=hw.detected_gpus??0;
  const groups=arena.device_groups||[];
  const gpus=d?.gpus||[];
  const groupHTML=groups.length?groups.map((group,i)=>`<div class="ws-device-group"><span>Device group ${i+1}</span><div>${group.map(id=>`<div class="ws-gpu"><span>GPU ${esc(id)}</span><b>${esc(gpus[id]?.name||'GPU')}</b><small>${esc(gpus[id]?.memory||'Memory not probed')}</small></div>`).join('')}</div></div>`).join(''):`<div class="ws-device-group"><span>${detected?'Detected accelerators':'CPU execution'}</span><div>${detected?Array.from({length:detected},(_,i)=>`<div class="ws-gpu"><span>GPU ${i}</span><b>${esc(gpus[i]?.name||'GPU')}</b><small>${esc(gpus[i]?.memory||'Memory not probed')}</small></div>`).join(''):'<div class="ws-cpu"><b>CPU</b><span>No GPU detected</span></div>'}</div></div>`;
  $('#ws-hardware').innerHTML=`<div class="ws-topology">${groupHTML}</div><div class="ws-hw-facts"><div><span>Processor</span><b>${esc(d?.hardware?.cpu_model||'Run host checks to identify')}</b></div><div><span>Detected GPUs</span><b>${num(detected)}${arena.count!==undefined&&arena.count!==detected?` · ${num(arena.count)} configured`:''}</b></div><div><span>Host memory · planning inventory</span><b>${arena.host_ram_gb!=null?num(arena.host_ram_gb)+' GB':'Unavailable'}</b></div></div><p class="msg">${groups.length?'Device grouping follows the arena configuration; it is not a measured interconnect map.':'Host inventory is separate from telemetry recorded during an experiment.'}</p>`;
  const checks=d?.checks||[], failures=checks.filter(c=>c.status==='fail'),warns=checks.filter(c=>c.status==='warn');
  $('#ws-readiness').innerHTML=`<span class="ws-orb ${failures.length?'warn':''}"></span><div><strong>${!d?'Host checks pending':failures.length?`${failures.length} host check${failures.length===1?'':'s'} need attention`:'Host checks complete'}</strong><p>${d?`${checks.filter(c=>c.status==='ok').length} passed · ${warns.length} warnings · checked ${new Date(state.doctorAt).toLocaleTimeString()}`:'Readiness depends on the chosen engine and workload.'} · Smoke testing remains a separate execution check.</p></div><button class="small" id="ws-recheck">Recheck</button>`;
  $('#ws-recheck').onclick=()=>$('#doctor-btn').click();
  const staged=(Engines.doc?.runtimes||[]).filter(r=>r.staged).length;
  $('#ws-setup-readiness').innerHTML=`<div class="ws-kv"><span>Host checks</span><b>${!d?'Not checked':failures.length?`${failures.length} need attention`:'Complete'}</b></div><div class="ws-kv"><span>Staged engines</span><b>${staged}</b></div><p class="msg">Review cached artifacts in Assets before starting. Downloads and engine pulls use the connected host.</p>`;
}
function renderProfiles() {
  if(!Control)return;
  const profiles=Object.entries(Control.catalogs.profiles||{});
  $('#ws-profiles').innerHTML=`<h2>Reusable launch profiles</h2><p class="msg">Profiles retain their existing configuration. Choose the corresponding model in Lab to apply a matching optimized launch.</p><div class="ws-profile-grid">${profiles.length?profiles.map(([name,p])=>`<details class="ws-profile"><summary><b>${esc(name)}</b><span>${esc(p.description||p.engine?.model_id||p.model_id||'Launch configuration')}</span></summary><pre>${esc(JSON.stringify(p,null,2))}</pre></details>`).join(''):'<p class="msg">No profiles returned by the service.</p>'}</div>`;
}
function renderScope() {
  if(!Roofline)return;
  const models=Roofline.plannedCandidates(),engines=[...Roofline.engines];
  const recovery=$('#ws-recovery').value,engineSelect=$('#ws-recovery-engine'),previous=engineSelect.value;
  engineSelect.innerHTML='<option value="">All selected engines</option>'+engines.map(e=>`<option value="${esc(e)}">${esc(Engines.label(e))}</option>`).join('');
  if(engines.includes(previous))engineSelect.value=previous;
  $('#ws-recovery-engine-wrap').hidden=!['retry','redo'].includes(recovery);
  const scope=engineSelect.value?[engineSelect.value]:engines,records=(Roofline.doc?.results||[]).filter(r=>scope.includes(r.engine));
  $('#ws-recovery-note').textContent=({resume:'Reuse recorded cells whose full conditions match. The service determines exact reuse when the plan is built.',retry:`Retry failed cells for ${scope.length} selected engine(s); ${records.filter(r=>r.error).length} failed attempt records are in scope. Successful results are retained.`,redo:`Remeasure ${scope.length} selected engine(s). Their ${records.length} existing attempt records will be removed from the active investigation and those cells will run again.`,fresh:'Replace the active investigation state and measure the selected scope afresh. Export any evidence you want to retain first.'})[recovery]+($('#ws-kt-scale').checked?' KTransformers scaling adds eligible TP, GPU-expert and stream combinations; the basic cell estimate below excludes those supplemental cells.':'');
  $('#ws-scope').innerHTML=`<div><b>${models.length}</b><span>models</span></div><div><b>${engines.length}</b><span>engines</span></div><div><b>${Roofline.cellCount(models,engines)}</b><span>planned cells</span></div>`;
}
function renderRoofline() {
  if(!Roofline)return;
  const d=Roofline.doc||{},s=d.summary||{},f=s.fastest||s.best,large=s.largest_served,attempted=s.largest_attempted;
  if(d.started_at&&optionsSyncedFor!==d.started_at){optionsSyncedFor=d.started_at;$('#ws-kt-scale').checked=!!d.plan?.cells?.some(c=>c.kt_scale);renderScope();}
  $('#ws-findings').innerHTML=[['Fastest selected',f?num(f.out_tok_s)+' tok/s':'—',f?`${short(f.model)} · ${f.confirmed?'Confirmed':'Exploratory'}`:'No settled result'],['Largest served',large?.approx_size_gb!=null?num(large.approx_size_gb)+' GB':'Size unavailable',large?short(large.model):'No successful measurement'],['Largest attempted',attempted?.approx_size_gb!=null?num(attempted.approx_size_gb)+' GB':'Size unavailable',attempted?`${short(attempted.model)} · ${attempted.status||'attempted'}`:'Boundary not tested']].map(([title,value,sub])=>`<div><span>${title}</span><strong>${esc(value)}</strong><small>${esc(sub)}</small></div>`).join('');
  renderMatrix();renderSpectrum();renderInspector();applyLens();
}
function renderSpectrum() {
  const d=Roofline.doc||{},s=d.summary||{};
  const rows=(s.spectrum||[]).filter(r=>state.tier==='all'||r.tier===state.tier);
  Roofline.renderSpectrum(d,{...s,spectrum:rows});
  const c=Roofline.charts.spectrum;
  if(c){c.options.scales.x.ticks.callback=v=>num(v);c.options.scales.x.ticks.maxTicksLimit=6;}
  if(c)c.options.onClick=(_,elements)=>{if(!elements.length)return;const hit=elements[0],ds=c.data.datasets[hit.datasetIndex];const record=rows.filter(r=>r.out_tok_s&&r.approx_size_gb&&(r.tier||'fast')===ds.tier)[hit.index];if(record){state.selection=[record.model,record.best_engine];matrixSignature='';renderInspector();}};
  const tableRows=$$('#rf-spectrum tbody tr');
  tableRows.forEach((row,i)=>{row.tabIndex=0;row.setAttribute('role','button');row.setAttribute('aria-label','Inspect '+rows[i].model);const select=()=>{state.selection=[rows[i].model,rows[i].best_engine||(d.plan?.engines||[])[0]];matrixSignature='';renderInspector();};row.onclick=select;row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};});
}
function renderMatrix() {
  if(!Roofline)return;
  const d=Roofline.doc||{},engines=d.plan?.engines||[];
  const models=(d.plan?.models||[]).filter(m=>state.tier==='all'||d.plan?.model_info?.[m]?.tier===state.tier);
  const signature=JSON.stringify([d.summary?.best_per_pair,d.results?.length,d.current,d.live,models,engines,state.metric,state.relative,state.selection]);
  $('#rf-matrix-panel').hidden=!(d.plan?.models?.length&&engines.length);
  if(signature===matrixSignature)return;matrixSignature=signature;
  if(!d.summary?.best_per_pair && d.results?.length){$('#rf-heatmap').textContent='This service needs the workspace update to provide canonical pairing results. Classic remains available.';return;}
  const metric=METRICS[state.metric],scale=metricScale(d,models,engines,state.metric,state.relative);
  if((!state.selection||!(d.plan?.models||[]).includes(state.selection[0])||!engines.includes(state.selection[1]))&&models.length&&engines.length){const model=models.find(m=>d.summary?.best_per_model?.[m])||models[0];state.selection=[model,d.summary?.best_per_model?.[model]?.engine||engines[0]];}
  const focusedPair=$('#rf-heatmap').contains(document.activeElement)?document.activeElement.dataset.pair:null;
  $('#rf-heatmap').innerHTML=models.length?`<div class="ws-table-scroll"><table class="ws-matrix"><thead><tr><th scope="col">Model / ${esc(metric.unit)}</th>${engines.map(e=>`<th scope="col">${esc(Engines.label(e))}</th>`).join('')}</tr></thead><tbody>${models.map(m=>`<tr><th scope="row"><b>${esc(short(m))}</b><small>${esc(({fast:'Fast',large:'Large on GPU',beyond_vram:'Beyond VRAM'})[d.plan?.model_info?.[m]?.tier]||'Selected model')}</small></th>${engines.map(e=>{const r=pairResult(d,m,e),status=pairState(d,m,e),v=r?.[state.metric],selected=state.selection?.[0]===m&&state.selection?.[1]===e;return `<td><button class="ws-cell ${status==='Failed'?'failed':''}" style="--intensity:${scale.fraction(m,v).toFixed(3)}" data-pair="${esc(JSON.stringify([m,e]))}" aria-pressed="${selected}" aria-label="${esc(short(m)+' on '+Engines.label(e)+': '+(v==null?'unavailable':num(v,metric.digits)+' '+metric.unit)+', '+status)}"><b>${v==null?(r?'Unavailable':status==='Not planned'?'—':status):num(v,metric.digits)}</b><small>${status}</small></button></td>`;}).join('')}</tr>`).join('')}</tbody></table></div><div class="ws-scale-note"><span>${state.relative?'Within each model':'Absolute scale'} · ${esc(metric.label)} (${esc(metric.unit)})</span><span class="ws-gradient"></span><span>0 → ${state.relative?'row maximum':num(scale.maximum,metric.digits)}</span></div><p class="msg">Color means magnitude, not quality. Lower latency or power is not automatically better without throughput and workload context.</p>`:'<div class="ws-empty"><p>No planned models in this tier.</p></div>';
  if(focusedPair)$$('#rf-heatmap [data-pair]').find(b=>b.dataset.pair===focusedPair)?.focus({preventScroll:true});
}
const kv=(label,value)=>`<div class="ws-kv"><span>${esc(label)}</span><b>${esc(value??'Unavailable')}</b></div>`;
function renderInspector() {
  if(!Roofline)return;
  const d=Roofline.doc||{},[model,engine]=state.selection||[];
  const r=pairResult(d,model,engine),info=d.plan?.model_info?.[model]||{};
  const attempts=(d.results||[]).filter(a=>a.model===model&&a.engine===engine);
  const signature=JSON.stringify([state.selection,state.inspector,r,attempts,info,state.pins]);
  if(signature===inspectorSignature)return;inspectorSignature=signature;
  $('#ws-inspector-title').textContent=model?short(model):'Select a pairing';
  $('#ws-inspector-head').innerHTML=model?`<span class="ws-badge">${esc(Engines.label(engine))} · ${pairState(d,model,engine)}</span><div class="ws-result-rate">${r?num(r.out_tok_s):'—'} <small>tok/s</small></div><p class="msg">Generation · selected recorded result</p>`:'<p class="msg">Select a heatmap cell to inspect configuration, measurement and evidence.</p>';
  for(const b of $$('[data-inspect]'))b.setAttribute('aria-pressed',String(b.dataset.inspect===state.inspector));
  let content='';
  if(!model)content='<p class="msg">No selection yet.</p>';
  else if(state.inspector==='configuration')content=kv('Model',model)+kv('Engine',Engines.label(engine))+kv('Tier',info.tier)+kv('Weight size',info.approx_size_gb!=null?num(info.approx_size_gb)+' GB':null)+kv('Total parameters',info.params_b!=null?num(info.params_b,1)+'B':null)+kv('Prompt / output',r?`${d.input_tokens??'—'} / ${r.output_tokens??'—'} tokens`:null)+kv('Tensor parallel / replicas',r?`${r.tp??'—'} / ${r.replicas??'—'}`:null)+kv('Max sequences',r?.max_num_seqs);
  else if(state.inspector==='measurement')content=kv('TTFT p95',r?.ttft_p95_ms!=null?num(r.ttft_p95_ms)+' ms':null)+kv('TPOT p95',r?.tpot_p95_ms!=null?num(r.tpot_p95_ms,1)+' ms':null)+kv('Offered concurrency',r?.concurrency)+kv('Held / in-flight',r?.in_flight)+kv('Queue depth',r?.queue_depth)+kv('Completed requests',r?.samples)+kv('Failed requests',r?.errors)+kv('Completion success',r?.success_rate!=null?num(r.success_rate*100,1)+'%':null)+kv('Total rate incl. prompt',r?.total_tok_s!=null?num(r.total_tok_s)+' tok/s':null);
  else if(state.inspector==='hardware')content=hardwareEvidence(r);
  else content=`<div class="ws-evidence-callout"><b>${r?.confirmed?'Full-resolution confirmation':'Search evidence'}</b><p>${r?.confirmed?'This selected row is marked confirmed by the search.':'No confirmation is recorded for this selected result.'}</p></div>`+kv('Steady state',r?r.steady_state===false?'Not settled':'Settled':'Unavailable')+kv('Scrape gaps',r?.scrape_gaps)+kv('Reasoning-only completions',r?.no_content)+kv('Source run',r?.run_dir)+`<h3>${attempts.length} recorded attempts</h3><div class="ws-attempt-list">${attempts.map(a=>`<details><summary>${a.error?'Failed':a.confirmed?'Confirmation':'Search'} · ${num(a.out_tok_s)} tok/s</summary><pre>${esc(JSON.stringify(a,null,2))}</pre></details>`).join('')||'<p class="msg">This pairing has not produced a result.</p>'}</div>`;
  $('#ws-inspector-body').innerHTML=content;
  $('#ws-recorded-hardware').innerHTML=`<h2>Hardware at the selected result</h2><p class="msg">${esc(model?short(model)+' · '+Engines.label(engine):'Select a pairing')} · recorded values, not live host readings.</p>${hardwareEvidence(r)}`;
  $('#ws-inspector-actions').innerHTML=r?'<button id="ws-open-evidence" class="primary small">Inspect recorded run</button><button id="ws-third-party" class="small">Run Third Party Benchmarks</button><button id="ws-pin" class="small">Pin comparison</button><button id="ws-export-pair" class="small">Export pairing evidence</button>':'';
  if(r){$('#ws-third-party').onclick=()=>Benchmarks.select(r);$('#ws-open-evidence').onclick=()=>openEvidence(model,engine);$('#ws-pin').onclick=()=>{const key=model+'|'+engine;if(!state.pins.some(p=>p.key===key))state.pins.push({key,model,engine,record:structuredClone(r),input_tokens:d.input_tokens,started_at:d.started_at});renderPins();$('#ws-pin').textContent='Pinned in History';};$('#ws-export-pair').onclick=()=>download('capsim-pairing-evidence.json',{kind:'roofline_pairing',model,engine,input_tokens:d.input_tokens,search_started_at:d.started_at,selection_policy:'service best_per_pair: completion success, confirmation, generation rate',selected:r,model_info:info,attempts});}
}
function hardwareEvidence(r) {
  return kv('KV cache occupancy',r?.kv_cache_pct!=null?num(r.kv_cache_pct,1)+'%':null)+kv('KV capacity',r?.kv_cache_tokens!=null?num(r.kv_cache_tokens)+' tokens':null)+kv('GPU package power',r?.gpu_power_w!=null?num(r.gpu_power_w)+' W':null)+kv('GPU-scope efficiency',r?.tokens_per_watt!=null?num(r.tokens_per_watt,2)+' tokens/J':null)+'<div class="ws-mini-note">Open Inspect recorded run for the selected concurrency ladder, hardware window and per-device readings. This summary carries only the metrics shown above. Missing readings are unavailable, never zero.</div>';
}
function groupHistory() {
  const box=$('#run-list'), rows=[...box.children];
  if(!Results.flat.length)return;
  const groups=new Map();
  rows.forEach((row,i)=>{const entry=Results.flat[i];if(!entry)return;row.dataset.search=[entry.run,entry.model_id,entry.cohort_name,entry.engine_type].join(' ').toLowerCase();row.dataset.mode=entry.mode||'pool';if(!groups.has(entry.run)){const group=element('details','ws-run-group');group.open=groups.size===0;group.append(element('summary','',esc(entry.run)));groups.set(entry.run,group);box.append(group);}groups.get(entry.run).append(row);if(entry.cohort_run_id===Results.openId)groups.get(entry.run).open=true;});
  filterHistory(false);
}
function filterHistory(reset=true) {
  if(reset!==false)state.historyPage=0;
  const q=$('#ws-history-search').value.toLowerCase().trim(),mode=$('#ws-history-method').value;let count=0;
  for(const row of $$('#run-list .run-row')) {row.hidden=!!((q&&!row.dataset.search.includes(q))||(mode!=='all'&&(mode==='pool'?['open_loop','headline_sweep'].includes(row.dataset.mode):row.dataset.mode!==mode)));if(!row.hidden)count++;}
  for(const group of $$('.ws-run-group')){group.hidden=![...group.querySelectorAll('.run-row')].some(r=>!r.hidden);if(q||mode!=='all')group.open=true;}
  const groups=$$('.ws-run-group').filter(g=>!g.hidden),pages=Math.max(1,Math.ceil(groups.length/12));
  state.historyPage=Math.max(0,Math.min(pages-1,state.historyPage));
  groups.forEach((g,i)=>{g.hidden=i<state.historyPage*12||i>=(state.historyPage+1)*12;});
  $('#ws-history-prev').disabled=state.historyPage===0;$('#ws-history-next').disabled=state.historyPage>=pages-1;
  $('#ws-history-page').textContent=`Page ${state.historyPage+1} of ${pages}`;
  $('#ws-history-count').textContent=`${count} measurements in ${groups.length} run directories`;
}
function renderPins() {
  const box=$('#ws-pins');box.hidden=!state.pins.length;
  box.innerHTML='<h2>Pinned Roofline evidence</h2><p class="msg">Snapshots pinned in this page session. Check conditions before comparing rates; confirmations and exploration remain labeled.</p><div class="ws-pin-grid">'+state.pins.map((p,i)=>`<article><h3>${esc(short(p.model))}</h3><span class="ws-badge">${esc(Engines.label(p.engine))} · ${p.record.confirmed?'Confirmed':'Exploratory'}</span><strong>${num(p.record.out_tok_s)} tok/s</strong>${kv('Prompt / output',`${p.input_tokens??'—'} / ${p.record.output_tokens??'—'}`)}${kv('Completion success',p.record.success_rate!=null?num(p.record.success_rate*100,1)+'%':null)}<button data-unpin="${i}" class="small">Remove pin</button></article>`).join('')+'</div>';
  box.querySelectorAll('[data-unpin]').forEach(b=>b.onclick=()=>{state.pins.splice(+b.dataset.unpin,1);renderPins();});
}

boot().catch(error=>{
  $('#ws-loading').innerHTML=`<strong>Workspace could not open</strong><p>${esc(error.message)}</p><a href="index.html">Open Classic UI</a>`;
  console.error(error);
});
