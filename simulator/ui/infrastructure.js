import {api, $} from './lib/api.js';
import {onShow, currentView} from './lib/tabs.js';
import {escapeHTML as esc} from './lib/workspace-data.js';

let projects=[], campaigns=[], selected='P04', requestKey=null, timer=null;
function message(text) { $('#infra-message').textContent=text; }
function render() {
  $('#infra-cards').innerHTML=projects.map(p=>`<button type="button" class="infra-card ${p.id===selected?'selected':''}" data-project="${esc(p.id)}" aria-pressed="${p.id===selected}"><span>${esc(p.id)}</span><h3>${esc(p.title)}</h3><p>${esc(p.feasibility_gate)}</p><strong>${p.implementation_status==='pilot_available'?'Pilot available':p.status==='deprioritized'?'Deprioritized':'Implementation pending'}</strong><small>${p.outputs.length} separate outputs</small></button>`).join('');
  const p=projects.find(p=>p.id===selected);
  if (!p) return;
  $('#infra-project-title').textContent=`${p.id} · ${p.title}`;
  $('#infra-method').textContent=p.baseline_and_treatment;
  $('#infra-controls').textContent=p.controls;
  $('#infra-acceptance').textContent=p.acceptance;
  $('#infra-outputs').innerHTML=p.outputs.map(o=>`<li><b>${esc(o.id)}</b> ${esc(o.title)}</li>`).join('');
  const cache=selected==='P03', diffusion=selected==='P05';
  $('#infra-diffusion-settings').hidden=!diffusion;
  $('#infra-tenant-settings').hidden=selected!=='P06';
  $('#infra-attention-settings').hidden=selected!=='P10';
  for(const id of ['infra-source','infra-duration','infra-warmup','infra-slo','infra-timeout','infra-quality','infra-slo-fraction','infra-min-requests','infra-context']) $('#'+id).closest('label').hidden=diffusion;
  $('#infra-source').required=!diffusion;
  for(const id of ['infra-records','infra-concurrency']) {
    $("#"+id).closest('label').hidden=cache||diffusion;
  }
  for(const id of ['infra-spec-method','infra-draft-tokens']) $("#"+id).closest('label').hidden=selected!=='P04';
  $('#infra-session-settings').hidden=selected!=='P02';
  $('#infra-prefill-label').hidden=selected!=='P01';
  $('#infra-search-settings').hidden=selected!=='P01';
  $('#infra-dynamo-note').hidden=selected!=='P01';
  $('#infra-transfer-settings').hidden=selected!=='P01';
  $('#infra-cache-records-label').hidden=!cache;
  $('#infra-cache-concurrency-label').hidden=!cache;
  $('#infra-cache-note').hidden=!cache;
  $('#infra-cache-samples-label').hidden=!cache;
  $('#infra-spec-note').hidden=selected!=='P04';
  $('#infra-spec-loads-label').hidden=selected!=='P04';
  $('#infra-mode').querySelector('[value="confirmation"]').disabled=cache||diffusion||['P02','P06'].includes(selected);
  if(cache||diffusion||['P02','P06'].includes(selected)) $('#infra-mode').value='pilot';
  $('#infra-form').hidden=p.implementation_status!=='pilot_available';
  $('#infra-pending').hidden=p.implementation_status==='pilot_available';
  $('#infra-history').innerHTML=campaigns.filter(c=>c.project_id===selected).map(c=>`<button type="button" class="infra-run" data-run="${esc(c.id)}"><b>${esc(c.model)}</b> · ${esc(c.phase)} · ${esc(c.outcome||'Not measured')}<small>${esc(new Date(c.created_at*1000).toLocaleString())}</small></button>`).join('')||'<p>No experiments recorded for this project.</p>';
  workflowControls();
}
function workflowControls(){
  $('#infra-token-settings').hidden=selected!=='P01';
  const goal=selected==='P01'&&$('#infra-goal-seek').checked;
  $('#infra-goal-settings').hidden=!goal;
  $('#infra-prefill-label').hidden=selected!=='P01'||goal;
  $('#infra-dynamo-note').textContent=goal?'Equal total GPU budget. Aggregate uses every GPU; split serving dedicates the first selected GPUs to prefill and the rest to decode. Capacity is searched within the chosen rate range.':'Same total GPU budget in both arms. Baseline uses one aggregate worker per GPU. Treatment separates prefill and decode with measured KV transfer.';
  const capacity=selected==='P02'&&$('#infra-session-mode').value==='capacity';
  $('#infra-session-capacity').hidden=!capacity;
  for(const id of ['tier','count','churn','gpu','disk','intervals']) $('#infra-session-'+id).closest('label').hidden=capacity;
  $('#infra-session-settings').lastElementChild.hidden=capacity;
  const calls=selected==='P06'&&$('#infra-tenant-workflow').value==='post_call';
  for(const id of ['infra-call-rates','infra-call-wer']) $('#'+id).closest('label').hidden=!calls;
  $('#infra-call-note').hidden=!calls;
  for(const id of ['infra-tenant-rate','infra-tenant-noise']) $('#'+id).closest('label').hidden=calls;
  $('#infra-records').closest('label').hidden=goal||calls||['P03','P05'].includes(selected);
  $('#infra-warmup').closest('label').hidden=calls||selected==='P05'||capacity;
}
async function refresh() {
  try {
    const data=await api('/api/infrastructure'); projects=data.projects;campaigns=data.campaigns;
    if(data.readiness) $('#infra-readiness').textContent=JSON.stringify(data.readiness.summary,null,2);
    render();
    const status=await api('/api/status');
    $('#infra-stop').disabled=!status.active_run?.running || status.active_run?.workload?.kind!=='infrastructure';
  } catch(e) {message(e.message);}
}
async function loadSources() {
  try {
    const data=await api('/api/roofline/benchmarks/results');
    const choices=data.results.filter(x=>!x.result.error&&x.result.out_tok_s&&String(x.result.engine).includes('vllm'));
    $('#infra-source').innerHTML='<option value="">Choose a recorded roofline configuration</option>'+choices.map(x=>`<option value="${esc(x.id)}">${esc(x.result.model)} · ${esc(x.result.engine)} · ${esc(x.result.out_tok_s)} tok/s</option>`).join('');
  } catch(e) {message(e.message);}
}
async function detail(id) {
  try {
    const data=await api(`/api/infrastructure/campaigns/${id}`);
    $('#infra-image-review').innerHTML='';
    $('#infra-recover').hidden=['complete','queued'].includes(data.state.phase);
    $('#infra-recover').dataset.run=id;
    $('#infra-result').textContent=JSON.stringify({state:data.state,comparison:data.comparison},null,2);
    tenantSummary(data);
    sessionCapacitySummary(data);
    if(data.state.outcome==='invalidated') {
      $('#infra-downloads').innerHTML=`<p>This comparison was invalidated. See the review above. Claim reports are disabled.</p><a href="/api/infrastructure/campaigns/${id}/export/zip">Original evidence ZIP (audit only)</a>`;
      return;
    }
    $('#infra-downloads').innerHTML=data.outputs.map(o=>`<section><b>${esc(o.id)} · ${esc(o.title)}</b><p>${esc(o.status)}: ${esc(o.reason)}</p>${['docx','md','json'].map(kind=>`<a href="/api/infrastructure/campaigns/${id}/export/${kind}?output_id=${encodeURIComponent(o.id)}">${kind.toUpperCase()}</a>`).join(' · ')}</section>`).join('')+(data.outputs.length?`<p><a href="/api/infrastructure/campaigns/${id}/export/zip">Download shared evidence ZIP</a></p>`:'');
    if(data.manifest.spec.project_id==='P05' && data.state.phase==='complete') {
      const button=document.createElement('button');button.textContent='Review generated images';button.onclick=()=>reviewImages(id);$('#infra-image-review').append(button);
    }
    if(data.manifest.spec.project_id==='P04' && data.state.phase==='complete') $('#infra-downloads').innerHTML+=`<p><a href="/api/infrastructure/campaigns/${id}/export/yaml">Download treatment configuration for a separate matched roofline test</a></p>`;
  } catch(e) {message(e.message);}
}
function sessionCapacitySummary(data) {
  let panel=$('#infra-session-results');
  if(!panel){panel=document.createElement('section');panel.id='infra-session-results';$('#infra-result').before(panel);}
  panel.hidden=data.manifest.spec.session_mode!=='capacity';if(panel.hidden)return;
  const rows=data.state.results||[];
  const num=x=>typeof x==='number'?x.toFixed(2):'—';
  panel.innerHTML='<h3>Document-analysis capacity</h3><p>Arrival-rate sweep through saturation for each document working set. Mean in-flight requests exclude the drain. Stable points and SLO thresholds are reported separately. Offered documents are not a verified physical cache-residency count.</p><div style="overflow:auto"><table><thead><tr>'+['Block','Arm','Documents','Offered /s','Mean in-flight','Correct /s','Output tokens/s','TTFT median s','TTFT p95 s','Stable','CPU reload'].map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+[r.block+1,r.arm,r.population,num(r.rate),num(r.metrics.mean_outstanding_requests),num(r.metrics.sustained_correct_requests_s),num(r.metrics.sustained_output_tokens_s),num(r.metrics.ttft_p50_s),num(r.metrics.ttft_p95_s),r.metrics.stable_capacity?'Yes':'No',r.arm==='baseline'?'—':r.metrics.cpu_reload_observed?'Yes':'No'].map(x=>`<td>${esc(String(x))}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function tenantSummary(data) {
  let panel=$('#infra-tenant-results');
  if(!panel){panel=document.createElement('section');panel.id='infra-tenant-results';$('#infra-result').before(panel);}
  panel.hidden=data.manifest.spec.project_id!=='P06'&&!data.manifest.spec.p01_prefill_grid?.length;
  if(panel.hidden){panel.replaceChildren();return;}
  const rows=data.state.results||[];
  const latency=value=>Number.isFinite(value)?value.toFixed(3):'Unavailable';
  if(data.manifest.spec.p01_token_search){
    const c=data.comparison||{},peak=c.peak_uplift;
    const heading=peak?`${peak.uplift_percent.toFixed(1)}% best observed throughput change · ${peak.shape} · ${peak.prefill} prefill GPUs`:'Matched throughput comparison pending';
    panel.innerHTML='<h3>Token throughput and TTFT search</h3><p>'+esc(heading)+'</p><p>Same token shape and GPU budget. Latency is reported alongside throughput. Historical roofline numbers require compatible settings before reuse.</p><div class="infra-table-scroll"><table><thead><tr>'+['Shape','Split','Streams','Output tokens / s','TTFT p50 ms','TTFT p95 ms','Stable','Errors'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.flatMap(r=>(r.rungs||[]).map(m=>'<tr>'+[r.shape,r.prefill?`${r.prefill}P${data.manifest.spec.gpu_ids.length-r.prefill}D`:'Aggregate',m.concurrency,latency(m.out_tok_s),latency(m.ttft_p50_ms),latency(m.ttft_p95_ms),m.steady_state?'Yes':'No',m.errors].map(x=>'<td>'+esc(String(x))+'</td>').join('')+'</tr>')).join('')+'</tbody></table></div>';
    return;
  }
  if(data.manifest.spec.p01_goal_seek){
    const c=data.comparison||{};
    const peak=c.peak_capacity_uplift;
    const absolute=c.peak_absolute_capacity, latencyPeak=c.peak_latency_reduction;
    const otherPeaks=[absolute?`Split-serving peak: ${absolute.passing_rate} requests/s · ${absolute.shape} · ${absolute.prefill} prefill GPUs · ${(absolute.placement||'pooled').replaceAll('_',' ')}`:'Split-serving peak not established yet',latencyPeak?`Latency peak: ${latencyPeak.reduction_percent.toFixed(1)}% lower p95 completion at ${latencyPeak.rate} requests/s · ${latencyPeak.shape}`:'Matched latency peak not established yet'];
    const headline=peak?`${peak.conservative_uplift_percent.toFixed(1)}% measured capacity advantage beyond the aggregate failing boundary · ${peak.shape} · ${peak.prefill} prefill GPUs · ${(peak.placement||'pooled').replaceAll('_',' ')}`:'Capacity peak not established yet';
    const failures=(data.state.attempts||[]).map(a=>`<li>${esc(a.allocation?.arm||'Serving')} · ${esc(String(a.allocation?.prefill??'?'))} prefill · attempt ${Number(a.attempt)+1}: ${esc(a.error)}. ${a.will_retry?'Retried with the same settings.':'Retry exhausted; remaining configurations continue.'}</li>`).join('');
    panel.innerHTML='<h3>Document shape and GPU allocation search</h3><p>'+esc(headline)+'</p>'+(failures?'<h4>Recorded serving failures</h4><ul>'+failures+'</ul>':'')+otherPeaks.map(x=>'<p>'+esc(x)+'</p>').join('')+'<p>Measured peaks within the selected workload and rate range. d = document records; a = extracted answer records. All failures remain visible.</p><div class="infra-table-scroll"><table><thead><tr>'+['Shape','Split','Rate / s','Qualified','Completed / s','Correct / s','Output tokens / s','TTFT p50 s','TTFT p95 s','Client queue p95 s','Input tokens','Output tokens','p95 total s','Gate'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>{const m=r.metrics;return '<tr>'+[r.shape,r.prefill===0?'Aggregate':r.prefill+'P'+(data.manifest.spec.gpu_ids.length-r.prefill)+'D · '+(r.placement||'pooled').replaceAll('_',' '),r.rate,`${m.qualified}/${m.attempted}`,latency(m.completed_requests_per_s),latency(m.correct_requests_per_s),latency(m.output_tokens_per_s),latency(m.ttft_p50_s),latency(m.ttft_p95_s),latency(m.queue_p95_s),`${m.prompt_tokens_min??'?'}–${m.prompt_tokens_max??'?'}`,`${m.output_tokens_min??'?'}–${m.output_tokens_max??'?'}`,latency(m.latency_p95_s),m.gates_passed?'Pass':'Fail'].map(x=>'<td>'+esc(String(x))+'</td>').join('')+'</tr>';}).join('')+'</tbody></table></div>'+goalCurves(rows)+'<p>Full curves are included as CSV and SVG in the evidence ZIP.</p>';return;
  }
  if(data.manifest.spec.tenant_workflow==='post_call'||data.manifest.spec.p01_prefill_grid?.length){
    const calls=data.manifest.spec.tenant_workflow==='post_call';
    panel.innerHTML='<h3>'+ (calls?'Post-call workflow results':'Prefill and decode search results')+'</h3><p>Every offered-load cell is shown. Failed arrivals stay in the denominator. These are explored rates, not confirmed maximum capacity.</p><div class="infra-table-scroll"><table><thead><tr>'+['Placement','Rate / s','Qualified / offered',calls?'Audio hours / hour':'p95 first token s','p95 total s','Gate'].map(x=>'<th scope="col">'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>{const m=r.metrics;return '<tr>'+[calls?r.arm:(r.prefill===0?'Aggregate':r.prefill+' prefill'),r.rate,`${m.qualified} / ${m.offered_requests??m.attempted}`,latency(calls?m.qualified_audio_hours_per_hour:m.ttft_p95_s),latency(m.latency_p95_s),m.gates_passed?'Pass':'Fail'].map(x=>'<td>'+esc(String(x))+'</td>').join('')+'</tr>';}).join('')+'</tbody></table></div>';return;
  }
  panel.innerHTML='<h3>Tenant service results</h3><p>Fixed offered demand per service. Burst failures remain in the totals. Passing normal demand does not establish maximum density or cost savings.</p>'+
    (rows.length?'<div class="infra-table-scroll"><table><caption>Every measured tenant and load phase</caption><thead><tr><th scope="col">Block</th><th scope="col">Placement</th><th scope="col">Load</th><th scope="col">Tenant</th><th scope="col">Qualified / offered</th><th scope="col">p95 response (s)</th><th scope="col">Service gate</th></tr></thead><tbody>'+rows.map(row=>{
      const m=row.metrics;
      return '<tr>'+[row.block+1,row.arm,row.phase,row.tenant,`${m.qualified} / ${m.offered_requests??m.attempted}`,latency(m.latency_p95_s),m.gates_passed?'Pass':'Fail'].map(value=>`<td>${esc(String(value))}</td>`).join('')+'</tr>';
    }).join('')+'</tbody></table></div>':'<p>No completed measurement windows yet.</p>');
}

function goalCurves(rows){
  if(!rows.length)return '';
  const colors=['#999','#64b5f6','#ffb74d','#81c784','#ba68c8','#f06292','#4db6ac','#dce775'];
  return '<details><summary>Capacity curves by document shape</summary><p>Dots pass the service targets; crosses fail. No values between measurements are inferred.</p>'+[...new Set(rows.map(r=>r.shape))].sort().flatMap(shape=>[...new Set(rows.filter(r=>r.prefill>0).map(r=>r.placement||'pooled'))].map(placement=>({shape,placement}))).map(({shape,placement})=>{
    const cells=rows.filter(r=>r.shape===shape&&(r.prefill===0||(r.placement||'pooled')===placement)), xmax=Math.max(...cells.map(r=>r.rate)), ymax=Math.max(1,...cells.map(r=>r.metrics.qualified_per_hour/3600));
    const marks=cells.map(r=>{const x=60+500*r.rate/xmax,y=190-150*(r.metrics.qualified_per_hour/3600)/ymax,color=colors[r.prefill];return r.metrics.gates_passed?`<circle cx="${x}" cy="${y}" r="4" fill="${color}"><title>${esc(r.prefill===0?'Aggregate':r.prefill+' prefill')}: ${r.rate}/s offered</title></circle>`:`<path d="M${x-4} ${y-4} l8 8 m-8 0 l8 -8" stroke="${color}" stroke-width="2"/>`;}).join('');
    return `<h4>${esc(shape)} · ${esc(placement.replaceAll('_',' '))}</h4><svg role="img" aria-label="${esc(shape)} qualified throughput versus offered demand" viewBox="0 0 760 240" style="width:100%;max-width:900px"><path d="M60 30 V190 H570" stroke="currentColor" fill="none"/><text x="10" y="25" fill="currentColor" font-size="12">Qualified requests/s</text><text x="10" y="45" fill="currentColor" font-size="12">${ymax.toFixed(2)}</text><text x="50" y="212" fill="currentColor" font-size="12">0</text><text x="540" y="212" fill="currentColor" font-size="12">${xmax}</text><text x="200" y="235" fill="currentColor" font-size="12">Offered requests/s</text>${marks}${[...new Set(cells.map(r=>r.prefill))].sort((a,b)=>a-b).map((p,i)=>`<text x="600" y="${35+i*22}" fill="${colors[p]}" font-size="12">${p===0?'Aggregate':p+' prefill'}</text>`).join('')}</svg>`;
  }).join('')+'</details>';
}

async function reviewImages(id) {
  try {
    const review=await api(`/api/infrastructure/campaigns/${id}/image-review`);
    $('#infra-image-review').innerHTML=`<h3>Blind concept-image review</h3><p>Check each criterion that passes. Unchecked criteria reject the image. Arm, steps and timing are hidden during review.</p><label>Reviewer name<input id="infra-image-reviewer" required maxlength="200"></label>`+review.images.map(r=>`<section data-review-image="${esc(r.blind_id)}"><img src="${esc(r.url)}" alt="Generated concept for review" loading="lazy" style="max-width:320px;width:100%"><p>${esc(r.prompt)}</p>${Object.entries(review.rubric).map(([key,label])=>`<label><input type="checkbox" data-rubric="${esc(key)}">${esc(label)}</label>`).join('')}</section>`).join('')+'<button id="infra-save-image-review">Save review and rebuild reports</button>';
    $('#infra-save-image-review').onclick=async()=>{
      try {
        const reviewer=$('#infra-image-reviewer').value.trim();if(!reviewer)throw Error('Enter the reviewer name.');
        const items=[...document.querySelectorAll('[data-review-image]')].map(e=>({blind_id:e.dataset.reviewImage,scores:Object.fromEntries([...e.querySelectorAll('[data-rubric]')].map(c=>[c.dataset.rubric,c.checked]))}));
        await api(`/api/infrastructure/campaigns/${id}/image-review`,{method:'POST',body:JSON.stringify({reviewer,reviewer_kind:'human',items})});
        message('Review saved with an audit record. Reports remain exploratory.');await detail(id);
      }catch(e){message(e.message);}
    };
  }catch(e){message(e.message);}
}
export const Infrastructure={init() {
  const tokenSettings=document.createElement('fieldset');tokenSettings.id='infra-token-settings';tokenSettings.innerHTML='<legend>Roofline token throughput search</legend><label><input id="infra-token-search" type="checkbox"> Search GPU splits and token shapes for output throughput</label><label>Input:output token pairs<input id="infra-token-shapes" value="128:256,4096:256,16384:256,4096:1024"></label><label>Concurrent streams<input id="infra-token-streams" value="128,512,2048,8192"></label><label>Maximum sequences per GPU<input id="infra-token-seqs" type="number" min="32" max="4096" value="256"></label><p>Uses the split and placement settings below. Keeps each allocation loaded across shapes. TTFT is reported alongside throughput.</p>';
  $('#infra-goal-seek').closest('label').before(tokenSettings);
  $('#infra-token-search').onchange=()=>{if($('#infra-token-search').checked){$('#infra-goal-seek').checked=false;$('#infra-mode').value='pilot';$('#infra-gpus').value='0,4,1,5,2,6,3,7';$('#infra-search-splits').value='2,4,6';}workflowControls();};

  $('#infra-cards').addEventListener('click',e=>{const b=e.target.closest('[data-project]');if(b){selected=b.dataset.project;requestKey=null;if(selected==='P10'){$('#infra-gpus').value='0';$('#infra-context').value=32768;$('#infra-records').value=64;}if(selected==='P06'){$('#infra-gpus').value='6,7';$('#infra-mode').value='pilot';$('#infra-blocks').value=1;$('#infra-records').value=16;$('#infra-context').value=8192;}if(selected==='P05'){$('#infra-gpus').value='0';$('#infra-mode').value='pilot';$('#infra-blocks').value=1;}if(selected==='P01'){$('#infra-gpus').value='0,1';$('#infra-context').value=32768;$('#infra-records').value=64;}if(selected==='P02'){$('#infra-gpus').value='0';$('#infra-context').value=8192;$('#infra-records').value=64;$('#infra-mode').value='pilot';$('#infra-mode').dispatchEvent(new Event('change'));$('#infra-duration').value=300;$('#infra-concurrency').value=2048;$('#infra-timeout').value=600;$('#infra-context').value=32768;$('#infra-records').value=192;$('#infra-session-cpu').value=128;}if(selected==='P03'){if($('#infra-mode').value==='confirmation'){$('#infra-mode').value='pilot';$('#infra-mode').dispatchEvent(new Event('change'));}$('#infra-context').value=65536;$('#infra-min-requests').value=32;}else{$('#infra-min-requests').value=selected==='P02'?8:30;if(selected==='P04')$('#infra-context').value=8192;}render();}});
  $('#infra-history').addEventListener('click',e=>{const b=e.target.closest('[data-run]');if(b)detail(b.dataset.run);});
  $('#infra-refresh').onclick=refresh;
  $('#infra-goal-seek').onchange=()=>{if($('#infra-goal-seek').checked){$('#infra-mode').value='pilot';if(!$('#infra-search-splits').value.trim())$('#infra-search-splits').value='1';if(!$('#infra-search-rates').value.trim())$('#infra-search-rates').value='1,2,4,6,8,10,12';}workflowControls();};
  $('#infra-recover').onclick=async()=>{try{await api(`/api/infrastructure/campaigns/${$('#infra-recover').dataset.run}/recover`,{method:'POST'});message('Owned resources recovered. Start a fresh experiment to remeasure.');await refresh();}catch(e){message(e.message);}};
  $('#infra-stop').onclick=async()=>{try{await api('/api/runs/stop',{method:'POST'});message('Stop requested. Waiting for owned deployment cleanup.');await refresh();}catch(e){message(e.message);}};
  $('#infra-check').onclick=async()=>{message('Reading host capabilities…');try{const r=await api('/api/infrastructure/preflight',{method:'POST'});$('#infra-readiness').textContent=JSON.stringify(r,null,2);message('Read-only readiness check saved.');}catch(e){message(e.message);}};
  $('#infra-form').addEventListener('input',()=>{requestKey=null;});
  $('#infra-spec-method').onchange=()=>{$('#infra-draft-tokens').value=$('#infra-spec-method').value==='mtp'?3:5;};
  $('#infra-mode').onchange=()=>{const confirm=$('#infra-mode').value==='confirmation';$('#infra-blocks').value=confirm?5:2;$('#infra-duration').value=confirm?600:60;$('#infra-warmup').value=confirm?120:15;};
  $('#infra-tenant-workflow').onchange=()=>{if($('#infra-tenant-workflow').value==='post_call'){$('#infra-min-requests').value=10;$('#infra-duration').value=120;}workflowControls();};
  $('#infra-session-mode').onchange=workflowControls;
  $('#infra-session-shape').onchange=()=>{
    const ladder=$('#infra-session-shape').value==='knowledge_worker_ladder';
    $('#infra-session-ladder').hidden=!ladder;
    for(const id of ['intervals','turns']) $('#infra-session-'+id).closest('label').hidden=ladder;
    if(ladder){
      // Preset used for the XE7740 KV-offload density measurements: 8 GPUs, 96 users, 96K-token padded contexts.
      $('#infra-gpus').value='0,4,1,5,2,6,3,7';$('#infra-mode').value='pilot';$('#infra-blocks').value=1;
      $('#infra-duration').value=1500;$('#infra-timeout').value=300;$('#infra-concurrency').value=8192;$('#infra-min-requests').value=30;
      $('#infra-context').value=131072;$('#infra-records').value=16;$('#infra-session-cpu').value=48;
      $('#infra-session-populations').value='96';$('#infra-session-arrivals').value='0.3';$('#infra-session-ceiling').value=64;
      $('#infra-session-slos').value='5,10,30,60,120,300';$('#infra-session-pending').value=16384;$('#infra-session-answers').value=2;
    }
  };
  $('#infra-form').onsubmit=async e=>{
    e.preventDefault();const button=$('#infra-start');button.disabled=true;
    try {
      const source=$('#infra-source').value;if(!source && selected!=='P05')throw new Error('Select a recorded roofline result.');
      requestKey=requestKey||Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');
      const spec={response_slo_s:Number($('#infra-slo').value),request_timeout_s:Number($('#infra-timeout').value),min_quality:Number($('#infra-quality').value),min_slo_fraction:Number($('#infra-slo-fraction').value),min_requests:Number($('#infra-min-requests').value),speculative_method:$('#infra-spec-method').value,speculative_tokens:Number($('#infra-draft-tokens').value),project_id:selected,mode:$('#infra-mode').value,paired_blocks:Number($('#infra-blocks').value),duration_s:Number($('#infra-duration').value),warmup_s:Number($('#infra-warmup').value),concurrency:Number($('#infra-concurrency').value),gpu_ids:$('#infra-gpus').value.split(',').map(x=>Number(x.trim())),document_records:Number($('#infra-records').value),context_limit:Number($('#infra-context').value)};
      if(selected==='P10'){spec.attention_baseline=$('#infra-attention-baseline').value;spec.attention_treatment=$('#infra-attention-treatment').value;spec.attention_kv_dtype=$('#infra-attention-kv').value;spec.attention_logprob_tolerance=Number($('#infra-attention-tolerance').value);}
      if(selected==='P06'){spec.tenant_workflow=$('#infra-tenant-workflow').value;spec.call_request_rates=$('#infra-call-rates').value.split(',').map(x=>Number(x.trim()));spec.call_max_wer=Number($('#infra-call-wer').value);spec.tenant_requests_per_second=Number($('#infra-tenant-rate').value);spec.tenant_noise_multiplier=Number($('#infra-tenant-noise').value);spec.tenant_kv_gib=Number($('#infra-tenant-kv').value);}
      if(selected==='P05'){spec.diffusion_samples=Number($('#infra-diffusion-samples').value);spec.diffusion_teacher_steps=$('#infra-diffusion-steps').value.split(',').map(x=>Number(x.trim()));}
      if(selected==='P04' && $('#infra-spec-loads').value.trim()) spec.speculative_concurrencies=$('#infra-spec-loads').value.split(',').map(x=>Number(x.trim()));
      if(selected==='P02') {spec.session_tier=$('#infra-session-tier').value;for(const [key,id] of Object.entries({session_ttft_slo_s:'ttft',session_count:'count',session_churn:'churn',session_gpu_gb:'gpu',session_cpu_gb:'cpu',session_disk_gb:'disk'}))spec[key]=Number($('#infra-session-'+id).value);}
      if(selected==='P02' && $('#infra-session-mode').value==='capacity') {
        spec.session_mode='capacity';spec.session_tier='cpu';spec.session_gpu_gb=null;spec.warmup_s=0;
        spec.session_arrival_rates=$('#infra-session-arrivals').value.split(',').map(Number);
        spec.session_max_arrival_rate=Number($('#infra-session-ceiling').value);
        spec.session_slo_grid_s=$('#infra-session-slos').value.split(',').map(Number);
        spec.session_population_grid=$('#infra-session-populations').value.split(',').map(Number);
        spec.session_rate_grid=$('#infra-session-intervals').value.split(',').map(x=>1/Number(x)).sort((a,b)=>a-b);
        spec.session_turns=Number($('#infra-session-turns').value);
        spec.session_answer_records=Number($('#infra-session-answers').value);
        spec.session_max_pending=Number($('#infra-session-pending').value);
        if($('#infra-session-shape').value==='knowledge_worker_ladder'){
          spec.session_workload='document';spec.session_cpu_backend='simple_cpu';spec.session_prime_documents=true;spec.session_document_padding=true;
          spec.session_context_tokens=Number($('#infra-session-context-tokens').value);spec.session_min_returns=Number($('#infra-session-min-returns').value);
          spec.session_refinement_rounds=Number($('#infra-session-refinement').value);spec.session_client_lag_s=Number($('#infra-session-client-lag').value);
          spec.session_engine_max_seqs=Number($('#infra-session-max-seqs').value);spec.max_output_tokens=1024;spec.max_hours=20;
          spec.session_rate_grid=[1/60];
        }
      }
      if(selected==='P01') {
        spec.p01_goal_seek=$('#infra-goal-seek').checked;
        if(spec.p01_goal_seek){spec.p01_placement_grid=$('#infra-goal-placement').value==='both'?['socket_local','cross_socket']:[$('#infra-goal-placement').value];spec.p01_rate_resolution=Number($('#infra-goal-resolution').value);spec.max_output_tokens=Number($('#infra-goal-cap').value);spec.max_hours=Number($('#infra-goal-hours').value);spec.p01_document_grid=$('#infra-goal-documents').value.split(',').map(x=>Number(x.trim()));spec.p01_answer_grid=$('#infra-goal-answers').value.split(',').map(x=>Number(x.trim()));}

        if($('#infra-search-splits').value.trim()){spec.p01_prefill_grid=$('#infra-search-splits').value.split(',').map(x=>Number(x.trim()));spec.p01_request_rates=$('#infra-search-rates').value.split(',').map(x=>Number(x.trim()));spec.p01_ttft_slo_s=Number($('#infra-search-ttft').value);spec.p01_queue_growth_s=Number($('#infra-search-queue').value);}
        spec.prefill_workers=Number($('#infra-prefill').value);
        if($('#infra-token-search').checked){spec.p01_token_search=true;spec.p01_goal_seek=false;spec.p01_request_rates=[];spec.p01_token_shapes=$('#infra-token-shapes').value.split(',').map(x=>x.trim().split(':').map(Number));spec.p01_streams=$('#infra-token-streams').value.split(',').map(Number);spec.p01_engine_max_seqs=Number($('#infra-token-seqs').value);spec.p01_placement_grid=$('#infra-goal-placement').value==='both'?['socket_local','cross_socket']:[$('#infra-goal-placement').value];spec.max_hours=Number($('#infra-goal-hours').value);}
        spec.p01_block_size=$('#infra-transfer-block').value?Number($('#infra-transfer-block').value):null;
        spec.p01_transport=$('#infra-transfer-mode').value;
        spec.p01_cuda_ipc_get=$('#infra-transfer-get').value;
        spec.p01_memtype_cache=$('#infra-transfer-memtype').value;
        spec.p01_cross_layer_blocks=$('#infra-transfer-layout').value==='cross';
        spec.p01_max_batched_tokens=$('#infra-transfer-tokens').value?Number($('#infra-transfer-tokens').value):null;
      }
      if(selected==='P03') {spec.cache_samples=Number($('#infra-cache-samples').value);spec.cache_record_counts=$('#infra-cache-records').value.split(',').map(x=>Number(x.trim()));spec.cache_concurrencies=$('#infra-cache-concurrency').value.split(',').map(x=>Number(x.trim()));}
      const r=await api('/api/infrastructure/campaigns',{method:'POST',body:JSON.stringify({result_id:source,idempotency_key:requestKey,spec})});
      requestKey=null;message(`Experiment accepted: ${r.campaign_id}. Use Stop active run to cancel.`);await refresh();
    }catch(e){message(e.message);}finally{button.disabled=false;}
  };
  onShow('infrastructure',()=>{refresh();loadSources();});
  timer=setInterval(()=>{if(currentView()==='infrastructure')refresh();},5000);
  window.addEventListener('pagehide',()=>clearInterval(timer));
}};
