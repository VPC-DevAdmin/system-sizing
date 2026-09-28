import { $, api } from './lib/api.js';

const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const list = id => $(id).value.trim().split(/[,\s]+/).filter(Boolean).map(Number);

export const Benchmarks = {
  rows: [], charts: [],
  init() {
    const settings = document.createElement('details');
    settings.id = 'rf-benchmark-settings';
    settings.className = 'ws-disclosure';
    settings.innerHTML = `<summary>Third-party benchmarks</summary>
      <p class="msg">After internal confirmation, test one winner across engines and quantizations per model family. Runs use the named harnesses; they do not certify results. Harness throughput is measured client-side over a fixed window with a synthetic corpus, so it re-measures the recorded shape rather than replicating the internal counter-based number.</p>
      <div class="row wrap"><label class="check"><input id="bm-aiperf" type="checkbox"> AIPerf 0.13.0</label><label class="check"><input id="bm-guidellm" type="checkbox"> GuideLLM 0.7.4</label><button id="bm-prepare" type="button">Prepare selected tools</button></div>
      <div id="bm-fields" hidden class="form-grid">
        <label>Recipe<select id="bm-recipe"><option value="reproduce">Reproduce selected roofline shape</option><option value="sweep">Matched serving sweep</option><option value="slo">SLO request-rate sweep</option><option value="sustained">Sustained confirmation</option></select></label>
        <label>Measured seconds per point<input id="bm-duration" type="number" min="10" max="86400" value="300"></label>
        <label>Warmup seconds<input id="bm-warmup" type="number" min="0" max="3600" value="30"></label>
        <label>Repetitions<input id="bm-repeats" type="number" min="1" max="20" value="3"></label>
        <label id="bm-concurrency-label">Concurrent requests <small>blank uses recorded point</small><input id="bm-concurrency" placeholder="32, 64, 128"></label>
        <label id="bm-rates-label" hidden>Requests per second <small>Poisson arrivals</small><input id="bm-rates" placeholder="1, 2, 4"></label>
        <label id="bm-input-label" hidden>Prompt tokens <small>blank uses recorded shape</small><input id="bm-input" type="number" min="8"></label>
        <label id="bm-output-label" hidden>Output tokens<input id="bm-output" type="number" min="1"></label>
        <label>Cache and launch policy<select id="bm-cache"><option value="warm">Reuse healthy launch and warm cache</option><option value="restart">Restart engine before each attempt</option></select></label>
        <label>TTFT p95 limit in ms <small>optional outside SLO</small><input id="bm-ttft" type="number" min="1"></label>
        <label>TPOT p95 limit in ms <small>GuideLLM; AIPerf unavailable</small><input id="bm-tpot" type="number" min="1"></label>
        <label>Maximum error fraction<input id="bm-errors" type="number" min="0" max="1" step="0.01" value="0.01"></label>
        <label>Minimum successful requests<input id="bm-min" type="number" min="1" value="100"></label>
        <label>Artifact processing allowance in seconds<input id="bm-processing" type="number" min="30" max="7200" value="900"></label>
        <label>Campaign time budget in hours<input id="bm-hours" type="number" min="0.1" max="720" value="72"></label>
        <label>Evidence disk budget in GB<input id="bm-disk" type="number" min="0.1" value="20"></label>
        <label>Seed<input id="bm-seed" type="number" min="0" value="42"></label>
        <label>Attempts per load point<input id="bm-attempts" type="number" min="1" max="4" value="2"></label>
      </div><p id="bm-estimate" class="msg"></p><p id="bm-message" role="status" class="msg"></p>`;
    $('#rf-confirm').closest('label').parentElement.after(settings);
    const panel = document.createElement('section');
    panel.className = 'panel'; panel.id = 'rf-benchmarks-panel';
    panel.innerHTML = `<h2>Third-party benchmark evidence</h2><p class="msg">Use the controls in Roofline Setup to choose harnesses and settings. Every load point and repetition retains its own artifacts.</p>
      <div class="row wrap"><label>Recorded result<select id="bm-result"><option value="">Choose a recorded result</option></select></label><label class="check"><input id="bm-fresh" type="checkbox">New experiment (repeat completed test)</label><button id="bm-configure" type="button">Benchmark settings</button><button id="bm-run" type="button">Run Third Party Benchmarks</button><button id="bm-stop" type="button" class="danger">Stop active run</button></div>
      <p id="bm-panel-message" role="status" class="msg"></p><label>Load curve metric<select id="bm-metric"><option value="output_tok_s">Output tokens/s</option><option value="requests_s">Requests/s</option><option value="ttft_p95_ms">TTFT p95 (ms)</option><option value="tpot_p95_ms">TPOT p95 (ms)</option><option value="itl_p95_ms">ITL p95 (ms)</option><option value="error_fraction">Error fraction</option></select></label><div id="bm-tools" class="msg"></div><div id="bm-search-export" class="row wrap" hidden><a style="color:var(--accent);text-underline-offset:.2em" href="/api/roofline/benchmark-report/docx">Download search DOCX</a><a style="color:var(--accent);text-underline-offset:.2em" href="/api/roofline/benchmark-report/zip">Download search evidence ZIP</a></div><p class="msg">Rates retain each harness’s definition. GuideLLM includes in-window tokens from partial boundary requests; AIPerf includes completion draining. Incomplete and proven boundary-cancelled counts remain visible.</p><div id="bm-history"></div><pre id="bm-detail" hidden style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto"></pre>`;
    const resultPanel=$('#rf-results-panel');
    (resultPanel.closest('details.ws-attempts')||resultPanel).after(panel);
    settings.addEventListener('input', () => this.settingsChanged());
    $('#bm-prepare').onclick = () => this.action(async () => api('/api/roofline/benchmarks/prepare', {method:'POST',body:JSON.stringify(this.spec())}));
    $('#bm-configure').onclick = () => this.configure();
    $('#bm-run').onclick = () => this.run();
    $('#bm-metric').onchange = () => this.refresh();
    $('#bm-stop').onclick = () => this.action(async () => api('/api/runs/stop',{method:'POST'}));
    panel.addEventListener('click', event => {
      const button=event.target.closest('[data-bm-resume],[data-bm-detail]'); if(!button)return;
      if(button.dataset.bmResume)this.action(()=>api(`/api/roofline/benchmarks/${button.dataset.bmResume}/resume`,{method:'POST'}));
      else this.action(async()=>{const d=await api(`/api/roofline/benchmarks/${button.dataset.bmDetail}`);$('#bm-detail').hidden=false;$('#bm-detail').textContent=JSON.stringify(d,null,2);});
    });
    this.settingsChanged();
    this.refresh();
    this.timer = setInterval(()=>this.refresh(),5000);
  },
  spec() {
    const recipe=$('#bm-recipe').value;
    const optional = id => $(id).value === '' ? null : Number($(id).value);
    return {harnesses:['aiperf','guidellm'].filter(h=>$('#bm-'+h).checked), recipe,
      duration_s:Number($('#bm-duration').value), warmup_s:Number($('#bm-warmup').value),
      repetitions:Number($('#bm-repeats').value), concurrency:recipe==='slo'?[]:list('#bm-concurrency'),
      rates:recipe==='slo'?list('#bm-rates'):[], input_tokens:recipe==='reproduce'?null:optional('#bm-input'),
      output_tokens:recipe==='reproduce'?null:optional('#bm-output'), cache_policy:$('#bm-cache').value,
      ttft_p95_ms:optional('#bm-ttft'), tpot_p95_ms:optional('#bm-tpot'),
      max_error_fraction:Number($('#bm-errors').value), min_requests:Number($('#bm-min').value),
      processing_grace_s:Number($('#bm-processing').value), max_hours:Number($('#bm-hours').value), disk_budget_gb:Number($('#bm-disk').value),
      seed:Number($('#bm-seed').value), max_attempts:Number($('#bm-attempts').value)};
  },
  settingsChanged() {
    const s=this.spec(); $('#bm-fields').hidden=!s.harnesses.length;
    $('#bm-rates-label').hidden=s.recipe!=='slo';$('#bm-concurrency-label').hidden=s.recipe==='slo';
    $('#bm-input-label').hidden=$('#bm-output-label').hidden=s.recipe==='reproduce';
    if(s.harnesses.length)$('#rf-confirm').checked=true;
    const points=Math.max(1,s.recipe==='slo'?s.rates.length:s.concurrency.length);
    $('#bm-estimate').textContent=s.harnesses.length?`At least ${Math.ceil(s.harnesses.length*points*s.repetitions*(s.duration_s+s.warmup_s)/60)} minutes per winning model, plus preparation, launches, draining and artifact processing. Tools run sequentially. ${s.harnesses.includes('aiperf')&&s.tpot_p95_ms?'AIPerf cannot satisfy a TPOT gate in this adapter; that result will be inconclusive.':''}`:'';
  },
  async action(fn) {
    try {await fn();this.message('Request accepted. Progress and evidence appear below.');await this.refresh();}
    catch(error){this.message(error.message);$('#rf-benchmark-settings').open=true;}
  },
  message(text) { for(const id of ['#bm-message','#bm-panel-message'])$(id).textContent=text; },
  configure() { const controls=$('#rf-benchmark-settings');controls.open=true;controls.scrollIntoView({behavior:'smooth'}); },
  async run() {
    const id=$('#bm-result').value;
    if(!id){this.message('Select a recorded result.');return;}
    if(!this.spec().harnesses.length){this.message('Select AIPerf or GuideLLM in benchmark settings.');this.configure();return;}
    await this.action(()=>api('/api/roofline/benchmarks',{method:'POST',body:JSON.stringify({result_id:id,spec:this.spec(),fresh:$('#bm-fresh').checked})}));
  },
  async select(row) {
    await this.refresh();
    const entry=this.rows.find(x=>x.result.run_dir===row.run_dir&&x.result.model===row.model&&x.result.engine===row.engine&&x.result.confirmed===row.confirmed);
    if(entry)$('#bm-result').value=entry.id;
    $('#rf-benchmarks-panel').scrollIntoView({behavior:'smooth'});
    $('#bm-run').focus();
  },
  async refresh() {
    if(this.refreshing)return;this.refreshing=true;
    try {
      const [data,results]=await Promise.all([api('/api/roofline/benchmarks'),api('/api/roofline/benchmarks/results')]);
      $('#bm-search-export').hidden=!data.search_report_ready;
      this.rows=results.results;const selected=$('#bm-result').value;
      $('#bm-result').innerHTML='<option value="">Choose a recorded result</option>'+this.rows.filter(x=>!x.result.error&&x.result.out_tok_s).map(x=>`<option value="${esc(x.id)}">${esc(x.result.model)} · ${esc(x.result.engine)} · ${Math.round(x.result.out_tok_s)} tok/s${x.result.confirmed?' · confirmed':''}</option>`).join('');
      $('#bm-result').value=selected;
      $('#bm-tools').textContent=data.tools.map(t=>`${t.id} ${t.version}: ${t.ready?'prepared':'not prepared'}`).join(' · ');
      this.charts.forEach(chart=>chart.destroy());this.charts=[];
      $('#bm-history').innerHTML=data.campaigns.map(c=>`<article class="panel"><h3>${esc(c.model)} · ${esc(c.engine)}</h3><p>${esc(c.phase)} · ${esc(c.outcome||'pending')} · ${c.attempts.length} attempts${c.error?' · '+esc(c.error):''}</p>
        <div class="table-scroll" style="max-width:100%;overflow-x:auto" tabindex="0" aria-label="Benchmark measurements"><table><thead><tr><th>Harness</th><th>Load</th><th>Repeat</th><th>Output tok/s</th><th>TTFT p95 ms</th><th>TPOT p95 ms</th><th>Requests/s</th><th>Error fraction</th><th>Incomplete / boundary</th><th>Outcome</th></tr></thead><tbody>${c.attempts.map(a=>`<tr><td>${esc(a.harness)}</td><td>${esc(a.point)}</td><td>${a.repetition}</td><td>${a.metrics?.output_tok_s?.toFixed(1)??'—'}</td><td>${a.metrics?.ttft_p95_ms?.toFixed(1)??'—'}</td><td>${a.metrics?.tpot_p95_ms?.toFixed(1)??'—'}</td><td>${a.metrics?.requests_s?.toFixed(2)??'—'}</td><td>${a.metrics?.error_fraction?.toPrecision(3)??'—'}</td><td>${a.metrics?.incomplete??'—'} / ${a.metrics?.boundary_cancelled??'—'}</td><td title="${esc((a.reasons||[]).join('; '))}">${esc(a.outcome)}</td></tr>`).join('')}</tbody></table></div>
        <div style="height:220px"><canvas data-bm-curve="${esc(c.id)}" aria-label="Per-repetition benchmark load curve" role="img"></canvas></div><div class="row wrap"><button data-bm-detail="${esc(c.id)}">Inspect methodology and attempts</button>${['interrupted','failed','queued'].includes(c.phase)?`<button data-bm-resume="${esc(c.id)}">Resume</button>`:''}${c.report_ready?`<a style="color:var(--accent);text-underline-offset:.2em" href="/api/roofline/benchmarks/${esc(c.id)}/export/docx">Download DOCX</a><a style="color:var(--accent);text-underline-offset:.2em" href="/api/roofline/benchmarks/${esc(c.id)}/export/zip">Download evidence ZIP</a>`:''}</div></article>`).join('')||'<p class="msg">No third-party measurements yet.</p>';
      const metric=$('#bm-metric').value;
      if(typeof Chart!=='undefined')for(const c of data.campaigns){
        const groups=new Map();
        for(const a of c.attempts){if(!a.metrics||a.metrics[metric]==null)continue;const key=`${a.harness} · repeat ${a.repetition} · ${a.outcome}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push({x:a.point,y:a.metrics[metric]});}
        const palette=['#4d8ff7','#dd8a3c','#46aa89','#a879de'];
        const canvas=document.querySelector(`[data-bm-curve="${c.id}"]`);
        this.charts.push(new Chart(canvas,{type:'line',data:{datasets:[...groups].map(([label,data],i)=>({label,data:data.sort((a,b)=>a.x-b.x),borderColor:palette[i%palette.length],borderDash:label.endsWith(' · measured')?[]:[6,4],backgroundColor:palette[i%palette.length],pointRadius:4}))},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{type:'linear',title:{display:true,text:c.recipe.recipe==='slo'?'Offered requests/s':'Concurrent requests'}},y:{title:{display:true,text:$('#bm-metric').selectedOptions[0].textContent}}}}}));
      }
    } catch(error) {$('#bm-tools').textContent=`Benchmark status unavailable: ${error.message}`;}
    finally {this.refreshing=false;}
  }
};
