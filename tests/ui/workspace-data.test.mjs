import test from 'node:test';
import assert from 'node:assert/strict';
import { pairResult, pairState, metricScale, number, escapeHTML, methodFromWorkload } from '../../simulator/ui/lib/workspace-data.js';
const doc = {
  plan: {cells:[{model:'m',engine:'a'},{model:'m',engine:'b'},{model:'n',engine:'a'}]},
  summary:{best_per_pair:{m:{a:{out_tok_s:100,kv_cache_pct:20,confirmed:true},b:{out_tok_s:50,kv_cache_pct:80}},n:{a:{out_tok_s:10,kv_cache_pct:null}}}},
  results:[],
};
test('pairing uses canonical service row, never an unrelated raw maximum',()=>{
  const d=structuredClone(doc);d.results=[{model:'m',engine:'a',out_tok_s:999}];
  assert.equal(pairResult(d,'m','a').out_tok_s,100);
  assert.equal(pairState(d,'m','a'),'Confirmed');
  assert.equal(pairState(d,'m','b'),'Exploratory');
});
test('metric scale follows metric and explicit normalization',()=>{
  const absolute=metricScale(doc,['m','n'],['a','b'],'out_tok_s');
  assert.equal(absolute.fraction('n',10),.1);
  assert.equal(metricScale(doc,['m','n'],['a','b'],'out_tok_s',true).fraction('n',10),1);
  const kv=metricScale(doc,['m','n'],['a','b'],'kv_cache_pct');
  assert.equal(kv.maximum,80);assert.equal(kv.fraction('m',20),.25);
});
test('unmeasured states and missing values are distinct from zero',()=>{
  const d=structuredClone(doc);d.summary={};
  assert.equal(pairState(d,'m','a'),'Untested');
  assert.equal(pairState(d,'n','b'),'Not planned');
  d.results=[{model:'m',engine:'a',error:'engine failed'}];
  assert.equal(pairState(d,'m','a'),'Failed');
  d.current={model:'m',engine:'a'};d.live=true;
  assert.equal(pairState(d,'m','a'),'Running');
  assert.equal(number(null),'—');assert.equal(number(0),'0');
});
test('remote text is escaped and saturation has a distinct method',()=>{
  assert.equal(escapeHTML('<img onerror="x">'), '&lt;img onerror=&quot;x&quot;&gt;');
  assert.equal(methodFromWorkload('persona:headline_generation'),'saturation');
  assert.equal(methodFromWorkload('cohort:chat_heavy'),'capacity');
});

test('recovery and KTransformers scaling serialize to distinct supported API modes',async()=>{
  const {rooflineRequest}=await import('../../simulator/ui/lib/workspace-data.js');
  const scope={models:['m'],engines:['a','b'],mode:'manual',limits:{model_limit:1,large_limit:0,beyond_limit:0},shapes:{max_num_seqs:[64],output_tokens:[128]},inputTokens:128,confirm:true,crossDomain:true};
  const resume=rooflineRequest(scope);
  assert.equal(resume.new_run,false);assert.equal(resume.workload.spec.resume,true);
  assert.equal(resume.workload.spec.retry_engines,null);
  const retry=rooflineRequest(scope,{recovery:'retry',engine:'b'});
  assert.deepEqual(retry.workload.spec.retry_engines,['b']);assert.equal(retry.workload.spec.redo_engines,null);
  const redo=rooflineRequest(scope,{recovery:'redo',ktScale:true,priority:['m']});
  assert.deepEqual(redo.workload.spec.redo_engines,['a','b']);assert.equal(redo.workload.spec.kt_scale,true);
  assert.equal(rooflineRequest(scope,{recovery:'fresh'}).new_run,true);
  assert.equal(rooflineRequest(scope,{recovery:'fresh'}).workload.spec.resume,false);
  assert.throws(()=>rooflineRequest(scope,{priority:['unknown']}),/Priority models/);
  assert.throws(()=>rooflineRequest({...scope,shapes:{max_num_seqs:[0],output_tokens:[128]}}),/positive whole/);
});
