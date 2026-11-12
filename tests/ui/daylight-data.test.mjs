import test from 'node:test';
import assert from 'node:assert/strict';
import { autoShape, barRows, ladderChart, lineChart, notAtCeiling, perUser, resultNotes }
  from '../../simulator/ui/lib/daylight-data.js';

const rung = (c, out, extra = {}) => ({ concurrency: c, out_tok_s: out, samples: 100, errors: 0, held: true, ...extra });

test('automatic shape uses the fewest GPUs per copy that hold the model', () => {
  const hw = { gpus: 8, vram_per_gpu_gb: 96 };
  assert.deepEqual(autoShape({ min_vram_gb: 40 }, hw), { tp: 1, replicas: 8 });
  assert.deepEqual(autoShape({ min_vram_gb: 255 }, hw), { tp: 4, replicas: 2 });
  assert.deepEqual(autoShape({ min_vram_gb: 900 }, hw), { tp: 8, replicas: 1 });
  assert.deepEqual(autoShape({}, {}), { tp: 1, replicas: 1 });
});

test('per-user speed is the inverse of the median token gap', () => {
  assert.equal(perUser({ tpot_p50_ms: 25 }), 40);
  assert.equal(perUser({}), null);
});

test('a peak at the highest load measured is flagged as not the ceiling', () => {
  const rising = { rungs: [rung(1024, 57360), rung(2048, 77622)], peak: rung(2048, 77622), status: 'ok' };
  assert.equal(notAtCeiling(rising), true);
  assert.deepEqual(resultNotes(rising).map((n) => n.title), ['Not yet the ceiling.']);
  const flat = { rungs: [rung(1024, 80000), rung(2048, 70000)], peak: rung(1024, 80000), status: 'ok' };
  assert.equal(notAtCeiling(flat), false);
  assert.deepEqual(resultNotes(flat), []);
});

test('false results are called out before anything else', () => {
  const failed = { rungs: [rung(512, 97)], peak: rung(512, 97, { samples: 0, errors: 256 }), status: 'ok' };
  assert.equal(resultNotes(failed)[0].kind, 'bad');
  assert.match(resultNotes(failed)[0].text, /256 of 256/);
  const reused = { rungs: [rung(512, 100)], peak: rung(512, 100, { prefix_reuse: 'above_template' }), status: 'ok' };
  assert.equal(resultNotes(reused)[0].title, 'Too much cache reuse.');
  assert.equal(resultNotes({ rungs: [], peak: null })[0].kind, 'bad');
});

test('charts label every value and escape names', () => {
  const rungs = [rung(1024, 57360), rung(2048, 77622)];
  const line = lineChart(rungs, rungs[1]);
  assert.match(line, /77,622 peak/);
  assert.match(line, /role="img"/);
  const ladder = ladderChart([1024, 2048, 4096], [rungs[0]], 2048);
  assert.match(ladder, /57,360/);
  assert.match(ladder, /2,048 requests: being measured/);
  assert.match(ladder, /4,096 requests: not started/);
  const bars = barRows([{ name: '<b>x</b>', value: 2, text: 'two', href: '#result/a' }, { name: 'y', value: 1, text: 'one' }]);
  assert.ok(!bars.includes('<b>x</b>'));
  assert.match(bars, /href="#result\/a"/);
});

import { bestRate, capacitySummary, rateChart, stepVerdict } from '../../simulator/ui/lib/daylight-data.js';

const step = (rate, stability, status, extra = {}) => (
  { arrival_rate_per_min: rate, stability, status, sample_size: 50, ttft_p95_ms: 200, ...extra });

test('each tested rate is judged kept up, slow, fell behind or pending', () => {
  assert.equal(stepVerdict(step(60, 'stable', 'pass')), 'ok');
  assert.equal(stepVerdict(step(120, 'stable', 'marginal')), 'slow');
  assert.equal(stepVerdict(step(480, 'divergent', 'pass')), 'over');
  assert.equal(stepVerdict({ arrival_rate_per_min: 960, capacity_status: 'pending', stability: null }), 'pending');
  // The live feed names the status field differently from the export.
  assert.equal(stepVerdict({ arrival_rate_per_min: 60, stability: 'stable', capacity_status: 'pass' }), 'ok');
  assert.equal(bestRate([step(60, 'stable', 'pass'), step(240, 'stable', 'marginal'), step(480, 'divergent', 'fail')]), 240);
  assert.equal(bestRate([step(60, null, 'pending', { sample_size: 0 })]), null);
});

test('capacity is the rate that met targets, with users derived from it', () => {
  const met = capacitySummary({ final_status: 'ok', open_loop: {
    rate_sla_per_min: 120, rate_max_per_min: 240, rate_ceiling_per_min: 480,
    derived_concurrent_sessions_at_sla: 19, measured_concurrent_sessions_at_max: 40 } });
  assert.deepEqual([met.rate, met.users, met.atTarget, met.notes.length], [120, 19, true, 0]);
});

test('capacity notes say when targets were missed or the ceiling was not reached', () => {
  const missed = capacitySummary({ final_status: 'cancelled', open_loop: {
    rate_sla_per_min: null, rate_max_per_min: 60, rate_ceiling_per_min: null,
    measured_concurrent_sessions_at_max: 5.33 } });
  assert.deepEqual([missed.rate, missed.users, missed.atTarget], [60, 5.33, false]);
  assert.deepEqual(missed.notes.map((n) => n.title),
    ['Response-time targets were missed.', 'The run did not finish.', 'Not yet the ceiling.']);
  const none = capacitySummary({ open_loop: {} });
  assert.equal(none.rate, null);
  assert.equal(none.notes[0].kind, 'bad');
});

test('the rate chart names each verdict in text, not colour alone', () => {
  const svg = rateChart([step(120, 'stable', 'pass'), step(60, 'stable', 'marginal'), step(480, 'divergent', 'fail')]);
  assert.match(svg, /60 per minute: slow turns/);
  assert.match(svg, />kept up</);
  assert.match(svg, />fell behind</);
  assert.match(rateChart([]), /No arrival rate/);
});
