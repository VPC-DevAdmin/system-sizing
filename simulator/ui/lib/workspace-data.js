/* Pure presentation helpers. Service summaries own result selection; UI filters
 * never silently rerank on a secondary metric or replace confirmations. */
export const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const number = (value, digits = 0) => value == null || !Number.isFinite(Number(value))
  ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });

export function pairResult(doc, model, engine) {
  return doc.summary?.best_per_pair?.[model]?.[engine] ?? null;
}

export function pairState(doc, model, engine) {
  const result = pairResult(doc, model, engine);
  if (result) return result.confirmed ? "Confirmed" : "Exploratory";
  const current = doc.current;
  if (doc.live && current?.model === model && current.engine === engine) return "Running";
  const attempts = (doc.results || []).filter(r => r.model === model && r.engine === engine);
  if (attempts.some(r => r.error)) return "Failed";
  if (attempts.length) return "Incomplete";
  return (doc.plan?.cells || []).some(r => r.model === model && r.engine === engine)
    ? "Untested" : "Not planned";
}

export const METRICS = {
  out_tok_s: { label: "Generation", unit: "tok/s", digits: 0 },
  kv_cache_pct: { label: "KV cache", unit: "%", digits: 1 },
  gpu_power_w: { label: "GPU power", unit: "W", digits: 0 },
  tpot_p95_ms: { label: "TPOT p95", unit: "ms", digits: 1 },
};

export function metricScale(doc, models, engines, key, withinModel = false) {
  const values = models.flatMap(model => engines.map(engine => pairResult(doc, model, engine)?.[key]))
    .filter(v => v != null && Number.isFinite(Number(v))).map(Number);
  const maximum = Math.max(0, ...values);
  return {
    maximum,
    fraction(model, value) {
      if (value == null || !Number.isFinite(Number(value))) return 0;
      const denominator = withinModel
        ? Math.max(0, ...engines.map(e => Number(pairResult(doc, model, e)?.[key]) || 0))
        : maximum;
      return denominator > 0 ? Math.max(0, Math.min(1, Number(value) / denominator)) : 0;
    },
  };
}

export function methodFromWorkload(value) {
  return value.startsWith("persona:headline_") ? "saturation" : "capacity";
}

export function rooflineRequest(scope, options = {}) {
  if (!scope.models?.length || !scope.engines?.length) throw new Error('Select at least one model and engine.');
  for (const values of [scope.shapes?.max_num_seqs, scope.shapes?.output_tokens]) {
    if (!values?.length || values.some(v => !Number.isInteger(v) || v <= 0)) throw new Error('Batch width and output lengths must be positive whole numbers.');
  }
  const recovery = options.recovery || 'resume';
  if (!['resume','retry','redo','fresh'].includes(recovery)) throw new Error('Choose a supported recovery mode.');
  const priority = options.priority || [];
  if (priority.some(m => !scope.models.includes(m))) throw new Error('Priority models must be included in the selected model scope.');
  const recoveryEngines = options.engine ? [options.engine] : scope.engines;
  if (recoveryEngines.some(e => !scope.engines.includes(e))) throw new Error('Recovery engines must be included in the search.');
  return { workload: { kind: 'roofline', spec: {
    models: scope.mode === 'manual' ? scope.models : null,
    ...scope.limits, cached_only: scope.mode === 'cached', engines: scope.engines,
    max_num_seqs: scope.shapes.max_num_seqs, output_tokens: scope.shapes.output_tokens,
    input_tokens: scope.inputTokens, confirm_winners: scope.confirm,
    allow_cross_domain_tp: scope.crossDomain, spectrum: options.spectrum !== false,
    diverse: options.diverse !== false, kt_scale: !!options.ktScale,
    priority_models: priority.length ? priority : null, resume: recovery !== 'fresh',
    retry_engines: recovery === 'retry' ? recoveryEngines : null,
    redo_engines: recovery === 'redo' ? recoveryEngines : null,
  } }, new_run: recovery === 'fresh' };
}
