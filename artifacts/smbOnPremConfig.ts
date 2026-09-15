// Static config for /demos/smb-on-prem — everything not in intel-sizing.json

export type LoadIntensity = 'light' | 'medium' | 'dark';

export interface PersonaCopy {
  id: string;            // matches JSON persona id
  name: string;
  description: string;
  example: string;
  inputLoad: LoadIntensity;
  outputLoad: LoadIntensity;
  // visual accent — used in cohort persona-mix bars
  color: string;         // hsl var fallback
  avgInputTokens: number;   // TODO replace with real per-persona measured averages
  avgOutputTokens: number;
}

export const PERSONAS: PersonaCopy[] = [
  { id: 'quick_lookup',  name: 'Quick lookup',  description: 'Frontline support, sales, inventory checks', example: "What's our return policy on opened software?", inputLoad: 'light', outputLoad: 'light', color: 'hsl(var(--chart-blue))',    avgInputTokens: 350, avgOutputTokens: 60 },
  { id: 'conversational',name: 'Conversational',description: 'Tutoring, coaching, customer dialogue',     example: 'Walk me through how to handle this objection…',         inputLoad: 'medium', outputLoad: 'medium', color: 'hsl(var(--chart-emerald))', avgInputTokens: 180, avgOutputTokens: 220 },
  { id: 'writer',        name: 'Drafter',       description: 'Email and short-form writing',              example: 'Draft a follow-up email to the prospect about pricing.', inputLoad: 'light', outputLoad: 'medium', color: 'hsl(var(--chart-amber))',  avgInputTokens: 500, avgOutputTokens: 350 },
  { id: 'document_qa',   name: 'Document Q&A',  description: 'Legal, finance, research over long docs',   example: 'Summarize the indemnification clauses in this contract.', inputLoad: 'dark',  outputLoad: 'medium', color: 'hsl(var(--chart-rose))',   avgInputTokens: 3500, avgOutputTokens: 280 },
  { id: 'code_assist',   name: 'Code assistance',description: 'Engineers pair-programming',               example: 'Why is this function returning null on edge cases?',     inputLoad: 'medium', outputLoad: 'dark',  color: 'hsl(262 83% 58%)',         avgInputTokens: 1200, avgOutputTokens: 450 },
  { id: 'long_form_generator', name: 'Long-form generator', description: 'Long-form drafts, articles, and reports', example: 'Draft a 1,500-word article on Q3 product launches.', inputLoad: 'light', outputLoad: 'dark', color: 'hsl(190 90% 45%)', avgInputTokens: 200, avgOutputTokens: 3000 },
];

export interface CohortCopy {
  id: string;            // matches JSON cohort id
  name: string;          // override JSON 'name'
  description: string;
  densityCoefficient: number;
  defaultMixPct: number;
  summaryLine: string;
  recommendation: string;
  status: 'comfortable' | 'tight' | 'at_capacity';
}

export const COHORTS: CohortCopy[] = [
  { id: 'chat_heavy',           name: 'Customer support team', description: 'Quick-lookup heavy. High request rate per active user.', densityCoefficient: 1.4, defaultMixPct: 15, status: 'tight',
    summaryLine: 'Up to 94 concurrent users get a responsive experience, with typical response speed around 6 tokens per second, near reading pace. Between 94 and 96 concurrent, the system stays operational but typical users start waiting on the model. Past 96, the experience degrades to failure.',
    recommendation: 'Output volume is what limits this team. The Higher Capacity on R470 profile or splitting the team onto a Two Socket R770 gives the most room.' },
  { id: 'general_knowledge',    name: 'General knowledge work', description: 'Mixed personas representing the realistic baseline office workload.', densityCoefficient: 1.0, defaultMixPct: 35, status: 'comfortable',
    summaryLine: 'Up to 44 concurrent users get a responsive experience, with typical response speed around 15 tokens per second, well above reading pace. Between 44 and 64 concurrent, the system stays operational but typical users start waiting on the model. Past 128, the experience degrades to failure.',
    recommendation: 'This is the typical enterprise case and the test scenario covers the 16 active users comfortably. To scale beyond this, a second CPU socket roughly doubles sustained capacity.' },
  { id: 'writer_dominant',      name: 'Marketing / content team', description: 'Drafter-heavy. Output-bound work pressures the decode pipeline.', densityCoefficient: 1.0, defaultMixPct: 10, status: 'tight',
    summaryLine: 'Up to 92 concurrent users get a responsive experience, with typical response speed around 11 tokens per second, comfortably above reading pace. Between 92 and 128 concurrent, the system stays operational but typical users start waiting on the model. Past 128, the experience degrades to failure.',
    recommendation: 'Similar pattern to support: a top-bin SKU helps most. Consider the Higher Capacity on R470 profile if writers run hot all day.' },
  { id: 'software_engineering', name: 'Software engineering team', description: 'Code-assist heavy. Long generated outputs sustain decode pressure.', densityCoefficient: 1.2, defaultMixPct: 15, status: 'at_capacity',
    summaryLine: 'Up to 52 concurrent users get a responsive experience, with typical response speed around 6 tokens per second, near reading pace. Between 52 and 56 concurrent, the system stays operational but typical users start waiting on the model. Past 56, the experience degrades to failure.',
    recommendation: 'Tightest envelope of the five. For >50 active engineers, plan a second server or step up to the Two Socket R770 profile.' },
  { id: 'analyst_team',         name: 'Analyst team', description: 'Document-QA + summarization. Input-heavy, fewer requests per active user.', densityCoefficient: 0.7, defaultMixPct: 25, status: 'comfortable',
    summaryLine: 'Up to 72 concurrent users get a responsive experience, with typical response speed around 12 tokens per second, well above reading pace. Between 72 and 128 concurrent, the system stays operational but typical users start waiting on the model. Past 128, the experience degrades to failure.',
    recommendation: 'Output heavy workload, additional compute is necessary to increase capacity of this team since it is decode limited.' },
];

// ----- Capability matrix (Section 3) -----

export interface CapabilityScore { capability: string; local: number; cloud: number; }
export interface CapabilityGroup { group: string; items: CapabilityScore[]; }

export const CAPABILITY_GROUPS: CapabilityGroup[] = [
  { group: 'Knowledge', items: [
    { capability: 'MMLU-Pro',    local: 78.4, cloud: 79.8 },
    { capability: 'MMLU-Redux',  local: 89.3, cloud: 91.3 },
    { capability: 'GPQA',        local: 70.4, cloud: 66.9 },
    { capability: 'SuperGPQA',   local: 53.4, cloud: 51.0 },
  ]},
  { group: 'Reasoning', items: [
    { capability: 'AIME25',      local: 61.3, cloud: 26.7 },
    { capability: 'ZebraLogic',  local: 90.0, cloud: 52.6 },
    { capability: 'LiveBench',   local: 69.0, cloud: 63.7 },
  ]},
  { group: 'Coding', items: [
    { capability: 'LiveCodeBench v6', local: 43.2, cloud: 35.8 },
    { capability: 'MultiPL-E',        local: 83.8, cloud: 82.7 },
  ]},
  { group: 'Instruction & Writing', items: [
    { capability: 'IFEval',             local: 84.7, cloud: 83.9 },
    { capability: 'Arena-Hard v2',      local: 69.0, cloud: 61.9 },
    { capability: 'Creative Writing v3',local: 86.0, cloud: 84.9 },
    { capability: 'WritingBench',       local: 85.5, cloud: 75.5 },
  ]},
  { group: 'Tool use / agent', items: [
    { capability: 'BFCL-v3', local: 65.1, cloud: 66.5 },
  ]},
];

export interface BinaryCapability { capability: string; local: boolean | string; cloud: boolean | string; }
export const BINARY_CAPABILITIES: BinaryCapability[] = [
  { capability: 'Tool use / function calling', local: true,                 cloud: true },
  { capability: 'Multimodal (vision)',         local: false,                cloud: true },
  { capability: 'Long context',                local: '262K native (8K tested)', cloud: '128K' },
  { capability: 'Languages',                   local: '119',                cloud: 'broad' },
];

export const CAPABILITY_CAPTION =
  'All scores reported by Qwen using a consistent evaluation methodology across both models. Source: Qwen3-30B-A3B-Instruct-2507 model card, July 2025. Higher is better on all numeric benchmarks (0–100 scale).';

// ----- Sample outputs (Section 5) — placeholders flagged for replacement -----

export interface SampleOutput { id: string; label: string; prompt: string; localResponse: string; cloudResponse: string; }

export const SAMPLE_OUTPUTS: SampleOutput[] = [
  { id: 'quick_lookup', label: 'Quick lookup',
    prompt: "What's the difference between an LLC and an S-corp for a 20-person consulting firm?",
    localResponse: '[TODO real sample] An LLC is a legal entity that provides liability protection with flexible tax treatment. An S-corp is a tax election available to LLCs and corporations that lets profits flow through to owners while allowing reasonable salary + distribution splits to reduce self-employment tax. For a 20-person consulting firm earning over ~$80k per partner, the S-corp election typically reduces overall tax liability. Talk to a CPA before electing.',
    cloudResponse: '[TODO real sample] An LLC offers liability protection and flexible taxation. An S-corp is a tax classification (not an entity) that lets owners pay themselves a salary and take remaining profits as distributions, avoiding self-employment tax on the distribution portion. For 20-person consulting firms with substantial profits, S-corp election often saves 5-10% in taxes. Consider the additional payroll and compliance overhead before electing.' },
  { id: 'code_assist', label: 'Code assist',
    prompt: 'This Python function is returning None when it should return a list. What\'s wrong?\n\ndef get_active(users):\n    [u for u in users if u.active]',
    localResponse: '[TODO real sample] The list comprehension is evaluated but never returned. Add a `return` statement:\n\n```python\ndef get_active(users):\n    return [u for u in users if u.active]\n```',
    cloudResponse: '[TODO real sample] You\'re missing the `return` keyword. The list comprehension produces a value but it\'s discarded. Fix:\n\n```python\ndef get_active(users):\n    return [u for u in users if u.active]\n```' },
  { id: 'doc_qa', label: 'Document Q&A',
    prompt: '[Contract excerpt: 14.1 Either party may terminate this Agreement upon thirty (30) days written notice…] Summarize the termination clauses.',
    localResponse: '[TODO real sample] Either party may terminate the Agreement with 30 days written notice. Termination for cause is allowed if a material breach is not cured within 15 days of notice. Surviving obligations include confidentiality, IP assignment, and payment of fees accrued before termination.',
    cloudResponse: '[TODO real sample] Termination provisions: (1) either party may terminate without cause on 30 days written notice; (2) either party may terminate for cause if a material breach goes uncured for 15 days after written notice; (3) confidentiality, IP, and payment obligations survive termination.' },
];

// ----- Test bench / software stack (Section 6.5) -----

export const TEST_BENCH: Array<[string, string]> = [
  ['Server',           'Dell R470, BIOS 1.6.4'],
  ['CPU',              'Intel Xeon 6 6761P (Granite Rapids)'],
  ['Cores / threads',  '64 / 128'],
  ['AMX support',      'Yes — BF16, INT8, FP16'],
  ['Memory',           '1 TB DDR5 (1,007 GiB usable)'],
  ['NUMA',             'Single node, single socket'],
  ['GPU',              'None'],
  ['OS',               'Ubuntu 24.04'],
];

export const SOFTWARE_STACK: Array<[string, string]> = [
  ['Inference engine',       'SGLang on CPU'],
  ['Attention backend',      'intel_amx'],
  ['Model',                  'Qwen3-30B-A3B-Instruct (FP8)'],
  ['KV cache',               '64 GB allocated'],
  ['Max model length',       '8,192 tokens'],
  ['Max total tokens',       '131,072'],
  ['CPU bind',               'cores 0–63'],
  ['Static memory fraction', '0.85'],
];

export const WORKLOAD_GENERATOR_PARAGRAPH =
  "For each team, a pool of independent virtual users was generated to match the team's persona mix. Each user behaved according to its persona profile: token counts, turn counts, and read/think times sampled from distributions calibrated to typical enterprise variability. Users issued requests to the server in their natural cycle: submit, wait for response, read, think, submit again. After the server reached steady state, we measured time to first token (TTFT) and time per output token (TPOT) for every request in a measurement window, then aggregated into the percentiles reported throughout this page. When a user completed its sessions, it was replaced with a fresh user to keep the pool at the target size. For each team, we repeated the test at progressively higher active-user counts to map how performance changed as load increased.";

export const WHAT_WE_DIDNT_TEST: string[] = [
  'Multi-node deployments. Every measurement is from one server.',
  'GPU-accelerated variants of the same model.',
  'Production retrieval pipelines (RAG) layered on top of inference.',
  'Sustained 24/7 load profiles beyond a single ten-minute measurement window per pool size.',
  'Quantization formats other than FP8 (BF16, INT8 may behave differently).',
];

// ----- TCO defaults + comparator presets (Section 6) -----

export interface ComparatorPricing { name: string; inputPerMtok: number; outputPerMtok: number; }

export const COMPARATORS: ComparatorPricing[] = [
  { name: 'GPT-4o',          inputPerMtok: 2.50, outputPerMtok: 10.00 },
  { name: 'GPT-4o-mini',     inputPerMtok: 0.15, outputPerMtok: 0.60 },
  { name: 'GPT-5.4-mini',    inputPerMtok: 0.75, outputPerMtok: 4.50 },
  { name: 'Gemini 2.5 Flash',inputPerMtok: 0.30, outputPerMtok: 2.50 },
];

export const TCO_DEFAULTS = {
  serverCost: 30000,
  powerW: 250,
  powerKwhCost: 0.12,
  coolingOverheadPct: 35,
  adminYearlyCost: 8000,
  rackMonthlyCost: 0,
  comparatorIndex: 0,        // GPT-4o
  blendedMidPct: 70,
  businessHoursPerMonth: 168,
  requestsPerInflightHour: 8,
};

// ----- Funnel defaults -----

export const FUNNEL_DEFAULTS = {
  orgSize: 1000,
  adoptionPct: 65,
  hourlyActivePct: 22,
  inFlightDensityPct: 11,
};

export const FUNNEL_TOOLTIPS = {
  adoption: "Gallup's Q3 2025 Workforce survey found 76% of technology workers, 58% of finance workers, and 57% of professional services workers use AI at work at least a few times per year. We use 65% as the weighted average across these enterprise knowledge-work sectors.",
  dailyActive: "Gallup's Q3 2025 Workforce survey found that 10% of U.S. employees use AI daily, representing 22% of those who use AI at any frequency.",
  peakHour: "Microsoft's 2025 Work Trend Index reports knowledge workers spend roughly 4-5 hours per week with AI tools. Spread across a 5-day workweek and 8-hour workday, that's about 11% of any single hour.",
};

// ----- Configuration profiles (Section 8) -----

export interface ConfigProfile {
  id: string;
  name: string;
  bestFit: string;
  cpu: string;
  memory: string;
  sockets: number;
  costMin: number;
  costMax: number;
  capacityEnvelope: string;
  isTested?: boolean;
  // per-cohort acceptable / fail thresholds (multipliers of the tested baseline)
  capacityMultiplier: number;
}

export const CONFIG_PROFILES: ConfigProfile[] = [
  { id: 'entry',     name: 'Entry',                      bestFit: '400–600 users',      cpu: 'Xeon 6 24–32C',           memory: '256 GB DDR5',  sockets: 1, costMin: 15000, costMax: 18000, capacityEnvelope: '0.5× standard',   capacityMultiplier: 0.55 },
  { id: 'standard',  name: 'Standard (tested)',          bestFit: '800–1,200 users',    cpu: 'Xeon 6 6761P (64C)',      memory: '1 TB DDR5',    sockets: 1, costMin: 30000, costMax: 30000, capacityEnvelope: '16–32 Active',     capacityMultiplier: 1.0, isTested: true },
  { id: 'capacity',  name: 'Higher Capacity on R470',    bestFit: '1,000–1,400 users',  cpu: 'Xeon 6 top-bin SKU',      memory: '1.5 TB DDR5',  sockets: 1, costMin: 40000, costMax: 50000, capacityEnvelope: '1.2× standard',    capacityMultiplier: 1.2 },
  { id: 'dual',      name: 'Two Socket R770',            bestFit: '1,600–2,000 users',  cpu: 'Dual Xeon 6',             memory: '1–2 TB DDR5',  sockets: 2, costMin: 50000, costMax: 60000, capacityEnvelope: '~1.8× standard',   capacityMultiplier: 1.8 },
];

export interface ConfigLever { rank: number; name: string; description: string; impact: string; }

export const CONFIG_LEVERS: ConfigLever[] = [
  { rank: 1, name: 'CPU frequency',  description: 'Decode-bound workloads spend most of their time waiting on per-core compute. A higher base/turbo frequency directly lowers TPOT and lifts the SLA ceiling.', impact: '+15–25% capacity' },
  { rank: 2, name: 'Socket count',   description: 'A second socket nearly doubles raw throughput, but adds NUMA cross-traffic and cost. The right answer when one server runs out of room.',                  impact: '~1.8× capacity, +40% cost' },
  { rank: 3, name: 'Memory capacity',description: 'Bigger KV cache means more concurrent users at long context. The tested server already sits near the ceiling for this model.',                              impact: 'WORKLOAD-DEPENDENT\nINCREASE FOR HEAVIER PERSONAS' },
  { rank: 4, name: 'Memory speed',   description: 'For this MoE model on Granite Rapids, memory bandwidth is not the binding constraint. DDR5-5600 vs DDR5-6400 is a wash.',                                  impact: 'Minimal impact for this workload' },
];

export const WHEN_NOT_RIGHT: string[] = [
  'You need full multimodal vision. Pair this server with a vision-capable cloud endpoint or a GPU node for image work.',
  'You need <1s end-to-end response with very long outputs. GPU systems still win on raw decode latency.',
  'Your usage spikes 10× from baseline regularly. Burst capacity is what cloud APIs are good at; keep them in the loop.',
  'You\'re under 50 concurrent users total. A smaller Entry-tier server is the better economic answer.',
  'You\'re in a regulated environment that requires specific certifications cloud providers already hold and you don\'t want to inherit them yourself.',
];

export const DELL_CTA_URL =
  'https://www.dell.com/en-us/shop/dell-poweredge-servers/new-poweredge-r470-rack-server/spd/poweredge-r470/pe_r470_tm_vi_vp_sb';
export const DELL_LOANER_URL = 'https://www.delldemodepot.com/';
