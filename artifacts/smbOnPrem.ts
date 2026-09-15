import {
  PERSONAS, COHORTS, CONFIG_PROFILES, COMPARATORS, TCO_DEFAULTS,
  type ComparatorPricing, type ConfigProfile,
} from '@/data/smbOnPremConfig';

// JSON typings (subset we read)
export interface CurvePoint {
  pool_size: number;
  ttft_p50_ms: number;
  ttft_p95_ms: number;
  tpot_p50_ms: number;
  tpot_p95_ms: number;
  kv_cache_used_pct: number;
  status: string;
  target_status?: string;
}

export interface BottleneckEvidence {
  kv_cache_used_pct: number;
  memory_bw_total_gb_s: number;
  effective_freq_ghz_mean: number;
  effective_freq_ghz_min: number;
  ttft_violation_rate: number;
  tpot_violation_rate: number;
}

export interface RawCohort {
  id: string;
  name: string;
  description: string;
  category: 'persona' | 'cohort';
  persona_weights: Record<string, number>;
  capacity_pool_size: number;
  soft_capacity_pool_size: number;
  fail_pool_size: number | null;
  deployment_band_shape: string;
  capacity_landing_zones: { fast: string; acceptable: string; degraded: string };
  curve: CurvePoint[];
  bottleneck_evidence: BottleneckEvidence | null;
  bottleneck?: string | null;
}

export interface SizingData {
  meta: { engine_config: { kv_cache_gb: number; max_model_len: number; max_total_tokens: number; cpu_bind: string } };
  cohorts: RawCohort[];
}

export function selectTeamCohorts(data: SizingData): RawCohort[] {
  return data.cohorts.filter(c => c.category === 'cohort');
}
export function selectPersonaCohorts(data: SizingData): RawCohort[] {
  return data.cohorts.filter(c => c.category === 'persona');
}
export function findCohortById(data: SizingData | null, id: string): RawCohort | undefined {
  return data?.cohorts.find(c => c.id === id);
}

// Derive the largest pool_size at which per-user throughput (tok/s) stays
// at or above the given threshold, using p50 TPOT from the measured curve.
// Linearly interpolates between the bracketing measurements.
export function poolSizeAtTps(cohort: RawCohort, minTps: number): number {
  const pts = cohort.curve
    .filter(c => c.tpot_p50_ms > 0 && c.pool_size > 0)
    .map(c => ({ pool: c.pool_size, tps: 1000 / c.tpot_p50_ms }))
    .sort((a, b) => a.pool - b.pool);
  if (pts.length === 0) return 0;
  if (pts[0].tps < minTps) return 0;
  let last = pts[0].pool;
  for (let i = 1; i < pts.length; i++) {
    const prev = pts[i - 1];
    const cur = pts[i];
    if (cur.tps >= minTps) { last = cur.pool; continue; }
    // crossing between prev (>=) and cur (<): interpolate in (pool, tps) space
    const span = prev.tps - cur.tps;
    const frac = span > 0 ? (prev.tps - minTps) / span : 0;
    return Math.max(0, Math.floor(prev.pool + (cur.pool - prev.pool) * frac));
  }
  return last;
}

// Funnel math
export interface FunnelInputs {
  orgSize: number;
  adoptionPct: number;
  hourlyActivePct: number;
  inFlightDensityPct: number;
}
export interface FunnelStages {
  total: number;
  adopted: number;
  hourlyActive: number;
  concurrent: number;
}
export function computeFunnel(i: FunnelInputs): FunnelStages {
  const adopted = i.orgSize * (i.adoptionPct / 100);
  const hourlyActive = adopted * (i.hourlyActivePct / 100);
  const concurrent = hourlyActive * (i.inFlightDensityPct / 100);
  return { total: i.orgSize, adopted, hourlyActive, concurrent };
}

// Demand vs capacity
export interface CohortDemand {
  cohortId: string;
  demand: number;
  green: number;
  yellow: number;
  red: number | null;
  exceeds: 'none' | 'green' | 'yellow' | 'red';
}
export function computeDemand(
  teamCohorts: RawCohort[],
  totalConcurrent: number,
  teamMix: Record<string, number>,
  densityFor: (id: string) => number,
): CohortDemand[] {
  return teamCohorts.map(c => {
    const pct = (teamMix[c.id] ?? 0) / 100;
    const demand = totalConcurrent * pct * densityFor(c.id);
    const green = c.capacity_pool_size;
    const yellow = c.soft_capacity_pool_size;
    const red = c.fail_pool_size;
    let exceeds: CohortDemand['exceeds'] = 'none';
    if (red != null && demand > red) exceeds = 'red';
    else if (demand > yellow) exceeds = 'yellow';
    else if (demand > green) exceeds = 'green';
    return { cohortId: c.id, demand, green, yellow, red, exceeds };
  });
}

// TCO
export interface TcoInputs {
  serverCost: number;
  powerW: number;
  powerKwhCost: number;
  coolingOverheadPct: number;
  adminYearlyCost: number;
  rackMonthlyCost: number;
  comparatorIndex: number;
  blendedMidPct: number;
  monthlyInputMtok: number;
  monthlyOutputMtok: number;
}
export function monthlyOpex(i: Pick<TcoInputs, 'powerW'|'powerKwhCost'|'coolingOverheadPct'|'adminYearlyCost'|'rackMonthlyCost'>) {
  const basePower = (i.powerW * 24 * 30 * i.powerKwhCost) / 1000;
  const cooling   = basePower * (i.coolingOverheadPct / 100);
  const admin     = i.adminYearlyCost / 12;
  return basePower + cooling + admin + i.rackMonthlyCost;
}
export function monthlyApiCost(input: number, output: number, p: ComparatorPricing) {
  return input * p.inputPerMtok + output * p.outputPerMtok;
}
export interface TcoSeries { month: number; onPrem: number; mid: number; blended: number; }
export function buildTcoSeries(i: TcoInputs, monthsMax = 36) {
  const opex = monthlyOpex(i);
  const mid = COMPARATORS[i.comparatorIndex] ?? COMPARATORS[0];
  const premium = COMPARATORS[0];
  const midRate = monthlyApiCost(i.monthlyInputMtok, i.monthlyOutputMtok, mid);
  const premRate = monthlyApiCost(i.monthlyInputMtok, i.monthlyOutputMtok, premium);
  const blendedRate = midRate * (i.blendedMidPct / 100) + premRate * (1 - i.blendedMidPct / 100);
  const series: TcoSeries[] = [];
  let breakevenMid: number | null = null;
  let breakevenBlended: number | null = null;
  for (let m = 0; m <= monthsMax; m++) {
    const onPrem = i.serverCost + opex * m;
    const midC = midRate * m;
    const blendedC = blendedRate * m;
    series.push({ month: m, onPrem, mid: midC, blended: blendedC });
    if (breakevenMid == null && m > 0 && onPrem < midC) breakevenMid = m;
    if (breakevenBlended == null && m > 0 && onPrem < blendedC) breakevenBlended = m;
  }
  return { series, midRate, blendedRate, opex, breakevenMid, breakevenBlended };
}

// Monthly token derivation
export function deriveMonthlyTokens(
  teamCohorts: RawCohort[],
  funnel: FunnelStages,
  teamMix: Record<string, number>,
  businessHoursPerMonth: number = TCO_DEFAULTS.businessHoursPerMonth,
  requestsPerInflightHour: number = TCO_DEFAULTS.requestsPerInflightHour,
) {
  const cohortAvg = teamCohorts.map(c => {
    let inSum = 0, outSum = 0, w = 0;
    for (const [pid, weight] of Object.entries(c.persona_weights)) {
      const persona = PERSONAS.find(p => p.id === pid);
      if (!persona) continue;
      inSum  += persona.avgInputTokens  * weight;
      outSum += persona.avgOutputTokens * weight;
      w += weight;
    }
    return { id: c.id, avgIn: w ? inSum / w : 0, avgOut: w ? outSum / w : 0 };
  });

  let weightedAvgIn = 0, weightedAvgOut = 0, totalPct = 0;
  for (const c of cohortAvg) {
    const pct = (teamMix[c.id] ?? 0);
    weightedAvgIn  += c.avgIn  * pct;
    weightedAvgOut += c.avgOut * pct;
    totalPct += pct;
  }
  if (totalPct > 0) { weightedAvgIn /= totalPct; weightedAvgOut /= totalPct; }

  const hours = funnel.hourlyActive * businessHoursPerMonth;
  const inFlightHours = hours * (funnel.hourlyActive > 0 ? funnel.concurrent / funnel.hourlyActive : 0);
  const monthlyRequests = inFlightHours * requestsPerInflightHour;
  const monthlyInputTokens  = monthlyRequests * weightedAvgIn;
  const monthlyOutputTokens = monthlyRequests * weightedAvgOut;

  return {
    monthlyRequests,
    monthlyInputMtok:  monthlyInputTokens  / 1_000_000,
    monthlyOutputMtok: monthlyOutputTokens / 1_000_000,
    weightedAvgIn,
    weightedAvgOut,
  };
}

// Sizing recommender
export function recommendProfile(
  teamCohorts: RawCohort[],
  totalConcurrent: number,
  teamMix: Record<string, number>,
  densityFor: (id: string) => number,
): ConfigProfile {
  const order = ['entry', 'standard', 'frequency', 'capacity', 'dual'];
  const sorted = order.map(id => CONFIG_PROFILES.find(p => p.id === id)!).filter(Boolean);
  for (const profile of sorted) {
    let passes = true;
    for (const c of teamCohorts) {
      const pct = (teamMix[c.id] ?? 0) / 100;
      const demand = totalConcurrent * pct * densityFor(c.id);
      const acceptable = c.soft_capacity_pool_size * profile.capacityMultiplier;
      if (demand > acceptable) { passes = false; break; }
    }
    if (passes) return profile;
  }
  return CONFIG_PROFILES.find(p => p.id === 'dual')!;
}

export function densityFor(id: string): number {
  return COHORTS.find(c => c.id === id)?.densityCoefficient ?? 1.0;
}

export function fmtUSD(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000)     return `$${(n / 1_000).toFixed(1)}K`;
  return `$${Math.round(n).toLocaleString()}`;
}

export function fmtCompactInt(n: number): string {
  return Math.round(n).toLocaleString();
}
