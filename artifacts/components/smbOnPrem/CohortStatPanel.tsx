import type { RawCohort } from '@/lib/smbOnPrem';
import { AnimatedNumber } from './AnimatedNumber';

export default function CohortStatPanel({ cohort }: { cohort: RawCohort }) {
  const b = cohort.bottleneck_evidence;
  if (!b) return null;
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <Stat label="Memory cache used at ceiling" value={`${b.kv_cache_used_pct.toFixed(1)}%`} />
      <Stat label="Memory bandwidth in use"      value={`${b.memory_bw_total_gb_s.toFixed(1)} GB/s`} />
      <Stat label="CPU frequency under load"     value={`Avg ${b.effective_freq_ghz_mean.toFixed(2)} GHz`} sub={`Floor ${b.effective_freq_ghz_min.toFixed(2)} GHz`} />
      <Stat label="Requests over SLA"            value={`${(b.tpot_violation_rate * 100).toFixed(1)}%`} />
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border bg-card p-3 shadow-sm">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

// re-export so PerCohortTabs can reuse animation if needed
export { AnimatedNumber };
