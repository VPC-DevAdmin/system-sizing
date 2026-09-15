import { COHORTS } from '@/data/smbOnPremConfig';
import { useSmbOnPrem } from '@/contexts/SmbOnPremContext';
import { poolSizeAtTps } from '@/lib/smbOnPrem';

export default function CohortRecommendation({ cohortId }: { cohortId: string }) {
  const c = COHORTS.find(x => x.id === cohortId);
  const { cohortById } = useSmbOnPrem();
  const data = cohortById(cohortId);
  if (!c || !data) return null;

  const comfortable = poolSizeAtTps(data, 10);
  const acceptable = poolSizeAtTps(data, 4);

  // Find tok/s at the comfortable threshold (interpolated from curve)
  const sortedCurve = [...data.curve]
    .filter(p => p.tpot_p50_ms > 0 && p.pool_size > 0)
    .sort((a, b) => a.pool_size - b.pool_size);
  const tpsAt = (pool: number): number | null => {
    if (sortedCurve.length === 0 || pool <= 0) return null;
    let prev = sortedCurve[0];
    for (const p of sortedCurve) {
      if (p.pool_size >= pool) {
        if (p.pool_size === pool || prev === p) return 1000 / p.tpot_p50_ms;
        const f = (pool - prev.pool_size) / (p.pool_size - prev.pool_size);
        const tpsPrev = 1000 / prev.tpot_p50_ms;
        const tpsCur = 1000 / p.tpot_p50_ms;
        return tpsPrev + (tpsCur - tpsPrev) * f;
      }
      prev = p;
    }
    return 1000 / prev.tpot_p50_ms;
  };

  const tpsComfortable = tpsAt(comfortable);
  const failPool = data.fail_pool_size ?? sortedCurve[sortedCurve.length - 1]?.pool_size ?? acceptable;

  const summary = (() => {
    const parts: string[] = [];
    if (comfortable > 0) {
      parts.push(`Up to ${comfortable} active users will see a responsive experience of 2x reading speed (>10 tokens per second).`);
    }
    if (acceptable > comfortable) {
      parts.push(`Between ${comfortable} and ${acceptable} active users, users will see output faster than reading speed (>4 tokens per second).`);
    }
    if (failPool && failPool > acceptable) {
      parts.push(`Past ${failPool}, users will start to wait on the model and experience degrades.`);
    }
    return parts.join(' ');
  })();

  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm space-y-3 animate-fade-in">
      <div className="text-sm font-semibold">{c.name}</div>
      <p className="text-sm leading-relaxed">{summary}</p>
      <div>
        <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">What gives this team more room</div>
        <p className="text-sm leading-relaxed">{c.recommendation}</p>
      </div>
    </div>
  );
}
