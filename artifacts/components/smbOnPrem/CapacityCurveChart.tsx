import { LineChart, Line, XAxis, YAxis, ReferenceArea, ReferenceLine, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import type { RawCohort } from '@/lib/smbOnPrem';

interface Props {
  cohort: RawCohort;
  greenMs?: number;
  yellowMs?: number;
  height?: number;
  compareCohort?: RawCohort | null;
}

export default function CapacityCurveChart({ cohort, greenMs = 200, yellowMs = 500, height = 300, compareCohort = null }: Props) {
  const data = cohort.curve.map(c => ({
    pool: c.pool_size,
    ttft: c.ttft_p95_ms,
    tpot: c.tpot_p95_ms,
    cmpTpot: compareCohort?.curve.find(x => x.pool_size === c.pool_size)?.tpot_p95_ms ?? null,
  }));
  const maxY = Math.max(...data.map(d => Math.max(d.ttft, d.tpot, d.cmpTpot ?? 0))) * 1.1;
  const yMax = Math.max(maxY, yellowMs * 1.6);

  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 36, right: 20, left: 10, bottom: 28 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          {/* SLA bands */}
          <ReferenceArea y1={0} y2={greenMs} fill="hsl(var(--chart-emerald) / 0.10)" />
          <ReferenceArea y1={greenMs} y2={yellowMs} fill="hsl(var(--chart-amber) / 0.10)" />
          <ReferenceArea y1={yellowMs} y2={yMax}    fill="hsl(var(--chart-rose) / 0.10)" />

          <XAxis dataKey="pool" tick={{ fontSize: 11 }} label={{ value: 'Active users', position: 'insideBottom', offset: -16, fontSize: 11 }} />
          <YAxis domain={[0, yMax]} tick={{ fontSize: 11 }} label={{ value: 'p95 latency (ms)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 12 }}
            formatter={(v: number, n: string) => [`${Math.round(v)} ms`, n]}
          />
          <Legend
            verticalAlign="top"
            align="center"
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, paddingBottom: 8 }}
          />

          {/* Markers */}
          <ReferenceLine x={cohort.capacity_pool_size}      stroke="hsl(var(--chart-emerald))" strokeDasharray="4 4" label={{ value: 'Comfortable', fontSize: 10, fill: 'hsl(var(--chart-emerald))', position: 'top' }} />
          <ReferenceLine x={cohort.soft_capacity_pool_size} stroke="hsl(var(--chart-amber))"   strokeDasharray="4 4" label={{ value: 'Acceptable',  fontSize: 10, fill: 'hsl(var(--chart-amber))',   position: 'top' }} />
          {cohort.fail_pool_size != null && (
            <ReferenceLine x={cohort.fail_pool_size} stroke="hsl(var(--chart-rose))" strokeDasharray="4 4" label={{ value: 'Failure', fontSize: 10, fill: 'hsl(var(--chart-rose))', position: 'top' }} />
          )}

          <Line type="monotone" dataKey="ttft" name="TTFT p95" stroke="hsl(var(--chart-blue))" strokeDasharray="5 3" dot={false} strokeWidth={2} isAnimationActive />
          <Line type="monotone" dataKey="tpot" name="TPOT p95" stroke="hsl(var(--primary))"     dot={false} strokeWidth={2.5} isAnimationActive />
          {compareCohort && (
            <Line type="monotone" dataKey="cmpTpot" name={`TPOT — ${compareCohort.name}`} stroke="hsl(var(--chart-amber))" dot={false} strokeWidth={2} isAnimationActive />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
