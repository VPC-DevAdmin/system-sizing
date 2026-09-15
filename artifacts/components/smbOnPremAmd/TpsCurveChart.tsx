import { ComposedChart, Line, XAxis, YAxis, ReferenceArea, ReferenceLine, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import type { RawCohort } from '@/lib/smbOnPrem';
import { dedupeSort, poolAxis } from './curveHelpers';

interface Props {
  cohort: RawCohort;
  height?: number;
}

export default function TpsCurveChart({ cohort, height = 300 }: Props) {
  const sorted = dedupeSort(cohort.curve);
  const raw = sorted
    .filter(c => c.ttft_p95_ms > 0 || c.tpot_p50_ms > 0)
    .map(c => ({
      x: c.pool_size,
      // When TTFT was recorded but TPOT was not (throughput collapsed below
      // measurable), fall back to 1 tok/s so the curve still extends to
      // match the TTFT chart's x-range.
      p50: c.tpot_p50_ms > 0 ? 1000 / c.tpot_p50_ms : 1,
    }))
    .filter(p => p.p50 <= 50);

  // Plot measured p50 directly so sharp knees (e.g. analyst team between
  // 16 and 32 users) aren't flattened by smoothing.
  const trendP50 = raw.map(p => ({ x: p.x, y: p.p50 }));

  const { domain: xDomain, ticks: xTicks } = poolAxis(raw.map(p => p.x));

  const dataMax = Math.max(...raw.map(p => p.p50), 1);
  // Round up to a clean 5-tok/s step so axis ticks read nicely
  const yMax = Math.max(5, Math.ceil((dataMax * 1.2) / 5) * 5);

  // Band thresholds (tok/s/user) — higher is better, so bands flip
  const greenTps = 10;
  const amberTps = 4;

  const trendMap = new Map<number, number>();
  trendP50.forEach(t => trendMap.set(t.x, t.y));

  return (
    <div style={{ height }} className="relative">
      <ResponsiveContainer>
        <ComposedChart margin={{ top: 12, right: 8, left: 12, bottom: 48 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          {/* Quality bands (higher tok/s = better) */}
          {yMax > greenTps && (
            <ReferenceArea y1={greenTps} y2={yMax} fill="hsl(var(--chart-emerald) / 0.10)" />
          )}
          <ReferenceArea y1={amberTps} y2={Math.min(greenTps, yMax)} fill="hsl(var(--chart-amber) / 0.10)" />
          <ReferenceArea y1={0} y2={Math.min(amberTps, yMax)} fill="hsl(var(--chart-rose) / 0.10)" />
          <XAxis
            type="number"
            dataKey="x"
            scale="log"
            domain={xDomain}
            ticks={xTicks}
            tickFormatter={(v) => String(v)}
            tick={{ fontSize: 11 }}
            label={{ value: 'Active users', position: 'insideBottom', offset: -16, fontSize: 11 }}
            allowDataOverflow={false}
          />
          <YAxis
            domain={[0, yMax]}
            allowDataOverflow
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => String(Math.round(v))}
            label={{ value: 'Tokens per second per user', angle: -90, position: 'insideLeft', offset: -4, fontSize: 11 }}
            width={68}
          />
          <Tooltip
            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 12 }}
            labelFormatter={(v) => `${v} concurrent users`}
            formatter={(v: number, n: string) => [`${Number(v).toFixed(1)} tok/s`, n]}
          />

          {/* Connectors */}
          {raw.map(p => {
            const t = trendMap.get(p.x);
            if (t == null) return null;
            return (
              <ReferenceLine
                key={`c50-${p.x}`}
                segment={[{ x: p.x, y: p.p50 }, { x: p.x, y: t }]}
                stroke="hsl(var(--primary))"
                strokeDasharray="2 2"
                strokeOpacity={0.55}
                ifOverflow="extendDomain"
              />
            );
          })}

          {/* Threshold reference lines, labels inside their bands */}
          <ReferenceLine y={10} stroke="hsl(var(--chart-emerald))" strokeDasharray="3 3"
            label={{ value: 'Conversational', position: 'insideTopRight', fontSize: 10, fill: 'hsl(var(--chart-emerald))' }} />
          <ReferenceLine y={amberTps} stroke="hsl(var(--chart-amber))" strokeDasharray="3 3"
            label={{ value: 'Reading speed (4 tok/s)', position: 'insideBottomRight', fontSize: 10, fill: 'hsl(var(--chart-amber))' }} />

          {/* Trendlines */}
          <Line data={trendP50} dataKey="y" name="Typical (p50)" type="monotone"
            stroke="hsl(var(--primary))" strokeWidth={2.5} dot={false} connectNulls isAnimationActive />

          {/* Raw dots */}
          <Line data={raw} dataKey="p50" name="Typical (p50) measured" legendType="none"
            stroke="transparent" dot={{ r: 3, fill: 'hsl(var(--primary))', strokeWidth: 0 }} activeDot={{ r: 4 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="absolute bottom-1 right-2 text-[11px] leading-[14px] pointer-events-none flex items-center gap-1.5">
        <svg width="18" height="2"><line x1="0" y1="1" x2="18" y2="1" stroke="hsl(var(--primary))" strokeWidth="2.5" /></svg>
        <span>Typical (p50)</span>
      </div>
    </div>
  );
}
