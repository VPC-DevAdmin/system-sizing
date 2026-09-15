import { ComposedChart, Area, Line, XAxis, YAxis, ReferenceArea, ReferenceLine, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import type { RawCohort } from '@/lib/smbOnPrem';
import { dedupeSort, filterOutliers, smooth, poolAxis } from './curveHelpers';

interface Props {
  cohort: RawCohort;
  height?: number;
}

export default function TtftCurveChart({ cohort, height = 300 }: Props) {
  const sorted = dedupeSort(cohort.curve);
  const raw = sorted.map(c => ({
    x: c.pool_size,
    p50: c.ttft_p50_ms / 1000,
    p95: c.ttft_p95_ms / 1000,
  }));
  const cleaned = filterOutliers(raw);

  const trendP50 = smooth(cleaned.map(p => ({ x: p.x, y: p.p50 })));
  const trendP95 = smooth(cleaned.map(p => ({ x: p.x, y: p.p95 })));

  const { domain: xDomain, ticks: xTicks } = poolAxis(cleaned.map(p => p.x));

  // Combined band data for the typical-range area
  const bandData = trendP50.map((t, i) => ({
    x: t.x,
    range: [t.y, trendP95[i]?.y ?? t.y] as [number, number],
  }));

  // Two-preset Y-axis: 0–12s when the data fits comfortably, otherwise 0–22s.
  // Switching (rather than auto-floating) keeps cohorts visually comparable
  // while still leaving room for the legible scale on smaller cohorts.
  const dataMaxS = Math.max(
    ...trendP95.map(p => p.y),
    ...cleaned.map(p => p.p95),
    0,
  );
  const useSmallScale = dataMaxS <= 8;
  const yMax = useSmallScale ? 12 : 22;
  const yTicks = useSmallScale ? [0, 3, 6, 9, 12] : [0, 5, 10, 15, 20];

  // Band thresholds (seconds) for TTFT
  const greenS = 10;
  const amberS = 20;

  const trendMap = new Map<number, number>();
  trendP50.forEach(t => trendMap.set(t.x, t.y));

  return (
    <div className="space-y-2">
      <div style={{ height }} className="relative">
      <ResponsiveContainer>
        <ComposedChart margin={{ top: 12, right: 8, left: 12, bottom: 56 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          {/* Quality bands */}
          <ReferenceArea y1={0} y2={Math.min(greenS, yMax)} fill="hsl(var(--chart-emerald) / 0.10)" />
          {yMax > greenS && (
            <ReferenceArea y1={greenS} y2={Math.min(amberS, yMax)} fill="hsl(var(--chart-amber) / 0.10)" />
          )}
          {yMax > amberS && (
            <ReferenceArea y1={amberS} y2={yMax} fill="hsl(var(--chart-rose) / 0.10)" />
          )}
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
            ticks={yTicks}
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => `${v.toFixed(v < 1 ? 1 : 0)}`}
            label={{ value: 'Seconds', angle: -90, position: 'insideLeft', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 12 }}
            labelFormatter={(v) => `${v} concurrent users`}
            formatter={(v: number, n: string) => [`${Number(v).toFixed(2)} s`, n]}
          />

          {/* Typical first-token response time band (p50 → p95) */}
          <Area
            data={bandData}
            dataKey="range"
            type="monotone"
            name="Typical first token response time (p50–p95)"
            legendType="none"
            stroke="none"
            fill="hsl(var(--chart-blue))"
            fillOpacity={0.12}
            isAnimationActive={false}
          />

          {/* Connector dotted lines: raw → trend at each measurement */}
          {cleaned.map(p => {
            const t = trendMap.get(p.x);
            if (t == null) return null;
            return (
              <ReferenceLine
                key={`c50-${p.x}`}
                segment={[{ x: p.x, y: p.p50 }, { x: p.x, y: t }]}
                stroke="hsl(var(--chart-blue))"
                strokeDasharray="2 2"
                strokeOpacity={0.55}
                ifOverflow="extendDomain"
              />
            );
          })}

          {/* Threshold reference lines */}
          <ReferenceLine
            y={greenS}
            stroke="hsl(var(--chart-emerald))"
            strokeDasharray="3 3"
            label={{ value: 'Comfortable ≤ 10s', position: 'insideBottomRight', fontSize: 10, fill: 'hsl(var(--chart-emerald))' }}
          />
          <ReferenceLine
            y={amberS}
            stroke="hsl(var(--chart-rose))"
            strokeDasharray="3 3"
            label={{ value: 'Slow > 20s', position: 'insideTopRight', fontSize: 10, fill: 'hsl(var(--chart-rose))' }}
          />

          {/* Smoothed trendlines */}
          <Line data={trendP50} dataKey="y" name="Typical (p50)" type="monotone"
            stroke="hsl(var(--chart-blue))" strokeWidth={2.5} dot={false} isAnimationActive />
          <Line data={trendP95} dataKey="y" name="Worst 5% (p95)" type="monotone"
            stroke="hsl(var(--chart-blue))" strokeWidth={1.75} strokeDasharray="6 4" dot={false} isAnimationActive />

          {/* Raw measurement dots (no connecting stroke) */}
          <Line data={cleaned} dataKey="p50" name="Typical (p50) measured" legendType="none"
            stroke="transparent" dot={{ r: 3, fill: 'hsl(var(--chart-blue))', strokeWidth: 0 }} activeDot={{ r: 4 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
        <div className="absolute bottom-1 right-2 text-[11px] leading-[14px] pointer-events-none flex flex-col items-end gap-0.5">
          <div className="flex items-center gap-1.5">
            <svg width="18" height="2"><line x1="0" y1="1" x2="18" y2="1" stroke="hsl(var(--chart-blue))" strokeWidth="2.5" /></svg>
            <span>Typical (p50)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <svg width="18" height="2"><line x1="0" y1="1" x2="18" y2="1" stroke="hsl(var(--chart-blue))" strokeWidth="1.75" strokeDasharray="4 3" /></svg>
            <span>Worst 5% (p95)</span>
          </div>
        </div>
      </div>
      <p className="text-[10px] leading-snug text-muted-foreground italic px-1">
        Latency thresholds based on Tan et al., "The Impact of Response Latency and Task Type on
        Human-LLM Interaction and Perception," CHI 2026, which found that ~10-second waits were
        perceived as more thoughtful and useful than 2-second responses, with degradation
        beginning around 20 seconds.
      </p>
    </div>
  );
}
