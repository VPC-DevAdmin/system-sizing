import { ChevronRight } from 'lucide-react';
import { useSmbOnPremAmd } from '@/contexts/SmbOnPremAmdContext';
import { FUNNEL_TOOLTIPS } from '@/data/smbOnPremAmdConfig';
import { AnimatedNumber } from './AnimatedNumber';

interface Stage {
  value: number;
  label: string;
  sub: string;
  transitionPct?: string;
  transitionLabel?: string;
  tone: 'blue' | 'emerald' | 'amber' | 'primary';
}

export default function FunnelSection() {
  const { funnel, funnelStages } = useSmbOnPremAmd();

  const stages: Stage[] = [
    {
      value: Math.round(funnelStages.total),
      label: 'Enabled seats',
      sub: 'Employees with a license',
      tone: 'blue',
    },
    {
      value: Math.round(funnelStages.adopted),
      label: 'Regular users',
      sub: 'Use AI most weeks',
      transitionPct: `${funnel.adoptionPct}%`,
      transitionLabel: 'adoption (Gallup Q3 2025)',
      tone: 'emerald',
    },
    {
      value: Math.round(funnelStages.hourlyActive),
      label: 'Daily active',
      sub: 'Use AI today',
      transitionPct: `${funnel.hourlyActivePct}%`,
      transitionLabel: 'of users are active daily (Gallup Q3 2025)',
      tone: 'amber',
    },
    {
      value: Math.round(funnelStages.concurrent),
      label: 'Active this hour',
      sub: 'Actively engaging the system',
      transitionPct: `${funnel.inFlightDensityPct}%`,
      transitionLabel: 'of daily users at peak hour (Microsoft Work Trend Index 2025)',
      tone: 'primary',
    },
  ];

  const total = stages[0].value || 1;

  const toneStyles: Record<Stage['tone'], { tile: string; num: string; bar: string; dot: string }> = {
    blue: {
      tile: 'bg-[hsl(var(--chart-blue)/0.06)] ring-1 ring-[hsl(var(--chart-blue)/0.18)]',
      num: 'text-[hsl(var(--chart-blue))]',
      bar: 'bg-[hsl(var(--chart-blue)/0.55)]',
      dot: 'bg-[hsl(var(--chart-blue))]',
    },
    emerald: {
      tile: 'bg-[hsl(var(--chart-emerald)/0.07)] ring-1 ring-[hsl(var(--chart-emerald)/0.20)]',
      num: 'text-[hsl(var(--chart-emerald))]',
      bar: 'bg-[hsl(var(--chart-emerald)/0.6)]',
      dot: 'bg-[hsl(var(--chart-emerald))]',
    },
    amber: {
      tile: 'bg-[hsl(var(--chart-amber)/0.08)] ring-1 ring-[hsl(var(--chart-amber)/0.22)]',
      num: 'text-[hsl(var(--chart-amber))]',
      bar: 'bg-[hsl(var(--chart-amber)/0.65)]',
      dot: 'bg-[hsl(var(--chart-amber))]',
    },
    primary: {
      tile: 'bg-gradient-to-br from-[hsl(var(--chart-rose)/0.20)] to-[hsl(var(--chart-rose)/0.08)] ring-2 ring-[hsl(var(--chart-rose)/0.45)] shadow-lg shadow-[hsl(var(--chart-rose)/0.15)] scale-[1.04]',
      num: 'text-[hsl(var(--chart-rose))] drop-shadow-sm',
      bar: 'bg-[hsl(var(--chart-rose))]',
      dot: 'bg-[hsl(var(--chart-rose))]',
    },
  };

  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">From total seats to active users</h2>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground max-w-[750px] leading-relaxed">
            A {Math.round(funnelStages.total).toLocaleString()}-seat team may have on average only{' '}
            <span className="font-medium text-foreground">{Math.round(funnelStages.concurrent)} active users</span>{' '}
            at any time. That's the number that sizes hardware with headroom to flex in case demand spikes.
          </p>
        </div>
      </div>

      <div className="rounded-xl border bg-card p-6 shadow-sm">
        {/* Numbers row */}
        <div className="grid grid-cols-2 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] gap-y-4 md:gap-y-0 items-center">
          {stages.map((s, i) => (
            <div key={s.label} className="contents">
              <div
                className={`flex flex-col items-center text-center px-2 py-3 rounded-lg transition-transform animate-funnel-tile-in ${toneStyles[s.tone].tile} ${s.tone === 'primary' ? 'relative' : ''}`}
                style={{ animationDelay: `${i * 180}ms` }}
              >
                {s.tone === 'primary' && (
                  <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-[hsl(var(--chart-rose))] text-white text-[9px] font-semibold uppercase tracking-wider shadow-sm whitespace-nowrap">
                    What hardware serves
                  </div>
                )}
                <div
                  className={`tabular-nums leading-none ${toneStyles[s.tone].num} ${s.tone === 'primary' ? 'text-5xl md:text-6xl font-bold' : 'text-3xl md:text-4xl font-semibold'}`}
                >
                  <AnimatedNumber value={s.value} />
                </div>
                <div className={`mt-2 text-[11px] uppercase tracking-widest font-medium ${s.tone === 'primary' ? 'text-[hsl(var(--chart-rose)/0.85)]' : 'text-muted-foreground'}`}>
                  {s.label}
                </div>
                <div className="mt-1 text-xs text-muted-foreground/80">{s.sub}</div>
              </div>
              {i < stages.length - 1 && (
                <div className="hidden md:flex justify-center text-[hsl(var(--chart-rose)/0.6)]">
                  <ChevronRight
                    className="h-5 w-5 animate-chevron-flow"
                    style={{ animationDelay: `${i * 200}ms` }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Shrink bar */}
        <div className="mt-6 space-y-2">
          <div className="relative h-2 rounded-full bg-muted overflow-hidden">
            {stages.map((s, i) => {
              const w = (s.value / total) * 100;
              return (
                <div
                  key={i}
                  className={`absolute top-0 h-full ${toneStyles[s.tone].bar}`}
                  style={{ right: 0, width: `${w}%`, zIndex: i }}
                />
              );
            })}
            <div
              className="pointer-events-none absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-white/40 to-transparent animate-bar-shimmer"
              style={{ zIndex: stages.length + 1 }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground px-1">
            {stages.slice(1).map(s => (
              <span key={s.label} className="tabular-nums inline-flex items-center gap-1.5">
                <span className={`h-1.5 w-1.5 rounded-full ${toneStyles[s.tone].dot}`} />
                <span className="font-medium text-foreground/80">{s.transitionPct}</span>
                <span className="whitespace-pre-line">{s.transitionLabel}</span>
              </span>
            ))}
          </div>
        </div>

      </div>

      {/* Progressive disclosure */}
      <details className="group rounded-lg border bg-card/50">
        <summary className="cursor-pointer list-none px-4 py-2.5 text-xs font-medium text-muted-foreground hover:text-foreground flex items-center gap-2 select-none">
          <ChevronRight className="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
          How we got here
        </summary>
        <div className="px-4 pb-4 pt-1 space-y-3 text-xs text-muted-foreground leading-relaxed">
          <p>
            <span className="font-medium text-foreground">Adoption ({funnel.adoptionPct}%).</span>{' '}
            {FUNNEL_TOOLTIPS.adoption}
          </p>
          <p>
            <span className="font-medium text-foreground">Daily active ({funnel.hourlyActivePct}% of regular users).</span>{' '}
            {FUNNEL_TOOLTIPS.dailyActive}
          </p>
          <p>
            <span className="font-medium text-foreground">Peak hour ({funnel.inFlightDensityPct}% of daily users).</span>{' '}
            {FUNNEL_TOOLTIPS.peakHour}
          </p>
        </div>
      </details>
    </section>
  );
}
