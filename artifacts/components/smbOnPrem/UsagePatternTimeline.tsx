import { Activity } from 'lucide-react';

type Phase = 'type' | 'wait' | 'read' | 'think';

interface Lane {
  name: string;
  T: number; W: number; R: number; Th: number; off: number;
}

const PHASE_STYLE: Record<Phase, { fill: string; label: string }> = {
  type:  { fill: 'hsl(var(--chart-blue) / 0.35)',    label: 'Type' },
  wait:  { fill: 'hsl(var(--chart-rose) / 0.55)',    label: 'System working' },
  read:  { fill: 'hsl(var(--chart-amber) / 0.40)',   label: 'Read' },
  think: { fill: 'hsl(var(--chart-emerald) / 0.40)', label: 'Think' },
};

const LANES: Lane[] = [
  { name: 'User #1', T: 1,   W: 4,    R: 4,    Th: 3, off: 0 },
  { name: 'User #2', T: 2,   W: 9,    R: 8,    Th: 6, off: 0 },
  { name: 'User #3', T: 1.5, W: 10.9, R: 9.6,  Th: 4, off: 0 },
  { name: 'User #4', T: 2.5, W: 17.1, R: 14.4, Th: 7, off: 0 },
  { name: 'User #5', T: 2,   W: 13.2, R: 8.8,  Th: 4, off: 0 },
];

// Deterministic pseudo-random pads (0–10s) so each user's cycle starts at a different time
const PADS: number[] = [1.4, 6.2, 3.7, 8.9, 4.5];

const DURATION = 30;
const TYPICAL: Lane = { name: 'Typical user', T: 3, W: 11, R: 8, Th: 5, off: 0 };

function buildSegments(l: Lane): Array<{ phase: Phase; start: number; end: number }> {
  const segs: Array<{ phase: Phase; start: number; end: number }> = [];
  if (l.off > 0) segs.push({ phase: 'type', start: 0, end: l.off });
  let t = l.off;
  while (t < DURATION) {
    const order: Array<[Phase, number]> = [
      ['type', l.T], ['wait', l.W], ['read', l.R], ['think', l.Th],
    ];
    for (const [phase, dur] of order) {
      if (t >= DURATION) break;
      const end = Math.min(t + dur, DURATION);
      if (end > t) segs.push({ phase, start: t, end });
      t += dur;
    }
  }
  return segs;
}

export default function UsagePatternTimeline() {
  const typicalSegs = buildSegments(TYPICAL);

  return (
    <section className="space-y-5">
      <div className="flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[hsl(var(--chart-amber)/0.12)] text-[hsl(var(--chart-amber))]">
          <Activity className="h-5 w-5" />
        </div>
        <div className="space-y-1">
          <h2 className="text-xl font-semibold tracking-tight">How people use AI</h2>
          <p className="text-sm text-muted-foreground max-w-[750px]">
            Typical users working in read, think, query cycles of different lengths
          </p>
        </div>
      </div>

      <p className="text-sm text-foreground/85 leading-relaxed max-w-[820px]">
        A user's interaction with AI is a cycle: type a prompt, wait for the model, read the response, 
        think about what to do next, type again. An active user stays in memory, but the CPU is only 
        working for a fraction of the time.&nbsp; Often multiple users don't overlap, reducing the demand on the CPU
      </p>

      <div className="rounded-xl border bg-card p-5 shadow-sm space-y-5">
        {/* Typical session — labels inside the bar */}
        <div className="space-y-2">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold">A typical session</h3>
            <span className="text-[11px] text-muted-foreground">One full cycle, repeating</span>
          </div>
          <div className="relative w-full overflow-hidden rounded-md ring-1 ring-border" style={{ height: 56 }}>
            {typicalSegs.map((s, j) => {
              const left = (s.start / DURATION) * 100;
              const width = ((s.end - s.start) / DURATION) * 100;
              const showLabel = width > 7;
              return (
                <div
                  key={j}
                  className="absolute top-0 h-full flex items-center justify-center text-center px-1"
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    background: PHASE_STYLE[s.phase].fill,
                  }}
                >
                  {showLabel && (
                    <span className="text-[11px] font-semibold leading-tight text-foreground/90">
                      {PHASE_STYLE[s.phase].label}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-muted-foreground">
            The <span className="font-medium text-[hsl(var(--chart-rose))]">System working</span> band
            is the only time the CPU is actually busy.
          </p>
        </div>

        {/* Persona stack — thinner lanes, shared colors */}
        <div className="space-y-2 pt-2 border-t">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold">Five personas, same cycle, different shapes</h3>
            <span className="text-[11px] text-muted-foreground">30s window</span>
          </div>
          <div className="space-y-1.5">
            {LANES.map((lane, i) => {
              const pad = PADS[i] ?? 0;
              const segs = buildSegments(lane);
              return (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-20 shrink-0 text-[11px] text-muted-foreground text-right truncate">
                    {lane.name}
                  </div>
                  <div className="relative flex-1 overflow-hidden rounded-sm bg-muted/40" style={{ height: 14 }}>
                    {/* Hashed idle pad */}
                    {pad > 0 && (
                      <div
                        className="absolute top-0 h-full"
                        style={{
                          left: 0,
                          width: `${(pad / DURATION) * 100}%`,
                          backgroundImage:
                            'repeating-linear-gradient(45deg, hsl(var(--muted-foreground) / 0.25) 0 2px, transparent 2px 5px)',
                        }}
                        title="Idle (not yet started)"
                      />
                    )}
                    {segs.map((s, j) => {
                      const left = ((s.start + pad) / DURATION) * 100;
                      const width = ((s.end - s.start) / DURATION) * 100;
                      if (left >= 100) return null;
                      const clippedWidth = Math.min(width, 100 - left);
                      return (
                        <div
                          key={j}
                          className="absolute top-0 h-full"
                          style={{
                            left: `${left}%`,
                            width: `${clippedWidth}%`,
                            background: PHASE_STYLE[s.phase].fill,
                          }}
                          title={`${lane.name} · ${PHASE_STYLE[s.phase].label}`}
                        />
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-3 pt-1">
            <div className="w-20 shrink-0" />
            <div className="relative flex-1 h-4 text-[10px] text-muted-foreground">
              {[0, 5, 10, 15, 20, 25, 30].map(s => (
                <span
                  key={s}
                  className="absolute top-0 -translate-x-1/2 tabular-nums"
                  style={{ left: `${(s / DURATION) * 100}%` }}
                >
                  {s}s
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Compact legend */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-muted-foreground border-t pt-3">
          <LegendSwatch fill={PHASE_STYLE.type.fill}>Type</LegendSwatch>
          <LegendSwatch fill={PHASE_STYLE.wait.fill}>
            <span className="text-foreground font-medium">System working</span>
          </LegendSwatch>
          <LegendSwatch fill={PHASE_STYLE.read.fill}>Read</LegendSwatch>
          <LegendSwatch fill={PHASE_STYLE.think.fill}>Think</LegendSwatch>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-5 rounded-sm"
              style={{
                backgroundImage:
                  'repeating-linear-gradient(45deg, hsl(var(--muted-foreground) / 0.45) 0 2px, transparent 2px 5px)',
                backgroundColor: 'hsl(var(--muted) / 0.6)',
              }}
            />
            Idle (not yet started)
          </span>
        </div>
      </div>
    </section>
  );
}

function LegendSwatch({ fill, children }: { fill: string; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-3 w-5 rounded-sm" style={{ background: fill }} />
      {children}
    </span>
  );
}