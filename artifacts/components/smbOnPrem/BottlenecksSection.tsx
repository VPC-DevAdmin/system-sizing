import { useMemo, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import {
  ComposedChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { useSmbOnPrem } from '@/contexts/SmbOnPremContext';
import type { RawCohort } from '@/lib/smbOnPrem';

// ---- Per-team static copy ----

interface TeamSpec {
  cohortId: string;
  tabLabel: string;
  workloadName: string;
  summaryLine: string;
}

const TEAM_SPECS: TeamSpec[] = [
  {
    cohortId: 'chat_heavy',
    tabLabel: 'Customer support',
    workloadName: 'customer support work',
    summaryLine: 'Quick-lookup heavy. High request rate per active user, short prompts and short outputs.',
  },
  {
    cohortId: 'general_knowledge',
    tabLabel: 'General knowledge',
    workloadName: 'general knowledge work',
    summaryLine: 'Mixed personas representing the realistic baseline office workload. Blend of quick lookups, conversational chat, drafting, and occasional long-form work.',
  },
  {
    cohortId: 'writer_dominant',
    tabLabel: 'Marketing / content',
    workloadName: 'marketing / content team work',
    summaryLine: 'Drafter and long-form heavy. Output-bound work, where long generated text pressures the decode pipeline.',
  },
  {
    cohortId: 'software_engineering',
    tabLabel: 'Software engineering',
    workloadName: 'software engineering work',
    summaryLine: 'Code-assist heavy with long-form generation. Mix of long-running generations and accumulated multi-turn context.',
  },
  {
    cohortId: 'analyst_team',
    tabLabel: 'Analyst',
    workloadName: 'analyst team work',
    summaryLine: 'Document Q&A dominant. Input-heavy work, with long prompts in and focused answers out.',
  },
];

type Zone = 'comfortable' | 'acceptable' | 'beyond';
const POOL_ZONES: Array<{ pool: number; zone: Zone; label: string }> = [
  { pool: 16, zone: 'comfortable', label: '16 users' },
  { pool: 32, zone: 'acceptable', label: '32 users' },
  { pool: 64, zone: 'beyond', label: '64 users' },
];

const ZONE_DOT: Record<Zone, string> = {
  comfortable: 'bg-emerald-500',
  acceptable: 'bg-amber-500',
  beyond: 'bg-rose-500',
};

const ZONE_INTERPRETATION: Record<Zone, string> = {
  comfortable:
    'This pool size has substantial operational headroom. Suitable for normal-load operation with room for usage spikes.',
  acceptable:
    'This pool size is at the edge of acceptable user experience. Suitable for steady-state operation, but tail latency widens during bursts.',
  beyond:
    'At this pool size, typical users wait on the model. Past the acceptable operating range for this team. Consider a higher-frequency CPU, a second socket, or load distribution across multiple servers.',
};

// ---- Build chart series from cohort curve at given pool size ----

interface Row {
  t: number;
  read_think: number;
  transitioning: number;
  decode: number;
  prefill: number;
  decodeRaw: number;
}

interface SeriesResult {
  rows: Row[];
  startT: number;
  decodeAvg: number;
  decodeMax: number;
  readThinkAvg: number;
  peakDecodeT: number | null;
}

// Per-(cohort, pool) hard overrides for the displayed time window, in absolute
// timeline seconds. When set, replaces the steady-state auto-detection.
const WINDOW_OVERRIDES: Record<string, [number, number]> = {
  'general_knowledge:16': [180, 300],
  'general_knowledge:64': [345, 465],
  'general_knowledge:32': [59, 179],
  'chat_heavy:64': [110, 230],
  'chat_heavy:32': [50, 170],
  'chat_heavy:16': [160, 280],
  'writer_dominant:16': [56, 176],
  'writer_dominant:64': [270, 450],
  'writer_dominant:32': [110, 230],
  'software_engineering:32': [216, 336],
  'software_engineering:16': [95, 215],
  'software_engineering:64': [218, 398],
  'analyst_team:32': [190, 310],
  'analyst_team:16': [83, 203],
  'analyst_team:64': [147, 267],
};

function buildSeries(cohort: RawCohort, pool: number): SeriesResult {
  const empty: SeriesResult = { rows: [], startT: 0, decodeAvg: 0, decodeMax: 0, readThinkAvg: 0, peakDecodeT: null };
  const curve = (cohort as unknown as { curve: Array<Record<string, unknown>> }).curve;
  const point = curve.find(p => p.pool_size === pool);
  if (!point) return empty;
  const tl = point.timeline as { schema: string[]; rows: number[][] } | undefined;
  if (!tl || !tl.rows?.length) return empty;

  const idx = (k: string) => tl.schema.indexOf(k);
  const iT = idx('t_offset_s');
  const iP = idx('prefill');
  const iD = idx('decode');
  const iTh = idx('think');
  const iI = idx('idle');

  const raw = tl.rows
    .map(r => {
      const prefill = r[iP] ?? 0;
      const decode = r[iD] ?? 0;
      const think = r[iTh] ?? 0;
      const idle = r[iI] ?? 0;
      const total = prefill + decode + think + idle;
      const transitioning = Math.max(0, pool - total);
      return { t: r[iT], prefill, decode, read_think: think + idle, transitioning, totalEngaged: total };
    })
    .sort((a, b) => a.t - b.t);

  const override = WINDOW_OVERRIDES[`${cohort.id}:${pool}`];
  let win: typeof raw;
  if (override) {
    const [a, b] = override;
    win = raw.filter(r => r.t >= a && r.t <= b);
  } else {
    const threshold = 0.9 * pool;
    const firstIdx = raw.findIndex(r => r.totalEngaged >= threshold);
    const window = firstIdx < 0 ? raw : raw.slice(firstIdx);
    let lastIdx = window.length - 1;
    for (let i = window.length - 1; i >= 0; i--) {
      if (window[i].totalEngaged >= threshold) { lastIdx = i; break; }
    }
    win = window.slice(0, lastIdx + 1);
  }
  if (!win.length) return empty;
  const startT = win[0].t;

  const W = 5;
  const half = Math.floor(W / 2);
  let decodeSum = 0, decodeMax = 0, rtSum = 0;
  let peakT: number | null = null;
  const smoothed: Row[] = win.map((_, i) => {
    const lo = Math.max(0, i - half);
    const hi = Math.min(win.length - 1, i + half);
    let prefill = 0, decode = 0, read_think = 0, transitioning = 0;
    const n = hi - lo + 1;
    for (let k = lo; k <= hi; k++) {
      prefill += win[k].prefill;
      decode += win[k].decode;
      read_think += win[k].read_think;
      transitioning += win[k].transitioning;
    }
    const dAvg = decode / n;
    decodeSum += dAvg;
    rtSum += read_think / n;
    if (dAvg > decodeMax) { decodeMax = dAvg; peakT = win[i].t - startT; }
    return {
      t: win[i].t - startT,
      prefill: prefill / n,
      decode: dAvg,
      read_think: read_think / n,
      transitioning: transitioning / n,
      decodeRaw: win[i].decode,
    };
  });

  return {
    rows: smoothed,
    startT,
    decodeAvg: decodeSum / smoothed.length,
    decodeMax,
    readThinkAvg: rtSum / smoothed.length,
    peakDecodeT: peakT,
  };
}

// ---- Telemetry stats from telemetry_samples ----

interface TelemetryStats {
  cpuUtilPct: number;
  freqAvgGhz: number;
  freqMinGhz: number;
  kvAvg: number;
  kvPeak: number;
  memAvgGb: number;
  freqMinT: number | null; // t-offset (relative to chart) of frequency floor
}

function computeTelemetry(
  cohort: RawCohort, pool: number, startT: number, endT: number,
): TelemetryStats | null {
  const curve = (cohort as unknown as { curve: Array<Record<string, unknown>> }).curve;
  const point = curve.find(p => p.pool_size === pool) as
    | { telemetry_samples?: Record<string, Record<string, number | null>>; measurement_started_at?: number }
    | undefined;
  if (!point?.telemetry_samples) return null;
  const samples = Object.values(point.telemetry_samples);
  if (!samples.length) return null;

  // Filter to steady-state window using sampled_at_ms relative to first sample.
  const t0Ms = (samples[0].sampled_at_ms as number) ?? 0;
  const inWindow = samples.filter(s => {
    const tRel = ((s.sampled_at_ms as number) - t0Ms) / 1000;
    return tRel >= startT && tRel <= endT;
  });
  const used = inWindow.length ? inWindow : samples;

  const num = (k: string) => used.map(s => s[k]).filter((v): v is number => typeof v === 'number');
  const cpuBound = num('cpu_util_bound_avg');
  const cpu = cpuBound.length ? cpuBound : num('cpu_util_avg');
  const fMean = num('freq_mhz_mean');
  const fMin = num('freq_mhz_min');
  const kv = num('kv_cache_used_pct');
  const mem = num('memory_used_gb');
  const avg = (a: number[]) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;

  // Find timestamp of frequency floor
  let freqMinT: number | null = null;
  if (fMin.length) {
    const minVal = Math.min(...fMin);
    const found = used.find(s => s.freq_mhz_min === minVal);
    if (found) freqMinT = ((found.sampled_at_ms as number) - t0Ms) / 1000 - startT;
  }

  return {
    cpuUtilPct: cpuBound.length ? avg(cpu) : avg(cpu) * 2,
    freqAvgGhz: avg(fMean) / 1000,
    freqMinGhz: (fMin.length ? Math.min(...fMin) : 0) / 1000,
    kvAvg: avg(kv),
    kvPeak: kv.length ? Math.max(...kv) : 0,
    memAvgGb: avg(mem),
    freqMinT,
  };
}

function fmtMmSs(t: number): string {
  const tt = Math.max(0, Math.round(t));
  const m = Math.floor(tt / 60);
  const s = tt % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function ticksFor(pool: number): number[] {
  const step = pool / 4;
  return [0, step, step * 2, step * 3, pool];
}

function phaseDescriptor(decodeAvg: number, rtAvg: number): string {
  if (decodeAvg < rtAvg * 0.7) return 'The system spent most of its time waiting on users between turns.';
  if (decodeAvg <= rtAvg) return 'Decode and read/think were roughly balanced.';
  if (decodeAvg <= rtAvg * 1.5) return 'Decode dominated, and the system continuously fed output streams.';
  return 'The system was running flat out, with read/think becoming the minority.';
}

function hardwareDescriptor(freqMin: number): string {
  if (freqMin > 3.0) return 'The chip had ample headroom in frequency and was not thermally constrained.';
  if (freqMin > 2.7) return 'The chip was working hard with moderate headroom in frequency.';
  return 'The chip ran at or near its sustainable AMX-load floor, the hardware throughput envelope for this workload class.';
}

function PoolPanel({
  team, cohort, pool, zone,
}: { team: TeamSpec; cohort: RawCohort | undefined; pool: number; zone: Zone }) {
  const series = useMemo(
    () => (cohort ? buildSeries(cohort, pool) : null),
    [cohort, pool],
  );

  const telemetry = useMemo(() => {
    if (!cohort || !series || !series.rows.length) return null;
    const endT = series.rows[series.rows.length - 1].t;
    return computeTelemetry(cohort, pool, series.startT, series.startT + endT);
  }, [cohort, pool, series]);

  if (!series || !series.rows.length) {
    return (
      <div className="rounded-xl border bg-card p-6 text-sm text-muted-foreground">
        No timeline data available for this pool size.
      </div>
    );
  }

  const data = series.rows;
  const ticks = ticksFor(pool);
  const peakDecodeUsers = Math.round(series.decodeMax);
  const showFreqMarker =
    !!telemetry && telemetry.freqMinGhz > 0 && telemetry.freqMinGhz * 1000 < 2700 && telemetry.freqMinT != null;

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="rounded-xl border bg-card p-4 shadow-sm">
        <div style={{ height: 360 }}>
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 28 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="t"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 11 }}
                tickFormatter={fmtMmSs}
                label={{ value: 'Time (m:ss)', position: 'insideBottom', offset: -16, fontSize: 11 }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                domain={[0, pool]}
                ticks={ticks}
                allowDecimals={false}
                label={{ value: 'Active users', angle: -90, position: 'insideLeft', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 12 }}
                labelFormatter={(t: number) => `t = ${fmtMmSs(t)}`}
                formatter={(value: number, name: string) => [
                  typeof value === 'number' ? value.toFixed(1) : value,
                  name,
                ]}
              />
              <Legend verticalAlign="top" align="center" wrapperStyle={{ fontSize: 11, paddingBottom: 8 }} />

              {/* Bottom to top: Read/think → Transitioning → Decode → Prefill */}
              <Area type="monotone" dataKey="read_think"   name="Read/think"   stackId="1" stroke="hsl(var(--chart-blue) / 0.6)"        fill="hsl(var(--chart-blue) / 0.30)" isAnimationActive />
              <Area type="monotone" dataKey="transitioning" name="Transitioning" stackId="1" stroke="hsl(var(--muted-foreground) / 0.5)" fill="hsl(var(--muted-foreground) / 0.25)" isAnimationActive />
              <Area type="monotone" dataKey="decode"        name="Decode"        stackId="1" stroke="hsl(var(--primary))"                 fill="hsl(var(--primary) / 0.80)" isAnimationActive />
              <Area type="monotone" dataKey="prefill"       name="Prefill"       stackId="1" stroke="hsl(var(--chart-amber))"             fill="hsl(var(--chart-amber) / 0.55)" isAnimationActive />

              {series.peakDecodeT != null && (
                <ReferenceLine
                  x={series.peakDecodeT}
                  stroke="hsl(var(--foreground) / 0.55)"
                  strokeDasharray="3 3"
                  label={{ value: `▼ peak decode ${fmtMmSs(series.peakDecodeT)}`, fontSize: 10, fill: 'hsl(var(--foreground))', position: 'top' }}
                />
              )}
              {showFreqMarker && telemetry && telemetry.freqMinT != null && (
                <ReferenceLine
                  x={telemetry.freqMinT}
                  stroke="hsl(var(--destructive) / 0.6)"
                  strokeDasharray="3 3"
                  label={{ value: `▼ freq floor ${fmtMmSs(telemetry.freqMinT)}`, fontSize: 10, fill: 'hsl(var(--foreground))', position: 'insideTopRight' }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <ul className="mt-3 space-y-1 text-[11px] text-muted-foreground">
          {series.peakDecodeT != null && (
            <li>
              <span className="font-mono text-foreground">{fmtMmSs(series.peakDecodeT)}</span>: Peak decode of {peakDecodeUsers} simultaneous users.
            </li>
          )}
          {showFreqMarker && telemetry && telemetry.freqMinT != null && (
            <li>
              <span className="font-mono text-foreground">{fmtMmSs(telemetry.freqMinT)}</span>: Frequency floor of {telemetry.freqMinGhz.toFixed(2)} GHz under sustained load.
            </li>
          )}
        </ul>

        <p className="mt-3 text-[11px] text-muted-foreground italic">
          Measured at {pool} active users running {team.workloadName}. The full pool is always
          accounted for. Users are actively engaged (prefill, decode, read/think) or in a brief
          gap between sessions (transitioning). Hardware utilization details are summarized in
          the cards below.
        </p>
      </div>

      {/* Hardware telemetry grid */}
      {telemetry && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard label="Physical core utilization" value={`~${Math.min(100, telemetry.cpuUtilPct).toFixed(1)}%`} />
          <StatCard label="CPU frequency under load" value={`Avg ${telemetry.freqAvgGhz.toFixed(2)} GHz / floor ${telemetry.freqMinGhz.toFixed(2)} GHz`} />
          <StatCard label="KV cache used" value={`${Math.round(telemetry.kvAvg)}% avg / ${Math.round(telemetry.kvPeak)}% peak`} />
          <StatCard label="System memory used" value={`${telemetry.memAvgGb.toFixed(1)} GB of 1 TB`} />
        </div>
      )}

      {/* Templated commentary */}
      <div className="rounded-xl border bg-card p-4 shadow-sm space-y-3">
        <p className="text-sm leading-relaxed text-foreground/90">
          At {pool} active users, the system carried an average of {series.decodeAvg.toFixed(1)} users in
          decode at any moment, with peaks to {Math.round(series.decodeMax)}. About {series.readThinkAvg.toFixed(0)} users
          were in read/think on average. {phaseDescriptor(series.decodeAvg, series.readThinkAvg)}
        </p>
        {telemetry && (
          <p className="text-sm leading-relaxed text-foreground/90">
            CPU cores ran at an average of {telemetry.freqAvgGhz.toFixed(2)} GHz under load, with a floor
            of {telemetry.freqMinGhz.toFixed(2)} GHz during peak decode. KV cache utilization averaged{' '}
            {Math.round(telemetry.kvAvg)}% (peak {Math.round(telemetry.kvPeak)}%). {hardwareDescriptor(telemetry.freqMinGhz)}
          </p>
        )}
        <p className="text-sm leading-relaxed text-foreground/90">{ZONE_INTERPRETATION[zone]}</p>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card p-3 shadow-sm">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-base font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function TeamPanel({ team, cohort }: { team: TeamSpec; cohort: RawCohort | undefined }) {
  const [pool, setPool] = useState<string>('32');
  return (
    <Tabs value={pool} onValueChange={setPool}>
      <TabsList className="flex flex-wrap h-auto mb-3">
        {POOL_ZONES.map(pz => (
          <TabsTrigger key={pz.pool} value={String(pz.pool)} className="text-xs gap-2">
            <span className={`inline-block h-2 w-2 rounded-full ${ZONE_DOT[pz.zone]}`} />
            {pz.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {POOL_ZONES.map(pz => (
        <TabsContent key={pz.pool} value={String(pz.pool)}>
          <PoolPanel team={team} cohort={cohort} pool={pz.pool} zone={pz.zone} />
        </TabsContent>
      ))}
    </Tabs>
  );
}

export default function BottlenecksSection() {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState('general_knowledge');
  const { cohortById } = useSmbOnPrem();

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="space-y-4">
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="w-full flex items-start justify-between gap-4 rounded-xl border bg-card p-4 text-left shadow-sm hover:bg-accent/30 transition-colors"
        >
          <div className="space-y-1">
            <h2 className="text-xl font-semibold tracking-tight">Additional Analytics &amp; Bottlenecks</h2>
            <p className="text-sm text-muted-foreground max-w-[820px]">
              What each user was doing during the measurement window and the hardware telemetry during that time
            </p>
          </div>
          <span className="flex items-center gap-1.5 shrink-0 rounded-full border border-primary/40 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            {open ? 'Collapse' : 'Expand'}
            <ChevronDown className={`h-4 w-4 transition-transform ${open ? 'rotate-180' : ''}`} />
          </span>
        </button>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-5 animate-fade-in">
        <p className="text-sm text-muted-foreground max-w-[820px]">
          This shows how many users were at each stage of their read, think, and process cycle during the measurement window for each team and density. The hardware utilization during this window was measured and is displayed below.
        </p>

        <Tabs value={active} onValueChange={setActive}>
          <TabsList className="flex flex-wrap h-auto">
            {TEAM_SPECS.map(s => (
              <TabsTrigger key={s.cohortId} value={s.cohortId} className="text-xs">
                {s.tabLabel}
              </TabsTrigger>
            ))}
          </TabsList>

          {TEAM_SPECS.map(s => (
            <TabsContent key={s.cohortId} value={s.cohortId} className="mt-4">
              <TeamPanel team={s} cohort={cohortById(s.cohortId)} />
            </TabsContent>
          ))}
        </Tabs>
      </CollapsibleContent>
    </Collapsible>
  );
}