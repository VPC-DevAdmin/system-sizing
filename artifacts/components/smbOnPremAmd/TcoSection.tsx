import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useSmbOnPremAmd } from '@/contexts/SmbOnPremAmdContext';
import { COMPARATORS } from '@/data/smbOnPremAmdConfig';
import { buildTcoSeries, fmtUSD } from '@/lib/smbOnPrem';

export default function TcoSection() {
  const { tco, setTco, monthlyInputMtok, monthlyOutputMtok } = useSmbOnPremAmd();
  const result = buildTcoSeries({
    serverCost: tco.serverCost, powerW: tco.powerW, powerKwhCost: tco.powerKwhCost,
    coolingOverheadPct: tco.coolingOverheadPct, adminYearlyCost: tco.adminYearlyCost,
    rackMonthlyCost: tco.rackMonthlyCost, comparatorIndex: tco.comparatorIndex,
    blendedMidPct: tco.blendedMidPct,
    monthlyInputMtok, monthlyOutputMtok,
  });

  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Cost over time vs cloud APIs</h2>
        <p className="text-sm text-muted-foreground max-w-[750px]">
          A $30K server vs the equivalent monthly token spend. Token volume is auto-derived from
          the funnel and team mix above; you can override or adjust pricing assumptions below.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        <div className="rounded-xl border bg-card p-4 shadow-sm">
          <div className="h-[340px]">
            <ResponsiveContainer>
              <LineChart data={result.series} margin={{ top: 10, right: 20, left: 10, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} label={{ value: 'Month', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={v => fmtUSD(v)} width={70} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number, n: string) => [fmtUSD(v), n]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="onPrem"  name="On-prem cumulative"   stroke="hsl(var(--primary))" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="mid"     name={`API · ${COMPARATORS[tco.comparatorIndex].name}`} stroke="hsl(var(--chart-rose))" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="blended" name="API · blended"        stroke="hsl(var(--chart-amber))" strokeWidth={2} dot={false} strokeDasharray="5 3" />
                {result.breakevenMid != null && (
                  <ReferenceLine x={result.breakevenMid} stroke="hsl(var(--chart-rose))" strokeDasharray="3 3"
                    label={{ value: `Breakeven vs API: m${result.breakevenMid}`, fontSize: 10, fill: 'hsl(var(--chart-rose))', position: 'top' }} />
                )}
                {result.breakevenBlended != null && result.breakevenBlended !== result.breakevenMid && (
                  <ReferenceLine x={result.breakevenBlended} stroke="hsl(var(--chart-amber))" strokeDasharray="3 3"
                    label={{ value: `Breakeven vs blended: m${result.breakevenBlended}`, fontSize: 10, fill: 'hsl(var(--chart-amber))', position: 'top' }} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border bg-card p-4 shadow-sm space-y-4">
          <div className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Inputs</div>

          <NumberRow label="Server cost"          value={tco.serverCost}        min={5000}  max={200000} step={500}  fmt={fmtUSD}              onChange={v => setTco('serverCost', v)} />
          <SliderRow label="Power draw"           value={tco.powerW}            min={100}   max={600}    step={10}   suffix=" W"               onChange={v => setTco('powerW', v)} />
          <SliderRow label="Power cost"           value={tco.powerKwhCost}      min={0.05}  max={0.30}   step={0.01} fmt={v => `$${v.toFixed(2)}/kWh`} onChange={v => setTco('powerKwhCost', v)} />
          <SliderRow label="Cooling overhead"     value={tco.coolingOverheadPct} min={0}    max={100}    step={5}    suffix="%"                onChange={v => setTco('coolingOverheadPct', v)} />
          <NumberRow label="Admin time / yr"      value={tco.adminYearlyCost}   min={0}     max={50000}  step={500}  fmt={fmtUSD}              onChange={v => setTco('adminYearlyCost', v)} />
          <NumberRow label="Rack / colo / mo"     value={tco.rackMonthlyCost}   min={0}     max={2000}   step={50}   fmt={fmtUSD}              onChange={v => setTco('rackMonthlyCost', v)} />

          <div className="space-y-1">
            <Label className="text-xs">Comparator</Label>
            <Select value={String(tco.comparatorIndex)} onValueChange={v => setTco('comparatorIndex', Number(v))}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {COMPARATORS.map((c, i) => <SelectItem key={c.name} value={String(i)}>{c.name}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>

          <SliderRow label={`Blend: ${tco.blendedMidPct}% mid / ${100 - tco.blendedMidPct}% premium`}
                     value={tco.blendedMidPct} min={0} max={100} step={5} suffix="%"
                     onChange={v => setTco('blendedMidPct', v)} hideValue />

          <div className="rounded-md bg-muted/40 p-2 text-[10px] text-muted-foreground">
            Auto-derived monthly volume:&nbsp;
            <span className="font-semibold text-foreground">{monthlyInputMtok.toFixed(2)} M input</span> /&nbsp;
            <span className="font-semibold text-foreground">{monthlyOutputMtok.toFixed(2)} M output</span> tokens.
          </div>
        </div>
      </div>

      {/* Three perspectives */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <Callout heading="Financial">
          <p className="text-xs leading-relaxed">
            Server pays back vs <span className="font-semibold">{COMPARATORS[tco.comparatorIndex].name}</span> in&nbsp;
            <span className="font-semibold">{result.breakevenMid != null ? `month ${result.breakevenMid}` : '—'}</span>.
            Monthly opex is <span className="font-semibold">{fmtUSD(result.opex)}</span>.
          </p>
        </Callout>
        <Callout heading="Strategic">
          <p className="text-xs leading-relaxed">
            Bringing inference on-prem moves the cost from variable to fixed. As the team grows
            into the server, marginal users are effectively free, the opposite of token-priced APIs.
          </p>
        </Callout>
        <Callout heading="Operational">
          <p className="text-xs leading-relaxed">
            Power, cooling, and admin are the only ongoing costs once the server is racked. The
            team that runs your other servers can run this one too.
          </p>
        </Callout>
      </div>
    </section>
  );
}

function Callout({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border bg-card p-4 shadow-sm">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-1">{heading}</div>
      {children}
    </div>
  );
}

function SliderRow({ label, value, min, max, step, suffix, fmt, onChange, hideValue }: {
  label: string; value: number; min: number; max: number; step: number;
  suffix?: string; fmt?: (v: number) => string; onChange: (v: number) => void; hideValue?: boolean;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        {!hideValue && <span className="font-medium tabular-nums">{fmt ? fmt(value) : `${value}${suffix ?? ''}`}</span>}
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={v => onChange(v[0])} />
    </div>
  );
}

function NumberRow({ label, value, min, max, step, fmt, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  fmt?: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{fmt ? fmt(value) : value.toLocaleString()}</span>
      </div>
      <Input type="number" min={min} max={max} step={step} value={value}
             onChange={e => onChange(Number(e.target.value))} className="h-8 text-xs" />
    </div>
  );
}
