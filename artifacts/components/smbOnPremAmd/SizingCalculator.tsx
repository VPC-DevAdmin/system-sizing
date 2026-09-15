import { Button } from '@/components/ui/button';
import { ArrowRight, ExternalLink } from 'lucide-react';
import { CONFIG_LEVERS, CONFIG_PROFILES, DELL_CTA_URL, DELL_LOANER_URL } from '@/data/smbOnPremAmdConfig';
import { fmtUSD } from '@/lib/smbOnPrem';

export default function SizingCalculator() {
  return (
    <section className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Configuration profiles & sizing</h2>
        <p className="text-sm text-muted-foreground max-w-[750px] leading-relaxed">
          A comparison of hardware options to fit your team
        </p>
      </div>

      {/* Profile table */}
      <div className="rounded-xl border bg-card shadow-sm overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              <th className="text-left p-3 text-muted-foreground font-medium"></th>
              {CONFIG_PROFILES.map(p => (
                <th key={p.id} className={`text-left p-3 ${p.isTested ? 'bg-primary/5' : ''}`}>
                  <div className="text-sm font-semibold">{p.name}</div>
                  {p.isTested && <span className="inline-block mt-0.5 text-[9px] uppercase tracking-wider bg-primary text-primary-foreground rounded-full px-2 py-0.5">this study</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            <Row label="Best fit"          values={CONFIG_PROFILES.map(p => p.bestFit)} />
            <Row label="CPU"               values={CONFIG_PROFILES.map(p => p.cpu)} />
            <Row label="Memory"            values={CONFIG_PROFILES.map(p => p.memory)} />
            <Row label="Sockets"           values={CONFIG_PROFILES.map(p => String(p.sockets))} />
            <Row label="Est. cost"         values={CONFIG_PROFILES.map(p =>
              p.costMin === p.costMax ? fmtUSD(p.costMin) : `${fmtUSD(p.costMin)}–${fmtUSD(p.costMax)}`)} />
            <Row label="Capacity envelope" values={CONFIG_PROFILES.map(p => p.capacityEnvelope)} />
          </tbody>
        </table>
      </div>

      {/* Levers */}
      <div className="space-y-2">
        {CONFIG_LEVERS.map((l, i) => (
          <div
            key={l.rank}
            className="flex items-start gap-4 rounded-xl border bg-card p-4 shadow-sm animate-fade-in transition-all"
            style={{ animationDelay: `${i * 100}ms`, animationFillMode: 'backwards' }}
          >
            <div className="shrink-0 flex items-center justify-center rounded-full font-bold tabular-nums h-10 w-10 text-base bg-muted text-foreground">{l.rank}</div>
            <div className="flex-1">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <h3 className="text-sm font-semibold">{l.name}</h3>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700 bg-emerald-500/15 px-2 py-0.5 rounded-full text-right whitespace-pre-line">
                  {l.impact}
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{l.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Final CTA */}
      <div className="rounded-2xl border border-primary/30 bg-gradient-to-r from-primary/10 via-primary/15 to-primary/10 p-6 text-center shadow-sm flex flex-wrap items-center justify-center gap-3">
        <Button size="lg" asChild>
          <a href={DELL_CTA_URL} target="_blank" rel="noopener noreferrer">
            Configure your Dell R7625 <ExternalLink className="ml-1.5 h-4 w-4" />
          </a>
        </Button>
        <Button size="lg" variant="outline" asChild>
          <a href={DELL_LOANER_URL} target="_blank" rel="noopener noreferrer">
            Request a loaner system <ExternalLink className="ml-1.5 h-4 w-4" />
          </a>
        </Button>
      </div>
    </section>
  );
}

function Row({ label, values }: { label: string; values: string[] }) {
  return (
    <tr className="border-b last:border-b-0">
      <td className="p-3 text-muted-foreground font-sans">{label}</td>
      {values.map((v, i) => (
        <td key={i} className={`p-3 ${CONFIG_PROFILES[i].isTested ? 'bg-primary/5' : ''}`}>{v}</td>
      ))}
    </tr>
  );
}
