import { CAPABILITY_CAPTION } from '@/data/smbOnPremConfig';

interface BenchCard {
  category: string;
  benchmark: string;
  blurb: string;
  local: number;
  cloud: number;
}

const CARDS: BenchCard[] = [
  { category: 'Knowledge',    benchmark: 'MMLU-Pro',    blurb: 'Broad academic & professional knowledge',  local: 78.4, cloud: 79.8 },
  { category: 'Reasoning',    benchmark: 'AIME25',      blurb: 'Competition-level math reasoning',         local: 61.3, cloud: 26.7 },
  { category: 'Coding',       benchmark: 'MultiPL-E',   blurb: 'Code generation across many languages',    local: 83.8, cloud: 82.7 },
  { category: 'Writing',      benchmark: 'WritingBench',blurb: 'Long-form writing quality',                local: 85.5, cloud: 75.5 },
  { category: 'Tool use',     benchmark: 'BFCL-v3',     blurb: 'Function calling & agent tool use',        local: 65.1, cloud: 66.5 },
];

const LOCAL_LABEL = 'Local · Qwen3-30B-A3B';
const CLOUD_LABEL = 'Cloud · GPT-4o';

function Card({ c, idx }: { c: BenchCard; idx: number }) {
  const parity = Math.abs(c.local - c.cloud) <= 3;
  const localWins = !parity && c.local > c.cloud;
  const localColor = parity ? 'hsl(var(--chart-blue))' : (localWins ? 'hsl(var(--chart-emerald))' : 'hsl(var(--chart-blue))');
  const cloudColor = parity ? 'hsl(var(--chart-blue))' : (localWins ? 'hsl(var(--chart-amber))'   : 'hsl(var(--chart-emerald))');

  return (
    <div
      className="rounded-xl border bg-card p-4 shadow-sm flex flex-col gap-3 animate-fade-in"
      style={{ animationDelay: `${idx * 60}ms`, animationFillMode: 'backwards' }}
    >
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{c.category}</div>
        <div className="text-sm font-semibold mt-0.5">{c.benchmark}</div>
        <div className="text-[11px] text-muted-foreground leading-snug mt-0.5">{c.blurb}</div>
      </div>

      <div className="space-y-2 mt-auto">
        {[
          { name: LOCAL_LABEL, value: c.local, color: localColor },
          { name: CLOUD_LABEL, value: c.cloud, color: cloudColor },
        ].map(b => (
          <div key={b.name}>
            <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-0.5">
              <span className="uppercase tracking-wider">{b.name}</span>
              <span className="font-mono tabular-nums text-foreground">{b.value.toFixed(1)}</span>
            </div>
            <div className="h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${Math.max(2, b.value)}%`, background: b.color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CapabilityMatrix() {
  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">A local model with GPT-4o-class capability</h2>
        <p className="text-sm text-muted-foreground max-w-[750px] leading-relaxed">
          We're running Qwen3-30B-A3B-Instruct on-prem.&nbsp; This is a local model that benchmarks on par 
          with GPT-4o across the categories that matter for everyday business work.&nbsp;
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {CARDS.map((c, i) => <Card key={c.benchmark} c={c} idx={i} />)}
      </div>

      <p className="text-[11px] text-muted-foreground italic max-w-[750px]">{CAPABILITY_CAPTION}</p>
      <p className="text-[11px] text-muted-foreground italic max-w-[750px]">
        Caveat: numbers as reported by Qwen on a single evaluation harness; third-party
        leaderboards may show different spreads.
      </p>
    </section>
  );
}
