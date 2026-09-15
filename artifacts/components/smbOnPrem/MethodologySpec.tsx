import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { TEST_BENCH, SOFTWARE_STACK, WORKLOAD_GENERATOR_PARAGRAPH } from '@/data/smbOnPremConfig';

function SpecList({ rows }: { rows: Array<[string, string]> }) {
  return (
    <dl className="divide-y divide-border/60">
      {rows.map(([k, v], i) => (
        <div
          key={k}
          className="grid grid-cols-[140px_1fr] gap-3 py-2 text-xs animate-fade-in"
          style={{ animationDelay: `${i * 30}ms`, animationFillMode: 'backwards' }}
        >
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="font-medium font-mono tabular-nums">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

const PERSONAS: Array<[string, string]> = [
  ['Quick lookup',        '350 in / 60 out · 1–4 turns (mostly 1) · ~18s read+think'],
  ['Conversational',      '180 in / 220 out · 3–12 turns (mostly 5–8) · ~53s read+think'],
  ['Drafter',             '500 in / 350 out · 1–5 turns (mostly 1–2) · ~114s read+think'],
  ['Document Q&A',        '3,500 in / 280 out · 1–7 turns (mostly 1–2) · ~150s read+think'],
  ['Code assistance',     '1,200 in / 450 out · 2–15 turns (mostly 4–8) · ~236s read+think'],
  ['Content generator', '200 in / 3,000 out · 2–3 turns · ~210s read+think'],
];

const TEAMS: Array<[string, string]> = [
  ['Customer support',         '60% quick lookup · 30% conversational · 10% drafter'],
  ['General knowledge',        '30% quick lookup · 30% conversational · 25% drafter · 10% content generator · 5% document Q&A'],
  ['Marketing / content',      '40% drafter · 30% content generator · 20% conversational · 10% quick lookup'],
  ['Software engineering',     '50% code assistance · 15% content generator · 15% quick lookup · 15% conversational · 5% document Q&A'],
  ['Analyst',                  '70% document Q&A · 20% drafter · 10% content generator'],
];

export default function MethodologySpec() {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="space-y-4">
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="w-full flex items-start justify-between gap-4 rounded-xl border bg-card p-4 text-left shadow-sm hover:bg-accent/30 transition-colors"
        >
          <div className="space-y-1">
            <h2 className="text-xl font-semibold tracking-tight">Methodology</h2>
            <p className="text-sm text-muted-foreground max-w-[750px]">
              Hardware, software, and workload definitions used for every measurement on this page.
            </p>
          </div>
          <span className="flex items-center gap-1.5 shrink-0 rounded-full border border-primary/40 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            {open ? 'Collapse' : 'Expand'}
            <ChevronDown className={`h-4 w-4 transition-transform ${open ? 'rotate-180' : ''}`} />
          </span>
        </button>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-5 animate-fade-in">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border bg-card p-5 shadow-sm">
          <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Test bench</h3>
          <SpecList rows={TEST_BENCH} />
        </div>
        <div className="rounded-xl border bg-card p-5 shadow-sm">
          <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Software stack</h3>
          <SpecList rows={SOFTWARE_STACK} />
        </div>
        <div className="rounded-xl border bg-card p-5 shadow-sm">
          <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Personas</h3>
          <SpecList rows={PERSONAS} />
        </div>
        <div className="rounded-xl border bg-card p-5 shadow-sm">
          <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Teams</h3>
          <SpecList rows={TEAMS} />
        </div>
        </div>

        <div className="rounded-xl border bg-card p-5 shadow-sm">
          <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Workload generator</h3>
          <p className="text-xs leading-relaxed text-foreground/80">{WORKLOAD_GENERATOR_PARAGRAPH}</p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
