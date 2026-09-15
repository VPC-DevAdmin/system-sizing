import { useState } from 'react';
import { Users, Layers } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { PERSONAS, COHORTS } from '@/data/smbOnPremAmdConfig';
import { useSmbOnPremAmd } from '@/contexts/SmbOnPremAmdContext';

const MAX_TOKENS = Math.max(
  ...PERSONAS.map(p => Math.max(p.avgInputTokens, p.avgOutputTokens)),
);

// Approximate read + think time per persona, in seconds
const READ_THINK_SECONDS: Record<string, number> = {
  quick_lookup: 5,
  conversational: 20,
  writer: 25,
  document_qa: 60,
  code_assist: 30,
  long_form_generator: 45,
};
const MAX_RT = Math.max(...Object.values(READ_THINK_SECONDS));

function TokenBar({
  label, tokens, color,
}: { label: string; tokens: number; color: string }) {
  const pct = Math.max(4, (tokens / MAX_TOKENS) * 100);
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className="font-medium tabular-nums text-foreground/80">{tokens.toLocaleString()} tok</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function TimeBar({ label, seconds, color }: { label: string; seconds: number; color: string }) {
  const pct = Math.max(4, (seconds / MAX_RT) * 100);
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className="font-medium tabular-nums text-foreground/80">{seconds}s</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

export default function PersonasAndCohorts() {
  const { teamCohorts } = useSmbOnPremAmd();
  const [hoverCohort, setHoverCohort] = useState<string | null>(null);

  const highlightedPersonas = hoverCohort
    ? new Set(Object.keys(teamCohorts.find(c => c.id === hoverCohort)?.persona_weights ?? {}))
    : null;

  return (
    <section className="space-y-8">
      {/* ============== PART 1: PERSONAS ============== */}
      <div className="space-y-4">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Users className="h-5 w-5" />
          </div>
          <div className="space-y-1">
            <h2 className="text-xl font-semibold tracking-tight">Typical ways people use AI</h2>
            <p className="text-sm text-muted-foreground max-w-[750px]">
              Different people use AI in ways that have different demands on the system.&nbsp;
              We outlined 6 typical use cases and personas to create unique demands common to business users
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
        {PERSONAS.map((p, i) => {
          const dim = highlightedPersonas && !highlightedPersonas.has(p.id);
          const glow = highlightedPersonas?.has(p.id);
          return (
            <div
              key={p.id}
              className={`rounded-xl border bg-card p-3 shadow-sm animate-fade-in transition-all duration-200 ${
                dim ? 'opacity-40' : ''
              } ${glow ? 'ring-2 ring-primary/40 shadow-md -translate-y-0.5' : ''}`}
              style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'backwards' }}
            >
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: p.color }} />
                <h3 className="text-sm font-semibold">{p.name}</h3>
              </div>
              <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{p.description}</p>
              <div className="mt-2 space-y-1.5">
                <TokenBar label="Avg input"  tokens={p.avgInputTokens}  color={p.color} />
                <TokenBar label="Avg output" tokens={p.avgOutputTokens} color={p.color} />
                <TimeBar label="Avg read/think time" seconds={READ_THINK_SECONDS[p.id] ?? 20} color={p.color} />
              </div>
            </div>
          );
        })}
        </div>
      </div>

      {/* ============== PART 2: TEAMS ============== */}
      <div className="space-y-3 rounded-2xl border border-border/70 bg-muted/20 p-4">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Layers className="h-5 w-5" />
          </div>
          <div className="space-y-1">
            <h2 className="text-xl font-semibold tracking-tight">Teams of AI users</h2>
            <p className="text-sm text-muted-foreground max-w-[750px] leading-relaxed">
              Real teams don't have just one use case, so we model teams that blend different use
              cases together for more real-world planning.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2.5">
        {teamCohorts.map((c, i) => {
          const copy = COHORTS.find(x => x.id === c.id);
          const weights = Object.entries(c.persona_weights) as [string, number][];
          return (
            <div
              key={c.id}
              onMouseEnter={() => setHoverCohort(c.id)}
              onMouseLeave={() => setHoverCohort(null)}
              className="rounded-xl border bg-card p-3 shadow-sm animate-fade-in transition-all duration-200 hover:shadow-md hover:border-primary/30"
              style={{ animationDelay: `${400 + i * 80}ms`, animationFillMode: 'backwards' }}
            >
              <h3 className="text-sm font-semibold">{copy?.name ?? c.name}</h3>
              <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                {copy?.description ?? c.description}
              </p>
              <div className="mt-2">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Persona mix</div>
                <div className="flex h-2.5 w-full rounded-full overflow-hidden border border-border/60">
                  {weights.map(([pid, w]) => {
                    const persona = PERSONAS.find(p => p.id === pid);
                    return (
                      <Tooltip key={pid}>
                        <TooltipTrigger asChild>
                          <div
                            className="h-full transition-all duration-700"
                            style={{ width: `${w * 100}%`, background: persona?.color ?? 'hsl(var(--muted))' }}
                          />
                        </TooltipTrigger>
                        <TooltipContent>
                          <span className="text-xs">{persona?.name ?? pid}: {(w * 100).toFixed(0)}%</span>
                        </TooltipContent>
                      </Tooltip>
                    );
                  })}
                </div>
                <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
                  {weights.map(([pid, w]) => {
                    const persona = PERSONAS.find(p => p.id === pid);
                    return (
                      <span key={pid} className="flex items-center gap-1">
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: persona?.color }} />
                        {persona?.name ?? pid} {(w * 100).toFixed(0)}%
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        })}
        </div>
      </div>
    </section>
  );
}
