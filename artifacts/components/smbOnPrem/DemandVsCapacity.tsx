import { useSmbOnPrem } from '@/contexts/SmbOnPremContext';
import { COHORTS } from '@/data/smbOnPremConfig';
import { poolSizeAtTps } from '@/lib/smbOnPrem';

export default function DemandVsCapacity() {
  const { teamCohorts } = useSmbOnPrem();
  const cards = COHORTS.map(copy => {
    const data = teamCohorts.find(c => c.id === copy.id);
    if (!data) return null;
    return {
      copy,
      comfortable: poolSizeAtTps(data, 10),
      acceptable: poolSizeAtTps(data, 4),
    };
  }).filter((x): x is NonNullable<typeof x> => x !== null);

  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Supported team size per server</h2>
        <p className="text-sm text-muted-foreground max-w-[750px] leading-relaxed">
          We tested how many active users can be supported per server by running pools of virtual 
          users matching the use cases and teams above.&nbsp; This gives you a range of users 
          where performance is comfortable as well as how far the hardware can stretch under 
          load while still providing output at typical reading speed.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {cards.map(({ copy, comfortable, acceptable }) => (
          <div
            key={copy.id}
            className="rounded-xl border bg-card shadow-sm p-5 flex flex-col gap-4"
          >
            <div className="min-h-[96px]">
              <div className="text-sm font-semibold leading-tight min-h-[2.5rem]">{copy.name}</div>
              <div className="text-[11px] text-muted-foreground mt-1 leading-snug">
                {copy.description}
              </div>
            </div>

            <div className="mt-auto rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2.5">
              <div className="text-[10px] uppercase tracking-widest text-emerald-700 dark:text-emerald-400 font-semibold">
                Comfortable
              </div>
              <div className="mt-0.5 flex items-baseline gap-1">
                <span className="text-xs text-muted-foreground">up to</span>
                <span className="text-2xl font-semibold tabular-nums">
                  {comfortable.toLocaleString()}
                </span>
                <span className="text-xs text-muted-foreground">people</span>
              </div>
            </div>

            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
              <div className="text-[10px] uppercase tracking-widest text-amber-700 dark:text-amber-400 font-semibold">
                Acceptable
              </div>
              <div className="mt-0.5 flex items-baseline gap-1">
                <span className="text-xs text-muted-foreground">up to</span>
                <span className="text-2xl font-semibold tabular-nums">
                  {acceptable.toLocaleString()}
                </span>
                <span className="text-xs text-muted-foreground">people</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border bg-muted/20 p-4 text-xs text-muted-foreground space-y-2 max-w-[900px]">
        <div className="text-[10px] uppercase tracking-widest font-semibold text-foreground">
          What "supported" means
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="flex gap-2">
            <span className="mt-1 h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
            <div className="leading-relaxed">
              <span className="font-medium text-foreground">Comfortable.</span> Time to first token
              is within 10 seconds and output is provided at 2x reading speed.
            </div>
          </div>
          <div className="flex gap-2 leading-relaxed">
            <span className="mt-1 h-2 w-2 rounded-full bg-amber-500 shrink-0" />
            <div>
              <span className="font-medium text-foreground">Acceptable.</span> Time to first token
              is within 20 seconds and outputs is provided at typical reading speed.
            </div>
          </div>
        </div>
        <div className="pt-2 text-[11px] text-muted-foreground/80 leading-relaxed border-t border-border/50">
          Latency thresholds based on Tan et al.,{' '}
          <a
            href="https://arxiv.org/html/2604.06183"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-foreground"
          >
            "The Impact of Response Latency and Task Type on Human-LLM Interaction and Perception,"
          </a>{' '}
          CHI 2026.
        </div>
      </div>
    </section>
  );
}
