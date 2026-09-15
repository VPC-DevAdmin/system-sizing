import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useSmbOnPremAmd } from '@/contexts/SmbOnPremAmdContext';
import { COHORTS } from '@/data/smbOnPremAmdConfig';
import CapacityCurveChart from './CapacityCurveChart';
import CohortStatPanel from './CohortStatPanel';
import CohortRecommendation from './CohortRecommendation';

export default function PerCohortTabs() {
  const { teamCohorts, cohortById } = useSmbOnPremAmd();
  const [active, setActive] = useState('general_knowledge');
  const [compare, setCompare] = useState<string>('none');

  const cohort = cohortById(active);
  const cmp = compare !== 'none' ? cohortById(compare) ?? null : null;
  const copy = COHORTS.find(c => c.id === active);
  if (!cohort || !copy) return null;

  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Per-cohort detail</h2>
        <p className="text-sm text-muted-foreground max-w-[750px]">
          Tab through each team to see its capacity curve, the bottleneck signature we recorded,
          and the sizing guidance that follows from it. Identical layout per tab, so differences
          stand out.
        </p>
      </div>

      <Tabs value={active} onValueChange={setActive}>
        <TabsList className="flex flex-wrap h-auto">
          {teamCohorts.map(c => {
            const cc = COHORTS.find(x => x.id === c.id);
            return <TabsTrigger key={c.id} value={c.id} className="text-xs">{cc?.name ?? c.name}</TabsTrigger>;
          })}
        </TabsList>

        {teamCohorts.map(c => {
          const cc = COHORTS.find(x => x.id === c.id);
          return (
            <TabsContent key={c.id} value={c.id} className="mt-4 space-y-4 animate-fade-in">
              <h3 className="text-base font-semibold leading-snug">
                {cc?.name}: {cc?.summaryLine}
              </h3>

              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground">Compare TPOT to:</span>
                <Select value={compare} onValueChange={setCompare}>
                  <SelectTrigger className="w-[260px]"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">— none —</SelectItem>
                    {teamCohorts.filter(x => x.id !== c.id).map(x => {
                      const xc = COHORTS.find(z => z.id === x.id);
                      return <SelectItem key={x.id} value={x.id}>{xc?.name ?? x.name}</SelectItem>;
                    })}
                  </SelectContent>
                </Select>
              </div>

              <div className="rounded-xl border bg-card p-4 shadow-sm">
                <CapacityCurveChart cohort={cohortById(c.id)!} compareCohort={cmp && c.id === active ? cmp : null} />
              </div>
              <CohortStatPanel cohort={cohortById(c.id)!} />
              <CohortRecommendation cohortId={c.id} />
            </TabsContent>
          );
        })}
      </Tabs>

      {/* avoid unused */}
      <span className="hidden">{cohort.id} {copy.id}</span>
    </section>
  );
}
