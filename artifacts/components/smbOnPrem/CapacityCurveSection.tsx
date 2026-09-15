import { useState } from 'react';
import { useSmbOnPrem } from '@/contexts/SmbOnPremContext';
import { COHORTS } from '@/data/smbOnPremConfig';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import TtftCurveChart from './TtftCurveChart';
import TpsCurveChart from './TpsCurveChart';
import CohortRecommendation from './CohortRecommendation';

export default function CapacityCurveSection() {
  const { teamCohorts, cohortById } = useSmbOnPrem();
  const [teamId, setTeamId] = useState<string>('general_knowledge');
  const cohort = cohortById(teamId);
  if (!cohort) return null;
  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Capacity curve by team</h2>
        <p className="text-sm text-muted-foreground max-w-[750px]">
          How latency and throughput degrade as more active users hit the server. Vertical markers
          show the comfortable, acceptable, and failure thresholds for each team.
        </p>
      </div>

      <Tabs value={teamId} onValueChange={setTeamId}>
        <TabsList className="flex flex-wrap h-auto">
          {teamCohorts.map(c => {
            const copy = COHORTS.find(x => x.id === c.id);
            return <TabsTrigger key={c.id} value={c.id} className="text-xs">{copy?.name ?? c.name}</TabsTrigger>;
          })}
        </TabsList>
      </Tabs>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl border bg-card p-4 shadow-sm">
          <div className="text-xs font-semibold mb-2">Time to first response</div>
          <TtftCurveChart cohort={cohort} />
        </div>
        <div className="rounded-xl border bg-card p-4 shadow-sm">
          <div className="text-xs font-semibold mb-2">Per-user response speed</div>
          <TpsCurveChart cohort={cohort} />
        </div>
      </div>

      <CohortRecommendation cohortId={teamId} />
    </section>
  );
}
