import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  FUNNEL_DEFAULTS, TCO_DEFAULTS, COHORTS,
} from '@/data/smbOnPremAmdConfig';
import {
  computeFunnel, deriveMonthlyTokens, densityFor, recommendProfile,
  selectTeamCohorts, selectPersonaCohorts, findCohortById,
  type FunnelInputs, type FunnelStages,
  type SizingData, type RawCohort,
} from '@/lib/smbOnPrem';

interface State {
  sizingData: SizingData | null;
  teamCohorts: RawCohort[];
  personaCohorts: RawCohort[];
  cohortById: (id: string) => RawCohort | undefined;
  funnel: FunnelInputs;
  setFunnel: (k: keyof FunnelInputs, v: number) => void;
  funnelStages: FunnelStages;
  teamMix: Record<string, number>;
  setTeamMix: (id: string, v: number) => void;
  resetTeamMix: () => void;
  teamMixSum: number;
  // TCO
  tco: {
    serverCost: number; powerW: number; powerKwhCost: number; coolingOverheadPct: number;
    adminYearlyCost: number; rackMonthlyCost: number; comparatorIndex: number; blendedMidPct: number;
    overrideTokens: boolean; manualInputMtok: number; manualOutputMtok: number;
  };
  setTco: <K extends keyof State['tco']>(k: K, v: State['tco'][K]) => void;
  monthlyInputMtok: number;
  monthlyOutputMtok: number;
  recommendedProfileId: string;
}

const Ctx = createContext<State | null>(null);

const defaultMix = COHORTS.reduce<Record<string, number>>((acc, c) => {
  acc[c.id] = c.defaultMixPct; return acc;
}, {});

export function SmbOnPremAmdProvider({ children }: { children: ReactNode }) {
  const [sizingData, setSizingData] = useState<SizingData | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch('/AMD_sizing_qwen3.json')
      .then(r => r.json())
      .then((d: SizingData) => { if (!cancelled) setSizingData(d); })
      .catch(err => console.error('Failed to load AMD_sizing_qwen3.json', err));
    return () => { cancelled = true; };
  }, []);

  const teamCohorts = useMemo(() => sizingData ? selectTeamCohorts(sizingData) : [], [sizingData]);
  const personaCohorts = useMemo(() => sizingData ? selectPersonaCohorts(sizingData) : [], [sizingData]);
  const cohortById = useCallback((id: string) => findCohortById(sizingData, id), [sizingData]);

  const [funnel, setFunnelState] = useState<FunnelInputs>({
    orgSize: FUNNEL_DEFAULTS.orgSize,
    adoptionPct: FUNNEL_DEFAULTS.adoptionPct,
    hourlyActivePct: FUNNEL_DEFAULTS.hourlyActivePct,
    inFlightDensityPct: FUNNEL_DEFAULTS.inFlightDensityPct,
  });
  const setFunnel = useCallback((k: keyof FunnelInputs, v: number) =>
    setFunnelState(prev => ({ ...prev, [k]: v })), []);

  const [teamMix, setTeamMixState] = useState<Record<string, number>>(defaultMix);
  const setTeamMix = useCallback((id: string, v: number) =>
    setTeamMixState(prev => ({ ...prev, [id]: v })), []);
  const resetTeamMix = useCallback(() => setTeamMixState(defaultMix), []);

  const [tco, setTcoState] = useState({
    serverCost: TCO_DEFAULTS.serverCost,
    powerW: TCO_DEFAULTS.powerW,
    powerKwhCost: TCO_DEFAULTS.powerKwhCost,
    coolingOverheadPct: TCO_DEFAULTS.coolingOverheadPct,
    adminYearlyCost: TCO_DEFAULTS.adminYearlyCost,
    rackMonthlyCost: TCO_DEFAULTS.rackMonthlyCost,
    comparatorIndex: TCO_DEFAULTS.comparatorIndex,
    blendedMidPct: TCO_DEFAULTS.blendedMidPct,
    overrideTokens: false,
    manualInputMtok: 0,
    manualOutputMtok: 0,
  });
  const setTco = useCallback(<K extends keyof State['tco']>(k: K, v: State['tco'][K]) =>
    setTcoState(prev => ({ ...prev, [k]: v })), []);

  const funnelStages = useMemo(() => computeFunnel(funnel), [funnel]);
  const teamMixSum = useMemo(() => Object.values(teamMix).reduce((a, b) => a + b, 0), [teamMix]);

  const derived = useMemo(() => deriveMonthlyTokens(teamCohorts, funnelStages, teamMix), [teamCohorts, funnelStages, teamMix]);
  const monthlyInputMtok  = tco.overrideTokens ? tco.manualInputMtok  : derived.monthlyInputMtok;
  const monthlyOutputMtok = tco.overrideTokens ? tco.manualOutputMtok : derived.monthlyOutputMtok;

  const recommendedProfileId = useMemo(
    () => recommendProfile(teamCohorts, funnelStages.concurrent, teamMix, densityFor).id,
    [teamCohorts, funnelStages.concurrent, teamMix],
  );

  const value: State = {
    sizingData, teamCohorts, personaCohorts, cohortById,
    funnel, setFunnel, funnelStages,
    teamMix, setTeamMix, resetTeamMix, teamMixSum,
    tco, setTco,
    monthlyInputMtok, monthlyOutputMtok,
    recommendedProfileId,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSmbOnPremAmd(): State {
  const v = useContext(Ctx);
  if (!v) throw new Error('useSmbOnPremAmd must be used within SmbOnPremAmdProvider');
  return v;
}