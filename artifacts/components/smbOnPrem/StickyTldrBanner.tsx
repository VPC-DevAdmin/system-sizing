import { useEffect, useState } from 'react';
import { useSmbOnPrem } from '@/contexts/SmbOnPremContext';
import { COMPARATORS } from '@/data/smbOnPremConfig';
import { buildTcoSeries, fmtUSD } from '@/lib/smbOnPrem';

export default function StickyTldrBanner() {
  const { funnelStages, monthlyInputMtok, monthlyOutputMtok, tco } = useSmbOnPrem();
  const [show, setShow] = useState(false);

  useEffect(() => {
    const onScroll = () => setShow(window.scrollY > 600);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const result = buildTcoSeries({
    serverCost: tco.serverCost, powerW: tco.powerW, powerKwhCost: tco.powerKwhCost,
    coolingOverheadPct: tco.coolingOverheadPct, adminYearlyCost: tco.adminYearlyCost,
    rackMonthlyCost: tco.rackMonthlyCost, comparatorIndex: tco.comparatorIndex,
    blendedMidPct: tco.blendedMidPct,
    monthlyInputMtok, monthlyOutputMtok,
  });

  return (
    <div
      className={`fixed top-14 left-0 right-0 z-40 border-b border-border bg-background/85 backdrop-blur-md shadow-sm transition-all duration-300 ${
        show ? 'translate-y-0 opacity-100' : '-translate-y-full opacity-0 pointer-events-none'
      }`}
    >
      <div className="mx-auto max-w-[1100px] px-4 py-2 flex flex-wrap items-center justify-center gap-x-6 gap-y-1 text-xs">
        <Stat big={fmtUSD(tco.serverCost)} small={`server replaces ${fmtUSD(result.midRate)}/mo API`} />
        <span className="text-muted-foreground/40">·</span>
        <Stat big={result.breakevenMid != null ? `Month ${result.breakevenMid}` : '—'} small={`Breakeven vs ${COMPARATORS[tco.comparatorIndex].name}`} />
        <span className="text-muted-foreground/40">·</span>
        <Stat big={`${Math.round(funnelStages.concurrent)}`} small="Concurrent users tested" />
      </div>
    </div>
  );
}

function Stat({ big, small }: { big: string; small: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-sm font-bold tabular-nums">{big}</span>
      <span className="text-[10px] text-muted-foreground">{small}</span>
    </div>
  );
}
