import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Server } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { SmbOnPremProvider } from '@/contexts/SmbOnPremContext';
import Hero from '@/components/smbOnPrem/Hero';
import FunnelSection from '@/components/smbOnPrem/FunnelSection';
import UsagePatternTimeline from '@/components/smbOnPrem/UsagePatternTimeline';
import PersonasAndCohorts from '@/components/smbOnPrem/PersonasAndCohorts';
import CapabilityMatrix from '@/components/smbOnPrem/CapabilityMatrix';
import DemandVsCapacity from '@/components/smbOnPrem/DemandVsCapacity';
import CapacityCurveSection from '@/components/smbOnPrem/CapacityCurveSection';
import TcoSection from '@/components/smbOnPrem/TcoSection';
import MethodologySpec from '@/components/smbOnPrem/MethodologySpec';
import SizingCalculator from '@/components/smbOnPrem/SizingCalculator';
import BottlenecksSection from '@/components/smbOnPrem/BottlenecksSection';

export default function SmbOnPremPage() {
  const navigate = useNavigate();
  const navItems: Array<{ id: string; label: string }> = [
    { id: 'active-users', label: 'Active users' },
    { id: 'personas',     label: 'Personas' },
    { id: 'teams',        label: 'Teams' },
    { id: 'model',        label: 'Model' },
    { id: 'capacity',     label: 'Capacity' },
    { id: 'sizing',       label: 'Sizing' },
  ];
  return (
    <SmbOnPremProvider>
      <div className="min-h-full bg-gradient-to-br from-muted/60 via-background to-muted/40">
        <div className="border-b border-border/50 bg-background/80 backdrop-blur-sm sticky top-0 z-30">
          <div className="mx-auto max-w-[1100px] px-4 py-3 flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={() => navigate('/')} className="gap-1.5">
              <ArrowLeft className="h-4 w-4" /> Demos
            </Button>
            <Separator orientation="vertical" className="h-6" />
            <div className="flex items-center gap-2">
              <Server className="h-5 w-5 text-primary" />
              <div>
                <h1 className="text-sm font-bold leading-tight">SMB On-Prem AI · Sizing Study</h1>
                <p className="text-[11px] text-muted-foreground">
                  Dell R470 · Intel Xeon 6 6761P · Qwen3-30B-A3B-Instruct (FP8) · SGLang on CPU
                </p>
              </div>
            </div>
          </div>
          <nav className="border-t border-border/40 bg-background/70">
            <div className="mx-auto max-w-[1100px] px-4 py-1.5 flex items-center gap-1 overflow-x-auto text-[11px]">
              {navItems.map((it, i) => (
                <span key={it.id} className="flex items-center gap-1">
                  {i > 0 && <span className="text-muted-foreground/50">·</span>}
                  <a
                    href={`#${it.id}`}
                    className="px-1.5 py-0.5 rounded text-muted-foreground hover:text-foreground hover:bg-accent/40 transition-colors whitespace-nowrap"
                  >
                    {it.label}
                  </a>
                </span>
              ))}
            </div>
          </nav>
        </div>

        <div className="mx-auto max-w-[1100px] px-4 py-8 space-y-12 [&>section]:scroll-mt-28">
          <Hero />
          <Separator />
          <section id="active-users"><FunnelSection /></section>
          <Separator />
          <UsagePatternTimeline />
          <Separator />
          <section id="personas"><PersonasAndCohorts /></section>
          <Separator />
          <section id="model"><CapabilityMatrix /></section>
          <Separator />
          <section id="capacity"><DemandVsCapacity /></section>
          <Separator />
          <CapacityCurveSection />
          <Separator />
          <section id="sizing"><SizingCalculator /></section>
          <Separator />
          <BottlenecksSection />
          <Separator />
          <MethodologySpec />
        </div>
      </div>
    </SmbOnPremProvider>
  );
}
