import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { SAMPLE_OUTPUTS } from '@/data/smbOnPremConfig';

export default function SampleOutputs() {
  return (
    <section className="space-y-5">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Sample outputs side by side</h2>
        <p className="text-sm text-muted-foreground max-w-[750px]">
          Three representative prompts. Click to expand and compare the local model's response
          against GPT-4o.
        </p>
      </div>
      <div className="space-y-3">
        {SAMPLE_OUTPUTS.map(s => <SampleCard key={s.id} sample={s} />)}
      </div>
    </section>
  );
}

function SampleCard({ sample }: { sample: typeof SAMPLE_OUTPUTS[number] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border bg-card shadow-sm overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full text-left p-4 flex items-start gap-3 hover:bg-muted/40 transition-colors"
      >
        <span className="mt-0.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground shrink-0 w-24">
          {sample.label}
        </span>
        <span className="flex-1 text-sm text-foreground line-clamp-2 whitespace-pre-wrap">
          "{sample.prompt}"
        </span>
        <ChevronDown className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-out"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-4 pt-0 border-t">
            <div className="rounded-lg border bg-muted/30 p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Local · Qwen3-30B</div>
              <p className="text-xs leading-relaxed whitespace-pre-wrap">{sample.localResponse}</p>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Cloud · GPT-4o</div>
              <p className="text-xs leading-relaxed whitespace-pre-wrap">{sample.cloudResponse}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
