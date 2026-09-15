import { useCountUp } from '@/hooks/useCountUp';

export function AnimatedNumber({ value, format }: { value: number; format?: (n: number) => string }) {
  const v = useCountUp(value);
  const fmt = format ?? ((n: number) => Math.round(n).toLocaleString());
  return <>{fmt(v)}</>;
}
