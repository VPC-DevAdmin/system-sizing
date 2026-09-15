import type { CurvePoint } from '@/lib/smbOnPrem';

export const POOL_TICKS = [8, 16, 32, 64, 128, 256];
export const POOL_DOMAIN: [number, number] = [8, 256];

/** Derive a log-scale x-domain and ticks that fit the cohort's measured pool sizes. */
export function poolAxis(xs: number[]): { domain: [number, number]; ticks: number[] } {
  if (!xs.length) return { domain: POOL_DOMAIN, ticks: POOL_TICKS };
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  const ticks = POOL_TICKS.filter(t => t >= min && t <= max);
  if (!ticks.includes(min)) ticks.unshift(min);
  if (!ticks.includes(max)) ticks.push(max);
  ticks.sort((a, b) => a - b);
  return { domain: [min, max], ticks };
}

export interface RawDot { x: number; p50: number; p95: number; }
export interface TrendPt { x: number; y: number; }

/** Dedupe by pool size (keep last), sort ascending. */
export function dedupeSort(curve: CurvePoint[]): CurvePoint[] {
  const m = new Map<number, CurvePoint>();
  curve.forEach(p => m.set(p.pool_size, p));
  return Array.from(m.values()).sort((a, b) => a.pool_size - b.pool_size);
}

/** Drop points whose value is an obvious deviation (Tukey fence on p95). */
export function filterOutliers<T extends { p50: number; p95: number }>(
  pts: T[],
): T[] {
  if (pts.length < 4) return pts;
  const sorted = pts.map(p => p.p95).sort((a, b) => a - b);
  const q = (frac: number) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * frac))];
  const q1 = q(0.25), q3 = q(0.75);
  const iqr = Math.max(q3 - q1, 1);
  const cap = q3 + 3 * iqr;
  return pts.filter(p => p.p95 <= cap);
}

/** Light triangular smoothing over sorted points. */
export function smooth(pts: TrendPt[]): TrendPt[] {
  return pts.map((p, i, arr) => {
    const window = arr.slice(Math.max(0, i - 1), Math.min(arr.length, i + 2));
    const y = window.reduce((s, v) => s + v.y, 0) / window.length;
    return { x: p.x, y };
  });
}

/** Smooth + enforce monotone non-increasing (used for throughput). */
export function smoothNonIncreasing(pts: TrendPt[]): TrendPt[] {
  const s = smooth(pts);
  for (let i = 1; i < s.length; i++) {
    if (s[i].y > s[i - 1].y) s[i].y = s[i - 1].y;
  }
  return s;
}
