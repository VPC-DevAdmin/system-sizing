import { $ } from "./api.js";

/* ── Chart.js theming ─────────────────────────────────────────── */

const css = getComputedStyle(document.documentElement);
const cvar = (name) => css.getPropertyValue(name).trim();
export const C = {
  text: cvar("--text"),
  muted: cvar("--muted"),
  line: cvar("--stroke"),
  accent: cvar("--accent"),
  gold: cvar("--gold"),
  teal: cvar("--teal"),
  blue: cvar("--blue"),
  purple: cvar("--purple"),
  ok: cvar("--ok"),
  warn: cvar("--warn"),
  fail: cvar("--fail"),
};
/* Soft area fill for a hex series color. */
export const fill = (hex, alpha = "2e") => hex + alpha;
Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.line;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.animation = false;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.line.borderWidth = 2;

export const PALETTE = [C.gold, C.teal, C.blue, C.purple, C.accent, "#e88a5c",
                 "#8fa8ff", "#5bc8c8"];

/* Vertical landing-zone markers drawn onto the knee chart. */
export const zoneLinesPlugin = {
  id: "zoneLines",
  afterDatasetsDraw(chart) {
    const zones = chart.options.zoneLines || [];
    const { ctx, chartArea, scales } = chart;
    if (!scales.x) return;
    for (const z of zones) {
      if (z.value == null) continue;
      const x = scales.x.getPixelForValue(z.value);
      if (x < chartArea.left || x > chartArea.right) continue;
      ctx.save();
      ctx.strokeStyle = z.color;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = z.color;
      ctx.font = "11px sans-serif";
      ctx.fillText(z.label, x + 4, chartArea.top + 12);
      ctx.restore();
    }
  },
};
Chart.register(zoneLinesPlugin);

/* The one chart factory. Every chart on the page shares the same
 * frame -- a canvas that fills its card (maintainAspectRatio off), a
 * bottom legend and a y axis that starts at zero -- and differs only
 * in what it plots and how its axes are labelled.
 *   canvas     a selector or the element
 *   x, y       scale options merged over the frame's defaults
 *   y2         adds a right-hand axis (grid off); datasets pick it
 *              with yAxisID: "y2"
 *   legend     merged over { position: "bottom" }
 *   animation  Chart.js animation config; omitted = the global
 *              default (off)
 *   options    anything else at the top level of the chart options
 *              (zoneLines, onClick, indexAxis) */
export function makeChart(canvas, { type = "line", labels = [], datasets = [],
                                    x = {}, y = {}, y2 = null, legend = {},
                                    animation, options = {} } = {}) {
  const scales = { x, y: { beginAtZero: true, ...y } };
  if (y2) {
    scales.y2 = { beginAtZero: true, position: "right",
                  grid: { drawOnChartArea: false }, ...y2 };
  }
  const opts = { maintainAspectRatio: false };
  if (animation !== undefined) opts.animation = animation;
  return new Chart(typeof canvas === "string" ? $(canvas) : canvas, {
    type,
    data: { labels, datasets },
    options: {
      ...opts,
      scales,
      plugins: { legend: { position: "bottom", ...legend } },
      ...options,
    },
  });
}
