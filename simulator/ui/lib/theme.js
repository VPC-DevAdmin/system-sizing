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
