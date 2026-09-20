import { $ } from "./api.js";

/* ── Tabs ─────────────────────────────────────────────────────── */

const TABS = [...document.querySelectorAll('#tabs [role="tab"]')];
const VIEWS = TABS.map(b => b.dataset.view);

export function currentView() {
  return TABS.find(b => b.classList.contains("active"))?.dataset.view
    ?? VIEWS[0];
}

/* Activate a view: class + ARIA state + URL hash, so a reload (or a
 * pasted link) lands on the same tab. Modules hook the tab button's
 * click event for their refreshes, which is why programmatic
 * switches go through btn.click() rather than calling this. */
export function showView(name) {
  const btn = TABS.find(b => b.dataset.view === name);
  if (!btn) return;
  for (const b of TABS) {
    const on = b === btn;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.tabIndex = on ? 0 : -1;          // roving tabindex: one tab stop
  }
  document.querySelectorAll(".view").forEach(v =>
    v.classList.toggle("active", v.id === `view-${name}`));
  if (location.hash !== `#${name}`) {
    history.replaceState(null, "", `#${name}`);
  }
}

for (const btn of TABS) {
  btn.addEventListener("click", () => showView(btn.dataset.view));
}

/* Arrow keys move between tabs (the WAI-ARIA tabs pattern, automatic
 * activation); Home/End jump to the ends. */
$("#tabs").addEventListener("keydown", (e) => {
  const i = TABS.indexOf(document.activeElement);
  if (i < 0) return;
  const step = { ArrowRight: 1, ArrowLeft: -1, Home: -i,
                 End: TABS.length - 1 - i }[e.key];
  if (step == null) return;
  e.preventDefault();
  const next = TABS[(i + step + TABS.length) % TABS.length];
  next.focus();
  next.click();
});

/* Deep link + back/forward: #results opens Results. */
export function viewFromHash() {
  const h = location.hash.replace(/^#/, "");
  return VIEWS.includes(h) ? h : null;
}
window.addEventListener("hashchange", () => {
  const v = viewFromHash();
  if (v && v !== currentView()) goTo(v);
});

/* Modules register their per-visit refreshes here instead of hooking
 * the button themselves. A hook is a click listener on the tab, so it
 * runs after showView, on a keyboard switch and on goTo() -- and not
 * on the default view's first paint, which has no click. */
export function onShow(view, fn) {
  const btn = TABS.find(b => b.dataset.view === view);
  if (!btn) throw new Error(`no tab for view "${view}"`);
  btn.addEventListener("click", fn);
}

/* Switch tabs the way a click would, refresh hooks included. */
export function goTo(view) {
  TABS.find(b => b.dataset.view === view)?.click();
}
