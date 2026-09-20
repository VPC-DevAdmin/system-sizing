/* capsim UI — no-build frontend for the control-plane service.
 *
 * Three views over the Phase 2 API:
 *   Run control — start/stop runs, doctor, run history   (/api/*)
 *   Live       — event-bus charts over /ws/telemetry
 *   Results    — knee curves, landing zones, bottleneck evidence,
 *                step drill-down, cross-run comparison
 *
 * ES modules, no bundler: lib/ holds what every tab shares (api,
 * theme, tabs, events); each tab is one module exporting its
 * controller object. This file only boots them, in dependency order.
 */

import { currentView, viewFromHash, showView, goTo } from "./lib/tabs.js";
import { Control } from "./control.js";
import { Headline } from "./headline.js";
import { Live } from "./live.js";
import { Results } from "./results.js";
import { Optimizer } from "./optimizer.js";
import { Storage } from "./storage.js";
import { Engines } from "./engines.js";
import { Roofline } from "./roofline.js";
import { Models } from "./models.js";
import { Editor } from "./editor.js";

/* ── boot ─────────────────────────────────────────────────────── */

Control.init();
// Headline before Live: on every status tick the saturation view
// flips first and the idle panel second (the live charts resize on
// that edge), which is the order the poll used to call them in.
Headline.init();
Live.init();
Results.init();
Optimizer.init();
Storage.init();
Engines.init();
Roofline.init();
Models.init();
Editor.init();

// Land on the tab in the URL (a reload keeps the operator where
// they were; a pasted #results link opens Results). Goes through
// click() so the module refresh hooks run. The default view runs
// its own first-visit hook.
const initialView = viewFromHash();
if (initialView && initialView !== currentView()) {
  goTo(initialView);
} else {
  showView(currentView());
  Control.onPrepareShow();
}
