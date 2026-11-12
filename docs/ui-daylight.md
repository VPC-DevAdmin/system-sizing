# The capsim UI (Daylight)

capsim opens at `/daylight.html`: opening `/` without naming a tool goes there. The UI has
two parts that share one look, one navigation bar and one service:

- **Guided screens** (`daylight.html`): Home, New run, Running, Result and Results, for
  the two most common questions: how fast the server can go (a saturation run) and how
  many users it supports (a capacity run). These are the simplified, chart-first screens.
- **Tools** (`index.html#<tool>`): Capacity run, Roofline, Optimize, Workloads, System,
  All results and Lab tests. These are the full original tools, restyled with
  `daylight-tools.css` and reached from the same bar ("More tools" on the guided screens,
  tabs on the tools page). Their layout and controls are unchanged.

`index.html?tools` opens the tools page without a tool named. The earlier experimental
workspace (`/workspace.html`, docs/ui-workspace.md) is still served but no longer linked.

## Principles

- **Organised by question.** Home asks what you want to find out: how fast the server
  can go, how many users it supports, or which model and engine is best.
- **Detail is opt-in.** Each screen leads with one number, one chart and at most three
  supporting figures. Everything else sits behind a "Show…" or "See the data" expander.
- **Charts before tables.** Tables appear only on request.
- **Plain labels.** "Requests at once", not concurrency rungs; the technical name
  appears once a control is opened.

## What it covers

| Screen | Route | Backed by |
|---|---|---|
| Home | `#home` | `/api/hardware`, `/api/models`, `/api/runs`, `/api/doctor` (on request) |
| New run, speed | `#new` | `POST /api/runs` with the `headline_generation` workload |
| New run, capacity | `#new/users` | `POST /api/runs` with a persona or cohort, open-loop |
| Running | `#run` | `/api/status` polled every 2.5 s (plus `/api/live/backfill` for capacity runs), `POST /api/runs/stop` |
| Result | `#result/<run>` | `/api/runs/<run>/headline`, or `/api/runs/<run>/export?slim=true` for a capacity run |
| Results | `#results` | `/api/runs` |

Speed and capacity runs have guided screens. Roofline, Optimize, the workload editor,
system preparation and lab tests are the tools above; so are a capacity run's full live
charts and detailed report, which the guided screens link to.

A capacity run needs two choices: the model, and who is using it (a team mix or one type
of user, shown with its typical question, answer and pause). Each arrival rate tested is
judged in words as well as colour: *kept up* (within the workload's response-time
targets), *slow turns* (kept up, some turns over target) or *fell behind* (the queue kept
growing). The result leads with the number of people it supports at the same time; that
is derived from the rate that met targets and the measured session length, or, when no
rate met targets, measured at the highest rate the engine kept up with, and the screen
says which.

New run sends three things Classic does not: `shape` (prompt and output tokens, applied
to the generation workload before the sweep), `prefix_reuse` (docs/prefix-cache-policy.md)
and an automatic engine shape: the fewest GPUs per copy that hold the model, and as many
copies as then fit.

The Result screen states when a number should not be quoted (most requests failed, cache
reuse above the design) and when the peak is probably higher than measured (output was
still rising at the highest load).

## Files

`simulator/ui/daylight.html`, `daylight.css`, `daylight.js`, the DOM-free helpers in
`lib/daylight-data.js`, and `daylight-tools.css` for the tools page (`node --test tests/ui/daylight-data.test.mjs`). No build step and
no external fonts, so it works on offline benchmark hosts.
