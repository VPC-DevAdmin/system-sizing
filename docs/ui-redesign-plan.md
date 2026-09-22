# capsim UI and UX redesign plan

Reviewed 22 September 2026. Audience: technical operators and solutions architects equally, with guided defaults and expert controls in the same workflow.

## Product direction

Make capsim an experiment workspace: ask a performance question, define a reproducible test, watch it execute, and turn the evidence into a defensible conclusion. Preserve the distinction between maximum generation throughput and workload capacity throughout the experience.

The visual direction is a precision instrument with dimensional glass navigation, quiet data surfaces, generous spacing, and purposeful motion. The app already has navy gradients, translucent panels, and entrance animations. Adding more effects alone will not resolve its hierarchy or workflow fragmentation.

This is a redesign proposal. No benchmark controls, running jobs, or production UI were changed during the review. The accompanying concept uses illustrative workflow states and explicitly labeled historical measurements; it is not connected to the server.

## Findings from the current UI

| Observed behavior | Consequence | Proposed change |
| --- | --- | --- |
| Six peer tabs mix tasks, methods, infrastructure and editing: Prepare, Optimize, Roofline, Workload, Results, Edit workloads. | Users must learn implementation boundaries before deciding where to start. | A shared experiment flow with supporting Library and System areas. |
| Prepare opens with a full doctor table, including repeated GPU names, before the model catalog and engine list. | Returning users must scan infrastructure they already configured. | Compact host readiness summary; expand only actionable problems. |
| Roofline begins with next-run configuration and a large candidate table while a run is active. | The current activity and best evidence are below setup material. | Separate the immutable active experiment from a new draft; open active experiments in Observe. |
| Roofline progress displayed 381 of 242 cells, or 157%. | Attempts, retries, confirmations and planned configurations appear to share an incompatible denominator. | Progress by unique planned configuration and phase; report attempts separately. Do not cosmetically cap an incorrect percentage. |
| The spectrum, matrix and every-cell table use different result selection paths. Llama FP8 appeared at about 16.2k in the spectrum and 21.1k in the matrix. | Users cannot tell which number is authoritative. | One result-selection policy and shared presentation model for every view. Explicitly label exploration versus validation. |
| The best-result heading says generation throughput while adjacent prose says highest total token rate. | The UI undermines the measurement definition. | Central metric dictionary used for labels, explanations, units and export. |
| Results exposed a history of 612 entries, with up to 500 displayed; many repeated the same persona heading. | A single search overwhelms the history and obscures the user's actual experiment. | Parent experiments, with candidates and attempts nested beneath them. |
| Workload shows capacity metrics and empty charts while a roofline experiment is active. | Empty telemetry can look like a broken or idle server. | One shared run workspace that renders the selected method's relevant metrics. |
| Optimize exposes fourteen dimensions and long cost/scoring explanations before a decision. | Advanced search-space mechanics dominate the common path. | Recommended scope summary with on-demand dimension editing and a concise budget preview. |
| Edit workloads says edits save immediately but also presents a Save button. | Persistence and the effect on an active run are unclear. | Explicit Save changes, saved state, and immutable workload versions for launched experiments. |
| Screens largely use the same small, muted type and similarly prominent rounded panels. | Surface decoration cannot establish a clear reading order. | Strong page titles, one primary action, fewer enclosing panels and a consistent type scale. |

Strengths to retain: plain-language workload summaries, contextual explanation of engine controls, the model/engine matrix, workload presets, existing chart narratives, persistent run state, keyboard-aware top navigation, and offline frontend assets.

## Information architecture

Use four persistent destinations: **Overview**, **Experiments**, **Compare**, and **Library**. Keep **System** and the current host's readiness in a utility area. New experiment is one consistent primary action, available wherever appropriate.

- Overview answers what is running, what needs attention, and what was learned recently. On a configured host, do not require preparation again.
- Experiments groups setup, execution and findings around a named question. Open an active experiment to Observe and a completed one to Findings; retain direct links to either view.
- Compare opens a deliberate comparison of two to four selected experiments or configurations, showing differences in methodology before performance deltas.
- Library holds models, engines, workload templates and saved launch configurations. Its default view is searchable and scoped to ready or recently used resources. The full catalog remains available.
- System holds hardware topology, storage, health checks and telemetry availability. Readiness also appears contextually during experiment creation.

The key hierarchy is `Experiment → Candidate configuration → Attempt → Measurement window`. A standalone benchmark is an experiment with one candidate. Search retries and confirmations belong to their candidate, rather than becoming unrelated history entries. Existing data may lack parent IDs: preserve it as legacy history unless a trustworthy relationship can be established from stored references; do not infer parentage from timestamps alone.

## Shared narrative flow

### 1. Question

Offer two prominent choices:

- **Size a workload:** How much demand can this configuration support at the chosen response targets?
- **Find peak performance:** What is the maximum sustained generation throughput under explicitly selected conditions? Preserve Roofline as the familiar secondary label.

Compare is a workspace action. Optimization is a method inside an experiment, not a compulsory separate destination. Expert users can duplicate a previous experiment or use a saved recipe directly.

The selected question determines metric meanings and the test method. Selecting a persona must never silently change from capacity measurement to a saturation benchmark.

### 2. Scope

Show a single focused decision area and a compact summary of choices. For capacity, choose workload, deployment and response targets. For peak performance, choose models, engines, prompt/output shapes and cache policy. Both offer use a saved configuration or tune automatically.

Default to a small recommended or explicitly selected scope. Provide a searchable selection drawer for large catalogs. Show availability and compatibility next to choices. Ineligible options include a reason and remedy; do not silently disappear. Model family, precision, version and full repository ID remain distinguishable.

Expert controls expand by topic: GPU placement, memory and cache, batching, engine-specific settings, measurement policy. Overrides are visible in the summary and survive switching between guided and expanded views. Changing presentation mode never changes the experiment objective.

Readiness is contextual: highlight missing resources for the chosen scope, estimated download size and available disk. A first-time host can open a preparation subflow and return to its saved draft.

### 3. Review

Present a concise, reproducible plan: question, host, models/engines, request shape, precision, cache policy, search budget, validation policy, relevant response targets and outstanding prerequisites. Label runtime estimates as estimates, preferably ranges; explain their assumptions on demand.

Separate required work from optional tuning. Name the primary action specifically, such as Start peak search or Start capacity test. Persist the draft, and capture an immutable configuration and workload version at launch. While another job is active, allow saving the draft and returning to the active experiment. Do not promise a queue or pause/resume capability the backend does not implement.

### 4. Observe

The first viewport answers: What is happening? Is it healthy? What has completed? Is intervention required?

Use a phase rail: Prepare resources → Search/measure → Validate → Complete. Show the current model/engine and useful work within the phase. Count unique completed configurations separately from attempts, retries and confirmations. If adaptive planning changes the denominator, explain the change. If remaining work cannot be predicted, show completed work without a misleading percentage.

One dominant chart should answer the active method's question. Peak: generation rate and validation window. Capacity: demand versus response target, with queue behavior available alongside it. Show no more than three or four primary metrics; put host telemetry, per-device data and raw events behind local views or a details drawer.

Group repeated failures by cause and model/engine. Explain whether the experiment continues, whether valid findings survive, and what action is available. Distinguish process state from telemetry connection state. A lost browser connection is not a failed run. Show last successful update and reconnect state while retaining the last known values visibly marked stale.

Use only supported lifecycle actions. Explain Stop's effect on the current measurement and saved evidence before execution. The active plan stays read-only; editing creates a new draft.

### 5. Findings

Lead with a concise conclusion, the selected measurement and its evidence status. Show the workload and hardware conditions immediately below. Capacity results lead with supported demand and target compliance; peak results lead with generation throughput and validation evidence.

Provide local views: **Summary**, **Compare candidates**, **Evidence**. Summary contains the conclusion and one explanatory chart. Compare candidates offers the matrix or ranked comparison, one at a time. Evidence provides exact windows, success/error accounting, methodology, configuration, logs and source run identifiers.

Use evidence labels such as Validated, Exploratory, Incomplete, Invalid measurement and Legacy evidence. Define each label with measurable backend criteria. A completed process does not imply a validated result. Old data missing newly required fields should say Not recorded or Legacy evidence, not silently pass.

Next actions are goal-specific: validate this candidate, use its launch configuration in a capacity test, compare with another experiment, or export findings with their assumptions. A maximum-throughput winner is not automatically a capacity recommendation or a model-quality recommendation.

## Replace repeated lists with views that answer questions

| Current surface | Default replacement | Detailed access |
| --- | --- | --- |
| Doctor table | Host summary plus actionable issues | System diagnostics |
| Model catalog | Search and a short selected-model summary | Full catalog drawer/page |
| Optimizer dimensions | A recommended recipe with overrides count | Grouped expert settings |
| Search attempts | Current candidate, phase and grouped exceptions | Paginated candidate/attempt table |
| Spectrum + matrix + model bars + engine bars | One selected comparison view | Switch view without duplicating content |
| Every-cell table | One authoritative row per candidate | Expand retries and confirmations |
| Hundreds of history cards | Named experiment rows, search and filters | Child runs within experiment |
| Many telemetry panels | One primary chart with contextual metrics | Diagnostics and per-device detail |

Tables remain valuable where precise comparison is the job. Use a shared table component with meaningful column sets, sorting, row actions, pagination, stable selection and an accessible details view. Do not replace a long table with an equally long wall of cards. Start with roughly five useful columns and expose additional columns deliberately. Preserve selection and scroll position on refresh.

## Data presentation contract

Create a shared metric definition and formatting layer. Each metric records its canonical key, label, unit, aggregation/window, direction, source, validity, missing-data reason and exact value. All summary cards, charts, tables and exports derive from the same selected result.

- Generation throughput: generated tok/s; prompt throughput is separately named. Total throughput is secondary and explicitly includes prompt tokens.
- Successful completion throughput: explicitly distinguish delivered successful work from engine generation where the data supports that distinction.
- Requests: Offered concurrency, Running requests and Queued requests. Avoid bare Streams or Concurrency when the meaning differs.
- Latency: Time to first token and Time per generated token, with percentile and measurement method. Use a consistent unit per table column or axis; do not mix scales within one column.
- Power: GPU power in W or kW with hardware scope. Energy efficiency is generated tokens/J (equivalent to tok/s divided by W), clearly identified as GPU-only when appropriate.
- Prompt shape: target user-text tokens versus actual engine input tokens; include chat-template overhead where measured. State output length, EOS and reasoning policy.
- Validation: measured duration, repeats, variability, scrape completeness and success/error breakdown, with unknowns visibly distinguished from zero.
- Display: compact figures in summaries (120.5k tok/s), exact values in details/export. Consistent significant digits, tabular numerals and right-aligned numeric columns.
- Names: readable model name with precision visible; full ID accessible without relying solely on hover. No unexplained gmu/mns/mbt in the guided path.
- Comparisons: show differences in hardware, method, workload, precision, cache and window policy before presenting relative performance. Only describe speedups as like-for-like when that is true.

Maintain two orthogonal states: execution lifecycle and evidence quality. Do not overload green to mean running, completed, best, healthy and validated simultaneously. Keep chart series colors stable across screens and pair status color with words/icons.

## Visual and interaction system

Use the current navy/teal identity as the starting point. Establish a 4/8px spacing system, a deliberate type scale, standard control heights, two density levels, and a small set of radii and elevations. Prefer sentence-case titles over widespread small tracked uppercase labels. Use a self-hosted readable sans-serif and tabular numerals; no runtime CDN dependency.

Three material levels:

1. A quiet background with restrained, static lighting.
2. Stable high-opacity content surfaces for charts, data, forms and long reading.
3. Dimensional glass for navigation, contextual action bars and detail drawers: subtle tint, a narrow highlight edge, layered shadow and limited backdrop blur.

Glass communicates hierarchy. Keep numerical content flat and sharp; avoid refraction through labels, nested blur stacks and animated backgrounds. Provide high-contrast and opaque fallbacks. Make depth most expressive in a hardware topology view, where actual GPU groups, replicas and memory placement make the dimensional treatment meaningful. A flat equivalent must remain available. No 3D data charts or tilting tables.

Motion specification (proposed product tokens): controls 120–160ms, selection/step transitions 180–240ms, drawers 220–280ms. Prefer transform and opacity. Retain spatial continuity as scope choices become the review summary. Never animate an unmeasured number through intermediate values or continuously reorder a live ranking beneath the pointer. Honor reduced-motion preferences; pause nonessential effects when hidden. Animate only meaningful state changes, and avoid repeatedly animating every telemetry sample.

Shared components: application shell, host status, experiment row, step navigator, field/help/error treatment, searchable resource selector, preset selector, settings section, metric display, evidence badge, progress-by-phase, chart frame, comparison table, details drawer, empty/error/stale state, and export summary. Each needs keyboard, loading, disabled, error, long-text and reduced-motion behavior before being reused.

## Implementation sequence

### Phase 0 — common language and trustworthy view models

Define experiment identity, lifecycle/evidence states, progress counters and the metric dictionary. Inventory existing endpoints and stored parent references. Centralize winner selection. Add legacy-data handling and eliminate contradictory metric labels and progress over 100%.

Done when the same selected result produces identical values and qualification in the summary, matrix, table and export; unknowns remain unknown; attempts cannot inflate completion progress.

### Phase 1 — shell and component foundation

Implement tokens, typography, surfaces, core controls, shared status and new navigation behind a versioned entry point. Keep existing services and offline packaging. Reuse the current chart library where it fits. A frontend framework change is optional and should follow component/state requirements rather than be a prerequisite for visual work.

Done when a reference page demonstrates all component states at 1440px, 1024px, 768px and a narrow viewport, with keyboard access and adequate contrast.

### Phase 2 — one complete peak-performance experiment

Build Question → Scope → Review → Observe → Findings for the current Roofline workflow. Use real fixtures for first run, active run, missing resource, interrupted run, partial failure and legacy evidence. Add deep links, persisted draft choices and immutable launched configuration.

Done when an operator can start a planned test, leave, return, identify its current phase, find its selected result and inspect the evidence without navigating unrelated setup pages.

### Phase 3 — capacity, optimization and grouped history

Bring capacity into the same shell with goal-specific setup and metrics. Make optimization an optional method with saved recipes. Migrate trustworthy parent/child history, implement comparisons, and move workload editing and assets into Library.

Done when a solutions architect can duplicate a benchmark, change the workload, see methodology differences, and export findings without confusing capacity and saturation claims. Experts can reach and inspect all launch settings without losing their draft.

### Phase 4 — visual polish, performance and accessibility

Apply final glass materials and motion after layout/state behavior is stable. Exercise reconnects, large histories, long model names, unavailable telemetry, failed confirmations and rapidly changing progress. Ship prebuilt local assets. Load tables/charts on demand and update affected elements instead of repainting entire pages on each poll.

Done when navigation and selection stay responsive with thousands of historical attempts; animation does not degrade telemetry responsiveness; keyboard and screen-reader checks pass; reduced motion and opaque mode retain full functionality.

## Evaluation tasks and success criteria

Test with both audiences. Ask them to create a peak test on a prepared host, resume observing an interrupted session, find the latest validated result, identify why a candidate failed, compare equivalent results, and reuse a launch recipe for workload sizing. Measure time, navigation detours and mistakes against the current UI rather than claiming unmeasured improvements.

Acceptance targets: active experiment phase and next action visible in the first viewport; no capacity metric shown for a peak-only mode; every headline has visible evidence status and workload context; every table and chart links back to the same measurement; successful checks are collapsed by default; no unbounded list of child runs on the landing screen; no draft setting can be mistaken for the configuration currently executing.

## Design references

The recommendation uses progressive disclosure and shared table behaviors described by [IBM Carbon's data-table guidance](https://carbondesignsystem.com/components/data-table/usage/). The material strategy draws on [Apple's materials guidance](https://developer.apple.com/design/human-interface-guidelines/materials), particularly using glass for navigation and controls and responding to reduced-transparency/high-contrast settings. These are design references, not a recommendation to import either platform's entire component system.
