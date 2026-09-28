# capsim: comprehensive UI redesign specification

22 September 2026. Supersedes `ui-redesign-plan.md` and the earlier five-step and three-tab concepts.

## 1. Scope, evidence and design commitment

capsim is a system preparation, hardware exploration, engine optimization and workload-capacity application. It must answer both “What can this machine do?” and “What can this deployment support?” without losing the evidence behind either answer.

The previous proposals jumped from a partial inventory to a layout. This specification starts from the current UI modules, service routes, request schemas, measurement/search code, deployment documentation, metric documentation and test inventory. Local source baseline: `c125cfc`. The checkout is being edited independently; existing engine/test changes were not touched. The live SSH-forwarded UI was unavailable during this fresh pass, so this is a source-grounded redesign informed by the earlier live-screen review, not a fresh verification of the deployed host.

The inventory includes all current UI modules: control, storage, models, engines, optimizer, roofline, headline, live, results and editor. It also covers important CLI/config capabilities that the existing UI does not fully expose. These are explicitly distinguished below.

**No forced wizard.** An experiment has one Setup page and one Workspace. A user can start from the hardware, a model, a saved workload, an existing result or a saved launch configuration. Preparation is visited when relevant; it is not a mandatory sequence on every run. Experts and guided users share state, with topic-specific disclosure rather than separate applications.

**No feature disappears because it is difficult to fit into a clean screen.** Each inventory item has a destination, a detail surface and a verification scenario. Reduce repeated presentation, not capability.

Mockups are interactive design fixtures with illustrative values, not live measurements. They demonstrate layout and selected interactions. The coverage matrix is the implementation contract; mockup controls do not imply that a backend operation already exists.

## 2. Navigation and responsibilities

Five stable destinations organize enduring jobs, rather than stages in a funnel:

| Destination | Primary job | Main surfaces |
| --- | --- | --- |
| System | Make the target ready and understand the hardware | Readiness, storage, dependencies, hardware/topology, collector coverage |
| Assets | Manage what can be run | Model catalog/discovery, cached artifacts/companions, engine runtimes, launch profiles |
| Lab | Configure and execute an investigation | Setup and Workspace; Roofline, launch optimization, workload capacity, saturation/shape search |
| Workloads | Define the traffic and experience requirements | Persona editor, distributions, cohorts/mixes, saved summaries |
| History | Find, compare, reuse and communicate evidence | Grouped investigations, individual runs, shared comparison and export |

The shell always names the actual target, connection freshness and current host job. It allows returning to an active experiment from any destination. A target is local CPU, local GPU, hybrid CPU/GPU, remote endpoint, or mock. This is not a claim that multi-host orchestration exists: a host switcher may initially only describe the current service and expose a configured remote endpoint through setup.

The current single-run lock remains global across Roofline, Optimize and benchmark modes. Drafting, reading history and editing a workload remain available while busy. No fake queue, universal pause or multi-host scheduler is introduced. Asset preparation that could affect benchmark resources must expose the conflict and follow the actual backend lock policy.

Navigation between screens preserves drafts, selected cell/model, metric, filters, scroll position, time window and comparison selection. URLs should deep-link to a run or candidate, not merely a top tab. Opening a historical run never substitutes live host metrics for its recorded metrics.

## 3. Capability coverage and traceability

Status: **Existing** = current UI/API supports the core capability; **Config/CLI** = current core supports it but UI/API work is needed; **New** = a proposed capability requiring an explicit data/service change. Mixed rows explain the boundary. This is a coverage inventory, not a promise that every listed action already has an endpoint.

| ID | Capability and important details | New home / detail | Source and status |
| --- | --- | --- | --- |
| S01 | Host doctor: CPU, NUMA, GPU, Docker/toolkit, disk, telemetry permissions, HF reachability/token presence, recommended profiles | System readiness; actionable issues, successful checks collapsed | `doctor.py`, `service/runs.py` `/api/doctor`; Existing |
| S02 | Hardware: detected vs configured device count, VRAM, RAM, PCIe/NUMA groups and unverified topology | System hardware; same topology component in Lab inspector | `arena.py`, `/api/hardware`, `/api/arena`; Existing; distinguish detected/configured/fallback |
| S03 | Mounted filesystems, free/total space, current HF cache path and configuration source | System storage; cache location editor | `ui/storage.js`, `service/prepare.py`; Existing |
| S04 | Unmounted drives, existing partitions, check-before-format, filesystem/mount/fstab instructions | System storage; guided operator review and host instructions | Existing instructions only. Current service explicitly does not execute format/mount. Privileged execution is New and separately scoped |
| S05 | Cache-location changes propagated to downloads, engines, doctor and search | System storage; impact review | Existing directory selection; moving old weights is not implied; migration/eviction is New |
| S06 | Runtime/package/driver setup and telemetry access remediation | System dependencies; exact remediation and recheck | `install.sh`, `docs/deploy.md`; Config/CLI/manual, not a universal package-install API |
| S07 | Smoke test and preflight, profile recommendations, mock pipeline check | System readiness; verification job detail | `cli.py`, `preflight.py`, docs; CLI smoke/preflight, UI bridge needed |
| S08 | Collector availability and degraded evidence | System coverage and every run inspector | `docs/metrics.md`, collector status exports; Existing data; shared presentation New |
| A01 | Catalog filtering: family/series, precision, size, sparsity, specialty, cached status, text search | Assets/models; scoped table and detail inspector | `ui/models.js`, `model_catalog.py`, `arena.py`; Existing across surfaces |
| A02 | Add by HF ID, existence verification/offline-unverified status, discovery, precision siblings, fit reasons | Assets/models; discovery drawer and add preview | `/api/models/add`, `/discover`, `discovery.py`; Existing |
| A03 | Download, partial/resume, retry, progress/logs, available space | Assets/models; persistent preparation activity | `/api/models/download`, `/api/models`; Existing; process restart durability must be specified separately |
| A04 | Safetensors vs GGUF companions, sharded files, config/tokenizer-only state, native KTransformers checkpoints | Model detail/artifact checklist and readiness predicates | `models.py`, `models.js`, `roofline.py`; Existing; never conflate config-only with complete weights |
| A05 | Engine image status, pull/build instructions, size, image store/free space, caveats, versions/logs | Assets/engines | `engine_runtimes.py`, `ui/engines.js`; Existing pull; CPU build/setup partly CLI |
| A06 | Curated/custom/optimized profiles, provenance and compatibility | Assets/profiles; setup selector; promotion review | `/api/profiles`, `/optimizer/promote`; Existing; versioned recipe history New |
| L01 | Local CPU, GPU, hybrid, remote and mock targets | Lab Setup/target; conditional controls and collectors | Engine registry, profiles, `engines/remote.py`; core Existing, complete target UI needs wiring |
| L02 | Manual launch controls: engine, replicas, TP, pack/spread, memory share, sequence and prefill budgets, KV dtype, EP, trust-remote-code | Setup grouped expert sections with visible overrides | `ui/control.js`, `engines/knobs.py`; Existing; preserve explicit edits |
| L03 | CPU binding, NUMA, CPU-expert/offload options and engine-specific levers | Target-specific expert sections and launch profile detail | CPU/hybrid engines, search/arena; Config/CLI plus existing arena levers |
| L04 | Feasibility and cost preview: eligible/excluded models, launch count, cold vs hot model reload cost, declared vs detected hardware | Setup scope summary; exclusions inspector | `/api/arena/preview`, `arena.py`; Existing for arena; shared preview across methods New |
| L05 | Launch engine without benchmark for manual endpoint testing | Lab Setup/advanced, launch-only job detail | `capsim launch-engine`; Config/CLI; UI lifecycle bridge needed, retaining single-host ownership/stop semantics |
| R01 | Roofline shortlist: auto/cached/manual, vendor diversity and selection rationale | Roofline Setup, model selection inspector | `/api/roofline/candidates`; Existing |
| R02 | Fast, large GPU-resident and beyond-VRAM tiers; largest served vs largest attempted | Roofline workspace lenses and spectrum | `roofline.py` Candidate/TIERS, `ui/roofline.js`; Existing |
| R03 | Cross-domain TP opt-in; GPU fit vs host RAM/companion/native-checkpoint compatibility | Roofline Setup/topology; excluded pairing reasons | `RooflineRequest`, planner; Existing |
| R04 | Engines × model × launch/request shapes, batch widths, output lengths, pinned input | Roofline Setup; matrix cell → shape grid | `RooflineRequest`, headline modules; Existing |
| R05 | Automatic staging, searching, confirmation; plan identity and persisted recovery | Roofline workspace phase header and event detail | `roofline.py`, `service/roofline.py`; Existing |
| R06 | Resume matching cells, fresh search, retry failed engines, redo engines, priority models | Setup/recovery review with preserved/invalidated counts | `RooflineRequest`; Existing core, not all current UI controls |
| R07 | Fastest, largest served, largest attempted, per-model/per-engine winners, every cell | Roofline matrix/spectrum with linked inspector | Existing; one consistent evidence/result selection layer required |
| R08 | Shape/concurrency goal-seeking, held vs offered requests, stopping reasons, convergence | Roofline model/run inspector; saturation workspace | `headline_search.py`, `headline_optimize.py`, `headline_sweep.py`; Existing algorithms differ; do not merge their scoring semantics |
| O01 | Full arena, model filters, dimensions, feasible subset, budget, resume/fresh | Optimize Setup, with compact grouped dimension editor | `ui/optimizer.js`, `/api/arena/preview`; Existing |
| O02 | Guided coverage/rank/refinement, seed, top-k, improvement stop, SLA-aware scoring; registry sweep | Optimize workspace convergence/coverage + candidates | `search.py`, optimizer driver; core Existing, some settings config-only |
| O03 | History, combined investigations, source scoring, promote winner | Optimize inspector and History, Use launch profile action | optimizer history/combined/promote APIs; Existing |
| C01 | Persona/cohort selection and workload summaries | Capacity Setup, link to Workloads | `control.js`, catalog API; Existing |
| C02 | Open-loop Poisson arrivals: stable limit λ_max, SLA-compliant limit λ_sla, derived session concurrency | Capacity workspace/report | `open_loop.py`, `rate_search.py`, exports; Existing |
| C03 | Closed-loop fixed pool / adaptive two-knee ramp, sweeps | Capacity Setup explicit method; load axes change accordingly | `StartRunRequest`, CLI sweep; Existing core; some UI exposure needed |
| C04 | Saturation benchmark, concurrency cap, input shape, quick/standard/thorough joint engine/shape search, saved optimal shape | Lab method: Saturation; Setup/Workspace | `control.js`, headline modules, `/api/headline-shape`; Existing |
| C05 | Targeted spot checks and repeated workload sweeps | Run inspector/re-measure review; Capacity Setup/sweep selection | `capsim spot-check`, `sweep`, audit plan scripts; Config/CLI; preserve append/supersession semantics when adding UI |
| T01 | Live sessions, in-flight, queue, prefill/decode, rolling TTFT/TPOT, warm/cold/read/think states, step progress | Method-specific Workspace; synchronized detail | `live.js`, websocket/backfill API; Existing |
| T02 | GPU per-device SM busy, DRAM busy, VRAM, power, temperature, clocks, throttle, PCIe traffic | Shared run inspector/Hardware lens | `collectors/gpu.py`, export; Existing where supported |
| T03 | CPU bound-set/per-core utilization, cycle breakdown, frequency, AMX/PMU, NUMA, RAM/cache/swap, bandwidth, disk and NIC pressure | Shared run inspector/Hardware lens, especially CPU/hybrid | host/PMU/IMC/RAPL collectors; Existing conditional coverage |
| T04 | Chassis/GPU/CPU package power, energy efficiency with correct measurement scope | Run inspector, result/compare views | collectors and export; Existing conditional; do not add GPU watts to chassis watts |
| T05 | Warmup/settling/measurement, measured window selection, scrape completeness, client limitation, errors/reasoning-only responses | Shared inspector/Evidence | headline and export data; coverage varies by legacy/method |
| W01 | Persona name/description; input/output, turns, read/think times, lognormal/discrete distributions | Workloads persona editor with behavior preview | `editor.js`, persona schema; Existing |
| W02 | Soft/hard TTFT and TPOT thresholds, request timeout, validation relationships | Workloads experience section | Existing |
| W03 | Cohort membership, relative weights, normalization, duplicates/zero-total validation | Workloads mix editor; visual shares | Existing |
| W04 | Save, new persona/cohort, next-run effect; preserve active-run settings | Workloads and setup references | Existing save/validation; immutable workload version and history linkage New |
| H01 | Run listing, partial/interrupted/no-window runs, child measurements and archived searches | History grouped investigations and drill-down | runs/optimizer APIs; reliable shared parent IDs New; retain legacy runs |
| H02 | Capacity narrative, saturation report, ladder/step detail, lower-bound and client-limited results, attribution | History → same Workspace/Inspector used live | `results.js`, export schema; Existing |
| H03 | Multi-run overlays, method-aware axes and methodology differences | History compare tray and shared comparison view | Existing comparison; canonical comparability contract New |
| H04 | Versioned full/slim JSON, source run IDs, preserved conditions, deletion | History exports/run actions | export/delete APIs; richer human report/bundle New; deletion stays explicit |
| H05 | Curve audit and prefix-cache analysis | Inspector/evidence diagnostics, source report links | `make audit`, `analyze-prefix-cache`, scripts; Config/CLI; UI execution bridge optional, conclusions retain source windows |
| X01 | Busy conflicts, reload/reconnect/backfill, stale state, unavailable/invalid data, interrupt/retry/resume | Shared shell and operation state components | service state/telemetry/search persistence; Existing pieces, unified semantics New |
| X02 | Offline operation, vendored assets, keyboard navigation, reduced motion/transparency, dense/comfortable layouts | Whole design system | Existing no-build/vendored app; expanded a11y/responsive checks required |

## 4. The Lab is four distinct instruments in one workspace

All methods share Setup, Workspace, run identity, job status, hardware context and the evidence inspector. They do not share a generic winner card or interchangeable axes.

### Roofline: characterize the machine

Default Workspace: compact capability summary → one dominant model × engine heatmap → selected-cell inspector. The matrix is the durable finding. Keep three independent summaries: fastest measured configuration, largest successfully served model, and largest attempted boundary. Each links to evidence and names its workload, precision and validation status.

**Views within the same workspace:** Matrix, Spectrum, Hardware. These are analytical lenses, not successive pages. Matrix shows engine differences by model; Spectrum shows speed across size/weight footprint and all three tiers; Hardware shows resource pressure and device balance in the selected measurement window. Changing a lens retains the selected model/candidate.

**Matrix semantics:** default to absolute generated tok/s, with metric and normalization selectors. Optional within-model normalization helps compare engines without hiding absolute values. Label every scale; do not color each model on an undisclosed relative scale. Use separate visual/text states for not planned/incompatible, eligible but untested, running, failed, incomplete, exploratory and confirmed. Color strength expresses the metric; small status markers express evidence. A failure is never a zero-throughput measurement. Users can inspect unsupported pairings to see why.

**Model inspector:** selected engine pairing, candidate shapes, batch/output heatmap, convergence or concurrency ladder, TP/replicas/placement, best exploration vs independent confirmation, attempts and relevant logs. The parent matrix cell is a documented aggregation of these candidates. One canonical result selector feeds it, the spectrum, capability summaries and export.

**Three breadth questions:** fast tier seeks high throughput and vendor/series diversity; large tier tests what fits GPU memory, possibly with cross-domain TP; beyond-VRAM tests hybrid/CPU-expert possibilities, native/GGUF artifacts and host RAM limits. Preserve total vs active parameters, weight size vs precision, KV cost and fit reasoning. These are not merely “small/large” filters. “Largest served” does not certify useful latency, quality or workload capacity.

**Goal seeking:** show the actual method's objective, units, eligibility rules, phase, evaluated/remaining/reused work, current candidate and reason for testing it. Existing Roofline/saturation/guided search algorithms have different objectives and stopping rules. Retain those identities. Goal changes make a new draft/revision, with a preview of reusable measurements and invalidations. A generalized arbitrary objective builder, energy optimizer or Pareto search is a New backend feature; energy/Pareto visualization of existing evidence can be a read-only addition.

**Recovery:** distinguish resume existing matching cells, retry failures after an engine fix, redo one engine and fresh measurement. Preview the exact retained and rerun sets; never imply that old measurements became comparable after a software/config change. Show persisted state separately from whether a live worker is currently driving it. No “running forever” from a stale state file.

**Actions:** open run evidence, pin comparison, use launch configuration in a capacity draft, inspect exclusion, resume/retry. Refinement around a manually selected cell and arbitrary single-cell confirmation need service support before they become active production controls. Mockups label these proposed actions and only create local draft previews.

### Launch optimization: find a deployment configuration

Keep Optimize as an explicit Lab method, not hidden behind a miscellaneous advanced switch. Setup shows model eligibility, dimension groups, fixed values, objective/caps, measurement ladder, budget, feasibility and relaunch cost. Guided coverage and neighborhood refinement differ from fixed registry sweeps and must be named.

Workspace shows best objective over evaluations, coverage/refinement phase, improvement/stop reason, candidate comparisons and a selected candidate's exact settings. Include total token/latency components behind composite scores; never show an unlabeled score that looks like tokens/sec. Use profile promotion with a provenance preview. Group history is available, but combining prior searches must preserve differences in objective, workload, hardware and software.

### Capacity: find a defensible service limit

Setup exposes target/profile, persona or cohort, methodology and relevant thresholds together. Default open-loop uses arrival rates; closed-loop pool ramp remains explicit for historical comparison and sweeps. Remote endpoints need URL/model/auth reference/optional metrics connection, with secrets masked and omitted from exports; setup must not imply capsim owns that engine.

Workspace anchors on the load-response relationship. Show λ_sla and λ_max separately, queue stability, SLA violations and latency, plus derived active sessions. A requests/sec number is not a sessions/min capacity. In-flight requests, offered users, active sessions and engine queue are independent measures. Mark lower bounds, unobserved knees, aborted windows, superseded measurements and client-limited windows.

Selecting a load point synchronizes queue/latency, session composition and hardware evidence to the same window. Live observations remain provisional until a window verdict exists. Finished results use the same workspace rather than a different chart/report implementation.

### Saturation and headline shape search: inspect the maximum-rate curve

Setup exposes fixed prompt/output conditions, concurrency ladder/cap and shape/joint-search preset. Workspace centers throughput versus offered concurrency and actually held batch, with latency, prefill/decode, power, queue and KV pressure available in linked views. Label the selected peak and stop reason. Keep fixed-output/EOS/reasoning/cache policy, target versus actual prompt length, successful completions and full-replica scrape coverage visible in evidence. A saturated peak is not a capacity/SLA claim.

## 5. Preparation is a durable operational workspace

System readiness summarizes blockers, warnings and usable capabilities for the selected target/workload. A host can be ready for a GPU throughput run with incomplete CPU power evidence. Readiness is not one unconditional green boolean.

Storage shows real device identity, partitions, mountpoint, filesystem, free/total space, cache path and source (environment override/config/default). Treat HF model cache, container image storage and result/log storage separately. An environment-pinned path is read-only with an explanation. Changing the selected directory does not move existing files; show what becomes unavailable and how to return.

The existing format/mount assistance remains an operator procedure: inspect contents, select the intended drive, review destructive consequences, show host commands, then recheck. Do not silently equate a successful refresh with a completed format. A future privileged execution service needs explicit authorization, stable device identity, mount collision checks, root-disk protection, logging and recovery; it is not part of a frontend reskin.

Dependencies show installed/required versions, exact remediation, access/reboot needs and recheck. Separate host prerequisites from engine-image staging. Doctor and smoke/preflight have different claims: detection/permissions are not proof that an engine can stream and export successfully. UI smoke execution needs a service bridge.

Assets has short working-set views and a searchable catalog, not a giant list embedded in preparation. An artifact has catalog identity, immutable revision where known, precision/format, engine compatibility, cached pieces and destination. Config/tokenizer-only and GGUF-companion requirements are explicit. Show partial/error/auth/disk/offline states and resume/retry. Existing downloads survive browser reload; service/process restart recovery must be verified before promising it.

A shared operation drawer displays downloads, pulls, builds/checks with phase, progress if available, start time, log tail and next action. Do not invent precise percentages when the process only supplies log text. Cancel/retry is only offered where backend lifecycle semantics support it. New downloads during measurement need explicit resource-interference policy.

## 6. Workload design and reusable launch knowledge

Workloads are first-class assets, not a form at the end of the navigation. Persona editing shows a behavior preview: ask → wait/prefill → stream → read → think → next turn. Controls preserve all current distributions (constant, lognormal, discrete), variability, turns/session and input/output length. Thresholds separate preferred experience from hard failure and timeout.

Cohorts show normalized traffic shares, member behaviors and total validity. Keep zero-total, duplicate members, missing references and invalid thresholds inline. Name and description lead; machine ID is secondary. Save changes has one unambiguous persistence model. Back/close with unsaved changes must preserve or explicitly discard them.

A launched experiment retains the actual workload snapshot. Adding immutable version identifiers is New; until then, show the captured configuration and clearly identify historical fields not recorded. Edits affect the next run, never rewrite earlier claims.

Profiles preserve optimization source, model revision, engine/image/version, CPU/GPU placement, launch levers and measurements supporting the recommendation. Applying a profile previews changed fields and respects manual overrides. A workload change may invalidate the relevance of the profile without making the underlying measurement invalid.

## 7. Shared run inspector and hardware perspective

One inspector is used from heatmaps, spectrum points, ladders, optimizer candidates, history rows and comparison charts. It retains its originating context and has four local sections:

1. **Configuration:** model/revision/artifact, engine/image/version, launch arguments, TP/DP/cpuset/NUMA, hardware snapshot, workload/method, cache/output policy and source profile.
2. **Measurement:** phase/window, load point, time-aligned throughput/latency/queue, session states, successful/failed/no-content/reasoning-only outcomes and settling.
3. **Hardware:** GPU topology and per-device details; CPU/per-core/bound-set/AMX, RAM/cache/swap/bandwidth, disk/NIC pressure, power/clock/thermal state.
4. **Evidence & attempts:** independent confirmations, selection policy, repeat variability, coverage, collector status, failures/logs, source IDs and export.

Hardware defaults to the selected run/window, with an explicit switch for live host state. Per-GPU averages cannot hide imbalance. NUMA and PCIe grouping show whether they were detected, configured or inferred. Dimensional topology can help explain placement, but a flat table/list equivalent is required.

Metric distinctions are mandatory: SM busy is not FLOP utilization; DRAM-controller busy is not measured GB/s; VRAM occupancy is not KV occupancy; warm sessions do not identify HBM-vs-DDR residency; aggregate MoE load is not per-expert routing. The current collectors do not support those stronger claims. “Roofline” here means empirical performance exploration; a theoretical FLOP/byte roofline would require additional measurements and methodology.

Power scope travels with every value. Chassis includes devices; do not sum chassis and GPU watts. Generation tok/s divided by W is tokens/J. Missing RAPL/IMC/PCIe/remote telemetry is unavailable with a reason, never zero. Host telemetry from the client must not be attributed to a remote endpoint.

## 8. Evidence, history, comparison and export

Data hierarchy: Investigation → method/revision → candidate → attempt → measurement window; workload/profile/model and host snapshots are referenced or captured. Existing optimizer groups, roofline cells and run IDs can be adapted. Do not guess legacy parentage from names/timestamps. Unlinked historical runs remain browsable as Legacy.

Maintain independent lifecycle and evidence states. Completed does not mean confirmed. Search-selected, confirmed, incomplete, invalid and legacy labels have method-specific definitions. An exploratory finding remains useful and visible; filtering to confirmed results must not make the search space appear unexplored.

A metric contract contains key, unit, numerator/denominator, aggregation/window, source/collector coverage, success/error inclusion, validity/missing reason and exact value. One selection policy feeds cards, matrices, charts and exports. Scales, significant digits, model names/precision and series colors remain consistent across screens.

History groups investigations and shows method, target, scope, time/status and one relevant outcome. Candidate attempts are detail, not hundreds of peer rows. Search/filter/sort/pagination and stable selection are necessary. Row actions are tucked into a consistent menu; deletion is not a prominent repeated cross icon.

Comparison begins with 2–4 selected runs/candidates and a methodology difference strip: target/hardware, model/precision, engine/version, shape/cache, workload, measurement method/window, validation. Comparable metrics can be overlaid; incompatible axes are split. Relative speedup is only described as like-for-like when justified. GPU-only efficiency cannot be ranked against chassis efficiency as if equivalent.

Export offers the existing full and slim versioned JSON with run/source identity. A new concise human report and reproducibility bundle can be added explicitly; preserve negative/partial findings, conditions, telemetry availability and raw references. Deletion must identify exactly which record/artifacts are affected, whether reversible, and any links that become unavailable.

## 9. Component and interaction inventory

| Component family | Required components and states |
| --- | --- |
| Shell/context | Target identity, connection freshness, active-job banner, method identity, persistent navigation, contextual readiness link |
| Preparation | Check summary, issue/remediation row, storage capacity/device table, path editor/source lock, operator procedure review, artifact checklist, download/pull job with logs |
| Selection/setup | Searchable model selector with fit reasons, artifact/precision selector, engine compatibility selector, profile/diff picker, method selector, grouped dimension editor, feasibility and cost preview, resume/fresh review |
| Workloads | Persona behavior preview, distribution editor, SLA/timeout fields, cohort share editor, save/unsaved/validation states |
| Exploration | Heatmap with state legend and labeled normalization, spectrum chart, candidate/shape grid, search progress/trajectory, dimension comparison, selection inspector and comparison tray |
| Execution | Phase progress, unique-work counts vs attempts, immutable active setup, stop/recovery control, recent events, grouped failures, reconnect/backfill/stale state |
| Telemetry | Time-window selector, linked cursor/selection, GPU topology + flat view, per-device/per-core grids, session-state timeline, latency/queue/throughput chart frames, collector coverage |
| Evidence | Metric with method/unit/source, lifecycle/evidence badges, selected-result policy, measurement/attempt table, confirmation comparison, source/log drawer |
| History/reuse | Grouped history, pagination/search/filter, selection tray, comparability summary, profile promotion/diff, rerun draft, export options, exact-scope deletion confirmation |

Every interactive component requires keyboard focus/labels, loading, empty, invalid, unavailable, long-name and narrow-layout states. Every job requires pending/running/success/partial/failure/disconnected behavior; cancel/resume only where implemented. Do not communicate either evidence or error solely by color.

The interface can be dense without becoming a list wall: short working sets, one primary analytical surface, a shared inspector, and user-selected additional columns. Use tables for precise comparisons; do not turn every row into a large card. History and attempt sets are paginated/windowed. Keep row position stable while live data changes.

## 10. Visual and motion system

Keep capsim's navy/teal identity, but establish strong page titles, readable sentence-case labels, tabular numbers and two explicit density settings. Use a 4/8 spacing scale, standard controls, restrained radii and three levels of elevation. Use consistent icons from one self-hosted set.

Glass is concentrated in the shell, contextual inspector and action surfaces; charts/forms/tables remain high-opacity and sharp. Use edge light, tint and shadow to convey depth. Do not blur nested numerical surfaces. GPU/NUMA placement can use restrained 3D layers with actual topology labels; no tilting data charts or looping decorative shaders.

Motion conveys selection, panel continuity and meaningful state changes: controls 120–160ms, tabs/inspector 180–240ms, larger drawer transitions 220–280ms. Prefer transform/opacity; no fake count-up of measurement values, no continuous ranking reorders, no repeated chart entrances on polling. Reduced-motion and opaque/high-contrast modes preserve functionality. Final contrast must be measured, including heatmap labels at every intensity.

Responsive behavior: desktop shows primary view and side inspector; medium width places inspector below; narrow width keeps the main actions and a simplified chart/table alternative. A wide heatmap may use labeled horizontal scrolling with a sticky model column rather than shrinking text. No mandatory phone-first workflow for a lab application, but every action remains reachable.

## 11. Delivery plan and completion gates

Do not start with a blanket stylesheet rewrite or a framework migration. Preserve the current offline packaging and reuse Chart.js unless a demonstrated interaction requires another library. The proposed UI can be implemented incrementally through adapters over existing endpoints; backend additions are separate tracked work.

| Stage | Deliverables | Completion gate |
| --- | --- | --- |
| 0: scope and contracts | Coverage matrix, endpoint/state map, metric glossary, parent identity plan, design fixture set, mockup walkthrough | Every ID above has a destination and existing/config/new classification; no hidden feature retirement |
| 1: shared foundation | Shell, tokens/components, active-job lock, entity/metric adapters, deep links, shared inspector and stable data selection | Same run/window/value across every representation; draft/live/history cannot be confused; offline startup works |
| 2: full system preparation | Readiness/dependencies/storage, assets/discovery/artifacts, download/pull state, operator formatting assistance, profile inventory | Fresh-host journey and pre-staged offline host both reach a truthful readiness state; partial/auth/disk failures recover |
| 3: full Roofline instrument | Single-page setup, three tiers, matrix/spectrum/hardware, model/shape/run drill-down, confirmation/resume/retry, history link | User can characterize speed and size, explain a selected cell, inspect resource use, and recover a search without losing evidence |
| 4: optimization and benchmark instruments | Guided/registry optimization, promotion; capacity open/closed/sweeps; saturation/shape search; shared telemetry | All supported methods retain their distinct objective, axes and stopping semantics; CPU/hybrid/remote/mock paths work |
| 5: workloads and evidence lifecycle | Full persona/cohort editing, immutable capture, grouped history, comparison, export, reuse/deletion behavior | Round-trip edits preserve distributions; historical evidence remains reproducible and comparable where appropriate |
| 6: polish and migration | Final materials/motion, responsive/a11y, large-history performance, legacy fallback and rollout | Coverage acceptance passes; no dropped controls/metrics; keyboard/offline/reconnect and historical fixtures pass |

Stages 2–5 share the foundation but represent equal product scope. Roofline remains the highest analytical priority; it does not substitute for the other workflows. Keep the existing UI available until the corresponding destination satisfies its coverage gate. Use a versioned entry point/feature flag and a rollback path; don't migrate stored results destructively to support a view.

Potential backend work: unified investigation/candidate/attempt identities; immutable host/config/workload snapshots; unified method-aware result selection; persisted preparation job lifecycle; UI smoke/preflight bridge; objective/measurement editing API; selected-cell refine/confirm actions; human report export. Privileged install/format execution, fleet scheduling, model quality evaluation and theoretical hardware roofline measurement are separate decisions, not implied deliverables.

## 12. Acceptance scenarios: proof the redesign is complete

1. **Fresh GPU host:** diagnose toolkit/perf/disk issues; inspect format/mount instructions; choose cache; stage models/images; see completion and recheck. No unearned Ready state.
2. **Prepared/offline host:** enter Lab immediately; use cached weights and images; discovery failure does not disable local work.
3. **Model variants:** add/discover a model; distinguish FP8 weights from FP8 KV; stage config-only + GGUF or native checkpoint; explain incompatible engines and partial artifacts.
4. **Roofline breadth:** include fast, large and beyond-VRAM tiers; inspect excluded pairings/cross-domain consequences; see fastest, largest served and largest attempted simultaneously.
5. **Roofline exploration:** switch absolute/within-model heatmap, select a cell, inspect shapes/rungs, correlate utilization and errors with the same window, compare exploration and confirmation.
6. **Recovery:** close/reopen browser; disconnect telemetry; restart service; resume compatible work, retry failed engine, redo a changed engine, preserve prior evidence/provenance. State what survives each failure.
7. **Goal seeking:** configure existing guided objective/caps, inspect coverage/refinement and stop reason, compare candidates and promote a launch profile with provenance.
8. **Capacity:** run a persona/cohort; separate λ_sla and λ_max, session concurrency and in-flight requests; inspect unstable/client-limited/lower-bound results.
9. **Saturation:** view offered vs held concurrency, generation/prefill, peak and plateau/abort; verify cache/EOS/shape conditions and confirmation.
10. **Targets:** CPU-only hides GPU-only controls; hybrid shows host RAM/CPU and GPU together; remote reports collector skips and never stops an unowned engine; mock is clearly synthetic.
11. **Workloads:** edit long-tail distributions and soft/hard SLAs; normalize a cohort; reject invalid/duplicate members; preserve a launched snapshot after edits.
12. **Compare/reuse/export:** select archived runs, see non-comparable differences, inspect a missing collector, reuse a configuration, export full/slim without losing conditions.
13. **Scale/accessibility:** thousands of attempts, long model IDs, empty catalogs, one failed image pull, stale connections; usable keyboard focus, narrow layouts and reduced motion; no hidden selection loss.

Testing combines service contract fixtures, a small number of workflow integration tests and visual/a11y checks. For this planning deliverable, validate mockup navigation, selection/drill-down and responsive behavior; do not run real benchmarks or alter the host. User walkthroughs must cover both technical operators and solutions architects, measuring task completion and interpretive errors against the existing UI.

## 13. Mockup walkthrough and limitations

The comprehensive mockup provides the shared shell with System, Assets, Lab, Workloads and History. Lab retains only Setup and Workspace, with method-specific workspaces for Roofline, Optimize, Capacity and Saturation. Roofline contains matrix/spectrum/hardware lenses and selection-linked run detail. Other screens demonstrate the preparation, asset, workload, comparison and evidence components that earlier sketches omitted.

Use the mockup to assess hierarchy, information placement and cross-workflow continuity. All numbers are illustrative. Preparation and launch actions only change mockup state or open explanatory previews. New service-dependent controls are labeled as proposed. It is not an exhaustive functioning implementation of every inventory row; the coverage matrix and acceptance scenarios above define that implementation scope.

Prototype verification completed: primary workspace navigation; Roofline metric switching, beyond-VRAM selection and linked hardware; all four Lab method views; single-page setup; model companion inspection; cohort share updates; history comparison with incompatible-method messaging; reopening the matching workspace from history; visual inspection at desktop, 1024px and 390px widths. JavaScript syntax and literal markup were checked. This is not a full accessibility audit or a test of production services. No benchmark, download, install, format or mount operation was executed.
