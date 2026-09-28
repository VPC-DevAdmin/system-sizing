# Parallel workspace UI

The Classic UI remains the default at `/` and `/index.html`. Choose **Try new UI** in its header to open `/workspace.html`. **Classic UI** in the new sidebar returns to the corresponding original section. Switching interfaces does not start, stop, or transfer a benchmark. Both use the same service and run lock.

The new workspace has five destinations:

- **System:** hardware inventory, detected versus configured device grouping, doctor checks and remediation, model cache location, and the existing guarded format/mount procedure.
- **Assets:** the model catalog and downloads, engine staging, compatibility notes, and existing launch profiles.
- **Lab:** Roofline, launch optimization, workload capacity, and saturation. Setup and Workspace are directly selectable; there is no wizard.
- **Workloads:** the existing persona/cohort editor, distributions, and experience thresholds, with explicit Save semantics.
- **History:** searchable, method-filtered runs grouped by their actual directory, paginated in sets of 12 groups, comparison, full/slim JSON export, and pinned Roofline evidence.

## Roofline and evidence

Setup exposes resume, retry failed engine cells, remeasure an engine, fresh investigation, model priority, and the optional KTransformers scaling matrix. Fresh and remeasure actions have an explicit impact confirmation; basic cell estimates label supplemental scaling work separately.

Matrix, Spectrum, and Hardware lenses share the selected pairing. Tier filtering and explicit absolute/within-model scaling apply to the matrix. Changing the metric displays that metric for the same selected generation result; it does not silently select a different candidate. Spectrum retains generation rate versus weight size, with keyboard-accessible rows for selection.

Pairing results come from `summary.best_per_pair`, using the same completion/confirmation/rate policy as the service summary. Failures, missing data, untested cells, and unplanned pairings remain distinct. Pins are session-only snapshots and preserve the measurement conditions.

**Inspect recorded run** opens a selection-scoped evidence dialog with the concurrency ladder, selected measurement telemetry, available per-device readings, attempt/source records, and JSON export. `/api/workspace/roofline-evidence` resolves its run from recorded state and restricts access to the configured runs directory. It opens databases read-only and selects the cohort and offered concurrency associated with the chosen result. At most 601 sampled points span that window; aggregate values remain the recorded values. Missing and legacy telemetry is explicit.

## Implementation boundary

The new entry point loads canonical form markup from `index.html`, then reorganizes its DOM before initializing the shared controllers. Only the new page loads `workspace.css`, `workspace.js`, and `workspace-evidence.js`. Existing launch, download, storage, stop, and export APIs retain their behavior. The original entry point, scripts, and stylesheet remain in use by Classic; its only markup addition is the switch link.

The new UI does not execute package installation or format drives, add unsupported run recovery modes, or synthesize hardware telemetry. Existing operator procedures and backend limits remain visible. API/CLI-only features from the wider redesign specification have not become active UI controls merely by being included in that specification.

## Deployment

Static UI additions are available immediately on the running service. The read-only evidence route requires the updated Python service to load at its next restart. When the service has not loaded that route, the dialog explains the limitation; the inline pairing evidence and exports remain available.

On 22 September 2026, the new files and switch were installed into both `/data/system-sizing` and the installed capsim package on XE7740. The existing Roofline search was left running. The Python router addition is staged for the next normal restart. Original modified files were backed up under `/tmp/capsim-workspace-20260923/originals` on that host.

## Verification

- `node --test tests/ui/workspace-data.test.mjs` checks result authority, metric normalization, state distinctions, and escaping.
- `pytest tests/test_workspace.py` checks both static entry points, canonical selection, measurement scoping, legacy handling, bounded samples, and path containment.
- Existing Roofline, service, storage/model, optimizer, and workload tests cover shared behavior.
- Browser checks cover the real XE7740 Roofline matrix, metric/tier controls, Spectrum, optimization history, grouped run history, assets, workload editing, Classic/New switching, and a 390-pixel mobile layout.
- The detailed evidence dialog was verified against an isolated recorded-run fixture. No production benchmark or download was launched during UI validation.
