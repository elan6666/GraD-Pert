# R50 build-all and smoke gate

User decision 2026-09-12 supersedes sequential full-run launch order:
finish authorized ablation code/configuration and tests first, then fresh
one-epoch integration gates; no new 50-epoch run before the build/smoke
campaign is reviewed. Existing results and failed roots remain immutable.

## Explicit pending rows

- G1/G2/G3: optimizer-only, schedule-only, combined (R50_GLM5_MUON.md).
- R1/R2/R3/R4: LR-low and official Scouter/GEARS/TxPert best/last reruns.
- C1: batch-condition-balanced expression MSE, no direction term.
- Remaining R50 families: EMA, capacity, prior, local and encoder/source
  interfaces and regression coverage per R50_E3_REDESIGN.md. Final method
  coordinates remain subject to its validation-only parent-selection and
  preregistration rules; do not invent a Cartesian product or revive A5/A1.

## Acceptance inventory

For every concrete row, record config hash, parent/diff, full source SHA,
unit/integration gates, upstream version when applicable, fresh smoke root,
actual train/validation completion, finite state, checkpoint role/hash,
resume behavior, resources and PKL cleanup. Missing evidence is pending,
never a pass. Smoke preserves the 50-epoch schedule horizon and accesses no
test truth. Verify dual-checkpoint evaluator wiring through synthetic tests;
actual best/last canonical test evaluation follows formal training.

The initial exact scope is eight already registered rows, not eight new
control runs. A row blocked by an official adapter or failed smoke remains
blocked while independent code work continues. No optimizer reset from
repeated one-epoch train calls. Official packages remain isolated runners.

Implementation uses bounded goals; they end before any CUDA smoke. Only
then launch capacity-safe smoke queues and restore the existing 30-minute
monitor. A failed smoke is preserved and repaired in a new published lineage.
GPU availability alone never authorizes a formal 50-epoch launch.

## Adapter audit at pre-change baseline 1e8630b (2026-09-12)

The external reruns cannot be launched merely by setting epochs=50:

- GEARS `benchmarks/gears/official_api.py:fit_one_epoch` explicitly permits
  full work only at 100/patience10 and calls official save_model afterward.
  Extend a separate R50 path, preserving a single continuous official train
  call and its optimizer state; independently audit best and final capture.
- TxPert `fit_with_validation` callback stops after ten non-improvements and
  loads best into the model at return. Capture actual last before that restore,
  and make no-early-stop/50 epochs explicit only for the new R50 policy.
- Scouter runner hardcodes full epochs=100 and writes only best.pt from the
  returned network. Audit the official fit helper's restoration behavior,
  then capture final before any best restoration and add dual evaluation.

These are verified code gaps, not completed fixes. The old adapters must retain
their existing semantics outside the explicit new R50 policy. No external
smoke or full run is ready until these paths and their tests are implemented.

## Build inventory (2026-09-12, before CUDA)

| Rows | Code/config state | Remaining gate |
| --- | --- | --- |
| G1/G2/G3 | Native split optimizer, endpoint schedule and self-contained configs implemented; targeted routing/update/resume and factor-diff tests pass | Native engine resume, full gates, diagnostic telemetry and clean server smoke |
| C1 | Condition-balanced squared-error reduction and isolated config implemented; analytic gradients/grouping tests pass | Native integration and clean server smoke |
| R1 | Existing LR-low config and native dual-checkpoint path available | Fresh identity and smoke, never relabel old missing last |
| R2/R3/R4 | Official adapter gaps audited above | Continuous50 training, actual last capture, dual evaluation and adapter tests, then smoke |
| Remaining EMA/capacity/prior/local/encoder | Existing native interfaces; exact future coordinates require parent freeze | Regression inventory and preregistered configs, not extra invented runs |

The mixed-optimizer CPU engine resume check uses a synthetic seven-node
Exphormer with a smaller projector (256/32), retaining native forward,
backward, optimizer, Teacher and center/checkpoint code. Full-size CPU attempts
were interrupted for cost; they are not passes. A first compact-capacity fixture
was rejected correctly because that whole-model profile is GAT-only; the final
fixture substitutes only its test projector modules, never a scientific config.
The corrected resume test passes with exact model/optimizer/center/RNG/gradient
state. Full E3 dimensions still require the fresh server smoke.

Current implementation local gate: 867 passed, four explicit skips (frozen
Scouter checkout, frozen TxPert reference, optional scLong reference, CUDA
checkpoint path), 91 warnings. Ruff, format349 files, strict mypy82 source
files and isolated wheel/sdist build passed. These skips are not silently
counted as external/CUDA coverage. Pre-change baseline is
`1e8630b6f862d3433fdd2bf51f42db908502d7c7`; the implementation publication
commit is discoverable from this file's Git history, not an old run's SHA.

Native R50 smoke already truncates the trainer to one epoch while the engine
keeps max_epochs times steps_per_epoch as its LR/EMA horizon. Its r50_selection
policy bypasses the test callback. Preserve these properties in every launcher;
do not rewrite max_epochs to1 to obtain a short run. External smoke adapters
must independently establish their matching properties, not inherit this claim.

## Frozen upstream inspection

Read-only server inspection on 2026-09-12 resolved Scouter to
`0cfddd000e19b72ff033ba67c8315f7bc3304932`, GEARS to
`f374e43e197b295016d80395d7a54ddb81cc6769`, and TxPert to
`08d82eea86746b044cf7531f4ec8c5f60e1cb73f`.
Scouter.py train lines123-232 creates Adam/ExponentialLR once, stores a shallow
best state and restores it at the end. The existing deepcopy adapter fixes
aliasing, but R50 still needs a snapshot immediately before that restore.
GEARS gears.py train lines478-582 keeps actual last in model and validation
best in best_model; save_model serializes best_model only. A new R50 adapter
can preserve those distinct states without rewriting training. Its optional
test_loader path must remain absent during fit. These observations do not
constitute completed adapter implementations or integration passes.

## External lifecycle implementation, first stage

Pre-change published baseline `a70d24cf769575704069fd81b480fa377381fef0`.
The three official API adapters now expose an explicit R50-only path with
one or50 requested epochs, no early stop and a fresh last-checkpoint target.
Scouter captures last immediately before its official best-state restoration;
TxPert saves the final Lightning checkpoint before loading best; GEARS saves
model (actual last), independently of best_model saved by official save_model.
Each still invokes the official training method exactly once. Old default
one/100-epoch behavior remains unchanged. No official source was modified.

Synthetic adapter tests verify first-epoch best versus fiftieth-epoch last,
non-aliased Scouter snapshots, validation-only TxPert callbacks and continuous
GEARS epoch traversal. The common runner/CLI/config wiring and dual-checkpoint
evaluation receipts are NOT yet complete. These APIs do not authorize a run;
all R2/R3/R4 smokes remain pending until that integration is implemented and
verified against the frozen official installations.

Next integration boundary found in `benchmarks/common/full_gate.py`: the old
external100 gate requires an evaluated smoke with one test call and exactly
one checkpoint. That contract must remain unchanged for historical paths.
New external R50 needs a distinct validation-only one-epoch/two-checkpoint
gate, explicit self-contained 50-epoch config policy, and per-role postfit
evaluation. Reusing the old gate would either read test truth during smoke
or incorrectly reject the new checkpoint retention. Merely changing epochs
in existing external-full YAML is not sufficient.

Lifecycle API gate: 882 tests passed, four explicit reference/CUDA skips;
Ruff, format350, strict mypy83 and isolated build passed. This is local
synthetic/regression evidence, not a frozen-package or server smoke pass.

## External fixed50 configuration boundary

Pre-change published baseline `3b47bf07b4b6614163463314250afb22145a060c`.
Three self-contained configs now live under `configs/r50-rerun/` for Scouter,
GEARS and TxPert public. The new `external_fixed_50` policy is restricted to
these Jurkat metrics-only integrations: exactly50 epochs, seed1, early stopping
disabled. The historical patience10 value is retained as inactive metadata;
the continuous-fit adapters use a non-triggering patience for this path.
Model, prior, data, split, evaluation, optimizer, LR, batch and scheduler values
are unchanged from each frozen external-full config. Config model/dataset IDs
remain schema-canonical; artifact roots and future run IDs distinguish reruns.

Sixteen config regressions check preserved scientific fields and reject
early stopping, changed seeds, invalid thresholds and non50 budgets. The old
external100 configurations remain unmodified. Runner integration is still
pending: current CLIs deliberately reject the new policy, so these configs
must not be treated as launch-ready. The distinct validation-only smoke gate
and per-role best/last evaluations remain the next implementation step.
