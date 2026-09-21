# v2 experiment package

The matrix in `groups.json` records factor definitions and validation-only group
selection dependencies. It is not a runnable queue until measured execution
profiles, frozen parent selections and launch preflight are recorded.

`PYTHONPATH=src python scripts/v2/generate_group.py --parent PARENT.yaml --group H1 --output NEW_DIRECTORY`

The parent must be an explicit standalone v2 config. The generator writes every
row as `NEW_DIRECTORY/ROW/gradpert_v2/DATASET.yaml`, preserving the required config
identity and all unchanged fields. `manifest.json` pins parent, generator and row
hashes. Existing directories are rejected; selecting a later parent creates a new
group directory. Within a group no row changes the parent of another row.

Supported training groups: B0, H1, H2, P1, L0, L1, L2, A1–A4, S1, S2. H1 changes
both peak and floor LR with the fixed project ratio. L1/L2 enumerate all eight
component combinations; all-off uses both external SSL weights zero. S1 changes
condition CE weighting only. S2 requires a mixed-condition parent and separates
sampling from the SSL2 KoLeo switch. Generated configurations still require
variant-specific resource preflight; capacity for one architecture is not proof
that another architecture fits.

H3 requires the pending measured batch profile selection. G1 and D1 are evaluation
protocols and are not silently generated as ordinary training configurations.
Neither a generated config nor a passing schema check constitutes an experiment
result or permission to bypass source/data/resource checks.

`PYTHONPATH=src python scripts/v2/capacity_report.py --probe RECEIPT.json CONFIG.yaml --output NEW_DIRECTORY`

Repeat `--probe` for multiple measurements. This writes capacity JSON/CSV from
completed, hash-matched 128-update probes with checkpoint continuation and
300-control inference. It reports observed points, not an inferred maximum or a
selected scientific batch. Preserve per-run source versions and KoLeo neighborhood
sizes when comparing physical, accumulated and distributed batches.

`PYTHONPATH=src python scripts/v2/collect_results.py --run SERVER_RUN_ROOT --output NEW_DIRECTORY`

Repeat `--run` to collect multiple runs. Run this on the server so selected
checkpoint hashes can be verified without transferring weights. Output is small
JSON/CSV suitable for dry-run-reviewed retrieval. Both best/last roles are kept,
even for identical checkpoint content. Missing manifests/checkpoints/tests and
invalid evidence remain explicit. Completion requires fifty committed validation
epochs, consistent minimum-loss selection, both test identities, checkpoint hashes,
and the zero-PKL postcondition. The table records training and evaluation SHAs
separately; it does not select hyperparameters from test scores.

`PYTHONPATH=src python scripts/v2/generate_group.py --parent PARENT.yaml --verify-manifest GROUP/manifest.json`

Verification checks parent and row hashes, complete registered factor levels,
owned config paths, strict schema, and the full configuration difference against
the frozen parent. Updating a checksum does not authorize an undeclared extra
factor. Supply the relocated parent path on the server; its hash must still match.
This verifies group configuration integrity, not source/data/GPU launch admission.

`capacity_probe.py --integration-only` is a distinct engineering mode: exactly one
complete optimizer update and checkpoint reload, without inference or sustained
capacity evidence. Its receipt kind is `integration_only`; the capacity report
rejects it. The normal mode still requires at least 128 updates, midpoint
continuation and 300-control inference. Use one-step receipts only for variant
integration preflight after a suitable sustained execution profile is established.

## Group queue

`PYTHONPATH=src python scripts/v2/run_group.py --manifest GROUP/manifest.json --parent PARENT.yaml --runtime SERVER_RUNTIME.json --gpu 0 --preflight-index PREFLIGHTS.json --queue-root NEW_SERVER_DIRECTORY`

By default this prints a read-only queue plan. `--execute` seals it and executes
rows sequentially in separate processes. `--resume --execute` uses the saved plans
and original run roots; completed rows are verified before skipping, committed
partial runs resume, and startup failures without an epoch commit require repair.
A queue never chooses a new parent during its execution. Group dependencies and
validation-only parent selection must be frozen before preparing the next group.

The preflight index has a `rows` list whose entries contain `config_sha256`, `seed`,
`receipt` (server path), and `sha256` (receipt hash). Every row needs an exact
config/seed probe from the same published source and data root/topology. This is
integration admission; the final execution profile must separately have sustained
capacity evidence. Queue, physical-GPU, and config/source/seed leases prevent
concurrent duplicate execution. Selected GPUs must be idle at launch. The current
runner uses one task per selected GPU; it does not assume two tasks fit safely.

Runtime and publication checks remain delegated to the existing train entry.
Checkpoint weights stay on the server. A failed row stops this group, preserving
its plan and logs for repair; invoking resume continues it before advancing.

### Shared v1 infrastructure and small artifacts

V2 reuses canonical training/evaluation data, frozen condition/control manifests,
metric implementations, the public train entry and source identity checks. The
curve adapter translates committed v2 epoch summaries to the existing native
`training.curves.render_curves` interface. Its training inputs explicitly remain
means over optimizer updates; they are not per-step observations. Native v1 code
and historical output conventions are unchanged.

V2 configurations require `metrics_only`. Best/last checkpoints remain on the
server, with metrics, curves and reproducibility receipts; no prediction matrix
or PKL is persisted. Exact ordered validation populations are stored once in
`validation_population.json` at the run root; epoch history retains population
hashes and a reference to this receipt. Curve outputs are under `fit/curves/`,
with an additional source-bound `fit/epoch_curves.csv` and curve receipt.

Remaining reuse audit: the v1 trainer and post-fit runner currently couple to
v1 model, checkpoint and optimizer interfaces. V2 lifecycle duplication must
still be reduced through compatible adapters/shared interfaces, preserving v1
behavior and v2 distributed/teacher-state recovery. This is not yet a completed
shared-lifecycle integration.

V2 checkpoint selection now calls the existing native `EarlyStoppingState` in
minimum-loss mode, including strict ties and replay on resume. Its stop signal
is ignored for the fixed epoch budget. V1 selection implementation is unchanged.

`select_parent.py` provides validation-only H1/H2/H3 parent selection from all
registered candidates and explicitly paired seeds. It verifies fifty committed
epochs, common source/data/reference identities and the best checkpoint hash;
selection uses mean best validation loss and never reads test receipts. Its CLI
requires `--manifest`, `--parent`, `--runs` (row-name to run-root lists), `--seeds`
and a new `--output`. Automatic dependency admission of this selection receipt
into subsequent groups is still pending; H3 also awaits measured batch levels.

### Validation-selected follow-up groups

Use `prepare_followup.py --selection SELECTION.json --manifest UPSTREAM/manifest.json
--parent UPSTREAM_PARENT.yaml --group H2 --output NEW_GROUP` after H1 completes.
The script recomputes the selection from original evidence, uses the winning
configuration as the actual generator parent, and seals the dependency paths and
hashes in the new manifest. Queue preparation and resume recheck this dependency.
Directly generated later-group configs remain useful for design review, but are
not launchable without the dependency. B0/H1 remain initial groups.

The required progression is H1→H2→H3, then structural/loss/sampling groups from the
H3-selected configuration. H3 materialization still awaits measured batch levels;
this does not make the later chain ready for execution. Upstream receipts and
configs must remain accessible at their recorded paths on the server.

Native v1 and v2 now share `training.epoch.execute_epoch` for batch iteration,
completed-update counting and optional pre-update excess-batch rejection. V1's
adapter retains its existing post-epoch count check and error-time receipt flush;
v2 retains its sealed pre-update bound. Model updates and distributed cell
slicing remain adapter responsibilities. Together with common selection, RNG
restoration, curves, data and metrics, this removes duplicated infrastructure
without changing v1 checkpoint payloads. Outer epoch commit/recovery orchestration
remains version-specific and still requires a final reuse audit.

`training.v2.context_evaluation.evaluate_contexts` scores a fixed, explicitly
ordered gene set under nested context budgets sampled from IDs alone. It reuses
canonical controls, v2 query inference and native mean-expression MSE. Results
contain context identities, row hashes and condition/macro MSE, never prediction
matrices. This is a separate G1 fixed-axis metric, not a substitute for the three
full-axis headline Pearson metrics. Tests perturb only unscored truth columns and
verify unchanged scores. A source/checkpoint-bound server CLI and real-data
acceptance remain pending; the API alone does not complete G1.

`training.v2.diagnostics.response_diagnostics` compares fixed-query responses
under control replacement, perturbation replacement and response CLS→gene
blocking. It reports population-mean RMS differences separately for prediction,
predicted delta, basal control CLS and response CLS. Optional fixed-truth MSE is
computed only for baseline versus the blocked response; replacement conditions
are not incorrectly scored against the original condition's truth. The API does
not claim arbitrary control cells are paired. Tests cover raw-control-residual
separation, distinct CLS roles and unchanged-input zero sensitivity. Frozen
population/checkpoint/source binding and real-server D1 execution remain pending.

H3 generation now accepts `--batch-probes PROBES.json` in both group generation
and `prepare_followup.py`. The JSON is an ordered list of two or three records
with `receipt`, `config` and `receipt_sha256`. Every receipt must pass sustained
capacity verification; probes must match the parent dataset/model/loss profile,
use one rank and no accumulation, and provide unique increasing physical batch
sizes. Generated rows change only microbatch and training batch; LR stays fixed.
Verification rechecks original receipt/config hashes. This enables H3 tooling;
it does not itself choose or freeze the final measured levels.

`evaluate_context.py` connects G1 evaluation to a completed fixed-50 run. Supply
`--training-run`, `--role best|last`, original `--config`, `--data-root`, `--gpu`,
`--publication`/`--publication-sha256`, and a new server `--output` JSON file.
The hash-pinned `--protocol`/`--protocol-sha256` JSON has exactly
`evaluation_gene_ids`, `budgets`, `context_seed`, and `split` (`val` or `test`).
Training identity/config/data and selected checkpoint ownership/hash are verified;
evaluation source/environment are recorded separately. No prediction matrices
are written. Current runtime construction supports the single-rank training
profiles; a distributed training config requires a later evaluation adapter.
Entry guards and CLI are tested, but real completed-checkpoint execution is still
unverified. Capacity checkpoints cannot masquerade as completed formal runs.

`evaluate_diagnostics.py` shares the completed-run/source/config/checkpoint/data
entry with G1. Its hash-pinned protocol has `query_gene_ids`, `condition_id`,
`alternative_condition_id`, and `split`. Both distinct conditions must belong to
the same frozen split. Their existing ordered control draws are loaded without
resampling; query embeddings are fixed across interventions. The response receipt
records both ordered control populations and the original condition's truth IDs,
with separate basal/response CLS and prediction/delta distances. Cell inference
is chunked with population-weighted aggregation. Real-server execution remains
unverified until an eligible checkpoint and its sealed protocol are available.

### Five-dataset integration candidates

`configs/v2/integration/gradpert_v2/` contains five self-contained engineering
configs. Each data section is copied exactly from its existing native v1 config,
including Norman's `norman_combo_seen2`; no split is regenerated. Graph manifest
hashes were read directly from the existing server `data-vnext-a942114` tree.
The batch2/full-objective profile is for `capacity_probe.py --integration-only`
(exactly one update plus checkpoint reload), not a formal reference batch or
sustained capacity claim. All five configs pass schema/data-contract comparison;
server integration receipts are still required before labeling them executable.
Do not use these small-batch configs as substitutes for final measured ablations.

For explicit engineering verification before formal training completes, all shared
evaluation entries accept `--engineering-receipt-sha256` instead of `--role`.
`--training-run` then names a passed capacity/integration probe root. The exact
receipt and `resume.pt` hashes, source and runtime data are checked. Only validation
truth is accessible; outputs are labeled `engineering_*`, `scientific_result:false`,
and retain the checkpoint's actual saved progress (not the probe's final step).
This does not satisfy formal50-epoch acceptance or provide scientific results.

`evaluate_validation.py` reuses that entry and the full native validation/reference
pipeline; its protocol is exactly `{"split":"val"}`. It evaluates every frozen
validation condition and the three existing metrics, with no condition truncation.
Real-server checks of these new engineering modes remain pending.
