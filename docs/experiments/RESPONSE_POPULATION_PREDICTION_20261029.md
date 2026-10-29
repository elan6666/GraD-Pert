# R16-GPM: complete-group supervision through the native prediction path

## Decision and evidence

This is one separately identified v2 representation research screening candidate,
not R50 and not a restart of the paused P1 queue. Its fixed budget is two epochs,
300 successful native updates, seed1, global320 (80/rank x2 ranks xaccum2), last.
Retain the existing R3-RI checkpoint as the primary reference; R14 and R15 are
mechanism references. Never retrain a baseline or use test values for selection.

R15 native val445 prediction loss .006719647891684144 exceeds RI
.005634920671810208 and R14 .006354849. The three all5000 Pearson values are
.1668531178/.1176982820/.0437648820 versus RI .189282232/.131629098/.059063965.
Common deep506 Response CLS relative variance .744879 versus RI .022535 can
coexist with effective rank5.8454 versus10.9848. Under the same-control crossed
comparison, R15 CLS condition MSD1.139835 versus RI .0520654 accompanies native
prediction MSD .000509296 versus .000643859. Teacher H remains near log16384.
These are negative evidence against accepting geometry alone as task improvement.
They motivate, but do not prove, proxy/task detachment. CLS can affect gene tokens
through self attention; it is not completely disconnected from prediction.

The main independent medium acceptance and versioned report are in the permanent
archive `representation-adaptive-20261028/r15-postfit-20261029/`. R15 model source
is aa65a42554d2f971adf0419bc1421e43c5a335d6; analysis source is
d3f31221cdcba04f9d179f782cb6f15f3589188c, not the current wrapper commit.

## Literature and mathematical rationale

CellOT (Nature Methods2023, https://doi.org/10.1038/s41592-023-01969-x) and CMonge
(Nature Machine Intelligence2026, https://doi.org/10.1038/s42256-026-01242-8)
address destructive, unpaired measurements with population comparisons. We borrow
that problem framing only; the present mean-MSE is neither OT nor a replication
of their algorithms, and their results do not establish expression-column holdout.

For independent unpaired controls C and treated cells Y within one condition,
E||f(C,p)-Y||^2 = ||E f(C,p)-E Y||^2 + tr Var f(C,p) + tr Var Y.
Thus the existing random-paired MSE already targets the treated mean in expectation.
The new term adds no truth source: it changes conditional weighting, estimator
noise and variance penalties, and could merely encourage a common-expression
mean. This is a falsifiable intervention, not proof of a discovered bottleneck.

## Single chosen intervention

Remove the R14/R15 CLS auxiliary and its trainable readout. Use native Softplus
absolute expression prediction from concat(Response gene token,e_p), with the
existing prediction MLP, inside each unchanged complete-group forward:

$$L_{GPM,p}=|Q|^{-1}\sum_{g\in Q}(n_p^{-1}\sum_i\hat y_{p,i,g}-\mu^{TRAIN}_{p,g})^2.$$

Q is the current native main query subset (1000 of allowed4775). Reduce mean
across cells BEFORE squaring, in FP32. Fixed targets have no gradients. No new
learned head, no CLS/rank/cosine term, no MMD/OT, no new output activation.
This simultaneously changes supervision path and target space from the old
projected256 response target to allowed absolute expression; it is not a pure
causal test of path placement. Do not run a second projected version as a sweep.

Keep primary MSE, all SSL weights, graph, RI residual, biases, norms, views, EMA,
main data sampler, update clock, private group RNG, original bank control slots
and hash schedule unchanged. Eligible TRAIN conditions1322, selected rows47807,
4--40/group, original context counts;13 smaller conditions remain in primary
training but skip only this auxiliary. Every eligible group occurs once/epoch.
Original coefficient per rank remains lambda*world*150/1322. Groups are serially
backpropagated, with private RNG active through checkpoint recomputation.
Teacher/centers receive no additional forward/tick. Evaluation loads no TRAIN
sidecar; partial medium auxiliary is NA, never approximated by a small batch.

## Versioned target sidecar and strict read boundaries

Retain the original six-array response bank byte-for-byte as the sampling bank.
Create a new server-only manifest plus NPZ with exactly allowed_gene_indices
(int64[4775]) and treated_mean (float32[1322,4775]). Its ordered conditions and
allowed axis must equal the sampling bank, and it binds the exact bank manifest
and array SHA, canonical/split/selection/exclusion identity, TRAIN metadata and
membership hashes, source/publication, numeric read audit and output hashes.
Build the mean from the SAME selected TRAIN treated rows and context weighting;
float64 accumulation then float32. Keep allowed own-target TRAIN values as the
primary task does; ownmask remains an independent validation endpoint. Never
read the225 excluded expression columns, even before masking/projection. No val
or test numeric rows. Do not invert Q or add cyclic-control means to old deltas.
Allowed-first reader must slice before materialization and enforce treated-row
whitelist. The sidecar and all expressions/weights stay under /data/yilangliu.
Reject symlinks, escaping paths, hash/condition/axis/split/selection mismatches,
nonfinite/negative values, incompatible field combinations or silent fallbacks.

## Calibration, validation, and rejection

Fresh seed1 initialization, original first8 native TRAIN batches/groups; zero
updates, full protected state/RNG restore. Shared parameter scope stays
student.response.parameters (exclude prediction head from norm selection for
comparability; gradients still train it). Signed gradients sum across groups
then rank-average vectors before norms. Lambda=min(1,.1*median Gpred/median Gaux).
No manual lambda choice, reused old lambda, validation-selected batch or sweep.
Record scalar gradient/input identities and actual throughput/memory. Require
native 10-update restore/complete40, parity and resource gates before fresh fit.
No source mutation in active server trees; clean local/GitHub/server commit must
match at every formal phase. Publish calibrated self-contained config separately.

After fit: native observer parity, frozen445/18 Pearson with ordered300 controls,
ownmask MSE/Pearson, raw/prehead RSA; same medium1335/5317 and crossed32/128/16
panels, student/teacher/prototypes/GG/GL/LL/context-condition contrasts. Corrected
CE/KL must be input-specific, not just lower CE. Report training loss split,
conditional downstream errors and all missing/NA scope, frozen source/config/CP
hashes, zeroPKL, report/figures/platform. Cross-version and single-seed limits stay.
If group training loss decreases but validation utility remains below retainedRI,
reject this candidate instead of extending epochs or sweeping coefficient. Any
provisional gain requires all four unchanged outcome criteria and later frozen
confirmation; scientific_complete remains false until independently established.

## Disjoint implementation ownership

- Main: this design, managed transitions, review, native preflight/launch/publication.
- Builder agent: new src/gradpert/training/v2/population_anchor.py,
  scripts/v2/prepare_population_anchor.py and their new tests only.
- Integration agent: response_anchor_runtime.py, runtime.py, config/v2.py,
  calibrate_response_anchor.py, generate_response_anchor.py and dedicated tests.
- Platform agent: ignored report delivery only; no scientific code/state changes.

Interface: load_population_targets(path, manifest_sha256, arrays_sha256, bank)
returns detached numpy float32[P,G]. Loader belongs to population_anchor.py and
checks server-only provenance/shape/axes. Mode response_anchor_loss is
population_prediction. New explicit fields response_anchor_population_path,
response_anchor_population_sha256, response_anchor_population_arrays_sha256 are
required together only for enabled population mode; forbidden in historical modes.
No anchor readout is attached in population mode. Historical defaults untouched.
