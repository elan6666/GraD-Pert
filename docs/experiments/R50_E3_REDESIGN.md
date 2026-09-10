# R50: E3-anchored validation-only experiment program

2026-09-10 delivery update: see R50_POSTFIT_TEST.md. Every full coordinate now
gets automatic best/final test reports after training; validation still alone
selects configurations. Preserve both checkpoints. Historical evidence below
describes the original selection-only contract, not the updated delivery policy.

Decision 2026-09-09. Supersedes pending A1/A3/A4/A5 launch plans, not their
historical evidence. Only R50 REF/LR-low/LR-mid are initially dispatchable.
All formal comparisons run exactly 50 epochs without early stopping. A distinct
one-epoch integration run precedes each new coordinate. R50 integration and
selection use train/validation only; final test evaluation is deferred until
the selected configuration and comparison set are frozen. This intentionally
differs from historical smoke runs that evaluated test after fitting.

## Historical identity, not a reused training run

Authoritative E3: server runs/formal-vnext-edlm-1b75b53-v1/
e3_genept_initialized/gradpert_b2/nadig_jurkat/seed-1, under
/data/yilangliu/GraD-Pert. Source 1b75b53e743f531c112fd1a825bd86aa161cc106;
resolved-config SHA 9bb31234112b2c72f2eecad720de30450ce3fdc24dee6ac0b8c44aa1fbcf647f.
Live training receipt verified 10 epochs/5820 optimizer steps; original config
declares fixed LR .001, batch256, EMA start .996, loss1/.8/.4/.1.
The old test was already inspected historically; it is NOT an untouched test
for new model selection. Repeated validation selection is also adaptive:
report its search budget and validate finalists across seeds before claims.

Runtime graph: HVG512+all targets, N2809, STRING+GO Exphormer-MG, four
RingInduced half-graph locals, zero local-anchor masking. Gene embedding128,
4 graph layers, head2048/256, prototypes16384, basal/decoder512; E3 means
Protein+Reactome+SIGNOR initialized embeddings, subsequently trainable.
Prior SHA 34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318.
Data, split, graph axis, ordered controls, precision, model forward/RNG order,
centering/temperatures, optimizer AdamW and weight decay0 stay fixed initially.
No later compact B0/B1 or step512 defaults silently enter this reference.

## Loss rationale and measurement

Use weights in prediction/condition/masked-node/spread order: **1/.8/.4/.1**.
These are a historical hypothesis, not demonstrated optimal coefficients.

- Prediction is mean squared expression error across cells and output genes.
- Condition is mean soft-target CE across cross-view pairs, then conditions.
- Masked-node is mean soft-target CE over masked nodes, not a sum over genes.
- Spread is the mean negative log nearest-neighbor distance of normalized
  condition representations, averaged over global views; it can be negative.
  It is unavailable with fewer than two unique conditions.

Their scales and units differ: weighted loss magnitudes are not gradient
importance. Existing train-step logs expose each raw loss, prediction vs total
weighted auxiliary graph-gradient norms, ratio, target entropy, prototype usage,
centers and condition counts. Aggregate these over warm/middle/late epochs,
not the first batch. Existing norms do NOT separate the three auxiliary terms.
If a weighting problem is indicated, use a bounded training-only diagnostic
to obtain per-term weighted gradient norms and cosine conflicts on the same
shared parameters, preserving state/RNG. Freeze any proposed weights before
its 50-epoch run. Do not equalize raw CE and MSE numbers or zero losses by default.
One possible controlled strength test scales all auxiliary weights together
by 0.5 or2 while preserving .8:.4:.1, but it is not queued without evidence.

## Sequential matrix

Each stage freezes its parent config and changes the declared factor only.
No full Cartesian product. REF is a fresh50 run, not the old10 result.

| Stage | Rows/candidates | Fixed context / decision |
|---|---|---|
| R50-LR | REF .001, LR-low .0001, LR-mid .0003 | batch256, constant LR, original EMA/capacity/loss |
| R50-SCHED | winning fixed LR vs warmup+cosine without restarts | same peak; warmup8/50 epochs expressed as steps, floor1e-6; EMA remains independently fixed to the parent schedule |
| R50-BATCH |128/256/512| same chosen LR; report steps/epoch and wall, not equal-compute claims; no accumulation/DDP |
| R50-EMA | starts .990/.994/.996, cosine toward1 | same LR schedule, batch, model; no joint LR/EMA change |
| R50-CAP | parent vs smaller projector vs fewer graph layers | keep embedding128, basal/decoder512, graph/view/prototypes; each row changes one capacity axis; explicit dimensional support and count/capacity checks before launch |
| R50-CHECK | at most2 adjacent LR checks for final batch/capacity | tests sequential-search interactions, not global optimality |

Later stages require new frozen configs and reviewed contracts, never edits to
active source. Lower-precision, different data/splits, expanded graph axes and
unrelated scientific modules are not authorized by this program.

## Selecting parents

Primary score: best validation macro delta Pearson across the fixed50 budget,
the same strict-improvement checkpoint rule for every run. Also report mean
validation over epochs41-50, best epoch, loss/entropy stability, parameter count,
peak memory and wall. Exclude only failed/non-finite/incomplete runs, preserving
failure evidence. If primary scores differ <=.002 (preregistered practical
tie, not a significance threshold), prefer lower fit wall; if still tied prefer
fewer parameters then lexicographic ID. Record this decision before next launch.
Same epochs with different batch sizes is equal data exposure, not equal updates.
Learning-rate/EMA horizons derive from50 epochs, including the reference EMA.
No test score or old test ranking enters any automated selection function.
Initial seed1 is screening; finalist repeated seeds are a separately materialized
verification stage, not a claim of significance from this single seed.

## Method comparisons after freezing the training recipe

1. Objective balance: retain1/.8/.4/.1 unless measured gradients justify a
   preregistered alternative. Separate strength from objective removal.
2. Prior: real E3 vs learned ID vs shuffled E3; same embedding width/capacity.
3. Local: parent RingInduced vs Fanout; Essential-only including every current
   perturbation anchor vs a condition-wise equal-size ordinary RingInduced
   control. The old half-size local is additional context. E=2023 of2809,
   so no truncation to1404; four essential locals may share the same node set.
   Expander connections remain computation edges, not biological neighbors.
4. Encoder/source: change encoder with source fixed, and source with encoder
   fixed. Do not label a source+encoder+local combination single-variable.
5. At most two justified combined candidates plus parent, all50 epochs.

Method-stage detailed rows are frozen after training selection, before their
results. No canceled A5 neighbor experiment and no old A1 combination launch.

## Execution and handoff

Fresh release from clean main, exact local/public/server identity and full
pytest/Ruff/format/mypy/build gates; use an isolated clean checkout, not the
uncommitted essential implementation at /tmp/gradpert-m2-fix.
Native R50 uses existing model smoke/full CLI with config policy r50_selection,
status trained, test_evaluations0, explicit selection receipt, no test metrics.
Supply a hash-verified GenePT availability receipt for the unchanged graph;
never bypass the missing-receipt guard that stopped old A1. Fresh run root per
row/phase; allocator expandable_segments:True, metrics_only, zero persistent
PKL, best.pt only after completion. No overwrite/automatic retry.

Planned first queue: GPU0 REF then LR-mid; GPU1 LR-low. One process per GPU,
only on verified idle resources; if other users are active, queue rather than
claim isolated timing. Loss/validation comparisons survive differing GPUs but
speed ties require compatible hardware/load evidence. One-epoch integration
must pass before each fresh50 run. Stop on source/input drift or a fatal row;
preserve logs, do not relaunch failed roots.

Before completing preparation, persist exact source/contract/config/input
hashes and launch commands. Complete goal before CUDA, then register one hourly
quiet monitor: check sessions/processes, bounded logs, epochs/steps/validations,
source identity, PKL/checkpoints, GPU/RAM/disk and terminal receipts. No busy
polling. Two unchanged hourly step snapshots trigger read-only stall diagnosis,
not a restart. Notify only failure, completion, material change or required input.
After the three rows finish, validate exactly50 epochs and steps50xsteps_per_epoch,
zero test calls, shared inputs, checkpoint/selection identities; disable monitor
before the next bounded analysis/next-stage-preparation goal. Whole-program
completion requires selected recipe, method comparisons and final evaluation,
not merely the first three terminal runs.
