# Product Specification

## Positioning

GraD-Pert is a standalone, reproducible research package for predicting
single-cell transcriptional responses from a basal cell profile and structured
perturbation context. v1 couples expression prediction with graph-view
self-distillation and evaluates all methods through one frozen population-level
protocol.

## Target users

- Research engineer preparing five perturbation datasets and launching formal
  server experiments.
- Method author implementing/debugging GraD-Pert without importing upstream
  research packages.
- Reviewer/collaborator checking split fairness, provenance, metrics, and
  result reproducibility.
- Analyst using small summaries locally and notebooks against explicitly
  mounted or server-side frozen artifacts.

## Problems and jobs to be done

- Prepare five heterogeneous public datasets without confusing raw, upstream-
  processed, cross-cell, and within-cell views.
- Train one well-defined GraD-Pert route and two learned baselines with known
  hyperparameter provenance.
- Prove every compared model saw the same condition split and evaluation
  controls.
- Recompute metrics and downstream analyses without retraining or trusting a
  notebook's hidden logic.
- Keep multi-GB artifacts on the server while retaining compact local evidence.

## MVP

- Installable `gradpert` package and CLI.
- Five-dataset registry/downloader/canonicalizer/QC/readiness state machine.
- Self-contained 30-file config matrix: six model IDs by five datasets.
- Native B2 model, training/resume, fit-test, condition inference.
- Three nonlearned models.
- Isolated official-code GEARS and TxPert runners.
- Frozen split/control manifests, two-stage artifacts, metric registry and three
  headline Pearson families.
- Server commit preflight, formal launch, result allowlist sync, artifact
  pointers, read-only analysis notebook examples.

## Non-goals

- Any ablation, B3, cross-cell/target-inductive protocol, extra dataset/model,
  paper-best private-graph reproduction, or performance claim before evidence.
- Embedding raw data, model weights, PKL/H5AD, or credentials in Git/local sync.
- Hiding related-work citations or rebranding copied upstream code.

## Requirements

1. Native source has no upstream model runtime dependency or upstream-named
   native classes.
2. Every dataset must pass source, preprocessing, gene, QC, split, control, and
   coverage gates before training.
3. Each model/dataset config is self-contained, strict, and hashable.
4. Every learned model/dataset pair must first pass an exact one-epoch smoke
   run. In the current phase, only GraD-Pert may continue to max 200 epochs
   with validation-only early stopping patience 10; GEARS and TxPert stop after
   the smoke checkpoint. Test is evaluated only after the selected checkpoint.
5. Every runner emits prediction-only condition artifacts using exact shared
   300-control row IDs.
6. The evaluator owns truth access and can independently recompute every
   published metric.
7. Formal compute is server-only and gated by exact three-way commit parity.
8. Local sync admits only small allowlisted outputs and server artifact pointers.

## Acceptance criteria

- All local unit/static/build/synthetic end-to-end tests pass from a clean env.
- Config-matrix verification proves 30 exact files and no hidden inheritance.
- Server preflight, environment locks, GPU fit, five dataset readiness receipts,
  learned/nonlearned run receipts, and metric recomputation receipts exist.
- Five datasets have results for all applicable models and paired run seeds, or
  are explicitly blocked by an upstream availability gate with no substituted
  data/result.
- A fresh review returns ship, followed by three evidence-led iterations and a
  final ship review.
