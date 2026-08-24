# Engineering Rules

This document expands the hard constraints summarized in root `AGENTS.md`.

## Identity, provenance, and licensing

- Native code and public APIs use GraD-Pert concepts and names. Product-facing
  text explains the problem, graph views, teacher/student roles, prediction
  path, and losses directly; it does not describe the package as a renamed or
  borrowed implementation.
- External names remain where truth requires them: isolated benchmark IDs,
  related work, result tables, licenses, and reference-alignment records.
- Never remove copyright/license notices or imply external code/results are
  ours. Independent behavioral alignment is not permission to copy and rename
  source. The TxPert checkout remains isolated under its non-commercial terms.
- Each alignment entry records repository URL, immutable commit, source file,
  symbol, shapes, update order, local counterpart, verification, license, and
  intentional divergence.

## Standalone package boundary

- `src/gradpert` cannot import/call an upstream model project or require its
  checkout. Static tests enforce forbidden imports and native class names.
- External learned models run in dedicated environments under `benchmarks/`
  and exchange only manifest-backed neutral artifacts with the common system.
- Upstream branch tips are never implementation evidence. Freeze commits.
- If behavior is unresolved after inspecting official code, record the gap and
  block only that component. Do not fill it from memory or a third-party port.

## Single-v1 model freeze

- Implement B2 only: a randomly initialized student prediction stack and an
  isomorphic graph teacher updated by EMA, optimized jointly from step zero.
- Ordinary graph nodes are post-processing HVGs; only known candidate targets
  may be forced outside the HVG boundary.
- STRING and GO are pruned independently to Top-20 incoming edges. The v1 W0
  route uses weights for selection only.
- Teacher: two global views, no basal/decoder/control/truth. Student: matching
  globals plus eight 512-node Local-RingInduced views; exactly one global uses
  node masking and exactly four locals mask all active anchors.
- Prediction uses a deterministic full Top-20 view, no stochastic graph/view
  operation, no projection head, and additive D0 decoding.
- Projection capacity is selected once by a one-GPU, 128-consecutive-step fit
  on every dataset at no more than 85% initially usable memory. It is then
  frozen across datasets and seeds.

## Training and hyperparameters

- The server integration gate is exactly one epoch for every learned
  model/dataset pair. Passing means the official/native model actually trains,
  validates, checkpoints, predicts 300 rows per condition, and enters the
  common evaluator; import-only or forward-only checks do not pass.
- Seed 1 is the paired run identity across every model and dataset, including
  deterministic nonlearned baselines. GraD-Pert-only replicate seeds remain
  additional runs, not substitutes for seed 1.
- After all one-epoch gates pass, only GraD-Pert may enter full training.
  GEARS and TxPert have `formal_run_policy=smoke_only`; nonlearned models use
  `inference_only`.
- `max_epochs=200`; validate each epoch; stop after 10 validation checks with
  no strict improvement in `val/txpert_macro_pearson_delta`; `min_delta=0`.
- Save best and last resumable checkpoints. Test only the sealed best checkpoint
  after training/selection is finished.
- Preserve frozen official LR/batch/optimizer/weight-decay/dataset differences.
  Every value has provenance `official`, `paper`, or `project_preregistered`.
- The public TxPert configs currently expose batch 64 and defaults AdamW,
  LR 1e-3, weight decay 0. The frozen GEARS code exposes train/test batch
  32/128 and Adam, LR 1e-3, weight decay 5e-4. Missing five-dataset recipes are
  not official facts.
- Random streams are separately named for split, run, training-control
  matching, view construction, evaluation-control sampling, and statistics.
- Every run logs losses, schedules, gradient norms/ratios, EMA/center states,
  config/environment/data hashes, hardware, source commit, and dirty status.

## Official learned benchmark boundary

- GEARS and TxPert are invoked from their frozen official checkouts in separate
  environments. Do not reproduce their model classes, losses, or optimizers in
  `src/gradpert` or in an adapter.
- Each external config points to the official commit and exact official config
  file/symbol. When no dataset-specific official recipe exists, record that
  absence and the selected published official profile; never label a locally
  invented value as official.
- Adapter code may only translate the canonical AnnData/split/control
  manifests, deny validation-time test truth, call the official training and
  forward APIs, retain the exact 300 predictions, and emit common artifacts.

## Config matrix

- Store exactly one complete resolved YAML for each model/dataset pair at
  `configs/experiments/<model_id>/<dataset_id>.yaml`.
- This applies to GraD-Pert, GEARS, TxPert, and every nonlearned baseline.
- No global experiment file, YAML merge key, Hydra defaults list, template
  inheritance, or implicit training default may determine a run.
- Shared schemas/validators may reject values but never supply hidden experiment
  values. A run stores the exact config bytes and SHA-256.

## Data, split, and leakage rules

- A dataset moves through downloaded, audited, canonicalized, split-frozen,
  control-frozen, and `canonical_ready`; only the last state is trainable.
- For K562/RPE1/Jurkat/HepG2, ordinary protocol is within-cell unseen singles.
  Norman uses the frozen predefined doubles protocol. Cross-cell is deferred.
- One condition-level split manifest is authoritative per dataset/protocol.
  Adapters export actual consumed IDs; mismatch fails before training/evaluation.
- Fit preprocessing, response-derived state, nonlearned deltas, graphs, early
  stopping, and hyperparameters without test expression or test metrics.
- The same data/split/gene/control/evaluation hashes apply across compared
  models; model-native training-control aggregation may differ when it is part
  of the frozen method and is declared.

## Prediction and evaluation boundary

- A prediction runner does not receive truth. It emits condition-keyed
  `PredictionArtifact` data: `Pred[300,G]`, exact input control row IDs, gene
  order, capabilities, and complete provenance.
- The common evaluator joins all real truth cells into an `EvaluationBundle`
  only after prediction is sealed.
- For each condition, use the identical manifest-selected 300 eligible control
  cells sampled with replacement. All models see the same ordered row IDs.
- Report TxPert macro Pearson delta, TriShift Pearson delta, and Systema Pearson
  as separate metric IDs. Do not alias one formula to another.
- Metrics are per condition, then equal-condition macro. Pair-dependent metrics
  are `not_applicable:no_paired_truth` under the unpaired population protocol.
- Nonlearned models fit train only. Declare whether output is population or
  per-control; never duplicate a mean to fabricate variance support.
- Notebooks are frozen-artifact consumers only; no hidden split/training/metric
  implementation lives solely in a notebook.

## Server and artifact topology

- Formal materialization runs only on the authorized server. Local work is
  source development, documentation, static/unit tests, and synthetic smoke.
- GitHub is the only source-code synchronization plane. Preflight requires the
  same clean commit locally, remotely, and on the server.
- Server-only: raw/processed data, graphs, `.h5ad`, prediction/evaluation PKL,
  checkpoints, weights, caches, and per-cell matrices.
- Server-to-local allowlist: `.txt`, `.json`, `.jsonl`, `.csv`, small `.md`, and
  explicitly approved small plots. Dry-run and inspect every transfer list.
- Local pointers record server path, commit, manifest/checksums, and reproduction
  commands without embedding machine credentials or secrets.

## Verification gates

- Contract tests: configs complete/self-contained; no forbidden native imports
  or classes; hashes and condition lists match; test truth unavailable to runner.
- Unit tests: split leakage, control sampling with replacement, graph pruning and
  views, losses, centers, EMA order, artifacts, metrics, baselines, resume.
- Integration: synthetic B2 train/restore/predict/evaluate, isolated runner
  schema conformance, server commit preflight, artifact sync dry-run.
- Evidence, not narrative, determines completion. Do not claim ready/trained/
  reproduced/publishable without current receipts.
