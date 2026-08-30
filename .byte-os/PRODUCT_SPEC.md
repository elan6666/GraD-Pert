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
- Frozen split/control manifests, metrics-only receipts with an optional single
  result bundle, metric registry and three headline Pearson families.
- Server commit preflight, formal launch, result allowlist sync, artifact
  pointers, read-only analysis notebook examples.
- Optional private Trackio dashboards for formal training/validation scalar
  curves and single-GPU telemetry, isolated from performance timing and all
  test/prediction/artifact surfaces.

## Active B2-vNext extension

The active extension keeps the standalone B2 training/evaluation product and
adds a single config-driven architecture surface for a preregistered one-
dataset ablation program. Its default is TxPert-style full-cell-line,
pre-split HVG512 plus all perturbation targets,
STRING+GO native multi-source sparse graph Transformer, four RingInduced
views capped at exact ratio `1/2` of the actual runtime graph, no local anchor
masking (`0/1` mask-view ratio), and direct losses `1.0/0.8/0.4/0.1`.

The successor program contains two direct-to-A0 modules. H changes only the
requested HVG count across 512/1024/2048/5000 while retaining a `1/2` local
coverage ratio. L changes exactly one of builder, local count, local node ratio,
or local mask-view ratio; its count row uses eight locals against the four-local
A0. Fixed local node budgets and fixed mask-view counts are legacy evidence
only; local-view count remains an explicit successor scientific factor.

The same main path must also select source-audited single/multi graph encoders,
STRING weight policies, decoder strategies, and GenePT feature modes. No
variant may bypass canonical data, split, control, evaluation, artifact, or
source-identity contracts.

## Non-goals

- Cross-cell/target-inductive protocol, an extra dataset, paper-best private-
  graph reproduction, an upstream runtime dependency, or any performance claim
  before evidence. B2-vNext ablations are limited to the frozen Nadig Jurkat
  program in `docs/design/GRADPERT_VNEXT_ABLATIONS.md`.
- Embedding raw data, model weights, PKL/H5AD, or credentials in Git/local sync.
- Hiding related-work citations or rebranding copied upstream code.

## Requirements

1. Native source has no upstream model runtime dependency or upstream-named
   native classes.
2. Every dataset must pass source, preprocessing, gene, QC, split, control, and
   coverage gates before training.
3. Each model/dataset config is self-contained, strict, and hashable.
4. Every learned model/dataset pair first passes an exactly one-epoch v1
   integration smoke. The separately preregistered vNext formal A/H rows use
   exactly 10 epochs; test is evaluated only after the selected checkpoint.
5. Every runner always emits prediction-only content receipts using the exact
   shared 300-control row IDs. The default mode persists no PKL; explicit
   `single_pkl` mode emits exactly one deduplicated `result.pkl`.
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
