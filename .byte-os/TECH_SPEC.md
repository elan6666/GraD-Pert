# Technical Specification

## Architecture

```text
strict per-pair YAML -> validated ExperimentConfig
dataset registry -> canonical data -> split/control manifests
                                |-> native GraD-Pert runner
                                |-> isolated GEARS runner
                                |-> isolated TxPert runner
                                |-> nonlearned runner
runner -> truth-free prediction receipt + inference recipe
evaluator joins truth -> metric tables/summaries
optional single deduplicated result PKL -> server only
allowlisted small sync -> notebooks/reports
```

## Package modules

- `gradpert.config`: strict YAML loading, completeness/provenance/hash checks.
- `gradpert.data`: registry, downloads, canonicalization, QC, splits, controls.
- `gradpert.graphs`: Top-K graph building, coverage, DropEdge, local views.
- `gradpert.modeling`: native encoder, views, projector, losses, teacher state,
  predictor, training/resume.
- `gradpert.baselines`: three train-only nonlearned models.
- `gradpert.artifacts`: schemas, safe trusted loading, sealing/checksums.
- `gradpert.evaluation`: truth join, metric registry, macro summaries.
- `gradpert.execution`: source identity, runners, deterministic matrix,
  completion/fairness gates, and small-result staging.
- `gradpert.cli`: thin command surface over tested services.
- `benchmarks/gears`, `benchmarks/txpert`: isolated upstream-dependent runners.

## Data model

- `DatasetSpec`, `SourceFile`, `PreprocessingManifest`, `QCReport`.
- `ConditionSplitManifest`, `EvaluationControlManifest`.
- `ExperimentConfig` with complete nested `data`, `model`, `training`,
  `evaluation`, `artifacts`, and per-field provenance.
- `PredictionArtifactManifest`, `EvaluationBundleManifest`, and
  `inference-recipe-v1` with condition records, ordered row IDs, and array checksums.
- `RunManifest`, `MetricRecord`, `Availability`, `ServerArtifactPointer`.

Canonical JSON hashing uses UTF-8, sorted object keys, compact separators, no
NaN, and SHA-256. Ordered lists remain ordered.

## Config matrix

Models:

- `gradpert_b2`
- `gears`
- `txpert_public`
- `matched_control_mean`
- `global_train_delta`
- `general_train_delta`

Datasets: the five canonical IDs. Exactly 30 YAML files; all fields resolved in
each file. The loader rejects merge keys, anchors, a `defaults` key, unknown
keys, and missing provenance.

## Integrations

- HTTP downloader with resume, explicit URL/checksum/size/license registry.
- STRING/GO public graph materializers with pinned source/version receipts.
- GEARS commit `f374e43...` in its own environment.
- TxPert commit `08d82ee...` in its own environment.
- GitHub `elan6666/GraD-Pert`, dry-run-first server orchestration, and
  hash-sealed small-result staging.

## Implementation risks

- Server dependency resolution across CUDA/PyG and upstream locks.
- Missing independent within-cell data sources/checksums.
- Memory from prototype head and ten views.
- Loader-side condition drops and label/gene conversions.
- Training recipe gaps for public TxPert.
- Metric semantic drift and unsafe truth access.

All risks become failing preflight/readiness/contract tests, not warnings that
formal runs can ignore.

## Testing strategy

- Pure Python unit tests for schemas, hashing, split/control sampling, artifact
  manifests, metric formulas, baselines, sync allowlist.
- PyTorch/PyG unit tests for graph views, shapes, losses, gradients, EMA/center,
  checkpoints and deterministic prediction.
- Golden tests against small frozen outputs from inspected official functions.
- Synthetic end-to-end data -> split -> train -> predict -> evaluate.
- Isolated runner contract tests with tiny fixtures and truth-access denial.
- Server smoke, worst-case fit, one-dataset one-seed pilot, then full matrix.
