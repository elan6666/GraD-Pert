# Data and Evaluation Contract

## Dataset protocols

| Dataset ID | Cell context | Task | Canonical preprocessing |
|---|---|---|---|
| `replogle_k562_essential` | K562 | within-cell unseen singles | signal filter, normalize 4000, log1p, independent Top-5000 HVG |
| `replogle_rpe1_essential` | RPE1 | within-cell unseen singles | same, independently fit |
| `nadig_jurkat` | Jurkat | within-cell unseen singles | same, independently fit |
| `nadig_hepg2` | HepG2 | within-cell unseen singles | same, independently fit |
| `norman` | K562 | predefined unseen doubles (`combo_seen2`) | verified GEARS-targeted processed artifact and frozen split |

No cross-cell protocol is active.

## State machine and files

```text
registered -> downloaded -> source_verified -> canonicalized -> qc_passed
           -> split_frozen -> controls_frozen -> canonical_ready
```

Server layout:

```text
data/<dataset>/<protocol>/
  source/
  canonical/adata.h5ad
  canonical/gene_ids.txt
  manifests/source.json
  manifests/preprocessing.json
  manifests/qc.json
  manifests/split.json
  manifests/evaluation_controls.json
  manifests/checksums.sha256
```

Only `canonical_ready` may train. Every transition is idempotent and receipt-
backed. Unknown expression scale, missing source checksum/license, gene
duplicates, ambiguous labels, split overlap, or unexplained condition loss is a
hard failure.

## Canonical splits

- K562/RPE1/Jurkat/HepG2: exclude control from perturbation conditions, sort
  canonical IDs, and generate one group split using NumPy PCG64 seed 42 with
  target fractions 0.5625/0.1875/0.25. Freeze IDs and hash once.
- Norman: import and normalize the official predefined doubles split; do not
  replace it with random fractions.
- Controls are accessible according to context but are never themselves a
  perturbation split condition.
- Every adapter writes consumed train/val/test condition IDs. The orchestrator
  compares ordered IDs and set hashes to the canonical manifest.

## Evaluation controls

- For each validation/test condition, determine the eligible control pool by
  identical cell line and experimental batch/context policy.
- Use PCG64 with stable input
  `evaluation_seed :: dataset_id :: split :: condition_id`.
- Sample exactly 300 row IDs with replacement, even when the pool exceeds 300.
- Store the ordered row IDs and source-pool hash. Every model uses these exact
  row IDs; a model never samples internally for the common benchmark.
- Truth is every real perturbed cell belonging to the condition. Pred and Truth
  are population samples and are not row paired.

## Artifact lifecycle

### PredictionArtifact (runner output, server-only PKL plus manifest)

- schema/model/dataset/protocol/run IDs;
- source and dirty-tree commits;
- config/environment/data/gene/split/control/checkpoint hashes;
- for each condition: `Pred[300,G]`, `InputCtrl[300,G]`, ordered control row
  IDs, gene IDs/order hash, capability flags;
- no Truth and no test-derived DE values.

### EvaluationBundle (evaluator output, server-only PKL/H5AD as needed)

- all PredictionArtifact fields;
- `Truth[N_condition,G]` and truth row IDs;
- `MetricCtrlPoolMean[G]` and control-pool manifest;
- frozen DE/Top-DE indices with derivation and information-boundary provenance;
- per-condition metric table and availability reasons.

### Small synchronized results

- `metrics_per_condition.csv`
- `metrics_summary.csv` and `.json`
- `coverage.json`, `failures.json`, `run_manifest.json`
- `server_artifact_pointer.json`, small `.txt`/`.md` receipts
- optionally user-approved small plots

Do not synchronize PKL, H5AD, checkpoints, per-cell matrices, or weights.

## Nonlearned models

- `matched_control_mean`: return the 300 actual input control rows unchanged.
- `global_train_delta`: add the training-only global batch-centered single-
  perturbation delta to every input control row.
- `general_train_delta`: exact seen-condition delta when available, otherwise
  global single delta; for unseen doubles, sum component-specific training
  single deltas with global fallback. No test truth is read.

All three emit `[300,G]`; their variance is inherited from real input controls,
not from duplicated mean vectors.

## Headline metrics

For condition `p`, let bars denote row means.

1. `txpert_macro_pearson_delta`

```text
c = mean(InputCtrl)
pearson(mean(Pred)-c, mean(Truth)-c) over all evaluation genes
```

2. `trishift_pearson_delta`

```text
c = MetricCtrlPoolMean
pearson((mean(Pred)-c)[DE_p], (mean(Truth)-c)[DE_p])
```

3. `systema_pearson`

```text
r = equal-weight mean of non-control train+validation condition centroids
pearson((mean(Pred)-r)[TopDE_p], (mean(Truth)-r)[TopDE_p])
```

Compute per condition, preserve undefined values with reasons, then macro-
average finite condition values while reporting numerator/denominator counts.
Do not coerce NaN to zero. Pair-dependent metrics are unavailable under this
protocol. All additional TxPert/TriShift metrics live in a versioned registry
with exact applicability and provenance.

