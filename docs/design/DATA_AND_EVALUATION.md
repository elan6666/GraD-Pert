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

## Frozen source and metadata boundary

- Each dataset has one registry file under `registry/datasets/`; source size,
  checksum, license, availability, observed raw columns, canonical columns, and
  condition transform are explicit. A source mapping in `requires_source_audit`
  state cannot enter canonicalization.
- Nadig raw files use `gene` as the perturbation column, `gem_group` as batch,
  `non-targeting` as control, and `adata.var.gene_name` as the gene-symbol
  axis. Canonicalization then emits `condition`, `batch`, `cell_type`,
  `control`, `condition_name`, and `gene_name`.
- The frozen preprocessing helper matches target labels against gene symbols.
  Therefore the lower-tail target-expression filter runs on raw target IDs
  before the `target -> target+ctrl` condition encoding. This resolves the
  public onboarding notebook's ordering inconsistency explicitly and is stored
  in the preprocessing receipt.
- The official RPE1 Figshare file was observed in `ic_checking` state and its
  download endpoint returned HTTP 202 with zero bytes on 2026-08-24. The frozen
  independent source is therefore scPerturb v1.4
  `ReplogleWeissman2022_rpe1.h5ad`: MD5
  `cc7f1ec50aeb3a3e1b4a6cfa713d80fa`, 1,236,886,900 bytes. A server audit
  confirmed 247,914 × 8,749, unique observation/gene IDs, 11,485 controls,
  `cell_line=RPE1`, complete `gene`/`batch`, and a full matrix of finite,
  nonnegative integer counts. Gene symbols are the var index.
- For raw within-cell sources, independently compute Top-5000 HVGs after
  normalize-total 4000 and `log1p`. Freeze two ordered axes: the expression
  input/output axis is exactly those HVGs; the graph axis is those HVGs plus
  every known candidate perturbation target. A forced non-HVG target is a graph
  node but is not silently added to decoder outputs. If a candidate label has
  no source-expression column, preprocessing appends a zero-valued graph-only
  column with explicit `forced_candidate_target=true` and
  `expression_output_gene=false`; the graph can still learn its identity and
  incident public relations, while no expression label is fabricated.
- The original K562 artifact has unique `var_names`, while display column
  `var.gene_name` aliases `TBCE-1 -> TBCE` and `HSPA14-1 -> HSPA14`. The
  canonical model axis therefore follows the official code-facing unique
  `var_names`; the non-unique display column is never used as an identifier.
- The two Nadig raw files each contain two Ensembl rows labelled `HSPA14`.
  Preprocessing follows the upstream filter's first-match behavior: the first
  row remains `HSPA14`, while each later collision is assigned the stable ID
  `HSPA14__<Ensembl var index>`. Both expression features remain present, but
  the perturbation anchor resolves only to the unsuffixed first-match row.

## State machine and files

```text
registered -> downloaded -> source_verified -> canonicalized -> qc_passed
           -> condition_eligible -> split_frozen -> controls_frozen
           -> canonical_ready
```

Server layout:

```text
data/<dataset>/<protocol>/
  source/
  canonical/adata.h5ad
  canonical/expression_gene_ids.txt
  canonical/graph_gene_ids.txt
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
  target fractions 0.5625/0.1875/0.25.
- Norman: import and normalize the official predefined doubles split; do not
  replace it with random fractions.
- After the source split is generated, apply the shared
  `gears_default_graph_intersection_v1` condition policy to train, validation,
  and test alike. It excludes conditions that the frozen official GEARS
  default perturbation graph cannot represent. Retained conditions preserve
  their original partition and order; they are never reshuffled after this
  filter. Every dataset registry records the exact exclusions, frozen GEARS
  commit, and SHA-256 hashes of the two official graph resources.
- Excluded rows may remain in the canonical H5AD for provenance but are outside
  all three benchmark partitions. This is intentional, explicit condition loss
  rather than a model-specific silent filter.
- Controls are accessible according to context but are never themselves a
  perturbation split condition.
- Every adapter writes consumed train/val/test condition IDs. The orchestrator
  compares ordered IDs and set hashes to the canonical manifest.
- The shared comparison run is seed 1 for GraD-Pert, GEARS, TxPert, and all
  deterministic nonlearned baselines. Extra GraD-Pert full-run seeds are
  replicate-only and never replace the paired seed-1 comparison.

## Evaluation controls

- For each validation/test condition, retain the context ID (cell line plus
  experimental batch) of every real perturbed cell. A 300-draw evaluation
  population is formed in two stages: sample 300 of those real-cell context
  IDs with replacement, then sample one control row with replacement from the
  exact matching context for each selected context ID. This preserves the
  condition's observed context mixture instead of collapsing all batches into
  one control pool.
- Use PCG64 with stable input
  `evaluation_seed :: dataset_id :: split :: condition_id`.
- Sample exactly 300 row IDs with replacement, even when the pool exceeds 300.
- Store the ordered context IDs, ordered row IDs, both order hashes, and the
  source-pool hash. Every model uses these exact 300 rows in this exact order;
  a model never samples internally for the common benchmark.
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

### Final ResultCatalog gate

The notebook-facing formal catalog contains exactly 45 runs:

- all six models/baselines x five datasets at paired seed 1;
- GraD-Pert-only seeds 2, 3 and 4 on all five datasets.

`scripts/results/build_final_catalog.py` consumes one explicit
`result-catalog-source-spec-v1` JSON. Every entry names the exact synchronized
`run_manifest.json`, `server_pointer.json`, and `metrics_summary.csv`; it never
discovers a newest directory. The default is a no-write plan. `--execute`
seals `result_catalog.json` plus `.json.sha256` only after proving:

- exact model/dataset/seed coordinate completeness and uniqueness;
- one formal published source commit;
- per-dataset equality of protocol, canonical data, split and 300-control
  hashes across all nine runs;
- one native config hash across seeds 1--4;
- the exact three metric rows/schema and equal condition denominators.

`benchmark_results.ipynb` uses `load_final_result_catalog`; a generic partial
catalog cannot silently become the formal comparison table.

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
