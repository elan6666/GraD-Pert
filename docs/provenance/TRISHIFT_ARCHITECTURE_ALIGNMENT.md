# TriShift Experiment-Architecture Alignment

## Audit boundary

- Read-only source: `/Users/elan/code/trishift`
- Audited commit: `87ac2c51c3c266391093f71a8bce2e6beaa81518`
- Purpose: reuse proven experiment-organization patterns, not model code or
  experiment values.
- GraD-Pert remains an independent package. External GEARS and TxPert runners
  call their frozen official packages in isolated environments.

## Patterns retained

### Thin dataset entrypoints over a shared runner core

TriShift's dataset entrypoints, for example
`scripts/gears/adamson/run_gears_adamson.py`, contain only argument/config
selection and delegate execution to `scripts/gears/_core/gears_eval_core.py`.

GraD-Pert keeps the same separation without one script per dataset:

```text
one exact model/dataset YAML
  -> model-specific isolated CLI
  -> shared model-specific adapter/core
  -> neutral PredictionArtifact
```

The CLI accepts exactly one config path. It does not merge a global config.

### Official-package isolation

TriShift dynamically imports the external `gears` package in its GEARS core.
GraD-Pert retains the process/package boundary and strengthens it:

- exact official repository commit and environment hash are preflight gates;
- the external package is imported only under `benchmarks/<model>/`, and the
  isolated import session remains active for the full official API lifecycle
  so lazy imports cannot resolve to another environment;
- `src/gradpert` never imports an external learned model;
- adapters call official construction, training, checkpoint, and forward APIs;
- no official model, loss, optimizer, or training loop is reproduced locally.

For the public TxPert package, the frozen `main.py` exposes checkpoint-based
inference/prediction but no fit entrypoint. The isolated smoke therefore lets
Lightning drive the official `PertPredictor.training_step` and
`configure_optimizers` for one epoch. Validation is disabled inside that fit
because the official validation hook also evaluates test; common validation
and test evaluation remain separate post-checkpoint phases.

### Condition-keyed reusable payloads

TriShift exports one dictionary entry per condition with full-gene prediction,
control, truth, DE indices/names, and metadata. Its
`recompute_metrics_from_pkl.py` demonstrates that metrics can be regenerated
without retraining.

GraD-Pert retains condition-keyed server PKLs but separates information phases:

1. runner writes truth-free `PredictionArtifact` with `Pred[300,G]`,
   `InputCtrl[300,G]`, ordered row IDs, gene IDs, and hashes;
2. evaluator alone joins all real `Truth[N_condition,G]`, metric-control state,
   DE/Top-DE state, and writes `EvaluationBundle`;
3. metrics CSV/JSON are regenerated from the sealed evaluation bundle.

This retains downstream convenience while preventing validation/test truth from
entering a model runner.

### One result adapter for notebooks

TriShift's `scripts/trishift/analysis/_result_adapter.py` centralizes model
labels, payload filename resolution, split discovery, and payload loading;
notebook helpers call that adapter instead of duplicating result parsing.

GraD-Pert exposes a typed `ResultCatalog` in
`src/gradpert/artifacts/catalog.py`. Each entry pins the exact
`run_manifest.json`, server-artifact pointer, and small metrics file by SHA-256.
The loader rejects directory discovery, symlinks, path escape, mutated files,
unevaluated runs, and mismatched run/pointer identities. Notebooks consume the
catalog/evaluation bundle and never implement training, splitting, control
sampling, or metric formulas.

The formal builder additionally requires an explicit 45-entry source spec and
rejects incomplete coordinates, fairness-hash drift, metric-schema drift and
condition-denominator drift before writing the catalog or trusted SHA sidecar.

### Small summaries plus rich server artifacts

TriShift pairs PKL payloads with `metrics.csv`, mean text, training-loss CSV,
and `run_meta.json`. GraD-Pert retains this two-tier output topology, adding
schema versions and content hashes. Optional result PKL, H5AD, and checkpoints remain on the
server; only allowlisted small receipts/results synchronize locally.

### Exact run snapshots and operational receipts

TriShift writes the materialized configs used by a run, `run_meta.json`,
per-split train logs, and a tabular training-loss history. These are useful
operational surfaces because a checkpoint alone cannot prove which data or
settings produced it.

GraD-Pert adopts the separation but strengthens identity:

- the one model/dataset YAML is copied byte-for-byte and its SHA-256 is placed
  in the run manifest; no merged temporary defaults file exists;
- epoch history, selected checkpoint, consumed condition IDs, peak GPU memory,
  and lifecycle events are small append-only receipts;
- the sealed run manifest changes state explicitly from `started` through
  `evaluated`, with at most one test evaluation;
- a ResultCatalog admits only `evaluated` runs with `test_evaluations=1`.

### Thin notebook layer, no experiment semantics

TriShift figure notebooks centralize much of their result loading in
`notebooks/_figure_helpers.py`, but that helper also contains extensive path
fallbacks and figure-specific result resolution. GraD-Pert keeps notebook code
smaller: it selects an explicit catalog entry and renders already-computed
tables/arrays. Dataset splitting, metric computation, result selection, and
fallback recomputation remain package/CLI responsibilities with tests.

## Patterns explicitly rejected

| TriShift pattern observed | Why it is rejected here | GraD-Pert contract |
|---|---|---|
| `configs/defaults.yaml` + `configs/paths.yaml` + recursive overrides | A run can depend on hidden global/inherited values | 30 complete independent YAML files; no merge/default/anchor/template |
| Python `DATASET_CONFIG` values beside YAML | Runtime defaults can disagree with the visible config | Config schema requires every experiment value; adapters do not supply experiment defaults |
| Runtime replacement of GEARS loss, analyses, and `GEARS.__init__` | Changes the benchmark implementation and ceases to be the official model | Call frozen official APIs; fail explicitly if canonical data is incompatible |
| External runner computes metrics and bundles Truth | Mixes model execution with evaluator-only information | Truth-free prediction artifact, followed by a separate common evaluator |
| GEARS `.predict()` mean path or an ambiguous subset helper | Can collapse the required 300 control-specific predictions | Official per-control forward on the exact ordered manifest rows; assert `[300,G]` |
| Search for the newest matching result directory | Can silently consume a stale or wrong run | Explicit sealed run ID/path plus config/code/data/split/control hashes |
| Alias files such as one variant copied to `metrics.csv` | Hides which variant produced the default result | One immutable run/metric identity; any view is a pointer with the source hash |
| Recompute scripts fall back between full summary and subsets | Formula/input surface may change silently | Versioned evaluator schema and exact metric applicability/reason fields |

## Target repository shape

```text
configs/experiments/<model>/<dataset>.yaml  # one complete source of run values

benchmarks/gears/
  runner.py                 # isolated CLI, imports official gears
  canonical_adapter.py      # AnnData/split/control translation only
  official_api.py           # narrow calls to frozen official API
  environment.lock

benchmarks/txpert/
  runner.py                 # isolated CLI, imports official gspp
  canonical_adapter.py
  official_api.py
  environment.lock

src/gradpert/artifacts/
  prediction.py             # truth-free neutral artifact
  evaluation.py             # evaluator-only truth join
  catalog.py                # implemented explicit sealed-run resolution

notebooks/
  benchmark_results.ipynb   # implemented, executed catalog consumer only

scripts/results/
  build_final_catalog.py    # explicit-source, dry-run-first 45-run catalog gate
```

## External one-epoch acceptance gate

For each GEARS/TxPert × dataset config, a passing smoke must prove all of the
following from receipts, not just exit code:

1. exact official commit and import origin;
2. exact config bytes/hash and official parameter provenance;
3. actual consumed train/val/test condition IDs equal the canonical manifest;
4. one completed optimizer epoch and a loadable official checkpoint;
5. no validation-time test-truth access;
6. prediction for every required condition with exact ordered 300 input-control
   row IDs and `[300,G]` output;
7. valid truth-free PredictionArtifact and common evaluator output;
8. no unexplained condition/gene coverage loss.

Only GraD-Pert proceeds from this gate to full `max_epochs=100`, validation-only
early stopping with patience 10. GEARS and TxPert remain `smoke_only` in the
current execution policy.
