# Plan 010: TxPert frozen-Anndata H5AD null compatibility

- Status: canonical-reader compatibility follow-up verified locally

## Evidence

- At clean repair commit `dddc767`, GEARS K562, RPE1, Jurkat, and Norman
  completed successfully, including the new 20-index graph gate.
- The first TxPert task then failed before model construction while frozen
  Anndata 0.11.4 read `/uns/log1p/base` written by Anndata 0.13.2 using the
  newer explicit `null` scalar encoding.
- A server synthetic cross-version check reproduced the failure with
  `base=None`. Removing only that null key produced a file the frozen reader
  loaded with the exact same expression values and shape.
- Commit `c127380` applied that write-side sanitation, but the hard gate proved
  the frozen runner first opens the immutable canonical H5AD directly inside
  `CanonicalTrainingData`, before any adapter copy exists. The full stack ends
  at that canonical read, so write-side sanitation alone cannot be reached.

## Write scope

- `benchmarks/txpert/runner.py`
- `tests/benchmarks/test_txpert_runner.py`
- Byte OS status, build, review, and execution evidence for this repair

## Non-goals

- Do not change canonical H5AD files, expression values, condition splits,
  300-control manifests, official model/training configuration, or metrics.
- Do not upgrade or modify the frozen TxPert environment/checkout.
- Do not rerun the immutable c240 GraD-Pert Nadig Jurkat B0 coordinate.

## Implementation

On the isolated TxPert adapter copy only, remove `/uns/log1p/base` exactly when
its value is `None` before writing `de_adata_test.h5ad`. Absence and null both
mean natural-log base, while absence is readable by the frozen Anndata version.
Preserve any non-null base and every other metadata/value. Record the policy and
removed paths in `official_data_adapter.json`.

Before opening the immutable canonical H5AD, register the exact Anndata 0.13.2
`null` 0.1.0 HDF5 reader behavior in the frozen runner's in-memory Anndata
registry: return Python `None`. Do not modify the frozen environment or source
H5AD. The existing write-side sanitation then removes that value from the
adapter copy so frozen Anndata does not need a corresponding null writer.

## Acceptance criteria

1. The synthetic file with explicit null fails under frozen Anndata 0.11.4,
   while the sanitized file loads with identical expression values and shape.
2. Only a null `/uns/log1p/base` is removed; non-null bases are preserved.
3. The adapter receipt records policy and exact removed metadata paths.
4. TxPert still receives no canonical test truth during fit and runs one epoch.
5. Full test/lint/format/type/build gates and a real server TxPert smoke pass at
   one new clean synchronized commit.
6. The dddc failure lineage remains recoverable under a precise superseded path.
7. The canonical-reader compatibility is registered in memory only, recorded in
   the adapter receipt, and reads `null` 0.1.0 exactly as the inspected newer
   Anndata implementation does.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_txpert_runner.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
