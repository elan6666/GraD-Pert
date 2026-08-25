# Plan 009: GEARS train+validation DE ranking repair

- Status: ranking-before-graph follow-up implementation in progress

## Evidence

- At source commit `c240157`, the official GEARS one-epoch jobs for Replogle
  RPE1 and Nadig Jurkat completed optimization and then failed inside
  `gears.inference.compute_metrics`.
- Their adapted H5AD files contain no `rank_genes_groups_cov_all`. The frozen
  official graph builder therefore emits the documented one-index sentinel
  `de_idx=[-1]`, and SciPy rejects Pearson correlation on a length-one vector.
- K562 contains a 5,000-gene ranking for each condition and passed the same
  official path, isolating the failure to skipped DE materialization.
- At repair commit `dc8e24a`, K562 passed after official rematerialization, but
  RPE1 and Jurkat exposed two and six train+validation conditions respectively
  with only one cell. Frozen Scanpy correctly refuses a t-test for these groups.
- At repair commit `5f82d73`, the singleton-safe ranking maps were correct, but
  frozen `new_data_process(skip_calc_de=True)` had already built and cached PyG
  graphs before those rankings were attached. RPE1/Jurkat therefore still
  trained with the official one-index missing-DE sentinel and failed in the
  same post-epoch Pearson calculation. K562 passed only because its cache had
  been built by the earlier full-ranking attempt, confirming cache construction
  order rather than ranking contents as the remaining defect.

## Write scope

- `benchmarks/gears/official_api.py`
- `benchmarks/gears/runner.py`
- `tests/benchmarks/test_gears_official_api.py`
- Byte OS status, build, review, and execution evidence for this repair

## Non-goals

- Do not modify the frozen GEARS checkout or reimplement GEARS metrics.
- Do not change canonical data, condition splits, 300-control manifests,
  optimizer/model parameters, or common evaluation metrics.
- Do not rerun the GraD-Pert Nadig Jurkat B0 optimization baseline.

## Implementation

Run frozen official condition-name preparation and DE-ranking functions on the
already truth-scoped train+validation AnnData. Canonical test rows remain absent
before model fitting. Record that scope in the adapter receipt.

For conditions with at least two cells, call the frozen official
`rank_genes_groups_by_cov` function unchanged. Keep singleton conditions in the
shared split, exclude them only from the undefined t-test call, and give the
official internal one-epoch metric path a stable full-gene-axis order. The
common evaluator continues to mark singleton DE metrics unavailable; the
fallback is not used by the shared three-metric evaluation.

Call frozen official `get_DE_genes(..., skip_calc_de=True)` first to materialize
condition names, attach safe ranking metadata, and only then call frozen
`PertData.new_data_process` once. Its own official graph builder therefore sees
the rankings on initial construction. Validate the official 20-index DE
contract for one graph from every retained condition; do not synthesize or edit
graph indices locally.

## Acceptance criteria

1. Frozen official condition-name, rank-by-covariate, and dropout/nonzero
   functions are called; singleton conditions never enter the undefined t-test.
2. The official training object still receives no canonical test loader.
3. Adapter receipts state that DE ranking is train+validation only.
4. Benchmark tests, full test/lint/format/type/build gates, and a real server
   GEARS smoke pass at one clean synchronized commit.
5. Failed c240 outputs remain recoverable under a precise `superseded` path.
6. The existing c240 GraD-Pert Nadig Jurkat B0 run is referenced, never rerun.
7. Every official condition graph exposes exactly 20 DE indices before
   split/dataloader construction, including singleton conditions.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_gears_official_api.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
