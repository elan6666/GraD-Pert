# Jurkat train-only cap40 dataset analysis

User-approved scope: keep at most 40 cells per training perturbation condition,
with proportional batch quotas and sampling without replacement. Keep every
nontraining row, all control cells, gene identity and expression values. Analyze
the original and one fixed seed42 sample with 100 independently seeded balanced
batch-stratified half partitions. No training default or experiment is launched.

Pre-change published source: `5d602a02158af679a4d384f677176ec910fa238a`.
Dataset parent: `/data/yilangliu/GraD-Pert/data/nadig_jurkat/within_cell_unseen_single`.
H5AD SHA256: `65b32637b24e6ee6d3399b3280d914dcedc34bf787098c3e9e14a47ebe80cbb5`.
Expected training rows 128266→47836; total rows 238977→158547; all 1335 training
conditions and all 6506 graph-axis variables retained, statistics on 5000 genes.

1. Implement/test strict cap, batch allocation, random singleton assignments,
   vectorized versus direct Pearson, and exact H5AD row/value/metadata preservation.
2. Publish only scoped code/docs on main; create a clean immutable server checkout,
   then run server tests. Reconcile current jobs before launching one CPU-only run.
3. Materialize fixed row IDs and a standalone analysis-derivative H5AD. It is not
   relabeled canonical-ready. Original controls and split hashes must remain exact.
4. Compute condition-equal split-half Pearson delta and raw-expression diagnostics;
   complement correlation and overlapping sample/full correlation for affected
   conditions. Use original condition-matched control pools for both versions.
   Record missing scores, count strata and paired-condition bootstrap intervals.
5. A Luna subagent owns bounded read-only waiting; main performs one terminal
   acceptance, pulls only small approved summaries/plots and writes a Chinese report.

Acceptance: 47836 selected train rows, no condition or nontrain row lost, exact
retained values and metadata, 100 split repetitions, source/config/data identities,
COMPLETE and exit0, original H5AD/manifests unchanged, small result report and plots.
This estimates repeatability, not a model score or mathematical performance ceiling.
It is not TxPert's raw-count multinomial sampling estimator. Cap40 changes row-mean
condition weights. Rare batches may disappear because their quota is below one;
record batch coverage instead of claiming exact preservation of every batch.
