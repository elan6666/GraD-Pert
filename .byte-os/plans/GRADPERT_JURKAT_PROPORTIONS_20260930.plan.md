# Jurkat proportion50 / proportion25 sampling comparison

User clarified that the additional two are sampling methods, not datasets.
Recovering the prior discussion confirms condition-wise 50% and 25% retention.
Published pre-change baseline: 071ba3e13db827aa1f1b8548f5fd9a88d8c771c4.

Only training perturbation rows change. For n cells, retain
min(n, max(min(n,2), floor(f*n+0.5))), f=0.5 or0.25. Use largest-remainder
proportional batch quotas then seed42 uniform sampling without replacement.
Every condition survives; singleton stays singleton, n>=2 stays splittable.
All control, validation, test and excluded rows, 6506 variables and retained
expression values remain exact. Derivatives remain analysis-only, not ready.

1. Add explicit fraction configuration with historical cap40 behavior preserved;
   verify rounding/minimum count, batch quotas, split/ID/matrix preservation.
2. Publish scoped main commit; create clean matching immutable server checkout
   and run relevant synthetic tests. Reconcile jobs; CPU-only4 threads, no GPU.
3. Run each method once sequentially with new IDs,100 balanced disjoint splits,
   same source5000genes and original context-matched control pools as cap40.
4. Luna owns bounded read-only waiting. Main independently validates terminal
   receipts/hashes/partitions/quotas and original reference-score parity, then
   retrieves reviewed small results only. Scientific data remain server-side.
5. Deliver original/cap40/50%/25% plots, all plotting data, Chinese result/PDF;
   update Byte and commit/push. Do not launch training or change defaults.

Primary: condition-equal mean of100 split-half Pearson delta correlations.
Diagnostics: raw-expression Pearson, retained/full (overlap optimistic),
retained/removed (disjoint), cell-count strata and batch coverage loss.
Acceptance: both complete exit0, all1335conditions and nontrain rows preserved,
exact obs/var/matrix, original hashes unchanged, compatible references checked,
zeroPKL, versioned report. This is dataset repeatability, not model scores.

## Completed acceptance 2026-09-30

Clean analysis/comparison SHA8638c6557d779ed2290a00b0882964597ca81d3b;
server source development/source-jurkat-proportions-8638c65. Queue
jurkat-proportions-8638c65-20260930T103836Z exit0,180.40seconds.
50%:64475 training rows,175186total, split-halfdelta=.262513065.
25%:32246 training rows,142957total, split-halfdelta=.166474252.
Original/cap40=.373931664/.261779657. All1335conditions/6506variables;
1331valid, same4singletons absent. Conditions losingbatches1309/1319.
12local and server tests pass; full retained matrix/dtype/obs/var exact;
originalhashesunchanged, all outputhashes/rows/quotas and267000CSVrows perrun
independently accepted. Reference/control/split/gene/assignment identities match
historicalcap40, originalscores atol1e-13.24smallfiles4.217MB reviewed and
verified; H5AD/selection/repeatCSV serveronly. Both Luna wait leases ended.
Report: docs/experiments/JURKAT_SAMPLING_COMPARISON_20260930.md;
PDF and all plot-data/full artifact index linked. No model run/default switch.
