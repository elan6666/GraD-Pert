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
