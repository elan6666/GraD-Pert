# Literature-guided data audit: first measured results

The first CPU audit of all five canonical datasets and the fixed-axis cross-cell
cache completed with source commit `c25fa0a4be1033d304b4e6bd927a686e883d167a`.
Receipt: `/data/yilangliu/GraD-Pert/development/systema-audit-c25fa0a/receipt.json`,
SHA256 `e50202206d7267ec90c2eed42ff4e3db258291e5e2d90220cf7103d229b760cb`.
The earlier source `102dfab` produced a complete pilot, but the `c25fa0a`
receipt is authoritative because it includes the centroid-accuracy diagnostic
distribution. The run used two CPU threads, low scheduling/I/O priority and no
GPU. Full definitions and literature sources are in
[LITERATURE_GUIDED_DATA_AUDIT.md](LITERATURE_GUIDED_DATA_AUDIT.md).

## Shared shifts and a train-only constant baseline

| Dataset | Mean cosine with common shift | Perturbed-mean baseline Pearson Δ, control reference | Same baseline, train perturbed-centroid reference | Split-half Pearson Δ, train perturbed-centroid reference |
| --- | ---: | ---: | ---: | ---: |
| Replogle K562 | 0.395 | 0.416 | -0.075 | 0.594 |
| Replogle RPE1 | 0.525 | 0.665 | -0.050 | 0.627 |
| Nadig Jurkat | 0.290 | 0.335 | 0.091 | 0.346 |
| Nadig HepG2 | 0.332 | 0.368 | 0.057 | 0.358 |
| Norman | 0.548 | 0.619 | 0.134 | 0.919 |

The second and third columns evaluate **the same constant, train-only
prediction** against held-out condition centroids, changing only the reference.
The drop in median Pearson Δ shows how a general perturbed-versus-control shift
can create a positive conventional score without identifying the held-out
perturbation. It is not a GraD-Pert model comparison. The split-half column is
an observed-data repeatability diagnostic using two disjoint halves of cells from each
test condition; it is not an attainable model ceiling because both halves
share a reference estimate. All correlations in this table are medians over
the frozen test conditions, unlike Systema's published multi-split means.

Across the four cell lines of the fixed-axis cache, mean cosine alignment was
0.410 (K562), 0.640 (RPE1), 0.399 (Jurkat), and 0.401 (HepG2). This is
descriptive within-line data geometry. No within-target train mean was
calculated: that would use test-only perturbation rows in a leave-one-line-out
fold. These four values share one 3,352-gene axis but still combine different
experimental sources and cannot by themselves establish why RPE1 is more
aligned.

## Cell counts and signal stability

The seeded split-half Pearson Δ generally rises as conditions contain more
cells, but the pattern is not monotonic in every dataset. K562 test conditions
with 20–49 cells had median 0.447 (26 conditions), versus 0.763 for at least
100 cells (163 conditions); HepG2 had 0.213 (277) versus 0.684 (53). RPE1
had 0.641 (194) in the 20–49 bin and 0.530 (176) in the at-least-100 bin.
These are observational strata: stronger or weaker biological effects may
also be unevenly distributed across count bins. They motivate the controlled
cell-count subsampling and E-statistics analysis proposed by scPerturb, not a
claim that increasing cell count causes these exact gains.

The constant baseline's centroid-accuracy mean is 0.5 and its 10th/90th
percentiles are approximately 0.1/0.9 in every tested dataset. This is a
rank identity for one prediction repeated across all conditions. It must not
be read as 50% successful perturbation identification. Once B0/B1 checkpoint
tests complete, report each model's paired per-condition centroid accuracy
against this baseline; meanwhile the reference-shift contrast is the more
informative data-only result.

This audit does not reproduce Systema's preprocessing, condition splits, gene
panels or three-run protocol. Later, separate observed-data audits measured
Norman's fitted interaction residuals and a bounded random-projection
energy-statistic on a small same-batch eligible subset. Neither reproduces a
paper's exact implementation, and individual causal responses remain
unmeasured. Their results, assumptions and receipts are consolidated in the
[Chinese dataset analysis report](PERTURBATION_DATASET_ANALYSIS_ZH.md).
