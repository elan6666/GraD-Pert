# Five-dataset descriptive analysis

This analysis is descriptive. It reads the frozen canonical test labels to
measure experimental reproducibility and must not select model variants,
hyperparameters, checkpoints, or training duration. The five within-project
datasets are Replogle K562 and RPE1, Nadig Jurkat and HepG2, and Norman.

## Visualization

For each independent dataset, sample up to 2,500 cells per perturbation class
with a 24-cell cap per non-control condition. Class means control, one
non-control target, or two non-control targets. Plot PCA and PCA-input UMAP,
colored separately by class, frozen condition split, and experimental batch.
Canonical conditions omitted by the official-graph representability filter
are labeled `excluded` rather than assigned to a benchmark split.
The plot is exploratory: distance and cluster shape are not a numerical
measure of perturbation prediction quality.

## Experimental reproducibility

TxPert describes two distinct estimators in its published Methods:

1. **Split-half:** for each perturbation, cell context, and batch, randomly
   partition observed test cells into roughly equal, disjoint halves. Compare
   their mean expression profiles. The project analysis repeats this five
   times, reports Pearson correlation on the complete frozen expression gene
   axis after subtracting the same frozen 300-control mean, and also reports
   raw-expression Pearson as a diagnostic. A condition with fewer than two
   cells has no split-half score. This is an empirical repeatability reference,
   not a mathematical upper bound on model performance.
2. **Sample-based extension:** fit a per-perturbation-and-batch multinomial
   gene distribution from the **original count matrix**; stochastically
   generate two independent datasets, each with the original number of cells;
   apply the original normalization, log transform and fixed HVG selection;
   then compute agreement. Sampling cells or adding noise directly to the
   canonical log-expression matrix is not this estimator. The exact count
   sampling and cell-library-size rules must be specified before implementation.

The frozen canonical H5ADs have normalized/log expression as `X`. K562's
frozen upstream-processed source does not supply verified raw counts through
the current canonical path. RPE1, Jurkat and HepG2 have raw-count source
files, while Norman has a `counts` layer whose raw-count semantics need an
explicit audit. Therefore the initial five-dataset run computes split-half
only and labels the sample-based estimate unavailable; it does not substitute
a log-expression bootstrap or imply the two estimators are equivalent.

The implementation writes small summaries, per-condition repeat scores and
plots under `/data/yilangliu`, binds output to the source commit and frozen
manifests, and leaves all source data and running training untouched.

Sources: [TxPert paper](https://www.nature.com/articles/s41587-026-03113-4)
and [frozen public implementation](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f).
