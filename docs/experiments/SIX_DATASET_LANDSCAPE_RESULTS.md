# Six-dataset landscape: first measured results

Analysis source commit: `846d2d7194e396d9c9845ad3d25b73794170ebc7`.
Server receipt: `/data/yilangliu/GraD-Pert/development/six-landscape-846d2d7/receipt.json`
(SHA256 `cff41058fcc3114e681fd8f63b5ed9aea15e6789bba3c2ed98111e33bbe75f63`).
The receipt has status `complete` and binds the fixed cross-cell fold manifest.
The code and measurement definitions are in [SIX_DATASET_LANDSCAPE.md](SIX_DATASET_LANDSCAPE.md).
The earlier five-dataset PCA/UMAP and test split-half analysis remains in
[FIVE_DATASET_EDA.md](FIVE_DATASET_EDA.md).

## Batch and perturbation identity

The uncorrected fraction of batch entropy associated with perturbation ID was
0.049–0.137 across the four single-line datasets. After subtracting the mean
of ten batch-label permutations, the excess was only 0.0005 (K562), 0.0019
(RPE1), 0.0006 (Jurkat), and 0.0035 (HepG2). This is an important
finite-sample correction: the raw statistic alone would exaggerate the
association. It does not establish that batch effects are absent; control
profile dissimilarity was usually higher across batches than within a batch.
Norman has one batch under this canonical metadata and so has no defined
between-batch comparison. [TxPert](https://www.nature.com/articles/s41587-026-03113-4)
motivated both diagnostics.

## Perturbation-specific information

Top-1 retrieval compares split-half control-subtracted perturbation means
among at most 500 eligible conditions with at least 20 cells. It is a data
repeatability diagnostic, not a model prediction result. Chance is 0.2% for
500 conditions; Norman's 236 eligible conditions give 0.42% chance.

| Dataset or fixed-axis line | Top-1 retrieval | Conditions |
| --- | ---: | ---: |
| Replogle K562 | 56.8% | 500 |
| Replogle RPE1 | 52.0% | 500 |
| Nadig Jurkat | 40.6% | 500 |
| Nadig HepG2 | 32.4% | 500 |
| Norman | 89.8% | 236 |
| Cross-cell K562 | 51.2% | 500 |
| Cross-cell RPE1 | 61.6% | 500 |
| Cross-cell Jurkat | 47.0% | 500 |
| Cross-cell HepG2 | 38.4% | 500 |

The strong differences should not be presented as a ranking of model
difficulty: the source studies, condition mixes, expression axes and cell
counts differ. Correlation with the mean response was also substantial
(median 0.32–0.73 across the nine contexts), so a general stress-like
component can coexist with perturbation-specific retrieval. This combination
motivates reporting both Pearson Δ and retrieval for trained models, as in
[TxPert](https://www.nature.com/articles/s41587-026-03113-4).

## Transfer and combinatorial response

For a held-out cross-cell line, the median Pearson Δ between its measured
response and the mean of available source-line responses to the same
perturbation was 0.507 (K562, 1,051 shared conditions), 0.501 (RPE1, 1,449),
0.427 (Jurkat, 1,413), and 0.480 (HepG2, 1,499). These are descriptive
measurements using target truth, not GraD-Pert or TxPert scores. The paired
line heatmap and its condition counts are in the server `crosscell.json`.

Norman has 131 observed double-gene conditions with both corresponding
single-gene responses. The median Pearson correlation between the observed
double shift and the sum of single shifts was 0.922; the median L2 residual
relative to the observed double shift was 0.443. This demonstrates substantial
additive structure plus non-additive remainder. It does not assign any
individual pair to synergy, suppression, redundancy, neomorphism or
epistasis; doing that requires the subtype definitions and uncertainty
analysis used by [GEARS](https://www.nature.com/articles/s41587-023-01905-6).

All four groups of measurements above are descriptive. The cross-cell cache
uses one K562-selected fixed gene axis and is not the paper's four
independently reselected HVG axes. Control subtraction uses each line's
overall control mean, so batch-dependent differences may still affect
response correlations. No result here should be used to select model
hyperparameters or checkpoints on the frozen test conditions.
