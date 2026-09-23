# Six-dataset perturbation landscape

This CPU-only, descriptive analysis extends the existing five-dataset PCA,
UMAP and split-half report. It covers the five independent within-cell
datasets and the separate 3,352-gene fixed-axis cross-cell cache. It does not
select model variants, training length or checkpoints. Test perturbations are
read only for descriptive statistics and are never fed back into training.

| Analysis | Scope | Interpretation and source |
| --- | --- | --- |
| Perturbation–batch association | Each of five datasets; each of four cross-cell lines | Fraction of batch entropy explained by perturbation identity, its mean under ten batch-label permutations, their difference, and fraction of conditions occurring in only one batch. Motivated by [TxPert Fig. 1](https://www.nature.com/articles/s41587-026-03113-4). The permutation reference controls finite-sample upward bias; it does not prove a biological cause. |
| Control consistency | Same nine contexts | Pearson correlation between independent halves of controls within a batch, compared with mean controls across batch pairs. Uses one seeded half split. High expression correlation alone does not imply absent batch effects. |
| Split-half retrieval | Same nine contexts | Among up to 500 conditions with at least 20 cells, retrieve the matching second-half perturbation by Pearson correlation of control-subtracted expression shifts. Report top-1/top-10 and chance. This adapts TxPert's perturbation-specific [retrieval analysis](https://www.nature.com/articles/s41587-026-03113-4); it is not a model score. |
| Shared response and effect magnitude | Same nine contexts | Per-condition RMS expression shift and correlation with the mean perturbation shift. A strong shared response can boost average Pearson without identifying the perturbation. |
| Cross-line transfer | Four cross-cell target folds only | Pearson correlation of the same perturbation's expression shift between lines, and of target shift against the mean of source-line shifts. This is a descriptive source-response reference, not a trained cross-cell prediction. |
| Double-gene additivity | Norman only | Compare observed double shift with the sum of its two measured single shifts; report Pearson and relative residual. Inspired by [GEARS genetic-interaction analysis](https://www.nature.com/articles/s41587-023-01905-6). A large residual is not, by itself, a validated synergy/epistasis subtype. |

All expression shifts subtract the mean control expression of the same cell
line, averaged across its batches. This makes the cross-line analyses simple
and consistent but does **not** remove batch confounding. The batch diagnostics
must therefore be read alongside response comparisons. Raw RMS magnitudes are
not directly comparable across the five independent gene axes or expression
processing histories. Within-dataset ranks and paired comparisons are the
defensible units. Cross-cell line comparisons share one fixed axis but may
still reflect line-specific experimental sources.

The script streams the backed sparse H5ADs and saves only aggregated JSON and
figures on `/data/yilangliu`. A single-batch dataset has no batch-association
or across-batch control comparison; plots mark these cases N/A. Control
dissimilarity is displayed as `1000 × (1 − Pearson)` to make small differences
visible. The five-dataset split-half estimates already in
`docs/experiments/FIVE_DATASET_EDA.md` remain a separate repeatability
analysis. The additional retrieval half split uses seed 42 and strata of
cell line, condition and batch; its correlation is not an upper bound on model
performance. Its control-reference estimate shares the same control pool
between halves, so the number is best read as a ranking diagnostic.
