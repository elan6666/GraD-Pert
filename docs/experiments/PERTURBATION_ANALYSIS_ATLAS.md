# Perturbation analysis atlas for future GraD-Pert evaluation

This is a literature-grounded **analysis menu**, not a claim that the listed
experiments have run. It extends the measured [five-dataset EDA](FIVE_DATASET_EDA.md),
[six-resource landscape](SIX_DATASET_LANDSCAPE_RESULTS.md), and
[Systema-inspired audit](LITERATURE_GUIDED_DATA_AUDIT_RESULTS.md). The five
canonical within-cell datasets and the separate fixed-axis cross-cell cache
remain distinct protocols. Model evaluation uses frozen best/last checkpoints,
ordered control/truth rows, the same feature axis, and condition-level units.

## Questions to answer in order

| Question | Concrete analysis and output | Literature rationale | Data needed / scope |
| --- | --- | --- | --- |
| **1. Is a perturbation detectable at all?** | E-distance from matched controls with label-permutation null; effect-size rank, confidence interval and FDR. Plot strength versus number of cells. Do not equate a high DEG count with a stronger effect without power control. | [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) uses E-statistics for perturbation distinctiveness and cell-count/depth studies. | Per-cell expression and controls: five + four cross-cell lines. Batch-stratified permutations where overlap permits. |
| **2. Is its signature reproducible?** | Repeated split-half correlations and top-gene sign/Jaccard agreement; between-batch and, if available, between-guide consistency. Pair each estimate with cell counts and a noise reference. | [FR-Perturb](https://www.nature.com/articles/s41587-023-01964-9) evaluates replicated perturbation effects and guide agreement; [pseudobulk DE research](https://www.nature.com/articles/s41467-021-25960-2) shows that cells are not substitutes for biological replicates. | Per-cell expression and batch now; guide/biological-replicate analysis only if audited metadata contains those IDs. Existing split-half work is a starting point. |
| **3. How hard is this dataset after controlling sample size?** | Repeat the same statistic after equal-cell downsampling at 10/20/50/100 cells per condition; show saturation curves, eligible-condition counts and bootstrap intervals. Cross-dataset ranks use matched gene intersections or within-dataset normalization. | [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) explicitly studies cell number and read-depth sensitivity; [CellOT](https://www.nature.com/articles/s41592-023-01969-x) uses observed-versus-observed comparisons as a noise reference. | Five + cross-cell; read-depth curves require audited raw counts and must not be manufactured from log-normalized H5AD. |
| **4. Which effects are shared versus target-specific?** | Decompose each centroid shift into common perturbed-control component and perturbation-specific residual; plot magnitude, angle, effective rank, nearest-neighbor recovery and score changes under the perturbed-centroid reference. | [Systema](https://www.nature.com/articles/s41587-025-02777-8) demonstrates metric inflation from systematic variation and supplies a perturbation-specific reference and centroid accuracy. | Five + cross-cell. The first common-shift/constant-baseline audit is already measured; do not mistake it for a model score. |
| **5. Are effects homogeneous across cells?** | Compare responder-score distributions, mixture proportions and conditional shifts across basal cell states; separately inspect whether a perturbation changes cell-state proportions. Report negative-control calibration. | [mixscape](https://www.nature.com/articles/s41588-021-00778-2) distinguishes KO-like and non-perturbed cells; [Perturbation Score](https://www.nature.com/articles/s41556-025-01626-9) treats response strength continuously; [CINEMA-OT](https://www.nature.com/articles/s41592-023-02040-5) separates response and confounder structure. | Per-cell expression and sufficient controls. Call inferred groups *response strata*, not causal individual treatment effects without assumptions/validation. |
| **6. What biological programs changed?** | Predeclare versioned cell-cycle, stress, DNA repair, ribosome and GO gene sets; compare observed and predicted program-score shifts, direction concordance and bootstrap uncertainty. Test whether removing a common program changes the ranking. | [Systema](https://www.nature.com/articles/s41587-025-02777-8) uses GSEA/AUCell to interpret systematic effects; [Replogle et al.](https://pubmed.ncbi.nlm.nih.gov/35688146/) map functionally coherent programs; [Ota et al.](https://www.nature.com/articles/s41586-025-09866-3) connect regulatory effects to programs and traits. | Five + cross-cell for transcriptomic programs. Independent CIN, viability or disease-trait claims require external measured labels. |
| **7. Are double-gene effects additive?** | For Norman, fit `delta_AB = a delta_A + b delta_B + residual`; report coefficients, residual direction/norm and bootstrap intervals. Compare 0/1/2-seen component strata and simple additive/matching-mean baselines. | [Norman et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/) fit interaction coefficients; [Systema](https://www.nature.com/articles/s41587-025-02777-8) defines matching mean for combinations. | Norman only. Existing sum-of-singles result does not establish synergy, suppression or epistasis subtype. |
| **8. Which effects transfer between contexts?** | In four leave-one-cell-line-out folds, stratify targets by source-line agreement, target effect size, program conservation, target seen/unseen status and source coverage. Compare source-average baseline, each model, and cell-line-specific controls. | [Nature Methods 27-method benchmark](https://www.nature.com/articles/s41592-025-02980-0) evaluates context generalization; [CellOT](https://www.nature.com/articles/s41592-023-01969-x) emphasizes context-dependent distributional response. | Fixed-axis cross-cell cache only; four folds are our own benchmark, not exact TxPert fold-specific HVG reproduction. |
| **9. Does a model match the full population, not just the mean?** | Alongside condition-centroid Pearson/RMSE, calculate matched-cell-count E-distance or MMD between predicted and observed post-perturbation populations; compare to observed-half/observed-half and identity baselines. Inspect variance, quantiles and multi-mode failures. | [CellOT](https://www.nature.com/articles/s41592-023-01969-x) explains why equal means may hide different distributions; [pertpy](https://www.nature.com/articles/s41592-025-02909-7) catalogs distance and permutation tests. | Requires per-cell model predictions and observed test cells; should use a fixed gene/latent axis and inspect sensitivity to dimensionality/kernel bandwidth. |
| **10. Does a model recover specific genes and relations?** | Condition-level top-k DEG precision/recall, signed rank correlation and sign agreement; stratify by gene expression, effect size, target gene, graph distance and whether the gene was excluded from training expression. Use training/validation data only for gene-panel or threshold choices. | [Systema](https://www.nature.com/articles/s41587-025-02777-8) contrasts all-gene and top-DE metrics; [PertEval-scFM, ICML](https://proceedings.mlr.press/v267/wenteler25a.html) stresses baseline and distribution-shift comparisons. | Five + cross-cell after model predictions; avoid defining an evaluation gene set from one model's outputs. |

## Recommended experiment order

1. **Data-quality audit:** E-distance/permutation, controlled cell-count curves,
   split-half/guide/batch reliability. This determines which conditions have a
   measurable effect and which metrics are interpretable.
2. **Effect taxonomy:** shared versus specific shifts, program responses,
   responder heterogeneity, and Norman interaction residuals. Store condition
   IDs and confidence intervals on the server; export only compact summaries.
3. **Model audit after best/last receipts:** evaluate B0/B1 and external baselines
   on the same frozen conditions with control and perturbed references, centroid
   accuracy paired by condition, top-gene/biological-program recovery, and
   distributional metrics when per-cell outputs exist.
4. **Cross-cell audit:** use the four sealed folds and stratify transfer by
   source agreement and target-line signal; keep this separate from within-cell
   ranking because it uses a fixed K562-selected gene axis.

For every stratum, report its condition count, cell count distribution, control
source, gene axis and split. Use condition-level bootstrap for uncertainty;
cell-level bootstrap alone can overstate precision when guides, batches or
biological replicates are the true independent units. A result on selected
high-effect conditions must also show the all-eligible-condition result, since
effect filtering can make a dataset appear artificially easy. No test-derived
effect stratum may guide model selection or tune a threshold.
