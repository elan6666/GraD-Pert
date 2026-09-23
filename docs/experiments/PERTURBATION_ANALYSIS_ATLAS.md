# Perturbation dataset analysis atlas

This is a literature-grounded **dataset-only analysis menu**, not a claim that
the listed experiments have run. This phase uses observed cells and metadata
only: it does not score GraD-Pert, external models, baselines or checkpoints.
It extends the measured [five-dataset EDA](FIVE_DATASET_EDA.md),
[six-resource landscape](SIX_DATASET_LANDSCAPE_RESULTS.md), and
[Systema-inspired audit](LITERATURE_GUIDED_DATA_AUDIT_RESULTS.md). The five
canonical within-cell datasets and the separate fixed-axis cross-cell cache
remain distinct protocols. Analysis units are conditions, cells, batches and
biological replicates when the corresponding metadata actually exists.

## Questions to answer in order

| Question | Concrete analysis and output | Literature rationale | Data needed / scope |
| --- | --- | --- | --- |
| **1. Is a perturbation detectable at all?** | E-distance from matched controls with label-permutation null; effect-size rank, confidence interval and FDR. Plot strength versus number of cells. Do not equate a high DEG count with a stronger effect without power control. | [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) uses E-statistics for perturbation distinctiveness and cell-count/depth studies. | Per-cell expression and controls: five + four cross-cell lines. Batch-stratified permutations where overlap permits. |
| **2. Is its signature reproducible?** | Repeated split-half correlations and top-gene sign/Jaccard agreement; between-batch and, if available, between-guide consistency. Pair each estimate with cell counts and a noise reference. | [FR-Perturb](https://www.nature.com/articles/s41587-023-01964-9) evaluates replicated perturbation effects and guide agreement; [pseudobulk DE research](https://www.nature.com/articles/s41467-021-25960-2) shows that cells are not substitutes for biological replicates. | Per-cell expression and batch now; guide/biological-replicate analysis only if audited metadata contains those IDs. Existing split-half work is a starting point. |
| **3. How hard is this dataset after controlling sample size?** | Repeat the same statistic after equal-cell downsampling at 10/20/50/100 cells per condition; show saturation curves, eligible-condition counts and bootstrap intervals. Cross-dataset ranks use matched gene intersections or within-dataset normalization. | [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) explicitly studies cell number and read-depth sensitivity; [CellOT](https://www.nature.com/articles/s41592-023-01969-x) uses observed-versus-observed comparisons as a noise reference. | Five + cross-cell; read-depth curves require audited raw counts and must not be manufactured from log-normalized H5AD. |
| **4. Which effects are shared versus target-specific?** | Decompose each observed centroid shift into common perturbed-control component and perturbation-specific residual; plot magnitude, angle, effective rank and split-half nearest-neighbor recovery before/after common-shift removal. | [Systema](https://www.nature.com/articles/s41587-025-02777-8) demonstrates shared systematic variation and proposes a perturbation-specific reference. | Five + cross-cell. The first common-shift audit is already measured; this phase describes data geometry only. |
| **5. Are effects homogeneous across cells?** | Compare responder-score distributions, mixture proportions and conditional shifts across basal cell states; separately inspect whether a perturbation changes cell-state proportions. Report negative-control calibration. | [mixscape](https://www.nature.com/articles/s41588-021-00778-2) distinguishes KO-like and non-perturbed cells; [Perturbation Score](https://www.nature.com/articles/s41556-025-01626-9) treats response strength continuously; [CINEMA-OT](https://www.nature.com/articles/s41592-023-02040-5) separates response and confounder structure. | Per-cell expression and sufficient controls. Call inferred groups *response strata*, not causal individual treatment effects without assumptions/validation. |
| **6. What biological programs changed?** | Predeclare versioned cell-cycle, stress, DNA repair, ribosome and GO gene sets; estimate observed program-score shifts and uncertainty for each condition. Test how much of the common response is explained by each program. | [Systema](https://www.nature.com/articles/s41587-025-02777-8) uses GSEA/AUCell to interpret systematic effects; [Replogle et al.](https://pubmed.ncbi.nlm.nih.gov/35688146/) map functionally coherent programs; [Ota et al.](https://www.nature.com/articles/s41586-025-09866-3) connect regulatory effects to programs and traits. | Five + cross-cell for transcriptomic programs. Independent CIN, viability or disease-trait claims require external measured labels. |
| **7. Are double-gene effects additive?** | For Norman, fit `delta_AB = a delta_A + b delta_B + residual`; report coefficients, residual direction/norm and bootstrap intervals. Stratify by whether both corresponding singles were measured well and by single-effect strength. | [Norman et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/) fit interaction coefficients and examine non-additive effects. | Norman only. Existing sum-of-singles result does not establish synergy, suppression or epistasis subtype. |
| **8. How context-dependent is the same perturbation?** | On the common 3,352-gene axis, compare the same observed condition across four cell lines: pairwise effect correlations, sign agreement, effect-strength changes, program conservation and source-line disagreement. Report shared-condition coverage. | [Replogle et al.](https://pubmed.ncbi.nlm.nih.gov/35688146/) contrast functional responses across cell lines; [CellOT](https://www.nature.com/articles/s41592-023-01969-x) emphasizes context-dependent population response. | Fixed-axis cross-cell cache only; source studies and line-specific controls must be identified. This does not score leave-one-line-out predictions. |
| **9. Does a mean effect hide a distribution change?** | Compare each observed perturbation population with matched controls using E-distance/MMD, covariance or variance change, quantiles and state proportions. Repeat with equal cell counts and an observed-half/observed-half noise reference. | [CellOT](https://www.nature.com/articles/s41592-023-01969-x) explains why equal means may hide different distributions; [pertpy](https://www.nature.com/articles/s41592-025-02909-7) catalogs distance and permutation tests. | Observed cells only, five + cross-cell. Fix the gene/latent axis and inspect sensitivity to dimensionality and kernel bandwidth. |
| **10. Which genes are reliable response markers?** | Recompute observed differential-expression ranks in independent cell halves or biological replicates; measure top-k overlap, sign consistency and effect-size uncertainty by baseline expression and detection rate. | [FR-Perturb](https://www.nature.com/articles/s41587-023-01964-9) estimates sparse regulatory effects; [single-cell DE analysis](https://www.nature.com/articles/s41467-021-25960-2) emphasizes replicate-aware uncertainty. | Five + cross-cell; formal DEG significance requires genuine replicate structure, not arbitrary cell halves treated as independent experiments. |

## Recommended experiment order

1. **Data-quality audit:** E-distance/permutation, controlled cell-count curves,
   split-half/guide/batch reliability. This determines which conditions have a
   measurable effect and which metrics are interpretable.
2. **Effect taxonomy:** shared versus specific shifts, program responses,
   responder heterogeneity, and Norman interaction residuals. Store condition
   IDs and confidence intervals on the server; export only compact summaries.
3. **Distribution and gene-level audit:** observed response heterogeneity,
   program shifts and repeated observed DEG rankings; distinguish a change in
   population composition from a shift within the same basal cell state.
4. **Combination and context audit:** Norman interaction residuals and the
   same perturbations across the four cross-cell lines. Keep the cross-cell
   cache separate from within-cell difficulty ranking because it uses a fixed
   K562-selected gene axis.

For every stratum, report its condition count, cell count distribution, control
source, gene axis and split. Use condition-level bootstrap for uncertainty;
cell-level bootstrap alone can overstate precision when guides, batches or
biological replicates are the true independent units. A result on selected
high-effect conditions must also show the all-eligible-condition result, since
effect filtering can make a dataset appear artificially easy. Preserve the
full-condition distributions alongside any selected examples. These dataset
descriptors can later define frozen strata, but this phase does not analyze
model outputs or choose model settings.
