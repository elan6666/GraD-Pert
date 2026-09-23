# Literature-guided perturbation-data audit

This expands the existing five-dataset EDA and fixed-axis cross-cell landscape.
The five within-cell datasets retain their frozen condition splits. The cross-cell
cache is a separate sixth resource with four cell-line folds on one K562-selected
3,352-gene axis. None of these descriptive analyses may select a model or tune
on test perturbations. Comparisons of raw expression scale across independent
gene axes are not valid.

## Sources and analyses

| Primary study | Question and executable analysis | Current status |
| --- | --- | --- |
| [Systema, Nature Biotechnology](https://www.nature.com/articles/s41587-025-02777-8) | Quantify mean cosine alignment of each control-subtracted condition centroid with the cell-weighted common shift; calculate a train-only perturbed-mean baseline under control and train-condition-mean references; compare split-half signal under both references; calculate Euclidean centroid accuracy for the constant baseline. | Implemented in `scripts/analysis/systema_data_audit.py`; not the paper's published dataset results. |
| [scPerturb, Nature Methods](https://www.nature.com/articles/s41592-023-02144-y) | Stratify split-half response agreement by cells per condition. Later, estimate E-distance and E-test against control with matched cell counts and permutations, plus cell-count saturation curves. | Count strata implemented; distributional E-statistics need a separate, bounded cell-level analysis. |
| [Norman et al., Science](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/) | Extend the existing observed-double versus sum-of-singles analysis to fitted coefficients and residuals, with cell-bootstrap uncertainty and held-out component logic. | Next CPU analysis; simple unscaled addition is already measured in the six-dataset landscape. |
| [Replogle et al., Cell](https://pubmed.ncbi.nlm.nih.gov/35688146/) | Inspect common cell-cycle, stress and ribosomal programs behind broad perturbation effects; compare target-specific modules between K562 and RPE1 using the shared-gene intersection. | Requires verified gene-set annotation and a separate gene-level audit; no phenotype labels will be inferred from expression alone. |
| [CINEMA-OT, Nature Methods](https://www.nature.com/articles/s41592-023-02040-5) | Analyze response heterogeneity and subpopulations within a perturbation using matched-control, batch-stratified cells. | Deferred until confounder and batch overlap are checked; centroid shifts alone cannot identify individual causal effects. |
| [PertEval-scFM, ICML](https://proceedings.mlr.press/v267/wenteler25a.html) and [27-method benchmark, Nature Methods](https://www.nature.com/articles/s41592-025-02980-0) | Audit evaluation axes: effect size, cell count, target seen status, cell-line transfer, baselines and reference-sensitive versus reference-insensitive metrics. | The current scripts cover several axes; model scoring waits for complete checkpoint receipts. |

## First executable audit and interpretation

For each condition `p`, let `O_p` be its observed centroid and `C` the
same-line control centroid. Systematic alignment is the average over conditions
of `cos(O_p - C, mean_all_perturbed_cells - C)`, matching Systema's shared-shift
definition. Zero-length shifts are excluded from the mean. Within-cell baseline
predictions use the average of **training perturbed cells only**. The alternative
Systema reference is the **equal-condition average of training condition
centroids**, applied identically to prediction and held-out truth. This
distinction matters when condition sizes vary. The script reports Pearson
correlation medians, which are descriptive summaries; Systema's figures often
report the mean across perturbations and three independent split repetitions.

The fixed cross-cell cache only receives descriptive within-line systematic
alignment in this first pass. It receives no within-target baseline: every
non-control target row is test-only in the sealed four-fold protocol. A
cross-cell baseline must average source-line training rows and use each target
line's control reference, rather than averaging target perturbation rows.

Split-half correlations reuse one seeded, batch-stratified half split. Both
halves share a control/reference estimate and therefore do not establish a
model performance ceiling. Count-bin values describe reliability and do not
substitute for controlled downsampling or an E-test. Centroid accuracy uses at
most 500 seeded held-out conditions per dataset for bounded computation and
must report that denominator. For a constant predictor, its mean accuracy over
the same candidate set is mathematically 0.5 (apart from ties); its per-condition
distribution, rather than that mean, is the useful baseline for later model
comparison. These analyses operate on the canonical feature
axis; they cannot be numerically equated to the paper's 10 processed datasets.

The next distribution-level analysis should use the same cell-count cap in
every condition and control, batch-stratified permutations when possible, and
report how many conditions have enough cells. For Norman, classify 0/1/2-seen
double perturbations from the frozen split before evaluating interaction
residuals. For gene programs, validate gene names and gene-set version before
scoring and keep inferred programs distinct from measured CIN labels.
