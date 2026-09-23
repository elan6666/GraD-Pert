# Dataset-only perturbation analysis

Overall outcome: complete feasible observed-data analyses in
`docs/experiments/PERTURBATION_ANALYSIS_ATLAS.md` on the five canonical
within-cell datasets and separate fixed-axis cross-cell cache, then publish a
Chinese report combining earlier EDA/landscape/Systema results and new audited
measurements. No model predictions, training or checkpoint evaluation.

Acceptance: code and meaningful tests pass; local/GitHub/server analysis source
is an identical clean commit; each server result has a new run ID and immutable
receipt; raw H5AD and per-cell matrices stay on `/data/yilangliu`; the report
distinguishes measured results, descriptive inferences and unavailable analyses.

Current stage: observed-data analysis and the Chinese report are complete.
The effect atlas source is `f7d58a27bc877d5904ba5500cff446aea187e03b`
with completed server receipt `dataset-effect-atlas-f7d58a2`. The matched-batch
distribution source is `ecdf8f14a11fde63f6f4cdaecda7c9456c3e96ee` with
completed server receipt `dataset-distribution-ecdf8f1`. Both ran on low-priority
CPU without touching active training or transferring raw scientific matrices.
The report is `docs/experiments/PERTURBATION_DATASET_ANALYSIS_ZH.md`. The
five-dataset EDA and six-resource landscape were rerun on the current canonical
data as `five-eda-current-ecdf8f1` and `six-landscape-current-ecdf8f1` because
the historical `data-vnext-a942114` H5AD files have different hashes. The
current EDA, landscape, Systema-inspired, effect-atlas and distribution audits
now agree on each of the five canonical H5AD hashes. In the canonical
H5AD, guide IDs exist for Replogle RPE1 and Nadig Jurkat/HepG2, while a count
layer exists only for Norman. Same-batch matched distribution tests have low
coverage in four single-gene datasets and are not generalized to all conditions.

Next: keep this report as the frozen data-only interpretation. Any later
model-based analysis, raw-count resampling or broader batch-aware distribution
test needs its own protocol and receipt.
