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

Current stage: metadata audit completed. Replogle RPE1 and Nadig Jurkat/HepG2
have guide identifiers; Norman has a count layer. Other resources lack those
fields in the audited canonical H5AD. Two training GPUs are fully occupied;
all analysis work uses low-priority CPU and bounded memory.

Next: implement and test the observed-data analysis, publish source, run it on
the server without touching active training, inspect receipts, then write and
verify the Chinese report. Earlier results are in
`docs/experiments/FIVE_DATASET_EDA.md`, `SIX_DATASET_LANDSCAPE_RESULTS.md`,
and `LITERATURE_GUIDED_DATA_AUDIT_RESULTS.md`.
