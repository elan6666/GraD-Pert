# Initial Jurkat experiment groups

B0 and H1 use physical batch64, accumulation1 and one GPU. The parent is an
exact copy of the completed full-objective capacity configuration. It runs50
epochs, selects best by validation prediction loss, retains last, and uses
metrics_only. This is an engineering batch choice, not a validation winner.

`batch_probes.json` pins the same-source batch32/64 completed receipts. These
are the H3 candidate levels; H3 must be generated from the actual H2 validation
winner with prepare_followup.py, not from an invented selected parent.
H2 and later groups similarly remain gated by actual upstream validation.

B0 and H1 generated manifests verify independent factor changes. Neither is
launch authorization evidence: each exact config/seed/source still needs the
runner capacity/integration preflight. Other datasets and architecture variants
have no implied batch64 capacity. One process per GPU is appropriate for this
profile: measured reserved memory is29691478016 bytes.

All rows retain the existing canonical v1 split and300-control evaluation
manifests. No split or evaluation population is regenerated.
