# v2 experiment package

The matrix in `groups.json` records factor definitions and validation-only group
selection dependencies. It is not a runnable queue until measured execution
profiles, frozen parent selections and launch preflight are recorded.

`PYTHONPATH=src python scripts/v2/generate_group.py --parent PARENT.yaml --group H1 --output NEW_DIRECTORY`

The parent must be an explicit standalone v2 config. The generator writes every
row as `NEW_DIRECTORY/ROW/gradpert_v2/DATASET.yaml`, preserving the required config
identity and all unchanged fields. `manifest.json` pins parent, generator and row
hashes. Existing directories are rejected; selecting a later parent creates a new
group directory. Within a group no row changes the parent of another row.

Supported training groups: B0, H1, H2, P1, L0, L1, L2, A1–A4, S1, S2. H1 changes
both peak and floor LR with the fixed project ratio. L1/L2 enumerate all eight
component combinations; all-off uses both external SSL weights zero. S1 changes
condition CE weighting only. S2 requires a mixed-condition parent and separates
sampling from the SSL2 KoLeo switch. Generated configurations still require
variant-specific resource preflight; capacity for one architecture is not proof
that another architecture fits.

H3 requires the pending measured batch profile selection. G1 and D1 are evaluation
protocols and are not silently generated as ordinary training configurations.
Neither a generated config nor a passing schema check constitutes an experiment
result or permission to bypass source/data/resource checks.

`PYTHONPATH=src python scripts/v2/capacity_report.py --probe RECEIPT.json CONFIG.yaml --output NEW_DIRECTORY`

Repeat `--probe` for multiple measurements. This writes capacity JSON/CSV from
completed, hash-matched 128-update probes with checkpoint continuation and
300-control inference. It reports observed points, not an inferred maximum or a
selected scientific batch. Preserve per-run source versions and KoLeo neighborhood
sizes when comparing physical, accumulated and distributed batches.

`PYTHONPATH=src python scripts/v2/collect_results.py --run SERVER_RUN_ROOT --output NEW_DIRECTORY`

Repeat `--run` to collect multiple runs. Run this on the server so selected
checkpoint hashes can be verified without transferring weights. Output is small
JSON/CSV suitable for dry-run-reviewed retrieval. Both best/last roles are kept,
even for identical checkpoint content. Missing manifests/checkpoints/tests and
invalid evidence remain explicit. Completion requires fifty committed validation
epochs, consistent minimum-loss selection, both test identities, checkpoint hashes,
and the zero-PKL postcondition. The table records training and evaluation SHAs
separately; it does not select hyperparameters from test scores.
