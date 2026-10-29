# R15 frozen medium submission snapshot

These are the exact metadata-only sealer, registry and orchestration wrapper bytes used by the 2026-10-29 R15 medium/crossed submission. They call the already published D3 diagnostic CLI with the original AA model checkpoint; no model, metric, sampling or numerical kernel is implemented here. The wrapper imports only the generic process scheduling helpers from the earlier controller and supplies its own explicit R15 plan.

The server release a1 failed metadata check-only before creating a diagnostic job because the inherited panel contains three additional metadata fields. Release a2 validates all five original panel fields, passed check-only, and launched job attempt1. Both releases remain immutable. The exact running wrapper files were published as server job artifacts before launch; this Git snapshot was added after launch for permanent source traceability. This later Git commit is not the D3 analysis source and must never be assigned retroactively to a checkpoint or result.

Native source: `aa65a42554d2f971adf0419bc1421e43c5a335d6`. Analysis source: `d3f31221cdcba04f9d179f782cb6f15f3589188c`. The registry binds the original epoch2-last checkpoint, resolved configuration, fixed medium/crossed manifests, allowed4775 axis and unchanged resource policy. The actual live owner is sealed at launch. No baseline is rerun.

Validation: both CLI help/compilation checks; exact inherited panel comparison positive case and seven mutation/missing/extra negative cases; real server check-only exit0; actual native seal completed and medium worker startup verified. These checks are execution evidence, not scientific success.
