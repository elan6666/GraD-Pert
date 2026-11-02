# Native RI transfer to other datasets, 300 updates

User authorization: 2026-11-02, native model only. Retain historical R3-RI Jurkat baseline; no repeat or external fits. This is a separate short research protocol, not R50 or a new claim of two full epochs.

## Update budget

`v2_updates_300` means exactly 300 successful optimizer updates, final-checkpoint selection, no early stopping or test tuning. LR, teacher EMA and temperature clocks use the 300-update horizon. `max_epochs` is the smallest data-epoch ceiling containing that budget; the final epoch may be partial and is labelled as such in history. Resume replays an interrupted data epoch from its committed checkpoint. Global320, micro80, world2, accumulation2, seed1; unchanged RI architecture/losses/precision.

All canonical TRAIN rows participate in the deterministic epoch sampler. No additional cell cap is applied. K562/RPE1/HepG2 use their original 5000 expression axes; Norman uses5045. Preserve original normalization, canonical partition, GEARS-representability intersection, controls and graph identities. Source/current data differ from historical RI where provenance says so; label cross-dataset/version comparisons descriptive.

Test condition identities determine the forbidden expression columns; test numerical values are never read during fitting or preparation. Enable restricted backed reads before any numerical TRAIN read (including control caches), then keep the existing Student/Teacher/query/target exclusion. Do not reuse Jurkat225: counts from current identity audit are K562117, RPE1238, HepG2209, Norman20. Forbidden values are inaccessible; gene identity and allowed priors remain.

## Frozen fit scope

Four new within-cell arms: replogle_k562_essential, replogle_rpe1_essential, nadig_hepg2, norman. Last-only frozen validation18/MSE and the shared medium/crossed representation template follow actual fit acceptance. Preserve NA/support counts. No test-based selection. Zero PKL throughout successful run root. Existing per-GPU limits, foreign-job identity gates, source/publication/config/data hashes and allocator contract remain unchanged.

Cross-cell remains part of the user's requested batch, but requires a new row-level adapter and fixed-axis GenePT/graph audit first. Never feed its folds into the condition-only within-cell interface. No cross-cell fit is approved by this implementation's within-cell preflight. Retain the four held-out-line scopes as pending, not completed.

## Required checks

Real complete native updates/restoration and 300-control inference for each exact config before fit. Each formal fit is source-clean/published and guarded by the shared GPU policy. Completion requires exact300 counters, source/data/config/checkpoint hashes, complete history, no test fit access and zero PKL. Reporting and independent verification remain separate stages.
