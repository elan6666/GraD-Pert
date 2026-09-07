# Plan039: official external Jurkat integrations

User authorized 2026-09-07: Scouter with GenePT-Seed, official TxPert and GEARS,
maximum100 epochs and validation patience10; preserve official hyperparameters.
Subsequent request: try two experiments per GPU, four including existing A3.

Contract: `docs/experiments/EXTERNAL_JURKAT_100_20260907.md`.

- [x] Read Scouter paper and freeze official source; verify K562/RPE1 recipe.
- [x] Audit official training, best-selection and test/validation boundaries.
- [x] Implement process-only common CLI dispatch and isolated official adapters.
- [x] Explicit standalone configs; no native-model parameter changes.
- [ ] Full tests/review including real official CPU synthetic integration.
- [ ] Clean main publication, isolated server source and exact-commit gates.
- [ ] One-epoch canonical smokes; compare capacity when adding second process.
- [ ] Best-checkpoint/zero-PKL/fairness smoke audit, then fresh full roots.
- [ ] Final common evaluation and small result delivery.

The existing primary worktree's tracking changes are unrelated and preserved.
No active source checkout is edited. The user-paused monitor remains paused.
