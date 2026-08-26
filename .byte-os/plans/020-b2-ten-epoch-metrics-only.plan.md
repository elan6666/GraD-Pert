# Plan 020 — B2 exact 10-epoch metrics-only run

## Objective

Run one fresh Nadig Jurkat B2 systems-only coordinate for exactly 10 epochs on
the server. Preserve the canonical full graph, all seven semantics-preserving
systems optimizations, seed 1, batch size 256, 16,384 prototypes, allocator,
split, ordered 300 controls, GPU class, validation-only checkpoint selection,
and project-wide zero-persistent-PKL policy.

## Boundaries

- Never overwrite or resume the completed one-epoch B2 coordinate.
- Use a new self-contained config, source commit, immutable contract, run ID,
  and run namespace.
- Add an explicit `fixed_epoch_pilot` lifecycle; do not weaken the one-epoch
  smoke or 100-epoch formal-run contracts.
- Validate every epoch and select the best validation checkpoint, but do not
  early-stop before epoch 10.
- Open canonical test truth only once after training and best-checkpoint load.
- Persist no PKL; retain only the selected checkpoint and small reconstruction,
  identity, timing, metric, and fairness evidence.
- Keep goal/continuous execution inactive while the long GPU run is active;
  monitor once every 10 minutes without busy-polling.

## Acceptance

- [x] New config is explicit B2/full-graph/all-seven/seed-1/10-epoch/
      `metrics_only` and passes targeted regression tests.
- [ ] Exact clean local/GitHub/server source identity and full gates pass before
      launch.
- [ ] Fresh run completes exactly 10 epochs and 5,820 optimizer steps.
- [ ] Exactly 10 validation receipts exist; test truth is absent during fit and
      test evaluation occurs exactly once from the selected checkpoint.
- [ ] Canonical/split/ordered-control/truth and graph identities match the
      sealed B2 one-epoch coordinate.
- [ ] The whole successful run root contains zero persistent PKL and only the
      selected checkpoint remains.
- [ ] Strict validation, reviewed small evidence, README/Byte OS status, final
      gates, public push, and clean synchronization are complete.
