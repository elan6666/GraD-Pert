# Plan 020 — B2/B3 exact 10-epoch metrics-only comparison

## Objective

Run fresh Nadig Jurkat B2 systems-only and B3 combined coordinates for exactly
10 epochs on separate server GPUs. Preserve each variant's sealed one-epoch
graph, all seven semantics-preserving systems optimizations, seed 1, batch size
256, 16,384 prototypes, allocator, split, ordered 300 controls, GPU class,
validation-only checkpoint selection, and project-wide zero-persistent-PKL
policy. The user added B3 as a concurrent comparison after the original B2
plan was created.

## Boundaries

- Never overwrite or resume the completed one-epoch B2 or B3 coordinate.
- Use a new self-contained config, source commit, immutable contract, run ID,
  and run namespace.
- Add an explicit `fixed_epoch_pilot` lifecycle; do not weaken the one-epoch
  smoke or 100-epoch formal-run contracts.
- Validate every epoch and select the best validation checkpoint, but do not
  early-stop before epoch 10.
- Open canonical test truth only once after training and best-checkpoint load.
- Persist no PKL; retain only the selected checkpoint and small reconstruction,
  identity, timing, metric, and fairness evidence.
- Keep goal/continuous execution inactive while either long GPU run is active;
  monitor at the user-selected 30-minute interval without busy-polling.
- Compare speed and record all three metrics without claiming effect
  equivalence. Record that the two GPUs share host CPU, RAM, and storage.

## Acceptance

- [x] New config is explicit B2/full-graph/all-seven/seed-1/10-epoch/
      `metrics_only` and passes targeted regression tests.
- [x] B3 uses the sealed reduced graph plus all seven systems groups with an
      explicit fixed-10-epoch, `metrics_only` contract.
- [x] Exact clean local/GitHub/server source identity and full gates pass before
      launch.
- [x] Both fresh runs complete exactly 10 epochs and 5,820 optimizer steps.
- [x] Each has exactly 10 validation receipts; test truth is absent during fit
      and test evaluation occurs exactly once from the selected checkpoint.
- [x] Canonical/split/ordered-control/truth and graph identities match the
      corresponding sealed one-epoch coordinate.
- [x] Each successful run root contains zero persistent PKL and only the
      selected `best.pt` remains.
- [x] A 71-check strict verifier passes and reviewed small evidence records
      B3's 1.383x speedup and the non-decisional metric deltas.
- [ ] README/Byte OS status, final
      gates, public push, and clean synchronization are complete.
