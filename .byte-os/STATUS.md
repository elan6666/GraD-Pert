---
schema_version: 1
mode: auto
project_kind: existing_codebase
stage: implementation
current_workflow: byte-auto
next_workflow: byte-auto
review_verdict: pending
hard_blocked: false
updated_at: 2026-09-14T03:22:00+08:00
---

# Current state

The active work is the R50 batch-1024 completion matrix in
`docs/experiments/R50_1024_FINISH_MATRIX.md`. This status is deliberately
short; prior status entries remain in Git history, while immutable run
receipts are the authority for scientific completion.

- Baseline before this change: clean/published `7efd44787c4023c9fde4ce81a2a110b8c69ac05d`.
- Existing batch-1024 reference: one full 50-epoch fit at source
  `bf938adc2465adda9697387882b87f420683297d`, with best and last tested.
  Its `concurrent` companion is one-epoch capacity evidence, **not** an
  independent repeat.
- New `r1024_ref_repeat`: second independent 50-epoch fit is **planned, not
  launched**. Prefer the original clean bf938 source; never overwrite the
  first run. Compare best-to-best, last-to-last and validation trajectories.
- Eleven previously recorded R1024 single-factor rows have terminal receipts.
  Old P1's declared projector width did not reach the model, so its receipt
  must not be interpreted as a valid projector-size ablation.
- New loss-weight/scale/LR-low rows and explicit batch-1024 TxPert override
  are in preparation. No new CUDA or formal 50-epoch fit has started for them.
  TxPert requires its own batch-1024 capacity and one-epoch smoke gates.
- Completed GEARS and Scouter official 50-epoch baselines retain their
  original official batch configurations; they are not relabeled 1024.

Next: finish projector wiring/tests, full gates and review; publish a scoped
main commit; verify clean source and capacity; then launch fresh roots on two
GPUs, targeting four concurrent tasks only when measured resource headroom
permits. Every row must retain 50 validations and test both best and true
epoch-50 last checkpoints. Keep the existing monitor current, quiet for
unchanged healthy training, and delete it only after all authorized rows pass.
