# Plan 019 — B0 metrics-only timing rerun

## Objective

Rerun the Nadig Jurkat B0 performance baseline exactly once without persistent
PKL output so its one-epoch training wall can be compared with B1, B2, and B3
under the same artifact policy and timing implementation.

The historical c240 B0 remains immutable evidence. This plan creates a new
performance coordinate in a fresh namespace; it does not overwrite, relabel,
or reuse the historical run.

## Write scope

- `configs/pilots/perf_b0_metrics_only/gradpert_b2/nadig_jurkat.yaml`
- `tests/pilots/test_graph_axis.py`
- `.byte-os/**` planning, status, review, and reviewed small evidence
- a fresh server contract, run root, launch log, and small validation receipts

## Frozen execution contract

- Dataset: Nadig Jurkat; model route: native `gradpert_b2`; seed: 1.
- Exactly one epoch and the same canonical split, ordered 300-control/truth
  identities, batch size 256, 16,384 prototypes, expandable allocator, and GPU
  class used by B1--B3.
- Canonical full graph axis: expected 6,506 nodes and 222,654 nonself edges.
- All seven systems optimizations disabled, including their helper flags.
- `result_mode: metrics_only`; a successful whole run root must contain zero
  `*.pkl` files and retain only the selected checkpoint.
- Primary speed measure: monotonic `one_epoch_training_wall_ms`; recompute
  full-epoch steps/s and cells/s from that wall.
- Record the three prediction metrics as non-decisional one-epoch evidence.

## Non-goals

- Do not delete, edit, or rerun the historical c240 B0 coordinate.
- Do not enable B1 graph reduction or any B2 systems optimization.
- Do not run more than one epoch or claim predictive-effect equivalence.
- Do not place checkpoints, H5AD, PKL, matrices, or large logs in Git.

## Acceptance criteria

- [x] Explicit self-contained B0 rerun config passes a regression test proving
      full graph, disabled systems, one epoch, batch 256, 16,384 prototypes,
      and `metrics_only`.
- [ ] Local and exact synchronized server pytest/Ruff/format/mypy/build gates
      pass before execution.
- [ ] The fresh run completes exactly 582 optimizer steps, uses no canonical
      test truth during fit, evaluates test exactly once, and retains a
      hash-pinned selected checkpoint.
- [ ] Exact canonical, split, ordered control/truth, expression-axis, and graph
      identities match the comparison contract.
- [ ] The completed run root contains zero persistent PKL files.
- [ ] A strict verifier seals the B0 rerun and a rebuilt B0--B3 comparison uses
      actual full-epoch wall time.
- [ ] README, Byte OS state, review, and reviewed small evidence are updated,
      committed, publicly pushed, and synchronized cleanly.

## Verification order

1. Targeted config/pilot and artifact-policy tests.
2. Full local pytest, Ruff, format, mypy where dependencies exist, and build.
3. Commit, public push, exact clean server synchronization, then the same full
   server gates including strict mypy.
4. Hash-pinned dry contract and fresh one-epoch server execution.
5. Strict identity/lifecycle/artifact/timing validation and zero-PKL scan.
6. Rebuild comparison, review the result, stage only reviewed small evidence,
   run final gates, commit, push, and synchronize.
