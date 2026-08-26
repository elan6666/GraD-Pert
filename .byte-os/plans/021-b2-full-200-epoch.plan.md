---
status: in_progress
created_at: 2026-08-26T21:01:14+08:00
updated_at: 2026-08-26T21:01:14+08:00
---

# Plan 021 — default B2 full run at 200 epochs with patience 10

## Goal

Make full-graph, all-seven-systems B2 the default native GraD-Pert execution
profile, restore the user-locked full-training ceiling to 200 epochs with
validation-only early stopping patience 10, and launch four fresh formal Nadig
Jurkat runs for registered seeds 1--4 after an exact-commit one-epoch
integration gate.

## Write scope

- `configs/experiments/gradpert_b2/*.yaml`
- `src/gradpert/config/schema.py`
- `src/gradpert/training/trainer.py`
- `src/gradpert/execution/native.py`
- focused config, matrix, and trainer tests
- active README/design/Byte OS policy and evidence files

## Non-goals

- Do not run B3 or the reduced graph.
- Do not rerun or overwrite any completed B0/B1/B2/B3 pilot.
- Do not launch the other four datasets. The user-requested "4 split" execution
  means registered run seeds 1--4 on the one frozen canonical split; do not
  invent four new data splits.
- Do not change data, split, controls, optimizer, loss, RNG ownership, update
  order, evaluation, batch size, prototypes, allocator, or artifact policy.
- Do not persist PKL output.

## Acceptance criteria

- All five native configs remain self-contained and explicitly request the
  canonical full graph plus all seven semantics-preserving systems groups.
- Native full configs require `max_epochs=200`, early stopping enabled,
  `patience=10`, `min_delta=0`, and the frozen validation metric.
- Fixed 10-epoch pilot support remains valid and unchanged.
- Trainer accepts only exact 10-epoch pilots or 200-epoch full budgets.
- Focused and full pytest, Ruff, format, strict mypy, build, config verification,
  and clean-tree gates pass locally/server as applicable.
- Local, public GitHub, and server source are the same clean commit before
  execution.
- A fresh exact-commit one-epoch Nadig Jurkat B2 gate passes before the full
  task starts.
- Four full tasks run with seeds 1--4, batch 256, 16,384 prototypes, expandable
  allocator, `metrics_only`, zero persistent PKL, maximum 200 epochs, and
  validation-only patience 10.
- Goal mode remains inactive during long server training and monitoring runs
  once per hour without busy polling.

## Verification

1. `python -m pytest -q tests/config/test_loader.py tests/execution/test_matrix.py tests/training/test_step_and_resume.py`
2. `python -m pytest -q`
3. `python -m ruff check .`
4. `python -m ruff format --check .`
5. `python -m mypy src`
6. `python -m build`
7. `gradpert config verify --all`
8. Repeat full gates on the exact clean server commit.
9. Validate the one-epoch gate before launching the new full namespace.

## Status

- [x] User selected B2 as the default and requested 200 epochs/patience 10.
- [x] Implement policy, configs, tests, and documentation.
- [ ] Pass local and exact-commit server gates.
- [ ] Commit, publicly push, and synchronize clean source.
- [ ] Pass fresh one-epoch Nadig Jurkat B2 integration gate.
- [ ] Launch the four formal full runs and install hourly monitoring.
- [ ] Validate and deliver the completed runs.
