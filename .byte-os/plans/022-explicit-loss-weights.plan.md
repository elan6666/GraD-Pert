---
id: 022
status: in_progress
wave: 1
depends_on: []
created_at: 2026-08-28T00:40:42+08:00
updated_at: 2026-08-28T03:15:00+08:00
---

# Plan 022 — explicit four-term B2 loss weights

## Goal

Set the self-contained formal GraD-Pert B2 objective to prediction,
condition-consistency, masked-node, and spread weights `1.0`, `0.8`, `0.4`,
and `0.1`, respectively, with fail-closed config validation and receipted
runtime identity.

## Write scope

- `configs/experiments/gradpert_b2/*.yaml`
- `src/gradpert/config/schema.py`
- `src/gradpert/training/step.py`
- `src/gradpert/execution/native.py`
- focused tests and active design/Byte OS documentation

## Non-goals

- Do not alter optimizer, architecture, data, split, controls, graph, systems
  optimizations, evaluation, or artifact policy.
- Do not rewrite or relabel completed results produced with the historical
  effective weights `1.0`, `0.1`, `0.1`, and `0.01`.
- Do not launch training as part of this source/config change.
- Preserve historical pilot configs and their exact legacy weight semantics.

## Acceptance criteria

- All five formal B2 configs explicitly carry the four requested weights.
- Formal config validation requires the exact registered values.
- The step objective and gradient composition use those four direct weights.
- Runtime health evidence records the exact effective weight mapping.
- Legacy `ssl_weight`/`spread_weight` pilot configs retain their old effective
  objective and cannot be mixed with the new formal lineage.
- Focused and full pytest, Ruff, format, mypy, build, and config verification
  pass in environments that provide the required Torch/PyG dependencies.

## Status

- [x] Implement explicit config/schema/runtime wiring.
- [x] Add objective and validation tests.
- [x] Update active design formula.
- [x] Pass full local non-Torch gates and record dependency-limited checks
  honestly: 195 passed, 9 skipped; strict mypy awaits Torch/PyG.
- [x] Review the final diff and remove unrelated API changes.
- [ ] Commit/push/synchronize only before a separately authorized run.
