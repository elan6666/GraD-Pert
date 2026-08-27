---
id: 026
status: in_progress
wave: 2
depends_on: []
updated_at: 2026-08-28T12:20:00+08:00
---

# Plan 026 — Dataset preprocessing scale audit

## Goal

Make the raw-versus-officially-processed expression boundary explicit and
fail-closed for all five registered datasets. Preserve the frozen K562 and
Norman expression matrices, and apply the TxPert normalization/HVG workflow
only to audited raw-count inputs.

## Non-goals

- Do not rewrite or relabel historical canonical H5AD/results.
- Do not change the frozen split, control manifests, or evaluation formulas.
- Do not introduce heuristic auto-transforms based only on observed ranges.

## Acceptance criteria

- Registry validation binds source semantics, input expression state, and the
  only permitted scale action.
- RPE1/Jurkat/HepG2 reject non-integer, negative, or non-finite raw matrices
  before filter/normalization.
- K562 preserves the pinned processed 5,000-gene matrix.
- Norman preserves the pinned GEARS processed 5,045-gene `X` exactly, while
  canonicalizing metadata and removing only stale response-derived caches.
- Tests prove Norman expression equality and raw transformed-input rejection.
- The corrected Jurkat vNext graph receipt proves full-cell-line pre-split
  normalize-4000/log1p/Seurat-HVG512 plus all perturbation targets.

## Write scope

- `registry/datasets/`, `src/gradpert/data/`, `tests/data/`
- data/design documentation and Byte OS evidence
