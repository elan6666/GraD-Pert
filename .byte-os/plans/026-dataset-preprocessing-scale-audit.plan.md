---
id: 026
status: completed
wave: 2
depends_on: []
updated_at: 2026-08-28T05:34:27+08:00
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

## Status

- [x] Formal server materialization at `a942114` preserved K562's 5,000-gene
  processed matrix and Norman's 5,045-gene GEARS matrix without a second
  normalize/log/HVG pass.
- [x] RPE1, Jurkat and HepG2 passed raw-integer fail-closed preparation using
  weak-signal filtering, normalize-total 4,000, log1p and Seurat Top-5000.
- [x] All five canonical split hashes, graph receipts and evaluation states
  were independently reloaded and verified.
- [x] Jurkat full-cell-line pre-split HVG512 produced a 2,809-gene exact union
  with topology SHA-256
  `ba22af6e9e9a558533aaae850f619840ea2d717310eb3362a52476c3c1ea9128`.
