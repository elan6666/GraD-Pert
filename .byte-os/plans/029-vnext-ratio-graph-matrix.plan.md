---
id: 029
status: in_progress
wave: 1
depends_on: []
updated_at: 2026-08-29T17:10:00+08:00
---

# Plan 029 — ratio-based graph-scale and local-view matrix

## Objective

Make the successor A0 and H/L matrices executable exactly as preregistered in
`docs/experiments/VNEXT_GRAPH_SCALE_AND_LOCAL_ABLATIONS.md`.

## Write scope

- `src/gradpert/config/native.py`, graph-view construction, training wiring,
  receipts and matrix validation needed for ratio semantics.
- Nadig Jurkat ablation generator/configs/matrix and focused tests.
- vNext design/experiment Markdown and Byte OS state.

## Non-goals

- No performance optimization in this plan.
- No training or test evaluation.
- No modification or reuse of superseded server run evidence.
- No asymmetric Teacher/Student graph axes.

## Acceptance criteria

- A0 is HVG512+targets, RingInduced, local node ratio 0.50, 8 locals, anchor
  mask ratio 0.
- Effective node budget is `floor(actual_runtime_node_count * p / q)` using
  exact integer fractions and is receipted with quotient, remainder, requested
  and realized coverage.
- Effective mask count is `local_count * mask_ratio`; non-integral formal
  products fail closed.
- H0/H1/H2/H3 change only graph HVG count 512/1024/2048/5000 while local ratio
  remains 0.50.
- L1/L2/L3/L4/L5 change only builder, local count, local ratio, mask ratio
  0.50, or mask ratio 0.25 respectively.
- Every row is generated from A0 and a semantic diff allowlist rejects an
  undeclared scientific difference.
- Old fixed-count/fixed-budget A/L coordinates cannot satisfy the new matrix.

## Verification

- Focused config, graph-view, receipt and matrix tests.
- Full pytest, Ruff, format, strict mypy and isolated build before publication.
