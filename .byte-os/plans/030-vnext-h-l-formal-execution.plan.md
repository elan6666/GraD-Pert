---
id: 030
status: pending
wave: 3
depends_on: [028]
updated_at: 2026-08-29T17:10:00+08:00
---

# Plan 030 — formal H and L module execution

## Objective

After the reviewed exact-effect performance commit is synchronized, materialize
the registered graph axes and complete the preregistered H/L 10-epoch runs.

## Non-goals

- No other ablation module.
- No 100/200-epoch work, extra seeds, test-guided rows or old coordinate rerun.
- No reuse of `42e`, `8221`, or `276d` evidence as a successor coordinate.

## Acceptance criteria

- Pre-split HVG512/1024/2048/5000 axis and topology receipts are exact and
  reviewed before launch.
- Existing graph destinations are reusable only when all current source,
  protocol, fit-scope, preprocessing, registry and requested-H identities
  match, not merely the requested integer H.
- A single cross-H production receipt proves common materialization lineage,
  nested selected-HVG sets `512 ⊂ 1024 ⊂ 2048 ⊂ 5000`, target
  coverage and each ordered union/topology hash before training.
- One shared H0/A0 plus H1/H2/H3 and L1/L2/L3/L4/L5 run exactly 10 epochs,
  5,820 ordered steps, 10 validations, no early stopping, seed 1.
- One evaluation from `best.pt`, exact three metrics, identical canonical split
  and ordered control/truth identities, zero persistent PKL and only best.pt.
- Results are reported descriptively without equivalence or test-tuning claims.
- Standalone experiment Markdown, README, Byte OS, review and delivery evidence
  are updated from reviewed small artifacts only.
