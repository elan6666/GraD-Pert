# Modeling AGENTS.md

## Module Purpose

Standalone GraD-Pert graph views, encoder, consistency objectives, teacher
state, expression predictor, and B2 training mechanics.

## Local Commands

- `python -m pytest -q tests/graphs tests/modeling tests/training`
- `python -m ruff check src/gradpert/modeling src/gradpert/graphs tests/modeling tests/graphs`
- `python -m mypy src/gradpert/modeling src/gradpert/graphs`

## Local Architecture Rules

- Follow `docs/design/GRADPERT_V1.md`; no alternative route or ablation switch.
- Native names only; no upstream model import/call/class name.
- Optimizer step precedes Teacher EMA; center updates use detached teacher logits.
- Deterministic prediction view never uses stochastic augmentation/head.

## Safe Edit Boundaries

Edit native `src/gradpert` and owned tests only. Frozen upstream code is evidence.

## Verification

Shapes, 18 pairs, masks, gradient ownership, EMA/center order, resume equivalence.

## Parent Context

Root `AGENTS.md` and `.byte-os/ENGINEERING_RULES.md` remain mandatory.

