---
schema_version: 1
plan_id: 035-vnext-decoder-fusion-width
status: complete
owner: repository-owner
created_at: 2026-09-02
---

# Plan 035 — Decoder fusion and perturbation-width factorial

## Goal

Prioritize a fresh Nadig Jurkat decoder module that independently measures
direct concatenation versus TriShift-aligned Transformer interaction at
64-wide and 256-wide perturbation states.

## Rows

- D3: `concat(b64,p64)`.
- D4: `concat(b64,p64,T([b64,p64]))`.
- D5: `concat(b64,p256)`.
- D6: `concat(b64,p256,T([b64,Wp(p256)]))`.

The Transformer has exactly two ordered tokens, one layer, four heads,
64-wide token dimension, 256-wide GELU FFN, pre-norm, dropout zero, no token or
position embeddings and concat readout. Only its D6 perturbation token uses a
learned 256-to-64 projection. Raw `p256` remains in the decoder input.

## Evidence boundaries

- Preserve completed E and old D1/D2 receipts without relabeling.
- Seal interrupted L1/L2 and do not resume their run roots; leave remaining
  L/M rows unstarted until the user reprioritizes them.
- Treat D3--D6 as one 2-by-2 factorial. D3/D5 and D4/D6 identify width;
  D3/D4 and D5/D6 identify Transformer interaction.
- A0 is contextual additive evidence, not a single-factor comparator for
  D5/D6.

## Gates

- [x] Implement config-selected model routes and deterministic matrix entries.
- [x] Test shapes, token projection, gradients, Teacher EMA and checkpoint
  restoration at both widths.
- [x] Pass full local pytest, Ruff, format, strict mypy and isolated build.
- [x] Commit/push and synchronize one clean exact source to the server.
- [x] Run bounded training-only capacity gates with no validation/test access.
- [x] Launch fresh formal D3--D6 only after capacity and identity gates pass.
- [x] Validate 5,820 ordered steps, ten validations, one best-checkpoint test,
  exact three metrics, zero persistent PKL and final-only `best.pt` per row.

## Completion

- Exact source `75a2c2b` completed D3--D6 in lineage
  `formal-vnext-decoder-75a2c2b-v2`.
- All four rows passed strict replay of source, config, step, validation,
  best-checkpoint evaluation, metric, canonical identity and zero-PKL gates.
- Neither Transformer interaction nor width 256 improved all three metrics
  consistently. A0 remains the preregistered default; comparisons are
  single-seed descriptive evidence only.
