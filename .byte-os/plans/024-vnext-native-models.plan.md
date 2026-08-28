---
id: 024
status: complete
wave: 3
depends_on: [023]
updated_at: 2026-08-28T05:34:27+08:00
---

# Plan 024 — B2-vNext native encoder, decoder, and GenePT variants

## Goal

Add native config-selected single- and multi-graph encoders, STRING numerical
weight routes, decoder variants, and explicit GenePT feature modes while using
the existing model/trainer/CLI lifecycle.

## Acceptance criteria

- Native code imports no upstream model package and copies no upstream source.
- Supported encoders are single STRING GATv2, single STRING sparse graph
  Transformer, STRING+GO multi-source sparse graph Transformer, and a clearly
  project-preregistered adaptive GAT fusion route.
- Student and Teacher share the resolved architecture contract and width.
- W0/W1/W2/W3/WS STRING routes are separate, receipted options.
- Decoder D0/D1/D2 and GenePT E0/E1/E2/E3/ES are config-only variants.
- Frozen synthetic golden tests bind source-derived public behavior and every
  trainable route passes gradient/EMA/checkpoint/resume tests.

## Write scope

- `src/gradpert/modeling/`, native execution/training wiring, tests and
  provenance documentation

## Status

- [x] Config-selected native encoder, decoder, and GenePT routes use the shared
  model/trainer/CLI lifecycle.
- [x] Official source-derived synthetic union golden is hash-pinned.
- [x] Training-mode Transformer views preserve independent BatchNorm semantics.
- [x] STRING numerical weights are frozen once on the full retained topology
  and remain invariant across global/local/prediction crops.
- [x] All trainable D2/GenePT routes pass server gradient, EMA, checkpoint and
  resume tests; resident sparse prediction unions are cached and numerically
  equivalent to rebuilds.
- [x] Commit/push/synchronize the exact reviewed source before formal CUDA use
  at `a9421142c086c4fe6b88cd48343a2cc03b1e408a`.
