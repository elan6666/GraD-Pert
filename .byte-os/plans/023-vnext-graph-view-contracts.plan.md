---
id: 023
status: in_progress
wave: 2
depends_on: [022]
updated_at: 2026-08-28T04:05:00+08:00
---

# Plan 023 — B2-vNext graph, feature, and view contracts

## Goal

Introduce one strict, receipted native architecture option object used by the
existing CLI. Materialize the TxPert-aligned full-cell-line pre-split
HVG512-plus-target graph axis,
support ordered STRING-only and STRING+GO topologies, implement deterministic
Local-Fanout views, and make local anchor masking configurable without changing
the frozen Top5000 expression/output/evaluation axis.

## Acceptance criteria

- Unsupported option combinations fail before model construction.
- HVG512 ranking and graph rebuild are frozen and hash-receipted.
- GenePT coverage preflight removes only missing non-target graph genes and
  fails the entire GenePT family when any perturbation target is missing.
- Fanout-256 is the default, uses `[20,10,5,5]`, retains all anchors, and does
  not complete induced edges.
- Global node masking remains active while local anchor masking defaults to 0.
- Unit/contract tests cover topology identity, budgets, masks, deterministic
  replay, and config serialization.

## Write scope

- `src/gradpert/config/`, `src/gradpert/graphs/`, `src/gradpert/modeling/`
- native execution wiring, configs, tests, design and Byte OS evidence

## Status

- [x] Strict config, HVG512-plus-target, GenePT filtering, and graph manifest
  contracts implemented.
- [x] Local and real server Torch/PyG/anndata tests cover view budgets,
  deterministic replay, mask policy, exact axis ordering and fail-closed target
  coverage.
- [x] Real development Jurkat materialization proved 512 direct HVGs and the
  exact target union on the full weak-signal-filtered pre-split cell line.
- [ ] Reissue and synchronize the corrected formal exact-commit receipt chain.
