---
id: 025
status: in_progress
wave: 4
depends_on: [024]
updated_at: 2026-08-28T05:34:27+08:00
---

# Plan 025 — Freeze, synchronize, and launch Nadig Jurkat ablations

## Goal

Generate the preregistered self-contained config matrix, verify the complete
system, synchronize one clean public commit to the server, run authoritative
GenePT coverage and one-epoch integration gates, then launch the eligible fixed
10-epoch seed-1 matrix through the existing CLI.

## Acceptance criteria

- Every matrix row is a self-contained YAML and differs from A0 only in the
  declared factor; the full matrix is frozen before any test result is read.
- Dataset/split/seed/control/truth/Top5000 axes are identical across rows.
- Local and server pytest/Ruff/format/mypy/build/config gates pass on one clean
  local/GitHub/server commit.
- GenePT target absence produces a sealed unavailable receipt and launches no
  GenePT row.
- Eligible rows train exactly 10 epochs, use no test truth during fit, evaluate
  once, retain only small receipts/checkpoint, and leave zero persistent PKL.
- Long server runs are handed to the existing scheduled monitor without busy
  polling or duplicate launches.

## Write scope

- `configs/experiments/`, server matrix tooling, tests, README, Byte OS plans,
  reviews, evidence and delivery records

## Status

- [x] Freeze 22 self-contained config rows and exact hashes before results.
- [x] Test that every row differs from A0 only by the declared factor/coupled
  graph-family requirements.
- [x] Add a hash-pinned queue orchestrator that delegates every executable row
  to the shared native `gradpert model pilot` entrypoint.
- [x] Fail closed before any process on source/config/matrix/GenePT receipt
  mismatch; missing GenePT targets create only a skip receipt.
- [x] Publish and synchronize reviewed implementation commit `a942114`.
- [x] Materialize and strictly verify the five data/graph/evaluation receipt
  chains and the full-cell-line Jurkat HVG512-plus-target graph.
- [x] Run exact GenePT `emb_b` coverage. Seventeen modeled perturbation targets
  are missing, so E1/E2/E3/ES are unavailable before model construction.
- [ ] Commit/push/synchronize the reviewed small-receipt mirror, pass the A0
  one-epoch CUDA gate, then launch only the 18 eligible fixed 10-epoch rows.
