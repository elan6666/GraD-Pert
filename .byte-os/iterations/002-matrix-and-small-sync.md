# Iteration 2 — Executable Matrix and Sealed Small Sync

## Evidence trigger

The server design promised a complete matrix and allowlisted result sync, but
the repository contained only a single official-smoke launcher and no safe
staging implementation.

## Change

- Add a deterministic matrix core and dry-run-first server entrypoint for exact
  15 learned smoke, 15 nonlearned seed-1, and 20 GraD-Pert full tasks.
- Gate full training on all 15 evaluated one-epoch receipts, matching
  checkpoint/commit/config identities, no test Truth during fit, and equal
  canonical-data/split/300-control hashes per dataset across learned models.
- Add default `small_results` and explicit receipt-root staging with extension,
  file-size, total-size, path, symlink, mutation and post-transfer hash gates.

## Measured result

Thirteen new focused tests passed initially; after integration and an explicit
receipt-root case, the full suite reached 131 passed with the same nine honest
optional/current-receipt skips. Thirty config paths and hashes remained exact.

## Regression evidence

Ruff lint/format, strict mypy over 19 non-Torch modules, Byte state validation,
and offline wheel/sdist build passed. No dry-run test created runs, receipts or
staging directories.
