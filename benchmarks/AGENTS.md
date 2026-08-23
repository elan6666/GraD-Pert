# Benchmarks AGENTS.md

## Module Purpose

Run externally named learned baselines in isolated environments and translate
their predictions to the truth-free common artifact contract.

## Local Commands

- `python -m pytest -q tests/benchmarks`
- `python -m ruff check benchmarks tests/benchmarks`
- Runner-specific `--preflight-only` and `--smoke` commands.

## Local Architecture Rules

- True upstream names and imports are allowed only inside the corresponding
  isolated runner/environment.
- Freeze exact commit/license/environment and record every changed behavior.
- Runner input contains no truth. Disable validation-time test access.
- Use exact canonical condition/control IDs and retain `[300,G]` predictions.

## Safe Edit Boundaries

Never edit the frozen checkout. Do not import benchmark packages from
`src/gradpert`.

## Verification

Commit/env/config/split/gene/control hashes; no-truth input; tiny artifact smoke.

## Parent Context

Root `AGENTS.md` and `docs/provenance/REFERENCE_ALIGNMENT.md` remain mandatory.

