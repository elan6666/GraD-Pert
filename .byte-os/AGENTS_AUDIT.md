# AGENTS Audit

## Root context

- Status: ready
- Purpose: navigation, hard boundaries, commands, edit safety, server topology.
- Detailed reusable rules moved to `.byte-os/ENGINEERING_RULES.md` to keep the
  always-loaded root file bounded.

## Module context files

- `src/gradpert/modeling/AGENTS.md`: native model/loss/gradient boundaries.
- `src/gradpert/graphs/AGENTS.md`: graph materialization/view invariants.
- `benchmarks/AGENTS.md`: isolated official environments, no fit-time test
  Truth, upstream licenses and adapter-only scope.
- Data/evaluation/execution commands remain centralized in `CODEBASE_MAP.md`
  and the active design documents.

## Coverage

- Scoped commands: active and current in `CODEBASE_MAP.md`.
- Noise paths: ready in `.gitignore`, settings, and root rules.
- LSP/symbol navigation: ready for the implemented Python/YAML stack.
- Subagent boundaries: ready; three read-only exploration handoffs recorded.
- Server and large-artifact boundaries: ready.
- Config matrix/no-global-config boundary: ready.

## Remaining update

- After the 15 learned server smokes: record exact post-fit runtime receipts and
  replace the current external-execution blocker with result pointers.
- After delivery: record the final catalog and artifact pointers.

## Freshness

- Last reviewed: 2026-08-24
- Next review: after first server smoke or 2026-11-24
- Owner/DRI: repository owner
