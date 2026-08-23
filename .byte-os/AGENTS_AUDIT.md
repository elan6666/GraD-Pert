# AGENTS Audit

## Root context

- Status: ready
- Purpose: navigation, hard boundaries, commands, edit safety, server topology.
- Detailed reusable rules moved to `.byte-os/ENGINEERING_RULES.md` to keep the
  always-loaded root file bounded.

## Module context files

- None yet because active implementation modules do not exist.
- Foundation must add local `AGENTS.md` only for:
  - `src/gradpert/modeling/`: model/loss/graph-gradient boundaries.
  - `benchmarks/`: isolated environments, no truth access, upstream licenses.
- Data/evaluation command detail is already covered by `CODEBASE_MAP.md`; add a
  local file only if implementation makes navigation materially cheaper.

## Coverage

- Scoped commands: ready in `CODEBASE_MAP.md`, activation gated by foundation.
- Noise paths: ready in `.gitignore`, settings, and root rules.
- LSP/symbol navigation: ready for planned Python/YAML stack.
- Subagent boundaries: ready; three read-only exploration handoffs recorded.
- Server and large-artifact boundaries: ready.
- Config matrix/no-global-config boundary: ready.

## Proposed updates from this session

- After plan 001: add actual package/module commands and modeling/benchmark
  local context files.
- After first server smoke: replace provisional environment commands with exact
  lock/install/launch commands and GPU evidence.
- After delivery: remove stale planned paths and record final artifact pointers.

## Freshness

- Last reviewed: 2026-08-24
- Next review: after first server smoke or 2026-11-24
- Owner/DRI: repository owner

