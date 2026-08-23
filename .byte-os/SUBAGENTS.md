# Subagent Strategy

Subagent mode: on

## Current read-only exploration

| Task | Scope | Write access | Status |
|---|---|---|---|
| `design_audit` | All GraD-Pert design and study Markdown under `TxPert/` | none | completed |
| `txpert_audit` | Frozen TxPert official code and local paper | none | completed |
| `gears_data_audit` | GEARS adapters and five-dataset integration/readiness | none | completed |

Completed handoffs:

- `.byte-os/subagents/exploration-design-audit.md`
- `.byte-os/subagents/exploration-txpert-audit.md`
- `.byte-os/subagents/exploration-gears-data-audit.md`

## Boundaries

- Exploration agents do not edit project or Byte OS files.
- The primary agent independently reads the selected design and reference files, resolves conflicts and owns all product decisions.
- Implementation subagents may be assigned only after plan files define disjoint directories, acceptance criteria and verification commands.
- Review subagents run only after implementation evidence exists.

## Required handoff

Every handoff must include Scope, Allowed files or directories, Files inspected, Files changed, Verification run, Result, Risks and Handoff.
