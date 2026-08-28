---
id: 027
status: in_progress
wave: 5
depends_on: [025]
updated_at: 2026-08-29T02:30:00+08:00
---

# Plan 027 — Queue-scoped publication receipt and Local Graph continuation

## Goal

Prevent a long formal queue from failing between coordinates solely because a
previously verified GitHub connection disappears. Verify the public ref once
immediately before launch, seal and hash-pin that evidence, require every row
to match the same clean commit/tree/remote/ref receipt, and continue only the
unfinished A0/L2/L4 Local Graph coordinates in a new synchronized lineage.

## Write scope

- `src/gradpert/execution/identity.py`, native/nonlearned wiring and CLI
- B2-vNext ablation orchestration and focused tests
- `docs/design/SERVER_EXECUTION.md` and Byte OS status/evidence

## Non-goals

- Do not weaken clean-tree, repository URL, commit, source-tree, or published
  ref equality.
- Do not resume across commits or overwrite the interrupted 8221 evidence.
- Do not rerun completed L1 or L3, and do not start another ablation module.

## Acceptance criteria

- The receipt can only be created after a live bounded `git ls-remote` proves
  the exact public ref.
- Formal rows accept it only when its exact SHA-256, repository, remote URL,
  ref, commit and source-tree hash match the current clean worktree.
- Missing paired arguments, tampered receipts, stale commits and stale trees
  fail before model construction.
- Local and exact-commit server pytest/Ruff/format/mypy/build gates pass.
- A new namespace runs only A0/L2/L4; completed L1/L3 remain immutable.

## Status

- [x] Reproduce the mid-queue `git ls-remote` failure and preserve evidence.
- [x] Implement and locally test the queue-scoped publication receipt.
- [ ] Publish and synchronize one clean successor commit.
- [ ] Launch and monitor only A0/L2/L4 in a new namespace.
