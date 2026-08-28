---
verdict: ship
created_at: 2026-08-29T02:58:00+08:00
scope: plan-027-local-review
---

# Queue publication receipt review

## Verdict

Ship to exact-commit server gates and controlled A0/L2/L4 continuation.

## Acceptance review

- The receipt creator performs the existing bounded, noninteractive live
  `git ls-remote` check; it does not trust a tracking ref or copied bundle.
- The destination must be new. Consumption requires a paired path and SHA-256
  and rejects malformed schema, wrong hash, repository/remote/ref drift,
  commit drift, tree drift and dirty worktrees before model construction.
- The ablation launcher validates the receipt before creating a row process,
  records its path/hash in the launch plan, and passes the same frozen pair to
  every native row.
- Direct formal execution without a receipt retains the prior live-check
  behavior, so this is an explicit queue contract rather than a silent global
  relaxation.

## Evidence

- 262 tests passed; eight dependency-only skips.
- Ruff, format, focused strict mypy, diff check and isolated build passed.
- A temporary real Git remote passed CLI receipt creation and offline receipt
  consumption with the exact emitted SHA-256.

## Remaining operational risk

The live publication check still requires working GitHub access once before
launch. That is intentional. The successor queue must be a new namespace and
must not resume or overwrite 8221 evidence; completed L1/L3 remain immutable.
