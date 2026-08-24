# Iteration 3 — Test-Truth Lifecycle and Source Identity

## Evidence trigger

Review showed that official fit inputs excluded canonical test rows, but both
runner functions constructed a canonical test reader before calling the fit
API. Development worktrees also replaced the declared snapshot commit with Git
HEAD without first requiring them to match.

## Change

- GEARS and TxPert now complete official fit, checkpoint hashing, and the
  one-epoch training receipt before a canonical test reader exists.
- Add a static lifecycle contract for both official runners.
- Reject development execution when `--development-commit` differs from the
  actual Git HEAD of a supplied worktree.
- Remove stale foundation-era harness/AGENTS/OKR statements.

## Measured result

The lifecycle tests prove no `CanonicalEvaluationData` construction appears
before official fit and that checkpoint plus training receipt precede the test
reader. The source-identity regression rejects declared commit drift.

## Regression evidence

Focused official/execution/server tests passed. The full local suite reached
137 passed with nine unchanged optional/current-receipt skips; Ruff remained
green. Server Torch/PyG regression is still required after synchronization.

The same iteration also closed the notebook handoff: the final builder requires
all 45 formal coordinates and fairness/metric gates from an explicit source
spec, and the re-executed notebook uses only that strict catalog loader.
