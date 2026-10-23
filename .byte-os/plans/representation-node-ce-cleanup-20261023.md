# Remove unused SSL2 node CE work

Pre-change source: 32216aea23c31491e393e3ea0d4caba00f80f7eb, clean and published before edits. Main owns only src/gradpert/training/v2/objective.py for this cleanup. Parallel path-audit work owns separate scripts/tests. Historical training checkouts remain unchanged.

In the reduction_weights branch of cell_loss, the initial unweighted node cross-entropy result is overwritten by weighted per-row terms. Its softmax/log_softmax are pure and consume no RNG or target/center state. Compute the unweighted node loss only in the non-weighted branch; preserve ibot_population rescaling there. Keep teacher target collection exactly once and preserve all branch precedence.

Verification uses existing unified/DDP/population and checkpoint tests, plus frozen pre-change/new method CPU forward, loss/gradient/pending-state parity on weighted, ordinary and population paths. Report operator-call reduction separately from wall-clock speed. No whole-step training gain claim without server ABBA measurement. Commit and push the scoped changes before future formal use; this change does not alter checkpoint or active-run identities.
