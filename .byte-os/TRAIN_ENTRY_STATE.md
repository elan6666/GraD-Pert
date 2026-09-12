# Simple train entry

Baseline verified clean/published before editing:
0143e705bf569f068c3809dddbb212d552e40c20.
Worktree /tmp/gradpert-r50-e3; primary user edits and server sources untouched.

Implements `python -m gradpert train` with optional config, single GPU,
config-listed seed, data-root and runtime overrides, and read-only dry-run.
Temporary default is unchanged historical configs/r50/batch1024, not sched512.
Preserves historical Pearson-best; explicit loss-selection configs stay min-loss.
Uses native fit and postfit; additionally reports all validation metrics and
renders curves without altering scientific parameters. Supports R50 metrics_only
only, fails closed for other protocols. See docs/TRAIN_CLI.md for runtime binding.

No CUDA run authorized by this feature request. Existing experiment monitor
grad-pert-r50 paused during implementation goal; restore it after goal ends.
The prior seven single-step jobs exited zero; strict terminal audit and formal
queue publication remain the monitor's next experiment task. Do not call those
formal 50-epoch results or reuse old evidence with this changed source commit.

Verification: 9 entry tests including no-write dry-run, invalid seed/GPU,
receipt mismatch/missing prior, unchanged default selection, unique IDs and
native-fit-before-postfit dispatch. Full regression: 988 passed, 4 honest
environment skips; Ruff/format (387 files), mypy (88 source files), isolated
wheel/sdist build passed. Publication and server dry-run are next; no CUDA run.
