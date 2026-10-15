# Cap40 full posthoc diagnostics

Outcome: one sealed diagnostic manifest, read-only last-checkpoint analysis for
DP/FN/FG/FB/PO, complete grouped Chinese reports, all figures and underlying
aggregate tables. No training, selection, EMA/center/optimizer updates, or edits
to original runs. Training/method source remains immutable 420d3a1; analysis has
its own published source identity. Baseline 0ee26e71a45bbbceff8007d9e0ce92571178b012
is committed and verified equal to GitHub main before edits.

Scope: all 47,836 cap40 training rows / 1,335 conditions, preserving numeric
expression exclusions and the 1,000-query budget. Fixed per-row control pairing,
query panels, view recipes and orders shared across arms. Full basic output/CLS
statistics; deep diagnostic subset up to eight cells per condition, stratified
by context; 64-condition x four-cell gradient subset. Existing frozen test
metrics remain separate; test-side feature probes use evaluation visibility.
Raw un-downsampled populations are out of scope.

Implementation ownership: main only, scripts/v2/cap40_posthoc*.py,
tests/test_cap40_posthoc*.py, docs/experiments/CAP40_POSTHOC*.md and common
analysis template. No native model or training changes. A bounded Luna helper
may supervise test waits read-only, without a second main poller.

Acceptance: exact unique-row coverage; aligned row/gene/view identities; separate
cell_line/cell_type/batch/condition and control treatment labels; stable streaming
moments/entropy/within-between decomposition and uncertainty; per-view independent
cell counts; no prefix sampling; checkpoint/state/center/RNG immutability;
original-method prediction parity; same manifest hashes across all arms; CPU
tests and guarded server integration before launch. Partial outputs never count
as completion. Original scientific artifacts are read-only; fresh output ID.

Execution: verify ownership and PO terminal evidence; implement/test and publish;
seal manifest on server; measure diagnostic chunk capacity/runtime, then execute
full jobs with approved GPU policy. Longer than one hour uses exact supervisor
handoff/20min monitor and verified return route. Report throughput, completed
rows/conditions and ETA without extrapolating old four-cell timings.

Deliver: common schema v2, each arm's report/receipt and full aggregate CSV index,
comparison report embedding every produced figure; missing/unidentifiable items
explicitly labeled. Per-cell matrices/checkpoints remain server-only. Persist
current run and next action in Byte status; no duplicate monitor.

## Rejected optional transformation (2026-10-15)

Shared manifest sealed: cc208dc56b2eaa45eb0ce4b005729c049f98349860fb6d148576f3fc99fbc6e1,
47836 rows, 1335 conditions, 10532 deep rows, 256 gradient rows, 11793 controls.
First ca65644 DP preflight preserved as FAILED: singleton-cell vs full-group
BF16 forward max tensor difference 0.04460430145263672 exceeded 2e-6.
Do not relax tolerance or adopt cell splitting. Full-group native vs cached
path had passed the preceding strict comparison. Repair only makes optional
chunk rejection explicit; every used path must still pass the original 2e-6
reference/cache comparison. Keep the exact sealed condition groups for all arms.
Record per-tensor errors, no claims about the cause until independently tested.
Fresh preflight/analysis IDs; old source/results untouched.
