# R1024 loss-selection preparation (in progress)

Implementation checkpoint: validation-only EvaluationStateLayout uses separate
validation_evaluation_v1 root, slices train/val/control before reading expression,
uses val-only DE masks and train-only Systema reference. Native loss-selection
runs load it (preparation still required on server), stream all three metrics,
write per-epoch records with run/source identity and print four values.
Curves now generate PNG/PDF plus validation_curves.csv and hash manifest after
fit. Plot smoke test passed, canonical three-metric parity fixture passed.
Full regression and isolated build are being rerun before publication.
Remaining safety review includes concurrent queue fatal-start race, full L3
reachability, original reproduction provenance, and server state preparation.

DEFAULT PLOTTING REQUIREMENT (latest user): each experiment automatically emits
training-loss curves and validation curves from persisted CSV/JSON. Training
plot includes total and separately labelled component losses when available;
validation plot has loss and three Pearson metrics in separate panels (do not
mix incompatible scales). Preserve raw values, epoch/global_step, run/source
identity, unavailable gaps/reasons. Save PNG and vector PDF as small artifacts,
plus plot provenance. Generate after training before terminal completion;
missing/failed plots must be explicit, never claim plot delivery from CSV alone.

LATEST METRIC REQUIREMENT: every validation epoch must compute and persist all
three canonical Pearson metrics (TxPert delta, TriShift delta, Systema) plus
prediction loss. Store per-epoch JSON and epoch-series CSV, including finite
condition counts/unavailable reasons and reference hashes. Console summary must
show all four. Existing implementation has only TxPert+loss; do not claim full
completion yet. Reuse canonical metric formulas with validation-only DE masks
and train-only Systema reference; audit evaluation state boundary before use.
Do not load test truth or use a test-derived reference during validation.

LATEST USER OVERRIDE: all future rows must run exactly 50 epochs, no early
stopping. Select best by minimum validation prediction loss, report Pearson
simultaneously, test best and true epoch-50 last. Seven new configs now disable
early stopping. Earlier early-stop requirements below are superseded; remove
premature-termination acceptance from launch validation before publication.

Published baseline verified main: 20a9f80847dc0fe0d219ca8ba39f0850e2efb485.
Work only in /tmp/gradpert-r50-e3; primary checkout has unrelated edits.
Monitor grad-pert-r50 is PAUSED, verified by update tool. Active preparation
goal ends before CUDA launch. No new CUDA launched.

Current four S1/S2/U1/T1 completed 50 epochs, 10800 steps, 50 validations,
automatic best/last tests, all output hashes verified and whole-root PKL zero.
Server source remains immutable source-r1024-20a9f80-v1. GPUs were idle.

Uncommitted implementation: condition-macro mean-expression MSE over all
expression genes, float64 means, validation-only 300 controls, no artificial
cell pairing. Report Pearson simultaneously; min loss selects best and stops
after ten nonimprovements. Old Pearson/max remains default. Postfit supports
early-stop terminal epoch but verifies reason and patience; true last required.
Tests added for min/max/ties/state roundtrip. Full pytest currently running.

Remaining before delivery:
- Test actual validation MSE against a hand-computed fixture, both metrics,
  trainer earlystop/checkpoint/receipt integration and premature postfit rejection.
- Review resume mode compatibility and selection metadata.
- New self-contained versioned configs for T2/P1/P2/C1/L1/L2/L3; do not mutate
  old configs/evidence. Extend full runner to derive valid actual terminal epochs.
- Four-slot automatic queue: two lanes/GPU, same slot through best/last tests;
  new task on slot completion, identity/input drift fail closed, failure preserved.
- L3 all training anchors reachability gate (not just first batch).
- R1024-REF-REPRO1 requires original batch1024 run/source/config identity audit;
  retain historical fixed .001 AdamW / 50 no-stop / Pearson-best seed1 exception.
- Full tests, Ruff/format/mypy/build, review, commit/push main, new clean server
  checkout/publication/full gates and new one-step evidence. No old smoke reuse
  as evidence for new behavior. Preparation goal completes before CUDA launch.
- Launch fresh pinned roots, verify dispatch, then reactivate same 30min monitor
  with concise reports and new identities, preserving all twelve authorized rows.
