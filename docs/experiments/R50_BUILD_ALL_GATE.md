# R50 build-all and smoke gate

User decision 2026-09-12 supersedes sequential full-run launch order:
finish authorized ablation code/configuration and tests first, then fresh
one-epoch integration gates; no new 50-epoch run before the build/smoke
campaign is reviewed. Existing results and failed roots remain immutable.

## Explicit pending rows

- G1/G2/G3: optimizer-only, schedule-only, combined (R50_GLM5_MUON.md).
- R1/R2/R3/R4: LR-low and official Scouter/GEARS/TxPert best/last reruns.
- C1: batch-condition-balanced expression MSE, no direction term.
- Remaining R50 families: EMA, capacity, prior, local and encoder/source
  interfaces and regression coverage per R50_E3_REDESIGN.md. Final method
  coordinates remain subject to its validation-only parent-selection and
  preregistration rules; do not invent a Cartesian product or revive A5/A1.

## Acceptance inventory

For every concrete row, record config hash, parent/diff, full source SHA,
unit/integration gates, upstream version when applicable, fresh smoke root,
actual train/validation completion, finite state, checkpoint role/hash,
resume behavior, resources and PKL cleanup. Missing evidence is pending,
never a pass. Smoke preserves the 50-epoch schedule horizon and accesses no
test truth. Verify dual-checkpoint evaluator wiring through synthetic tests;
actual best/last canonical test evaluation follows formal training.

The initial exact scope is eight already registered rows, not eight new
control runs. A row blocked by an official adapter or failed smoke remains
blocked while independent code work continues. No optimizer reset from
repeated one-epoch train calls. Official packages remain isolated runners.

Implementation uses bounded goals; they end before any CUDA smoke. Only
then launch capacity-safe smoke queues and restore the existing 30-minute
monitor. A failed smoke is preserved and repaired in a new published lineage.
GPU availability alone never authorizes a formal 50-epoch launch.
