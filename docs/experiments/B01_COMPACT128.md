# B0/B1 compact128 architecture

User-confirmed 2026-09-09. This supersedes the unlaunched compact20 proposal.
The user selected this architecture after correction of the earlier parameter
estimate; 20--30 parameters/cell is no longer an acceptance requirement.

## Frozen coordinate

- Gene embedding128; STRING and GO each use two GATv2 layers, two heads128,
  learned residual concatenation and output64. Source-adaptive fusion is retained.
- Student/Teacher projection: 64 -> 256 -> 256 -> 32, L2 normalization,
  weight-normalized16384-prototype layer. Condition/masked-node head sharing is unchanged.
- Basal: 5000 -> 128 -> 64. Additive decoder: (b+p)64 -> 128 -> 5000.
  Existing BatchNorm, activation and dropout behavior is retained.
- Student and Teacher each have their own graph encoder/projector; Teacher is
  frozen to gradients and updated by EMA. No weight sharing between them.
- B0 uses the existing E3 seeded initialization into128-dimensional gene
  embeddings, then trains them. B1 uses learned-ID initialization. No other
  scientific factor differs between the pair.
- Graph6506, expression5000, eight512-node locals and four masked locals,
  loss1/1/.1/.1, seed1, batch1024, eval batch256, max100 epochs, patience10,
  metrics_only and zero persistent PKL remain unchanged.
- LR/EMA: see [step schedule](B01_BATCH1024_STEP_SCHEDULE.md). Peak2e-4 at
  nominal global batch1024;16% step warmup then cosine, EMA.994 toward1.

## Exact complete-model parameter count

| Component | Parameters |
| --- | ---: |
| Student graph encoder | 1,886,848 |
| Student projector | 631,328 |
| Teacher graph encoder | 1,886,848 |
| Teacher projector | 631,328 |
| Basal encoder | 648,640 |
| Decoder | 653,576 |
| Total | 6,338,568 |

Trainable3,820,392; frozen Teacher2,518,176. Against128,266 training perturbed
cells the complete ratio is49.417367:1. Historical total30,252,744;
parameter reduction79.048%. Counts include all heads and frozen Teacher,
not buffers. This is architecture compression, not scientifically exact-effect
optimization; no throughput improvement or accuracy preservation is claimed.

## Configuration and execution boundary

Standalone configs are `configs/combinations/b0_compact128_step_batch1024/`
and `b1_compact128_step_batch1024/`, each under `gradpert_b2/nadig_jurkat.yaml`.
Historical profiles/checkpoints are not overwritten. Architecture receipt binds
the explicit widths and capacity profile. Native full-shape execution requires
exact6,338,568 parameters and the recorded Jurkat dimensions, writing
`model_capacity.json`; old full-size checkpoints are not compatible.

Implementation and CPU regression verification precede any CUDA launch.
Formal launch still requires a clean published/synchronized commit, new run
roots and CUDA integration verification. No new training is implied by this document.

## Verification (2026-09-09)

Isolated server CPU snapshot `b01-compact128-cpu-hFD4Wm`: full suite661 passed,
4 skipped (unconfigured frozen Scouter/TxPert/scLong references and CUDA-only
checkpoint test). Ten focused tests cover full-size B0/B1 parameter counts,
unchanged historical count, pair-only prior difference, and next-step resume
for historical and compact ID/E3 paths with both schedules. Synthetic GenePT
matrices in unit tests are not a validation of the production prior artifact.
Ruff/format, strict mypy79 source files, and wheel/sdist build passed.
Logs remain on server: `cpu-tests-final.log`, `cpu-build.log`.
No CUDA speed/memory or scientific accuracy result exists for this profile yet.

## Authorized 200-epoch successor

User requested both GPUs on2026-09-09: B0 onGPU0, B1 onGPU1. New standalone
`*_compact128_step_batch1024_epoch200` configs use explicit
`vnext_combination_200`, max200, unchanged validation early stopping patience10.
The200 value is an upper bound, not a promise to ignore early stopping.
Each row first runs its own one-epoch smoke, validates evaluated/formal source
and config identity, one evaluation and zero PKL, then starts a fresh full run.
Runner: `scripts/server/run_compact200_pair.py`; any failed phase stops that row.

Monitoring handoff: reuse paused `grad-pert-h4-formal-monitor`, hourly after
launch with no active goal. Check exact tmux/PID identity, smoke/full logs,
step/validation progress, source cleanliness, both GPUs and zero persistent PKL.
Notify on phase transition, completion, failure, or required action only.
No log advance for two hourly checks is a stall investigation trigger, not
permission to kill/relaunch. On terminal evidence disable monitor before the
next bounded analysis/repair goal; preserve old attempts and do not auto-retry.
Expected duration remains unestimated until real step timing is available.
