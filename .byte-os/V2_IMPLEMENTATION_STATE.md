# GraD-Pert v2 implementation state

Goal active. User explicitly authorized completing the entire implementation,
capacity and ablation-package plan before ZCode takes ablation supervision.
No model training or capacity run has launched. ZCode session choice remains
pending and blocks only the eventual supervision handoff.

## Workspace and baseline

Isolated checkout: `/Users/elan/code/grad-pert-v2-build`.
Published pre-change baseline: `19342809e0cacfe70196da32c356471f2f78d9e0`.
Root checkout `/Users/elan/code/grad-pert` has unrelated dirty user changes;
do not reset, merge, or commit them. Relay owner remains Codex in root coordination.

## Implemented, partial system only

- Native sparse graph read, additive expr/gene, separate control/response CLS,
  3 KDA + 1 noncausal latent attention, mHC per sublayer, output delta residual.
- Native KDA reference and equivalent block solve, no upstream runtime dependency.
- Two weighted SSL paths, EMA/centers, scoped matrix/AdamW optimizer, atomic
  v2 checkpoint with RNG restoration, single-process accumulated step.
- Strict v2 architecture/options; model.version serialized only if present,
  distinct v2_fixed_50 policy. Legacy run dispatch still rejects v2: real runner pending.
- Resumable epoch lifecycle implemented with atomic selection journal, best/last links,
  exact replay after epoch interruption, and role-specific test receipts. Synthetic tests
  verify uninterrupted/resumed parameter and RNG equality and test receipt reuse.
- Ordered-query inference and common evaluation adapter implemented; not yet wired into lifecycle.
  Tests preserve full output coverage, control row order and cell-batch prediction parity;
  validation rejects a reference containing test conditions.
- Deterministic views and 16-group design matrix; H3 batch levels deliberately null.
- User confirmed project warmup+cosine from g2_schedule: .16 warmup and .2 floor,
  endpoint schedule. This is project-preregistered, not official GLM5 timing.

## Verified

- 27 synthetic v2 tests pass (including inference/evaluation and epoch lifecycle tests): recurrence/chunk outputs and gradients, sparse/dense
  graph parity, masked expression nonleakage, loss switches, Teacher/EMA,
  actual Muon+AdamW resume, accumulation gradient scaling.
- 136 legacy modeling/native-step tests pass.
- 147 existing configs have identical validation results and resolved JSON hashes
  versus the published baseline extracted to `/tmp/gradpert-v2-baseline`.
- Config/train-entry suite: 81 passed, 4 failed. All four failures reproduced on
  unmodified baseline in test_r1024_finish_matrix.py; do not fix unrelated configs.
- Scoped mypy: 11 files pass; scoped Ruff passes (rerun after subsequent edits).
- Server SSH works. Last read both RTX5090: 32607 MiB total, 2 MiB used each;
  occupancy is mutable. Server source env Torch2.13.0+cu130.
- Local synthetic test env `/tmp/gradpert-v2-test-env` Torch2.14.0, pytest, mypy,
  Ruff installed. `OMP_NUM_THREADS=1` used. No real scientific data downloaded.

## Required next work (not complete)

1. Tests for view-builder invariants and strict version/config identity; audit
   exact loss reductions (row reduction currently applies to condition term only).
2. Complete v2 data/runtime/validation/best-last path through existing train CLI;
   new v2 callbacks must preserve identity/allocator/resource/evaluation gates.
3. Implement and test missing ablation switches (per-gene MLP, holdout/diagnostics),
   accurate disabled-loss work skipping, full batch/KoLeo/DDP semantics and resumes.
4. Run genuine CUDA numerical checks and complete-path capacity (128 sustained
   updates, save/resume/validation/inference); source must first be clean/published
   and match server. Do not fabricate batch maxima from CPU tests.
5. Generate full independent configs only from verified capacity profiles;
   implement group runner/validator/result collection and baseline preflight.
6. Publish final source, verified server snapshot and experiment launch package;
   then transfer only supervised ablation execution to confirmed ZCode conversation.

Plan: `.byte-os/plans/GRADPERT_V2_IMPLEMENTATION_AND_EXPERIMENTS.plan.md`.
Current implementation is not a ready full v2 training release.
