# Unified MLP first batch: authorized implementation and execution

Owner: codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b. Supervisor:
codex:01a0df0b-4142-7df1-86c0-d959471d80a1. Baseline clean and published:
40fb032365a7d61b8de6caae7e4fb15b78c2ee24 (GitHub main verified).

## Outcome and limits

Implement and publish N0/U24/MR1/P1/C1/O1/VH/S1-L4/CG1/S12-L4;
10-update CUDA restore/inference gates; run these ten configurations from
scratch for six epochs on the existing Jurkat train-only cap40 selection.
No per-epoch validation, last-only frozen 300-control test, all 18 Pearson
variants with effective condition counts, zero PKL. No second batch or V0/V1.
One GPU per experiment, two independent lanes, accumulation=1; freeze a
common feasible batch from engineering gates, never reuse old DDP capacity.
Historical source/config/checkpoint identities remain unchanged.

## Method contract

Unified nonlinearity: pre-RMSNorm epsilon1e-6; bias on gate/up/down;
gate upper clip10, up clip[-10,10], SiLU(gate)*up. Keep role-specific dropout,
linear prediction/prototype outputs and attention projection/normalization
paths. No duplicate FFN pre-norm. U4 raw frozen2048->sharedLinear256->
unified256/1024/256. Expression scalar lifts to256 before normalization.
Keep single-pass final-state legal-neighborhood KDA, random order, sources,
mHC, all five active loss weights1, spread/KoLeo0. U24 reads the same raw
prior through a shared2048/64/256 correction, zero output initialization.
MR1 masks25% training control genes, weight1, blocks residual/cross leakage.
P1 replaces random cell MSE with matched-context same-condition population
mean MSE plus unbiased multiscale RBF MMD. Freeze MMD coefficient1 as a
project setting, not an official or tuned value; bandwidth derives detached
from training truth only, never test. Log each component and degeneracy.
CG1 keeps graph layers1-3 shared; masked visible-control feature/weight/
weighted-sum/summary modulates layer4 source key gate only. Student/Teacher
pair the same(control,p/gene), with no unmasked-summary shortcut.
SSL1 default Local GO/STRING; S1-L4 adds two random-node locals; S12-L4 uses
two GO and two STRING. SSL2 S12-L4 has four independent random locals.
All CLS pairs equal, excluding same Global pair; VH enlarges both branches'
Global .8-1 and Local .4-.65, independently of Local counts and masks.

## Disjoint implementation delegation

- Main: config/v2.py, model.py/operators.py integration, views/objective/
  engine/runtime integration, configs, first-batch runner, documentation.
- MLP helper agent: ONLY new modeling/v2/mlp.py and
  tests/v2/test_unified_mlp.py; report API and handoff, do not commit.
- Population helper agent: ONLY new training/v2/population.py and
  tests/v2/test_population.py; pure sampler/loss API; report handoff, no commit.
- Bounded tests: Luna read-only supervision; main never duplicates its polling.
- Matrix helper: ONLY scripts/v2/generate_unified_group.py,
  tests/v2/test_unified_group.py and generated configs/v2/unified_first_20261010/.
- Queue helper: ONLY scripts/v2/run_unified_group.py and
  tests/v2/test_unified_queue.py; main integrates/publishes.

## Acceptance and continuation

Verify historical config/checkpoint identity, masking and IDs, per-control
graph behavior, equal view pairs, nonzero-LR multi-update optimizer/EMA/center
and RNG checkpoint restore, queue terminal collection. Publish scoped code
to main, immutable matching server checkout/config/publication/runtime.
Keep source/config hashes in every preflight and fresh run. Resource gate
must observe both independent lanes; launch only after passing all arms.
Long queue handoff requires exact supervisor ACK, active20min checks and
main return wake fallback. Failure returns to main for safe repair, never
mutates active source or silently repeats a completed experiment.

Current phase: ten CUDA preflights passed; N0/U24 formally active under c8e26d0.
User's 2026-10-10 scheduling update adds bounded fit/postfit overlap for the
remaining eight arms through a fresh published controller. Adopt the two live
inline children without restart. Preserve the sealed parent queue/preflights;
new deferred rows get new run IDs and exact native source-tree equivalence
evidence. One fit plus one capped evaluator per GPU, idle-only evaluation retry
on capped OOM, no batch/metric changes. Synthetic pipeline/retirement tests,
publication/migration and durable supervisor ACK precede closing the main turn.
This plan supersedes
the earlier GELU-only nineteen-configuration draft for active execution.
