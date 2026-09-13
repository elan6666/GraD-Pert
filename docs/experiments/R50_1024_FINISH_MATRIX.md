# R50 batch-1024 completion matrix

Decision: 2026-09-14. Status: preparation; no row in this document is a
completed scientific result merely because its configuration or smoke passes.

## Scope and parent

All **new** 50-epoch rows use train batch 1024, eval batch 256, seed 1,
validation every epoch, no early stop, and automatic canonical test of both
validation-selected best and true epoch-50 last checkpoints (native
`best.pt`/`last.pt`, external runner's own checkpoint format). Test scores are
reporting only, never row-selection inputs. Preserve the canonical Jurkat
condition split, ordered 300 controls/truth, three metric definitions,
metrics_only and zero persistent PKL. The parent is the completed independent
batch-1024 E3 run at source `bf938adc2465adda9697387882b87f420683297d`.
The new source may be compared to that parent only with exact configuration,
split, evaluation and native-implementation parity receipts; label the
comparison cross-commit. Never relabel old batch-64/256/512 results as 1024.

## Independent batch-1024 reproduction

`r1024_ref_repeat` is a second **full, independent** 50-epoch fit of the
batch-1024 parent configuration, with a fresh run root. It changes no
scientific factor; the configuration test requires equality of model, data,
training, and evaluation fields. The earlier `concurrent` record was a
one-epoch capacity probe, not a repeat. Best and last are two checkpoints of
one fit, not two fits. Record each run's full source SHA, config SHA, ordered
split/control/truth hashes, seed, environment, all epoch validation values,
best/last checkpoint hashes and tests. Compare run-to-run best and last
separately, including absolute metric deltas and epoch-to-epoch validation
trajectories. If the repeat necessarily uses a newer source, require native
default architecture/state and one-step parity against the original source,
and label the comparison **cross-commit replication**, not exact-source
repeatability. Never overwrite the completed parent or capacity-probe roots.

The baseline has loss weights prediction/condition/masked-node/spread
`1/.8/.4/.1`, four RingInduced half-graph locals, four graph layers,
projector hidden 2048/bottleneck 256, gene embedding 128, basal/decoder
hidden 512 and 16,384 prototypes. All GraD-Pert rows below independently
inherit this parent, fixed LR .001 and EMA .996-to-1 except the declared
single factor. They do not inherit another ablation's winning setting.

| ID | Single changed factor | Why this coordinate |
| --- | --- | --- |
| r1024_lr_low_retest | fixed LR .001 → .0001 | New batch-1024 LR-low; does **not** fill the historical batch-256 missing-last checkpoint |
| r1024_weight_condition_half | condition weight .8 → .4 | Test whether condition CE pressure dominates prediction |
| r1024_weight_masked_double | masked-node weight .4 → .8 | Test greater gene-neighborhood self-distillation pressure |
| r1024_weight_spread_half | KoLeo weight .1 → .05 | Test weaker dispersion without changing its sample pool |
| r1024_scale_projector4096 | projector hidden 2048 → 4096 | Isolate wider DINO-style projection head |
| r1024_scale_graph6 | graph layers 4 → 6 | Isolate deeper graph encoder |

The old R1024-P1 configuration declared a 1024-wide projector but its sealed
source did not route that value into the model. Its result remains preserved
but cannot count as a projector-capacity ablation. The new 4096 row requires
model-parameter/count tests and an exact legacy-default architecture hash.

The three loss rows are sensitivity probes, not an assertion that the new
weights are superior or that raw loss magnitudes should be equalized. Report
unweighted and weighted terms, gradient/finite health, best validation loss,
all three validation Pearson metrics, training loss curves, test best/last,
wall and peak memory. The scale rows report exact total/trainable parameter
counts and model/data ratio before launch; keep embedding, basal/decoder,
head dimensions and graph topology unchanged.

## External TxPert coordinate

The previously completed official TxPert 50-epoch rerun used its frozen
official batch 64 and stays labeled that way. A fresh batch-1024 TxPert row
is a **user-requested batch override**, not an official-default comparison.
The isolated adapter retains the frozen upstream checkout/config SHA and
passes 1024 only to the official data module. The model, graph, optimizer,
LR, weight decay and evaluation remain unchanged. Record both official
batch 64 and effective batch 1024 in the preflight and run receipts.
The earlier batch-64 one-step gate does not authorize this new coordinate.

## Execution gates and four-slot scheduler

Before CUDA, publish a new clean main commit and verify identical clean local,
GitHub and a fresh server checkout. Do not edit the active 7efd source.
Run one real training update for each new GraD-Pert coordinate and TxPert
batch-1024 capacity, on the full 50-epoch schedule horizon with validation/test
closed. TxPert then requires a fresh exactly-one-epoch smoke before its full
50-epoch run. Preserve failures without changing batch or silently retrying.

Use two physical GPUs with at most two training/evaluation processes per GPU.
Try to keep four independent slots filled **only while** measured GPU memory,
allocator headroom, host resources and source identity allow it. No fixed
four-process promise can override OOM/other users; leave an unsafe slot idle
and record why. Never overlap a row with its own best/last test. Each run has
a fresh hash-pinned root/contract and explicit training/evaluation source SHA.
Run full tests, Ruff, format, strict mypy, build and review before formal
launch; do not treat one-step or one-epoch gates as 50-epoch results.

The older batch-512 G1/G2/C1 concepts are treated as covered by the completed
batch-1024 U1/S1/S2/C1 rows for queue accounting, at the user's direction;
this is administrative closure, **not** a claim of numerical or scientific
equivalence. The combination G3 remains excluded by the no-combinations
decision. KoLeo batch-cell K1 is diagnostic-only after its zero-distance,
zero-gradient step and is not in this 50-epoch queue.
