# GraD-Pert v2 B0: mHC and joint-only epoch validation

Effective 2026-09-30, the next fresh Jurkat B0 uses
`configs/v2/mhc_joint_only_jurkat/three_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`.
This supersedes the no-mHC config as the next B0 choice. Historical no-mHC
three-epoch and 3+3 results retain their original identities and protocols.
The pre-change source is the published main commit
`6d5cb7e4d05dcaeb7c8105dc04562a2627e9459a`.
The new config SHA256 is
`156c7a0314073fb2901f6d56f8667c657a1d4878c283a1d28ac67b3bdd1ba01c`.

## Model and batch

`streams=4` restores the existing mHC residual routing around Cell/Response
attention and FFN sublayers, with the same architecture in the EMA teacher.
The graph encoder retains its established residual path. This reuses the
native mHC implementation; no attention operator is replaced.
The complete prediction + SSL1 + SSL2 objective is retained. Both SSL stage
multipliers are 1; their internal coefficients remain 0.8, 0.4, 0.1. Single-pass
random-order final-state KDA, graph edge source key gates, graph scan32,
graph target rows64 and sequence scan16 are retained.

The config uses two GPUs, microbatch68 per rank, accumulation2, global batch272,
seed1, three epochs, and the project warmup/cosine schedule. Prior four-stream
m74 formal training OOMed; the earlier m68 four-stream run completed three
epochs. The m68 choice is based on that evidence, not on the no-mHC m74 capacity.
This config still needs a new-source server integration check before a formal
launch; this update does not start training or establish new sustained capacity.

## Validation and checkpoint selection

`model.parameters.validation_mode` explicitly selects `joint_only`. The
historical fallback is `joint_and_prediction`; parsing old config files does
not change their resolved serialization or architecture identity.

Each epoch runs only the existing fixed-view full joint-loss pass:

\[
L_{\mathrm{val}}=L_{\mathrm{MSE}}
+0.8L_{1,\mathrm{condition}}+0.4L_{1,\mathrm{node}}+0.1L_{1,\mathrm{spread}}
+0.8L_{2,\mathrm{DINO}}+0.4L_{2,\mathrm{iBOT}}+0.1L_{2,\mathrm{KoLeo}}.
\]

Its fixed validation batch identities, view seed, actual batch/cell counts,
component values and reduction remain in `fit/history.json`. Teacher and centers
are read-only and validation does not advance the training RNG. Joint-validation
reduction and its nearest-neighbor population follow the existing implementation;
this change only removes the separate population-prediction validation pass.
`best.pt` is selected by minimum validation joint loss; the final epoch is
`last.pt`.

Loss-only validation does not prepare the val 300-control population/reference,
call condition prediction, or compute Pearson. It does not emit a validation
population prediction loss or a fabricated zero/null Pearson. The MSE component
of joint validation is labeled `joint_component_prediction` in curve data,
which is distinct from the 300-control condition-mean `prediction_loss`.
CSV/PDF/PNG curves retain joint loss and all available components; result
collection and parent selection accept joint-only evidence. Same-run resume
continues to require the original source and configuration.

## Final evaluation

After the fixed training budget, both best and last still use the frozen
300-control test protocol. TxPert, TriShift and Systema each retain all-gene
and the shared DEG version, including expression-exposure groups. Systema's
train+validation condition-mean reference and the zero-PKL result contract
are unchanged. No test result participates in checkpoint selection.

## Verification

Targeted validation, lifecycle resume, curve export, result collection, parent
selection, config and execution orchestration tests: 107 passed locally.
The orchestration test asserts no val
population/reference/prediction construction in joint-only mode and two terminal
test calls. Synthetic multi-update lifecycle tests retain best/last and exact
resume state in both modes. Scoped Ruff check/format and typechecking of all six
changed source modules pass; wheel and source distribution build successfully.

Expanded v2/config/training/execution/evaluation checks: 538 passed, 11 skipped,
5 failed. All five failures also reproduce on the exact pre-change commit:
one missing historical `single_pass_jurkat/one_epoch_m66_a2` config, three
historical R1024 training-config comparison failures, and one TxPert batch
expectation mismatch (64 versus 1024). The 11 skips require CUDA/Triton.
These unrelated failures were preserved; a clean full-suite pass is not claimed.
No new GPU training or capacity result is claimed. A future formal launch must
use a clean published server checkout and pass the same-config CUDA preflight.
