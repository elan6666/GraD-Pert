# Unified MLP: restored dual-GPU accumulated first batch

This supersedes only the single-GPU scheduling section of the unified first-batch
plan. Owner remains `codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b`; long-run
supervisor remains `codex:01a0df0b-4142-7df1-86c0-d959471d80a1`.

Pre-change source: `64a9401fa53cbd04811baca2e65820976272fc9a`, verified on GitHub
main. Old N0/U24 single-card processes and dispatcher were explicitly stopped;
checkpoint epoch0, interrupted epoch1, no completed scientific result. Preserve
all old source, configs, run IDs and evidence. Every restored run starts fresh.

## Scope and order

All ten self-contained configs restore world2, accumulation2 and the prior
per-card micro68, nominal global272. Evaluation cell batch stays32; inference
still uses the exact frozen300 control manifest. Run only the five
confirmed arms: **N0 → CG1 → U24 → MR1 → P1**. The user explicitly deferred the
ambiguous sixth. C1/O1/VH/S1-L4/S12-L4 configs are generated in the same family
but remain deferred; do not dispatch them.

- N0: unified biased pre-RMSNorm clipped-SwiGLU MLPs and shared learned raw-GenePT
  projection; complete prediction+SSL1+SSL2, mHC.
- CG1: N0 plus control-conditioned fourth graph MLA/source key gate; graph prefix
  shared, no new graph edges, no masked-control bypass.
- U24: N0 plus shared raw-GenePT-conditioned prediction readout correction.
- MR1: N0 plus masked-control to perturbed-expression supervision, ratio0.25,
  auxiliary weight1; hidden raw-expression residual excluded.
- P1: N0 with same-condition grouped mean-MSE+unbiased multiscale MMD(weight1),
  replacing random paired-cell MSE. Actual group rows, never duplicate truths
  just to fill272. Conditions with <2 distinct rows explicitly skip MMD.

All arms retain six epochs from scratch, seed1, cap40 train-only manifest,
no validation loss/Pearson, epoch6 last-only test. Three Pearson families,
all/DEG × overall/seen-expression/unseen-expression, effective condition counts,
frozen common DEG and control manifests, zero-PKL postcondition. No test tuning.

## DDP correctness

CG1 SSL1 inputs are rank-local (control,p) rows. Condition CE uses weights from
the full global batch; graph node CE remains masked-node mean. Chunking the
control dimension scales each chunk by its actual population, not an equal
average of chunks. Source, masks and EMA/center update boundary are unchanged.

P1 concatenates accumulated prediction rows, gathers them differentiably across
ranks, then computes mean-MSE and MMD once on the complete effective population.
It does not average independent rank/microbatch MMDs. Each rank backpropagates
the complete objective; gather backward sums remote-row contributions and the
parameter average divides by world. Empty local tails join the same collective
order without making a fake observation. Record this distributed scope in runtime.

Local deterministic tests compare outputs, gradients, optimizer moments,
Student/Teacher and centers after nonzero-LR updates. Use high precision and
matched micro shapes for reduction parity; this does not claim bitwise CUDA
parity across different batch shapes or the same stochastic masks after topology
changes. Existing real float32/dropout/checkpoint tests remain required.

## Launch, capacity and scheduling

Publish a clean main commit before CUDA work; use a new immutable server checkout,
publication receipt, runtime and run IDs. Exactly10 full optimizer updates,
checkpoint restore and frozen300-control inference for each active arm. Check
CG1 and N0 before any formal fit to validate the common candidate batch. A failed
candidate is retained and does not permit formal dispatch. Any common rebatching
must be documented and applied uniformly; no arm silently picks its own maximum.

One dual-GPU fit at a time. Preserve the user's fit/evaluation pipeline request
only under explicit resource admission. A measured reserved peak plus15% margin
inside60% of device memory admits a sealed65% fit allocator cap and25% standalone
GPU0 evaluator; otherwise drain prior evaluation on idle resources before fitting.
Resource cap is recorded in the final immutable launch plan, separate from method
and data config. A capped evaluator OOM gets exactly one new-root, idle-only retry;
other failures stop later dispatch. The next untested config requires idle dual
GPUs, so drain an older active evaluation before that preflight. Never duplicate
preflights, overwrite failed artifacts, or run two dual-GPU fits concurrently.

This admission proves only bounded feasibility, not a full-run speedup or long-term
stability. Record actual overlap wall time and failures for later comparison.
Short preflight waits use the existing Luna helper, without main duplicate polling.
When formal fitting starts, hand the exact queue/run/attempt to the designated
supervisor, reactivate its20min monitor and main return fallback, and verify ACK.
