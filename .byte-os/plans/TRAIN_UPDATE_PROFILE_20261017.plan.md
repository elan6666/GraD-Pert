# Current-update profiling and semantic-preserving optimization

User authorization: full CPU/CUDA training profile and measured end-to-end
optimization; keep authorized ablations advancing. Main owns this independent
build scope. Supervisor retains the live PT1 lease and unchanged 20-minute check.
Baseline local/GitHub main: a0acaee6b05959093de2ae645fa54fa6ef3f35f3, clean.
Live PT1 source: ec32671f4c37bc3d1d0f935ddca87fb89a55fa3e (never edit).

## Acceptance and sequence

1. Diagnostic-only labels separate KDA writes/reads, projection/prototypes/CE,
   ordinary backward, checkpoint recomputation, optimizer, statistics and
   communication. Attribute backward kernels using forward sequence AND forward
   thread, launches using correlation IDs. Retain unknown coverage; inclusive
   parent times cannot be added. Stream traces on the server.
2. Validate instrumentation with nonzero-LR multi-update optimizer/Teacher/center/
   RNG comparisons, checkpoint early-stop/exception cleanup and trace fixtures.
   Profiling cannot become a formal capacity gate or change native execution.
3. Publish clean source; at PT1's complete terminal return, collect bounded dual
   5090 profiles at the existing m80 x2GPU xaccum2/global320 workload. PT1 SSL and
   MSE phases need separate captures; the next joint ablation needs a joint
   capture. Ten-update engineering runs, startup/export cost recorded separately.
   Never co-run probes with active training or modify its source/config.
4. Immediately advance DP2 and the existing authorized queue after the short
   capture; analyze CPU traces and implement candidates while it trains. Preserve
   PT1 -> DP2 -> S12-L4 -> CG1 -> U24 -> MR1 -> P1 -> C1 -> O1 -> K2-fixed-prefix.
5. Choose a small number of mechanisms from measured dominant costs and primary
   literature/official implementation. Explain wasted work, removable upper bound,
   implementation cost and new bottleneck. Preserve all failed/no-gain evidence.
6. An adopted mechanism needs complete nonzero-LR multi-step output/gradient/
   optimizer/EMA/center/RNG checks and unprofiled dual-GPU ABBA whole-step cells/s,
   startup/compile cost, peak memory and 10-step stability. Do not loosen gates.
   A faster kernel or profiled wall time is not end-to-end speedup evidence.

## Research informing measurement (read before implementation)

- PyTorch 2.13 profiler source: https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/profiler/README.md
  Forward/backward sequence IDs are thread-local: use the backward forward-thread
  ID too. CUDA correlation connects CPU launches to kernels, not CPU duration.
- Checkpoint 2.13: https://github.com/pytorch/docs/blob/site/2.13/checkpoint.md
  Non-reentrant context_fn has separate forward/recompute contexts and early-stop;
  compose existing contexts without changing RNG preservation or saved tensors.
- Profiler tutorial: https://docs.pytorch.org/tutorials/beginner/profiler.html
  Warm up; shape/stack recording adds overhead. Keep profiler off for adoption ABBA.
- Mechanism literature registry: docs/experiments/GRADPERT_V2_PERFORMANCE_LITERATURE.md.
  Revisit only the mechanisms matching the new trace (old trace costs are historical).

## Current evidence / limits

Previous topology packing: 12.49x component speedup; frozen eval graph reuse:
~1.58x single-condition inference. Training update ~20.4s SSL / ~8.2s MSE, data
wait reduced but training wall only ~0.8% better, not stable evidence of faster
updates. Current inclusive profiler cannot yet reliably separate operator backward
and recomputation. No current operator percentages or new speedup claimed.

Store aggregate report/CSV/figures/provenance in the ignored permanent archive
experiment-analysis/train-update-profile-20261017/. Raw traces/checkpoints stay
on /data/yilangliu. Update Byte state at launch, terminal evidence and adoption.

## Build verification

Local relevant suite44passed; final diagnostic22passed with exact (atol=rtol=0)
nonzero-LR complete-update/RNG tests, including checkpoint recomputation and
2SSL+2MSE transition. Ruff/format/diff checks pass. No src/ or configs/ changes
relative to live ec32671. Server CPU50passed (CUDA hidden); clean release17f2194bd03f0c1f4a9cae6ac031c756104c5618.
BUILD_READY SHAed63bb362099a2e106c409da126ec4c8ae96caec28f2a5b32e9d924766173867.
PREPARED_GPU_PROBES SHAfc388457fb91fca9d3d9d8fc4a5fb00e17b061819a3e5d1113816e75e4e0dfad.
Real CUDA captures await full PT1 terminal/main acceptance; no new gain claimed.

## 2026-10-17 scheduling correction

PT1 fit4/last18 independently passed, repaired real saved CPU report passed. Comparable medium sealing exposed a parent protocol metadata mismatch (staged_training only). This CPU-only repair is not a numerical prerequisite for independent profile_only captures. Main revised the self-imposed wait, retaining all actual source/config/data/resource checks. Fresh diagnostic root train-update-profile-17f2194-20261017-r2, plan SHA189c921536bd1cddffa2c1ae89d3f534d0047963c9731be6b8b89284e65c0c09, parent prepared-plan SHAfc388457fb91fca9d3d9d8fc4a5fb00e17b061819a3e5d1113816e75e4e0dfad. Short Luna supervision only; no concurrent medium GPU job. All scientific analysis and formal preflight gates remain required.
