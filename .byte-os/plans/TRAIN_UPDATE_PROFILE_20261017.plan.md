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

Priority correction: after short captures, immediately DP2 formal10/restore+eval128 gates and exact main acceptance, then fit. PT1 same-medium backfill moves to next idle GPU window; its already accepted fit/last18/CPU report and exact data metadata proof do not require delaying the next fit. Full scientific acceptance still requires the missing medium; no result is relabeled complete. Cost attribution runs on CPU in parallel with later fit.

## Current measured stage and next mechanism gate

Three workloads and both ranks completed; current ignored report links all806 cost rows. Scan/project/read dominate (~69% joint rank0 kernel time); prototype+CE <1%. Current profiled windows include substantial recording overhead; no native speedup inferred. Next CPU-only analysis keeps original per-module labels and allocator traffic/peak separately, without new GPU capture or raw transfers. Do not equate allocation churn with simultaneously saved activation bytes.

DP2 formal8882 started, all acceptance gates passed and supervisor ACK/20minACTIVE verified. Main disjoint CPU work continues; no GPU probes during formal fit. PT1 medium pending at idle window, no re-training. Pending candidate acceptance remains full update/RNG plus independent unprofiled dualGPU ABBA/10step; no optimization has been adopted from this profile yet.

## Candidate H: preserve the ATen DAG, move host scan orchestration

The profile motivates separating Python binding/loop overhead from CUDA arithmetic, rather than assuming GPU idle intervals are all removable. First bounded candidate executes the current final-state chunk loop and block ATen calls from a native C++ composite implementation. This retains the same casts, cumsum, masked-before-exp decay, left-associated products, sum, triangular solve, matmuls and state update. It does not fuse CUDA arithmetic, introduce a new backward, change chunk length, reorder targets, cache RNG, add virtual nodes or reuse a state across legal neighborhoods. All graph/self/cross final-state calls are reached via one diagnostic-only scope; no native default or active source changes.

The mechanism can remove repeated Python→C++ boundaries and Python loop/intermediate bookkeeping during forward and checkpoint replay. It cannot remove the remaining CUDA launch APIs, GPU memory traffic, ordinary autograd backward or communication. Thus no large speedup is assumed and kernel count should remain essentially unchanged. Extra implementation/build dependency and cold compilation are explicit costs; reject if whole-step gains do not justify them. This is an application of mature PyTorch C++/ATen dispatch, not a new attention algorithm or originality claim.

Sources checked before implementation: PyTorch native ATen overview https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/README.md ; official extension/dispatcher guidance https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html and https://docs.pytorch.org/tutorials/advanced/dispatcher.html . KDA paper/FLA remain formula provenance, not a runtime dependency. Unlike rejected fused Gram/recurrent/CUDA Graph candidates, this hypothesis keeps the eager operator DAG and autograd implementation; its own numerical checks remain mandatory.

Acceptance: synthetic FP32 and BF16 autocast outputs/all input-state gradients (single/tail/multiple chunks), exact CPU nonzero-LR three-step full optimizer/Teacher/center/RNG and exception cleanup; then published isolated server source and same-checkpoint/RNG dual5090 full-update gates at existing tolerances. Only in a safe idle window: unprofiled ABBA at representative current micro80/global320, initialization/compilation recorded, cells/s and wall plus peak memory/10step stability. No GPU probe while DP2 fit runs; no adoption without end-to-end evidence.

CPU-only attribution source ba7620583f4405a60dc5dd06e1b8390dae4dedf0 is clean/pushed and matches server. Auxiliary analysis publication identity was checked by local live GitHub ref and remote clean HEAD; server direct GitHub HTTPS timed out and no formal CUDA publication receipt was fabricated. No proxies/routes/DNS changed. CPU module/allocator jobs use original saved trace SHA and disjoint output train-update-profile-ba76205-modules-20261017-r1, plan SHA9629229a9a5c56631752421d51f218dfc8b38b104abdd0382fa15fc1a6c94e8e. This job does not satisfy or bypass any formal CUDA/source-publication gate.

Candidate H implementation is isolated in scripts/performance/kda_aten.cpp and scripts/v2/kda_aten_candidate.py. capacity_probe --aten-kda is benchmark-only and rejects profiling/mixed candidates/active shared queues; update_parity --candidate-aten-kda uses unchanged configs/checkpoint/RNG, compiles before the reference snapshot, restores the global probe scope on exceptions, and records source/cold-load identity. Native src/config remain untouched. Initial synthetic CPU12passed; related suite56passed; final direct block tests changed to actual4heads64channels and await their targeted run. No CUDA adoption or throughput claim. Module attribution PLAN/COMPLETE and output hashes verified; no second capture.

Final CPU checks:4head×64 at both actual C16/C32 with BF16 input gradient casts passed25tests in4.15s; related56tests passed68.33s. Complete checkpointed3-step nonzeroLR exact CPU optimizer/objective/Teacher/center/RNG and probe exception cleanup included. Ruff/format/diff passed. No src/config/numerical medium changes. Pending Linux CPU build and then published-source dualGPU numeric/ABBA gates in an idle window; no adoption yet.

## User correction: finite precision and candidate I (current)

The user explicitly accepts numerical differences and prioritizes performance.
Do not spend optimization time reproducing the exact reduction/accumulation DAG.
Mathematical equations, gene IDs, legal neighbors, random draws, masks and update
timing remain fixed. Float CPU checks now use the existing full-update GPU
atol=3e-5/rtol=3e-4, with RNG/integers exact and nonfinite values rejected; this
supersedes the self-imposed CPU zero-tolerance requirement above, not scientific
or evaluation contracts. Old failed runs are never relabeled.

Candidate H Linux927aa3a r1 failed before compilation (missing Ninja). Isolated
Ninja1.13.2 was installed outside the active training venv. Linux r2 failed the
old zero test on key gradients (maxabs9.5367e-7, 1858/8704 unequal), not the
existing GPU tolerance. Mac exact results remain historical device-specific
evidence. Changed primitive view/accumulation ordering is a possible explanation,
not a proven Linux root cause. Do not repeat unchanged zero-gate probes.

Priority I: remove channel-expanded Gram tensors in BOTH forward and backward.
Earlier forward-only Gram fusion kept eager five-dimensional backward, so this
is a new specific IO hypothesis, not an unchanged retry of that rejected version.
Keep C16/C32, triangular solve, inter-chunk final-state propagation, FP32 state
and gate formula. Reuse the old forward implementation, implement a new diagnostic
Triton backward that owns a token/head and reduces incoming/outgoing contributions
without atomics or global [B,H,C,C,D] intermediates. No native default change.

For strict i>j, P_ij=sum_d k_id*k_jd*exp(g_id-g_jd), u=dL/dP. Then
dk_id=sum_(j<i)u_ij*exp(g_id-g_jd)*k_jd
      +sum_(j>i)u_ji*exp(g_jd-g_id)*k_jd;
dg_id=sum_(j<i)u_ij*k_id*k_jd*exp(g_id-g_jd)
      -sum_(j>i)u_ji*k_jd*k_id*exp(g_jd-g_id).
Mask before exp; do not use exp(g_i)/exp(g_j), clip gates or change the solve.
Float summation order may differ. For B64,H4,C32,D64, one expanded FP32 tensor
is64MiB; this is a shape calculation, not measured peak/HBM traffic saved.
The kernel trades repeated on-chip neighbor loads and reductions for fewer HBM
temporaries/launches; register pressure, occupancy and contiguous input copies
may erase the gain. Scan is ~58% of joint kernel time, but Gram is only part of
scan and kernel percentages cannot bound end-to-end gain without critical path.

Sources checked before this implementation: Kimi Linear
https://arxiv.org/abs/2510.26692, FlashAttention
https://arxiv.org/abs/2205.14135, frozen FLA954438d1fcb5e1bb05c22f9908de9c5c2df74ae5
fla/ops/kda/chunk_intra.py and chunk_bwd.py. Borrow IO-aware tiling and explicit
backward principles; independently derive our final-state block subexpression.
No upstream import/copy or originality claim. FLA all-token output pipeline is
not a drop-in replacement for our final-state training path.

Implementation: scripts/performance/gram_io_cuda.py, scripts/v2/gram_io_candidate.py;
--gram-io isolated unprofiled benchmark-only and --candidate-gram-io paired update.
They restore original functions on exceptions and reject mixed candidates.
CPU float64 derivative oracle covers tails/noncontiguous upstream/strong decay;
checkpointed three nonzero-LR full updates cover loss, gradients, optimizer,
Teacher, center and RNG with CPU kernel oracles (not CUDA verification).
Local r3:52passed,6 explicit kernel tests skipped. Original r1 scalar exact failure
and r2 missing-Ninja-PATH failure retained. Next published immutable Linux source:
CPU interpreter kernel derivatives and offline sm120 compile, then idle-window
dual5090 local forward/backward, full optimizer/EMA/center numeric gates, unprofiled
10-update ABBA and measured startup/cold compilation/steady cells/s/peak memory.
Candidate H is a secondary host-only control. Reject candidates without whole-step
gain; do not substitute isolated kernel speed for acceptance.

DP2 formal source8882, its unique20min supervisor and authorized scientific queue
remain unchanged. Main does not poll DP2 or occupy its GPUs. At terminal handback,
independently accept DP2, run a bounded useful optimization test window if ready,
and immediately continue S12-L4 and remaining authorized arms. PT1 medium backfill
remains pending at an idle window; no re-training or new scientific groups.
