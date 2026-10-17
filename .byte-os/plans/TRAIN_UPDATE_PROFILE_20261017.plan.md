# Current-update profiling and semantic-preserving optimization

## Native opt-in integration (not yet cleared for a scientific run)

Completed diagnostic ABBA38f0987: A pooled13.806636 vs B14.998951 cells/s
using the slower rank wall, +8.6358%; full process wall including startup
+8.0341%. Peak allocated22,041,441,280 ->18,547,902,976 bytes (-15.85%).
All four10-update arms passed, same ordered batches/view RNG/config/data.
Triton cache was warmed by separately recorded derivative/parity work; do not
claim cold-compile timing. Old script/runs remain immutable.

Positive-LR57295c3:3 updates/side at.001 completed, exact input/RNG/structure,
all finite. Original3e-5/3e-4 gate FAILED and remains failed: Student final
state relativeL2 .0011444, maxabs .0034479; Teacher .000018237; centers at most
.00046072 relativeL2. Max relativeL2 across tiny individual tensors is larger,
not the whole model ratio. This is not proof of identical training trajectories
or long-term predictive quality. The user's explicit acceptance of finite
floating differences allows a disclosed opt-in performance integration, not
rewriting these original failures as passes or changing evaluation thresholds.

Native candidate weighted_gram_backend=triton_io shares one canonical Triton
backward with probes; fused forward/triangular solve/state math are unchanged.
Default=eager is omitted from historical architecture payloads; no parameters
or initialization/RNG added. Explicitly require W64,C<=32,eager final-state KDA;
reject mixed/unsupported modes rather than silently fallback. All9 graph/Cell/
Response self/cross KDA operators receive it. No scripts/upstream runtime import.
Self-contained B0 IO engineering config is separate from historical B0/PT2.

Before new PT2: published clean native source, actual CUDA forward/backward and
three positive-LR full updates with raw Student/Teacher response CLS and decoded
prediction outputs in addition to gradients/optimizer/objective/EMA/centers/RNG.
The original numeric result remains separately reported; nonfinite, identity,
integer, missing-gradient or structure changes fail closed. Then actual PT2
10-update switch/restore plus original eval128/300control gates. Native local
130 relevant tests passed, lint/format/diff clean. No formal PT2 dispatch yet.

## Real CUDA follow-through after DP2 handback

DP2 accepted:4/4,600 updates,last18,same-medium1335/5317,deep128/506,
gradient32/125,updates0,all1566 artifact bytes/SHA verified,zeroPKL.
Source8882,checkpoint46cbba75...,report2b5b6042... . Main ACK recorded;
both long monitors PAUSED. Permanent DP2 review/10figures/data archived.

Published clean38f0987 and live-GitHub publication receipt
738505131edcd2b6e668353813b2655fde8d05d80d2f85c35c7c73158080d391
match immutable source-perf-38f0987. Actual CUDA derivative12 passed.
Old full-update parity r1 failed at first LR=0 step:three gradient tensors,
maxabs2.316311e-4,exact inputs/RNG,objective/optimizer passed. Preserve the
failure; this is not multi-step nonzero-LR acceptance.

Independent unprofiled ABBA A1/B1/B2/A2 uses10 updates/warmup2,global320,
same seeded B0 joint config/source38f0987. Root gram-io-38f0987-abba-
20261017-r1. Cache reuse after separately timed kernel/parity is explicit;
do not call it cold compilation. Luna owns short read-only supervision.
Finite-only failure does not suppress timing; no native adoption follows.

Diagnostic repair only: --updates(2..10), strictly positive explicit
--diagnostic-learning-rate, --record-finite-drift for isolated candidates.
The unchanged3e-5/3e-4 check remains failed when exceeded. Opt-in recording
continues finite floating discrepancies only, with status finite_drift_recorded,
never passed. Integer values, gradient presence, shape/dtype, structure,
ordered inputs and all RNG remain exact; nonfinite values always stop.
Record original failure counts/maxabs/relativeL2. Relevant local45passed,
12opt-in CUDA checks skipped(real CUDA12 independently passed on38f0987).
Next fresh published-source three positive-LR full updates and ABBA evidence.

Latest user addition (2026-10-17): after this performance stage, prioritize new
PT2=2SSL+4MSE with both epoch2 and epoch6 medium diagnostics, then continue S12-L4
and remaining authorized arms. Preparation and unimplemented endpoint orchestration
are tracked in PT2_STAGED_2PLUS4_20261017.plan.md; do not launch it before performance
acceptance or edit current DP2. This supersedes older remaining-order lines below.

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

Linux CPU candidate88420ca/r1 is preserved as failed:51passed/7failed, no CUDA
context. Six actual-kernel interpreter checks failed because CUDA libdevice.exp
is not patched by this Triton interpreter; explicit sm120 offline compilation
itself passed (PTX SHA dfa697db...b8927799). The remaining full-update CPU tiled
oracle test exceeded the existing elementwise gate by one weight element
(5.193e-5); with an explicit fixture seed, a third-update control_cls gradient
also differed. This is a real finite-precision counterexample, not resolved by
the exact float64 derivative test, and is not relabeled passed.

Repair the interpreter with tl.exp (documented approximate rounding), not a
CUDA-side test bypass. Backward now reads original key/gate/upstream strides;
do not save unnecessary contiguous copies. The CPU full-update wiring oracle
uses the original eager derivative ordering to validate hooks/update timing;
the new reduction has separate float64 formula and real interpreter/CUDA tests.
CPU wiring success cannot establish new-kernel multi-update numerical acceptance.
The real dualGPU full-update gate remains mandatory for adoption; a failed
elementwise comparison does not erase performance measurements or its error
statistics. Evaluate numerical stability and actual throughput separately as
requested, without pretending old tolerance passed.

CPU full-update mock comparison is now explicitly a wiring/finite/RNG diagnostic,
not a numerical acceptance gate: its JUnit records per-tensor maxabs/relative L2
and pass/fail at the unchanged original elementwise gate. It must not be reported
as a new-kernel numerical pass. Keeping an early assertion there concealed the
remaining error distribution and spent time chasing CPU autograd accumulation
rather than measuring the target CUDA mechanism. Float64 mathematical derivative
checks and opt-in actual-kernel comparisons remain numerical gates. Real dualGPU
full optimizer/EMA/center comparisons retain their independent original checks.
This change does not relabel the r1 or seeded r4/r5 failures or authorize adoption.

Linux75009a5 CPU r1 completed64checks including12actual Triton interpreter
shapes (tails, strong decay, contiguous/noncontiguous inputs and upstream).
Explicit-target offline sm120 compilation passed without CUDA initialization;
no GPU execution. Main verified every listed output hash/bytes and clean source.
RESULT SHA a8b76f597a036c45372d7255bd3cee951839fd60f4a97c18f02f271a6c505cff;
PLAN SHA1b4816809881928226adc3a8aebab07b60f49dbdb1620e72e1344a7d151ff5d7.
The recorded CPU mock has2239float comparisons,4outside the original element
gate, maxrelativeL2~0.001433; these are NOT a CUDA numerical pass. Permanent
archive includes old88420ca failure and new75009a5 evidence separately.

At next idle window verify formal publication separately (current CPU-only
source proof is not its replacement), measure real local-kernel and full-update
numeric drift and unprofiled10step ABBA. Under the user's latest instruction,
finite-only floating differences may be recorded while proceeding to a separately
labeled performance diagnostic; do not block all timing work on bitwise identity.
Nonfinite/RNG/ID/update-order errors still require repair. Any failed existing
acceptance gate remains failed; no adoption or model-quality claim follows from
CPU interpreter, offline compilation or a fast isolated kernel. Main already
notified the designated supervisor of this pending bounded window; no new monitor
or GPU job was created and the scientific queue was not modified.
