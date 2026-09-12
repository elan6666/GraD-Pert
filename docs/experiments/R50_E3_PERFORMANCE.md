# R50 E3 bounded performance diagnosis

Initial diagnosis completed, 2026-09-12. No optimization accepted.
Pre-edit clean local/GitHub main baseline verified:
`bf938adc2465adda9697387882b87f420683297d`.

The existing profiler now supports an explicit `r50_e3_batch512` coordinate.
Its original `a0` default remains unchanged. The E3 coordinate pins the complete
original config byte hash
`839c7518d79bafd592ca3587d07442179590976da2bd5405ba4af3541ba78505`;
changing the launcher-provided hash cannot authorize a different configuration.
Run the full native R50 lifecycle with its original 50-epoch schedule horizon,
but stop after exactly five optimizer steps (two warmup, three profiled). The
bounded wrapper raises immediately after step five, before validation/testing.
Evaluation constructors are replaced by fail-closed guards. This is training-
only diagnostic evidence, never a five-step scientific completion.

The frozen E3 prior must use the existing availability receipt:
`/data/yilangliu/GraD-Pert/contracts/r50-f4bad63-v1/genept-preflight.json`,
SHA256 `1b8f3e7b84ba7dbe7a688fb1e750640fe6b9230e2617e23988b33e8f70c47dd7`.
The profiler checks receipt bytes before GPU initialization; the native loader
also validates prior content, graph identity and ordered coverage before fit.

Before launch: pass local/server tests, lint, format, mypy and build; push the
scoped change to main; create a new clean server checkout, never pull active
batch128 source. Seal source/config/prior/canonical/split/graph hashes and
launcher bytes in a fresh contract. Bind physical GPU0 UUID, confirm idle and
headroom, use expandable_segments:True, cpu_vectorized unions, one CPU thread
per library and CUBLAS_WORKSPACE_CONFIG=:4096:8. Do not alter scientific config.
End the preparation goal before any CUDA launch, then resume the existing
grad-pert-r50 monitor at 30-minute intervals after successful submission.

Capture all-GPU and host resource snapshots: GPU1 may still run batch128.
Concurrent diagnosis is attribution only, not speed acceptance. Require exact
five steps, no evaluator access, no OOM/retry, sufficient memory and zero PKL.
Preserve failed attempts without overwrite or automatic relaunch.

After receipt validation, choose the smallest optimization supported by actual
profiling. Preserve forward/BatchNorm/dropout and sampling order, gradients,
optimizer, Teacher, centers and RNG. Require exactness and three-epoch matched
old/new comparison using the original 50-epoch schedule, then isolated same-GPU
ABBA timing under low shared host load. Only accepted changes become defaults;
this preparation does not establish a performance improvement.

## Sealed initial evidence and next diagnostic

Source c492547ca897f964dbef52abaca4669154da5f15 ran on physical GPU0;
GPU1 concurrently trained batch128. Receipt SHA256
e68d6e3388c4b452489c0a0e7ce9aea39dd9a44ccaeea2b4e8e057653aa4c7d4,
server root `/data/yilangliu/GraD-Pert/development/r50-perf-c492547-v1`.
All five steps completed, all resource/runtime predicates passed, zero PKL,
no evaluation access. Trace/table hashes were independently checked.

Measured steps 2/3/4 wall: 2723.847/3033.376/2737.653 ms;
student-local: 1368.704/1495.332/1364.597 ms;
view-build: 350.046/515.249/355.830 ms. These are instrumented attribution
measurements, not accepted timing comparisons.

Across three active steps the trace records 105236 cudaLaunchKernel calls
(399.18 ms inclusive runtime time), 4799 cudaLaunchKernelExC (16.17 ms),
11157 cudaMemcpyAsync (77.05 ms), and 1530 cudaStreamSynchronize (31.29 ms).
CPU operator totals are nested and must not be added: e.g. NativeDropoutBackward0
and its autograd wrapper overlap. Counts suggest launch overhead, but do not
establish a specific implementation target or expected speedup.

Next: a fresh five-step E3 diagnostic with cProfile around the same bounded
worker, preserving all configuration and guards, to separate cumulative Python
view/union/encoder preparation costs. Retain the initial attempt untouched.
Python/Torch instrumentation changes overhead, so this second run is also
attribution only. Select an optimization only after narrower evidence exists.

## Measured channel preparation target (not accepted optimization)

Python diagnostic source `ac279f81283f87f0377078bc0036580d4217b061`, receipt
`057b6cf885127134cdb9a032fc84837b5fb4d4c20dd2fb18a833e32d1be1557f`,
completed five steps without evaluation. Across that instrumented run,
182 ordered-pair union calls accumulated 7.325 seconds. Startup, profiler
export and nested cumulative times are not steady-state training timings.

A separate CPU-only preparation benchmark was committed/pushed before use at
`526ab594192a4c5557e5dd7c704831121e04c405`. Eight synthetic tests passed both
locally and on the server. It reconstructed the first three actual epoch-0
batch512 schedules, with 34 distinct global/local views each, preserving their
ordered source pairs. Frozen input SHA256:
`7f9bd01561646717397bffeef741402088718f0286605970e4e4d7ba36e3e368`.
Evidence remains at
`/data/yilangliu/GraD-Pert/development/r50-union-526ab59-v1/preparation.json`.
Independently read-back SHA256:
`d47aa05246e8cd2770314039f3a01f99b6b10deba1aa1178b3a8904013c22533`.
The preceding Python pstats SHA256 is
`340b9ccbf38994f3e17dfc43f996bfd3f4dd979b62a9b9c6ba2acb9a43554d4c`.

Warmup-excluded alternating preparation timings, milliseconds per batch:

- Original: 602.070, 601.684, 601.289, 600.118, 599.726.
- Array-native: 265.785, 265.629, 266.213, 266.671, 265.825.

Medians 601.289 versus 265.825 ms justify testing this narrow target. The
benchmark checks all concatenated pair/channel arrays and Torch global RNG;
it does not establish full union, training-state or end-to-end speed parity.
GPU1 was concurrently training; isolated acceptance timing is still required.

Candidate `cpu_array` retains the `cpu_vectorized` default and reference path.
It converts each source once, constructs reverse channels as array views,
and vectorizes bounds/self-edge preparation. It leaves union sorting,
membership, expander generation and model forwards unchanged. Candidate remains
unaccepted until full edge/gradient/state/resume gates, three-epoch comparison
and serial same-GPU ABBA pass. No scientific row may select it yet.

### Next CUDA contract (must be published and sealed before launch)

Run serial on physical GPU0 with three fresh roots: `cpu_vectorized`,
`cpu_array`, and `cpu_array` with an in-place checkpoint serialization roundtrip
after zero-based step 2. Each is six training steps (capacity phase),
deterministic algorithms, `--capture-exact-state`, identical frozen E3 config,
seed, prior, batches and original 50-epoch schedule. No validation/test access.
Capture initial and every post-step model/Teacher, every gradient, optimizer,
both centers and Python/NumPy/CPU/CUDA RNG hashes. Require initial equality,
all six ordered non-timing metrics and state hashes exactly equal; require
complete receipts, resource predicates and zero PKL. First-step health remains
required. Roundtrip deliberately zeros persistent model/center tensors and
clears optimizer state before loading, then demands exact restoration. It is
explicitly not a fresh-process restart; fresh-engine resume is separately
covered by the synthetic multistep test and must not be misrepresented.

Full-state hashing/checkpoint I/O invalidates timing acceptance. Following
these gates, prepare the original-horizon three-epoch matched comparison and
isolated ABBA (5 warmup +20 measured per arm, A/B/B/A), without state hashing.
Acceptance requires >=10% and >=100 ms median wall reduction, p90 and peak
memory no more than 5% worse, no allocator retry/OOM and exact-effect gates
still passed. Current profiler's default timing length (2+10) is not that ABBA
contract and must not be silently substituted. Preserve all failed attempts.

## Six-step CUDA result (2026-09-12)

Source `0f0c08eb7829d4cf7988b58ad954dd598ce097df`, server root
`/data/yilangliu/GraD-Pert/development/r50-array-0f0c08e-cuda-v1`.
All three arms finished with six ordered steps, complete predicates, queue RC0,
no evaluation access and zero PKL. Independent comparison checked initial and
every step's six full-state hashes, non-time metrics, view stats and first-step
health; all matched exactly. The roundtrip arm also passed its explicit
serialization-restoration check. Receipt SHA256 values:

- reference: `06b5f0e7c5b5ed66ce660eb53c650669404322f6c4ab8dd26d729eb2d59d18d3`
- candidate: `8dc6349af8b36a4f7df31283c12d61b7a57bb6866ef7302da6fd7d674a48d65f`
- roundtrip: `ffdbc71b732af8f23aa0de37b264c4adf3415b2ba917bfefc9aea0d7ac352899`

This proves the bounded deterministic trajectory, not three-epoch equivalence,
fresh-process CUDA resume, or accepted end-to-end speedup. Default remains
`cpu_vectorized`. GPU1 batch128 remained active during these diagnostics.

Next bounded execution must stop at the next epoch's data-factory entry,
after the preceding validation/logging/checkpoint transaction. Keep trainer
max_epochs=50 and total_schedule_steps unchanged. An independent third arm
stops after epoch1 and resumes in a new process to epoch3; no epoch replay and
no test access. Validation and checkpoint identity must be audited at each
boundary before a stop can be declared successful. First-epoch and final
state digests must match the uninterrupted paths, with configuration and
ordered data identities bound separately from run-specific paths.

## Original-horizon three-epoch result and ABBA protocol

Source `b25825e2cf9ebc55f26f5b22db7e8789e92626a6` completed all three
trajectories (reference, candidate, fresh-process resume) at 1005 steps and
three validations, retaining the original 50-epoch schedule horizon. Queue
RC0 and the independent comparison passed; no test access or PKL. Evidence
remains at `development/r50-epochs-b25825e-v1` on the server. Terminal hashes:

- reference: `208f13d371661aa29dcdf43ad3fec5ff8c0120bc84cc77e00ff35cf91ae11b13`
- candidate: `e1050bb8343815871bc47d676d0388e1c1ae9f21eb91b8aa56d109f576f1ce9c`
- resume prefix: `ced22ee68fd0a0a65240b2280d4d07c74e13194693766e6a5cd909450e101a69`
- resume completion: `a78c9a26442aa6d781353a226b1fa5dff98eef8ca669e71748c3ef93a0085aeb`

The separate batch128 run at training/evaluation commit
`72d75463db2d93d531743cb7229419104c0b63f0` completed 50 epochs and its
best/last postfit receipt passed `verify_existing`. These test results are
not used to select this performance implementation.

New timing uses explicit `--phase timing --timing-protocol abba_5_20`:
five warmup plus twenty measured steps, serial A1/B1/B2/A2 on one physical
GPU. Legacy timing remains 2+10. Each arm uses the same scientific config,
schedule horizon and instrumentation, with ordered perturbation/control row
hashes. Full-state hashing and Torch profiler are disabled. Resource snapshots
are collected outside the native step timer before and after the run and
after each step. No other compute process is permitted on either GPU; other
GPU utilization must be <=5%, one-minute host load <=2. These sampled checks
do not prove uninterrupted exclusivity between samples and that limitation
must accompany timing evidence. Failed attempts are preserved, not retried
in place.

Report all raw times and p50/p90/p95/p99. Pair A1/B1 and A2/B2, take the median
of optimized/reference median ratios and median absolute improvements.
Require ratio <=0.9 and improvement >=100 ms; both pairs' p90 and allocated
and reserved peak memory ratios must be <=1.05. Resource/identity/evaluation
gates are prerequisites, not substitutes for timing acceptance. Candidate
remains opt-in until all gates and review pass; no new scientific run is
authorized by a timing-only receipt.
