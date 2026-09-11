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
