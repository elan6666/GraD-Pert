# R50 E3 bounded performance diagnosis

Preparation in progress, 2026-09-12. No optimization accepted or CUDA launched.
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
