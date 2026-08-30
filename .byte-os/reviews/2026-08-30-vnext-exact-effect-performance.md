# B2-vNext exact-effect performance review

## Verdict

Accepted for the frozen successor A0 implementation. The change is restricted
to sparse-union preparation and is not scientific ablation evidence.

## Equality evidence

- Deterministic CUDA reference-versus-optimized attempt 004 passed exact
  non-timing metrics, complete view and union identities, every gradient,
  Student/Teacher state, optimizer state, centers, and CPU/CUDA RNG.
- The sealed equality receipt SHA-256 is
  `10327bdd6fbd226a0daf44e18d74991f5bca62a0d3ebc89f4ac8819e8a4dbda9`.
- Every ABBA arm used source commit `7332cc1`, the same A0 config, the same
  frozen 25-batch sequence, one physical GPU, no validation or test access,
  and zero persistent PKL.

## Matched timing

The serial order was A1 reference, B1 CPU-vectorized, B2 CPU-vectorized, A2
reference. Each arm used five warmups and twenty measured steps.

| Arm | p50 ms | p90 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| A1 reference | 11,166.911 | 21,047.256 | 21,776.374 | 22,391.192 |
| B1 optimized | 4,842.376 | 4,922.064 | 5,013.519 | 5,105.427 |
| B2 optimized | 4,890.634 | 4,927.345 | 4,955.981 | 4,956.310 |
| A2 reference | 11,082.219 | 11,180.119 | 11,228.973 | 11,315.403 |

The paired median optimized/reference ratio is `0.437470`, corresponding to a
`56.253%` wall reduction, `2.286x` speedup, and `6,258.060 ms` median absolute
reduction per step. Both optimized p90 values are lower than their paired
reference. Peak allocated and reserved GPU memory rose only `0.017%` and
`0.041%`, respectively, below the 5% regression gate. CUDA retry/OOM counters
were zero.

## Boundary

This review accepts an implementation optimization only. The eight-row
sentinel and ABBA runs are training-only performance evidence and do not
complete A0 or any H/L scientific row. Formal A0/H1--H3/L1--L5 must still run
the preregistered ten-epoch protocol with validation-only checkpoint selection
and one test evaluation from `best.pt`.
