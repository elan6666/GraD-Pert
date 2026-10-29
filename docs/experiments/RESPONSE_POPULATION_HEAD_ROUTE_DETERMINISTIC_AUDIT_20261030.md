# R17 route audit: isolate CUDA backward repeatability

This is an engineering diagnostic amendment, not a new training objective or a
successful experiment. The frozen original design remains
`RESPONSE_POPULATION_HEAD_PROTECTION_20261030.md`. The original d6 design and the
later faaff calibration observations have separate versioned documents; do not
edit an already hash-pinned design to append later results.

## Evidence and bounded hypothesis

Native attempts A1, A2 and A3 made zero optimizer updates and failed the original
CUDA Response-gradient comparison, atol=2e-6 and rtol=0. A2 added failure-only
within-route replays. A3 additionally set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before CUDA initialization. On A3, head-input gradients were identical for TT,
FF and TF; Response differences were respectively 1.220703125e-4,
6.103515625e-5 and 6.103515625e-5. T/F denotes whether the auxiliary loss may
differentiate prediction-head parameters. Within-route differences therefore
already exceed the cross-route gate. They do not identify the responsible
kernel, establish a faulty gradient route, or validate its scientific value.

PyTorch documents deterministic alternatives for some CUDA operations and a
strict error for operations without a supported deterministic implementation:
https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
The actual Response path contains indexed reads and a final SDPA attention
layer. Source inspection alone does not establish which operation caused the
observed variability. In particular, a permutation gather need not have repeated
destinations, and the largest error at an injection weight does not prove that
its Linear operation is the origin.

## Audit-only intervention

On CUDA only, the complete native head-route audit executes with
`torch.use_deterministic_algorithms(True, warn_only=False)` and a legal, observed
cuBLAS workspace setting. Both the prior enabled flag and warn-only flag must
be restored in `finally`, including failures. CPU audit behavior remains
unchanged. An unsupported operation is a terminal diagnostic failure: no
warn-only mode, fallback, tolerance change or implicit retry is permitted.

Score/loss equality remains exact. Head-input and Response gradients retain the
original 2e-6 absolute / zero relative tolerance. A successful CUDA audit must
also pass TT and FF replay checks under that same gate, preserve existing main
head gradients and leave auxiliary head-parameter gradients zero. Native
receipts record the actual before/inside/restored settings, workspace,
repeatability comparisons, and any original exception traceback.

Audit replays are excluded from all eight initialization measurements and the
lambda calculation. The context exits before those measurements continue.
Neither the calibration gradient-norm computation nor formal training is put
into deterministic mode by this change. A pass establishes route consistency
under the recorded diagnostic execution only; it does not establish bitwise
repeatability of the default training backend or scientific improvement.

The next native attempt uses a fresh published source and run ID. Only after
its original calibration and resource/state/RNG/zero-PKL checks pass may a
measured lambda be sealed and the existing ten-step/restore preflight proceed.
Historical attempts, sources and failure receipts remain unchanged.
