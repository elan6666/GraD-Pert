# Functional B0/K/A execution plan

Overall deliverable: five fresh Jurkat cap40 six-epoch no-validation runs and
final epoch6 frozen tests, source-bound results and comparisons.

- Predecessor epoch10 evaluation: main independently accepted terminal receipt,
  all592 conditions, zeroPKL, exact checkpoint/source/config; epoch6 remains missing.
- Build: native config-selected K1 two-pass, K2 prefix self-read, A1 full softmax,
  A2 complete retention adaptation; preserve current mHC/E23 losses/task flow.
- Protocol: disabled validation has no fake best/loss. Recoverable last per epoch;
  only finalepoch6 test. Collector/curves/dispatch use existing execution lifecycle.
- Verification: operator output/gradient tests, legal graph masks, checkpoint/full
  update parity, EMA/center ordering, interruption resume, strict config/collector.
- Publication: scoped main commit, immutable server checkout; local/GitHub/server
  match; exact configs under configs/v2/cap40_functional_b0_ka/manifest.json.
- Capacity and run: all five exact128-update probes first, commonbatch272 unless
  an explicitly recorded shared fallback is needed; then B0->K1->K2->A1->A2.
  Any failure stops, preserves evidence, and returns to main. No other group starts.
- Long supervision: existing supervisor01a0df0b-4142-7df1-86c0-d959471d80a1,
  exact queue/run/attempt handoff, one20min monitor while active, return fallback.
- Final acceptance: six complete epochs each, finaltest all/DEG and exposuregroups,
  exact source/config/data/checkpoint, zeroPKL. Compare paired seed descriptively.

Authoritative mutable Byte state:
/Users/elan/.codex/worktrees/v2-gene-exposure-eval/grad-pert/.byte-os/coordination/state.json.
Implementation worktree: /Users/elan/.codex/worktrees/v2-functional-ablation/grad-pert.
Baseline pre-change commit5d6db5735c1bcb3f6a62ab88dc40a67227026a2f (published,
code identical to e27f49166dae688eb35444331a1ef1916238efe7).
Method/source audit: docs/experiments/GRADPERT_V2_FUNCTIONAL_B0_K_A_20261006.md.

Local verification (2026-10-06): directed202 passed; broader590 passed, five
failures reproduced on clean pre-change source, eleven unavailable CUDA/Triton
skips. Final affected20 tests, Ruff, format and six-file mypy pass. Evidence:
.byte-os/coordination/receipts/functional-local-verification-20261006.json.
Published-source server CPU checks and exact128-update CUDA gates remain pending.
