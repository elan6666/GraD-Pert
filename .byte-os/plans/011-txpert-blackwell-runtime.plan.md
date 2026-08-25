# Plan 011: TxPert Blackwell runtime compatibility

- Status: complete; CUDA build verified and execution proceeds under plan 012

## Evidence

- The `9207e8c` TxPert RPE1 hard gate passed canonical H5AD loading and reached
  the frozen official data module, then failed on its first CUDA `torch.cat`.
- The isolated TxPert environment used `torch 2.6.0+cu124`, whose wheel reports
  support only through `sm_90`; both server GPUs are RTX 5090 devices with
  capability `sm_120`.
- PyTorch's official 2.7 release adds Blackwell support in its CUDA 12.8 wheels.
  The official PyG wheel index supplies matching 2.7/cu128 builds for the four
  extension packages frozen by TxPert.

## Write scope

- `benchmarks/environments/txpert-cu128.json`
- `benchmarks/txpert/runtime.py`
- `benchmarks/txpert/runner.py`
- `scripts/server/build_txpert_environment.py`
- focused tests, server execution docs, and Byte OS evidence

## Non-goals

- Do not alter the frozen TxPert checkout, model, optimizer, split, control
  rows, evaluation, or one-epoch policy.
- Do not change GraD-Pert or GEARS environments.
- Do not rerun the immutable c240 GraD-Pert Nadig Jurkat B0 coordinate.

## Acceptance criteria

1. The official TxPert lock commit and SHA remain hash-bound and unchanged.
2. A new isolated environment preserves every non-CUDA official requirement
   while replacing only PyTorch/PyG CUDA distributions with 2.7/cu128 builds
   and the exact SymPy 1.13.3 version required by Torch 2.7.
3. Preflight requires exact module versions, CUDA 12.8, active capability
   `sm_120`, wheel architecture `sm_120`, a core CUDA kernel, and a PyG CUDA
   extension kernel.
4. The previous environment and failed `9207e8c` lineage remain recoverable.
5. Local and server test/lint/format/type/build gates pass at one synchronized
   clean commit before a fresh single-task TxPert RPE1 hard gate.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_txpert_runtime.py tests/benchmarks/test_txpert_runner.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
