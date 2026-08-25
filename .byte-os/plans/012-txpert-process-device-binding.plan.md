# Plan 012: TxPert process-local GPU binding

- Status: local implementation verified; synchronized server verification pending

## Evidence

- The `d75d93a` TxPert RPE1 hard gate passed the Blackwell runtime and data
  adapter gates, then failed before optimizer step 1 with tensors split across
  `cuda:0` and `cuda:1`.
- The frozen official entrypoint selects `device="cuda"`, and its Lightning
  trainer selects the first visible GPU. Internal Exphormer tensors retain the
  construction-time device string, so passing a physical `cuda:1` conflicts
  when Lightning moves the module to its default visible `cuda:0`.
- The complete failed lineage and log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-d75-txpert-device-binding`.

## Write scope

- `src/gradpert/execution/matrix.py`
- `benchmarks/txpert/runtime.py`
- focused tests and Byte OS evidence

## Non-goals

- Do not edit the frozen TxPert checkout or alter its model, optimizer, data,
  split, controls, one-epoch policy, or evaluation.
- Do not change GraD-Pert or GEARS GPU behavior or environments.
- Do not rerun immutable B0.

## Acceptance criteria

1. Each TxPert subprocess sees exactly its assigned physical GPU through
   `CUDA_VISIBLE_DEVICES=<physical index>` and receives local `--device cuda:0`.
2. The matrix task retains the physical device identity and records the process
   environment mapping; GEARS and GraD-Pert device commands remain unchanged.
3. The TxPert runtime receipt records requested local device, visible physical
   mapping, visible device count, and current local device index.
4. Invalid or ambiguous TxPert device identifiers fail before execution.
5. Local and server test/lint/format/type/build gates pass at one synchronized
   clean commit before a fresh RPE1-only hard gate.

## Verification

```bash
python -m pytest -q tests/execution/test_matrix.py tests/benchmarks/test_txpert_runtime.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
