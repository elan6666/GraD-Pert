---
id: 015
status: complete
wave: 1
depends_on: []
updated_at: 2026-08-28T03:15:00+08:00
---

# Plan 015: TxPert post-fit inference device restore

- Status: complete at commit `c6418df`

## Evidence

- Commit `a5f3473` completed RPE1 training for exactly one epoch and wrote a
  training receipt, then failed at the first canonical test prediction.
- The complete run/log lineage is preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-a5f3-txpert-postfit-device-teardown`;
  log SHA-256 is
  `965c05ffcb545a9ea7d6bbf7b2c18540836d30b5f33bd7c420bf3265b99a77f9`.
- The failure reported an embedding weight on CPU and its frozen Exphormer
  indices on local `cuda:0`. Lightning's fit teardown moves registered module
  state to CPU; the frozen official inference entrypoint explicitly applies
  `.to(device)` after loading a checkpoint.

## Write scope

- `benchmarks/txpert/official_api.py`
- `benchmarks/txpert/runner.py`
- focused tests and Byte OS evidence

## Non-goals

- Do not change frozen upstream code, checkpoint contents, parameters, model,
  optimizer, training, split, controls, prediction formula, or evaluation.
- Do not reimplement official inference or use test truth during restoration.

## Acceptance criteria

1. After official fit and checkpoint serialization, call the trained official
   module's standard `.to(requested_device)` before canonical test prediction.
2. Fail before test data is opened unless every registered parameter and every
   registered buffer is on the requested device.
3. Add a small training-receipt section with policy, requested device,
   observed parameter/buffer devices, and the inspected official reference.
4. Unit tests prove CPU parameters return to the requested device. Full server
   gates and a fresh RPE1 one-epoch hard gate must pass at one synchronized
   clean commit.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_txpert_official_api.py tests/benchmarks/test_txpert_runner.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```

## Completion evidence

- TxPert RPE1 completed exactly one epoch and 2,143 optimizer steps.
- The post-fit receipt records all registered parameters and buffers restored to
  local `cuda:0` before canonical test access.
- Checkpoint hash, one test evaluation, exact three-metric schema, and shared
  protocol/canonical/split/ordered-300-control hashes passed strict validation.
