# Plan 013: TxPert control-only training index compatibility

- Status: adapter translation verified; superseded at fit boundary by plan 014

## Evidence

- The `b06f29f` RPE1 gate passed source, CUDA 12.8, `sm_120`, PyG, canonical
  data, split, and process-local device gates, then failed before optimizer
  step 1 in frozen `gspp/models/txpert.py` at `z_p[p]` with `IndexError: too
  many indices for tensor of dimension 2`.
- The complete failed run and launch log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-b06f-txpert-pert-index-shape`;
  the launch-log SHA-256 is
  `6c72355d09d2a28dae771903f90a8de55c0528bf54dc859c0fc919d9cb2511f2`.
- A read-only replay of the frozen official data module observed 137,169
  training rows: 125,684 treatment rows contain numeric IDs such as
  `[3166, -1]`, while 11,485 control-only rows created by the official dataset
  extension contain `["ctrl"]`.
- The same frozen data module defines `pert2id["ctrl"] = -1`, constructs the
  perturbation graph with `len(pert2id)` nodes, and the frozen model indexes its
  perturbation tensor with every component. The observed string is therefore a
  representation mismatch at the adapter boundary, not a model redesign.

## Write scope

- `benchmarks/txpert/official_api.py`
- `benchmarks/txpert/runner.py`
- focused tests and Byte OS evidence

## Non-goals

- Do not edit the frozen TxPert checkout, graph, model, loss, optimizer,
  training rows, split, controls, one-epoch policy, or evaluation.
- Do not remove control-only rows or translate any non-control label.
- Do not alter GraD-Pert or GEARS behavior and do not rerun immutable B0.

## Acceptance criteria

1. After the frozen official data module is constructed, map only the exact
   official string control label through `data_module.pert2id`; preserve all
   already numeric components and row/index order.
2. Fail before model construction for a missing/non-numeric official control
   ID, an empty/malformed condition, a non-control string, boolean, or numeric
   ID absent from the official mapping.
3. Emit a small adapter receipt with the policy, official label/ID, condition
   and component counts, conversion counts, valid-ID count, and deterministic
   before/after hashes.
4. Unit tests prove the expected conversion and failure paths without mutating
   input on failure; a real server preflight proves the resulting official
   first batch contains numeric components only and can index the official
   perturbation tensor.
5. Local and server test/lint/format/type/build gates pass at one synchronized
   clean commit before a fresh RPE1-only one-epoch hard gate.

## Follow-up evidence

- Commit `687681f` passed all implementation, real-data index, GPU tensor, and
  server quality gates. Its RPE1 hard gate nevertheless failed at the same
  first-step expression because Lightning invoked `setup("fit")` again after
  the adapter receipt had been sealed.
- A read-only replay observed `train_data` object identity change across the
  second setup and all 11,485 control-only rows revert from `[-1]` to
  `["ctrl"]`. Plan 014 closes this lifecycle boundary without changing the
  official loader or training algorithm.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_txpert_official_api.py tests/benchmarks/test_txpert_runner.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
