# Plan 014: preserve the adapted TxPert training dataset through Lightning fit

- Status: implementation and verification in progress

## Evidence

- Commit `687681f` converted every official control-only training row and
  sealed a correct adapter receipt, but its RPE1 gate failed at optimizer step
  zero with the same frozen tensor-index exception.
- The complete failed run and log are preserved under
  `/data/yilangliu/GraD-Pert/superseded/20260825-formal-v2-6876-txpert-lightning-reset`;
  the launch-log SHA-256 is
  `9602ae9bdfbdf883a78826511ff4d0ceb8d3b463c802216b23dc2c63bb3a83ca`.
- Direct replay showed that passing the frozen module as `datamodule=` makes
  Lightning call `setup("fit")` again. The frozen setup allocates a new
  `train_data` and restores exactly 11,485 `["ctrl"]` rows after adaptation.
- The frozen module's `train_dataloader()` already returns the intended
  dataset, official `collate_fn`, `shuffle=True`, configured batch size, and
  `drop_last=True`. Lightning accepts that loader directly without invoking
  the data module lifecycle again.

## Write scope

- `benchmarks/txpert/official_api.py`
- `benchmarks/txpert/runner.py`
- focused tests and Byte OS evidence

## Non-goals

- Do not edit or subclass the frozen TxPert checkout/data module.
- Do not implement a local dataset, collate function, sampler, training step,
  optimizer, scheduler, model, or loss.
- Do not change rows, ordering policy, shuffle semantics, batch size, split,
  controls, one-epoch policy, evaluation, GEARS, or GraD-Pert.

## Acceptance criteria

1. Construct the official training loader only after strict index adaptation,
   by calling the frozen data module's own `train_dataloader()` exactly once.
2. Call Lightning `Trainer.fit` with that official loader rather than the data
   module, so no second `setup("fit")` can replace the adapted dataset.
3. Preserve the frozen official collate, shuffle, batch, drop-last,
   `training_step`, AdamW optimizer, and exactly one epoch; validation remains
   disabled and canonical test truth remains unopened during fit.
4. The training receipt names the exact official loader path used.
5. Unit tests prove the loader, not the data module, is passed to Lightning. A
   real server preflight proves loader iteration retains numeric indices, then
   full server gates and a fresh RPE1-only hard gate must pass at one clean
   synchronized commit.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_txpert_official_api.py tests/benchmarks/test_txpert_runner.py tests/benchmarks/test_runner_lifecycle.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
