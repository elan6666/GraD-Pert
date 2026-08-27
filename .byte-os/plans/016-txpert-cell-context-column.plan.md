---
id: 016
status: complete
wave: 1
depends_on: []
updated_at: 2026-08-28T03:15:00+08:00
---

# Plan 016: TxPert official cell-context column adapter

- Status: implementation and verification in progress

## Evidence

- Commit `c6418df` passed the TxPert RPE1 one-epoch hard gate, including the
  post-fit device restore, but the later HepG2 task failed before model
  construction in frozen `PertDataModule._split_control_data` with
  `KeyError: 'cell_line'`.
- Frozen TxPert commit `08d82ee` defines `cs.CELL_TYPE = "cell_line"` and reads
  that exact observation column throughout its official data module.
- The real failed HepG2 adapter cache contains canonical `cell_type=hepg2` but
  no `cell_line`; the passing RPE1 adapter cache contains both columns with
  identical `RPE1` values. The failure and interrupted peer are preserved at
  `/data/yilangliu/GraD-Pert/superseded/20260825-c641-txpert-hepg2-missing-cell-line`.

## Write scope

- `benchmarks/txpert/runner.py`
- `tests/benchmarks/test_txpert_runner.py`
- focused Byte OS plan/status/build evidence

## Non-goals

- Do not edit or subclass frozen TxPert, change expression values, conditions,
  row order, split, controls, model, optimizer, training, prediction, or
  evaluation.
- Do not infer an alternative biological context or overwrite a conflicting
  pre-existing official column.

## Acceptance criteria

1. Before the isolated adapter H5AD is written, require canonical `cell_type`.
2. If official `cell_line` is absent, materialize it from the exact canonical
   `cell_type` values without changing row order; if it exists, require exact
   equality and preserve it.
3. Receipt the canonical/official column names, whether the column was added,
   policy, and the ordered context-value hash.
4. Tests cover missing, matching, conflicting, and missing-canonical columns.
5. A real HepG2 server preflight must prove frozen `PertDataModule.prepare_data`
   passes this boundary before any new formal one-epoch lineage is launched.
6. Full server pytest, Ruff, format, strict mypy, build, Git publication, and
   clean local/GitHub/server parity must pass at the new commit.

## Verification

```bash
python -m pytest -q tests/benchmarks/test_txpert_runner.py
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src benchmarks
python -m build
```
