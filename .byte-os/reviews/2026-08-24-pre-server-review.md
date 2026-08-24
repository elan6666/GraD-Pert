# Pre-Server Completion Review — 2026-08-24

## Verdict

`not_ship`: local implementation is coherent and materially verified, but the
requested end state is not complete without the server experiment receipts,
formal result catalog, source parity and final post-result review.

## Scope inspected

- Active product/design/engineering contracts and all eight plans.
- Native model, graph views, losses, state updates, training/resume/inference.
- Canonical data/split/control contracts and datasets-v2 evidence.
- Nonlearned implementations and official GEARS/TxPert adapters.
- Prediction/evaluation artifacts, metrics, ResultCatalog and notebook.
- Server source identity, matrix orchestration and small-file staging.
- Full local tests, Ruff, available-dependency mypy, build, Byte state, Git
  diff/large-file/credential searches.

## Resolved findings

1. `P1` Test evaluator existed in official-runner scope during fit.
   - Evidence: pre-repair runner outer context constructed test data before
     `api.fit_one_epoch`.
   - Repair: test reader now starts only after fit, checkpoint hash and training
     receipt; parameterized lifecycle test covers both runners.
2. `P1` Development commit declaration was not bound to Git HEAD.
   - Evidence: `inspect_source_identity` overwrote the argument when `.git`
     existed.
   - Repair: mismatch is a hard error with a Git fixture regression test.
3. `P1` Server matrix and allowlisted sync were documentation-only promises.
   - Repair: iteration 2 added exact task/fairness gates and sealed staging.
4. `P2` Foundation-era harness/AGENTS/OKR text contradicted current files.
   - Repair: navigation, commands and current baseline now describe observed
     implementation and explicitly absent results.

## Open release blockers

1. Resolved after review: the clean server checkout passed 153 tests, Ruff,
   full strict mypy, isolated build, config verification, and Git-clean gates.
2. The 15 learned model/dataset one-epoch gates do not yet have current receipts.
3. Seed-1 nonlearned formal results and GraD-Pert full seeds 1--4 are absent.
4. datasets-v2 small receipt chains have not been allowlist-staged and verified
   after transfer.
5. Resolved after review: local, GitHub, and the clean server checkout were
   verified at `1ec6bd6a60a6480ec641806318ca5efab0e3ac90`.
6. The strict 45-run ResultCatalog builder and executed notebook are ready, but
   no actual formal catalog, post-result metric recomputation audit, final ship
   review or delivery record exists.

## Current verification

- `pytest -q`: 137 passed, 9 skipped. Skips are Torch/PyG/anndata unavailable
  locally or datasets-v2 receipt sync pending.
- Ruff lint/format: pass.
- Strict mypy on 20 available-dependency modules: pass.
- Config matrix: 30/30 files and 30 unique hashes.
- Offline wheel/sdist build: pass.
- Byte state validation: pass.
- Git diff whitespace check and credential scan: pass; the source snapshot is
  committed and publicly published without large scientific artifacts.

## Required next review

After server access/source publication are available, inspect actual task and
training receipts, fairness hashes, evaluator recomputation, staged sync tree,
catalog and notebook. Only that post-result state can receive a `ship` verdict.
