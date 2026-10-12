# Official expression-holdout baselines: Scouter -> scDFM -> GEARS

User authorized main chat implementation/config/tests while the existing immutable
F1/N1/GM1 queue remains supervised by codex:01a0df0b-4142-7df1-86c0-d959471d80a1.
Base published commit 4bfa1f13b95c73b482954e4efd396cfc82cc6afe verified origin/main.
Independent workspace /Users/elan/.codex/worktrees/v2-official-expression/grad-pert.
The existing supervisor owns only old-run coordination; this build is owned by
codex:01a0c01a-0611-7a90-b3e8-8ad7e017748b under the user's new explicit instruction.
Never modify old run/source/queue or silently start further native ablations.

Acceptance: frozen official code imports and training routines (no local models);
exact Jurkat cap40 train rows/conditions, excluded expression identities, no test
truth in fit, context-matched control pairing, no heldout numeric leakage through
loss/path/MMD/statistics; frozen 300-control and evaluation-state-v2 references;
18 metrics plus counts/NA/provenance and zero persistent PKL; fresh run identities.
Use official model-specific recipe, document task-only adaptations and budget.
CPU official integration + local contract tests now; GPU 10-update capacity and
real common inference only after original queue is scientifically complete.
No pass is a scientific result until true final test receipts exist.

Disjoint implementation ownership:
- main: contract.py/data.py/evaluation.py/runner.py/queue.py/configs/docs/CLI/tests
  excluding agent test paths; owns publication, deployment, scheduling acceptance.
- adapter agent: benchmarks/expression_holdout/scouter.py and gears.py;
  tests/benchmarks/expression_holdout/test_scouter.py and test_gears.py.
- scDFM agent: benchmarks/expression_holdout/scdfm.py;
  tests/benchmarks/expression_holdout/test_scdfm.py.
Agents may read references, fetch official source into /tmp for inspection, not
edit official evidence; write a concise handoff. No server GPU jobs or Git changes.

Shared adapter interface (numpy arrays, no evaluator/truth access):
PreparedData: sanitized adata with full expression gene ID axis, train_conditions,
val_conditions, allowed_indices, excluded_indices, train_row_ids, val_row_ids,
control_row_ids, row_context_ids, train_control_indices/val_control_indices mapping
(each target row's compatible candidate controls), and receipt.
fit(prepared, config:dict, runtime:dict, output:Path, max_updates:int|None=None)
-> FittedAdapter with model:any, receipt:dict; official recipe dict in config['training']
and config['model']; runtime has official_checkout, device, seed, optional prior_path.
predict(fitted, controls:np.ndarray, target_genes:tuple[str,...],
 gene_ids:tuple[str,...], *, cell_batch:int, query_count:int, seed:int) -> [300,G]
(default interface also permits small synthetic inference for diagnostics).
Dependencies should be lazy; official import origin/commit verified by main runner.
If needed define FittedAdapter locally through contract import (main writes it).
Adapter must fail closed on unsupported frozen version, identity/context mismatch,
or impossible legal targets; do not invent model architecture or modify loss form.

Next: inspect frozen source/environment and recipes; implement adapters and common
protocol; run tests + real upstream CPU integrations; publish immutable version;
seal dependent queue and supervision handoff; after dependency success run GPU gates,
then official Scouter -> scDFM -> GEARS and common tests in serial. Long supervision
uses designated chat, 20min checks and tested return fallback; no duplicate monitoring.
