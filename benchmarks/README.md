# External baselines

GraD-Pert's native package lives in `src/gradpert`. External models run through
this directory from pinned upstream checkouts in their own environments.
The adapters bind inputs, split manifests, evaluation, and receipts.

For new C2 runs, invoke the corresponding module from the repository root in
its pinned environment:

| Workflow | Module |
| --- | --- |
| Shared cohort manifest and audit | `benchmarks.crosscell.study_manifest`, `benchmarks.crosscell.study_audit` |
| Baseline queue | `benchmarks.crosscell.study_queue` |
| TxPert Cross-cell C2 | `benchmarks.txpert.crosscell_c2` |
| TxPert Jurkat and natural K562 replay | `benchmarks.txpert.replay_jurkat`, `benchmarks.txpert.natural_k562` |
| scDFM Cross-cell C2 | `benchmarks.expression_holdout.scdfm_crosscell_c2`, `benchmarks.expression_holdout.scdfm_crosscell_budgeted` |
| scDFM Jurkat replay | `benchmarks.expression_holdout.scdfm_replay_jurkat` |

```sh
PYTHONPATH=src:. python -m benchmarks.crosscell.study_queue --help
```

The C2 controller accepts `--parallel 2` when both GPUs are available. Bind
each ready descriptor to one physical GPU using `CUDA_VISIBLE_DEVICES=0` or
`CUDA_VISIBLE_DEVICES=1`. Each runner must hold the shared per-GPU lease during
its GPU work. Smoke and formal occupy the same queue slot. A missing or blocked
descriptor leaves the other GPU available for any ready C2 arm. The status
file lists all running arms under `active_arms` in two-GPU mode.

Historical run receipts refer to the script paths at their captured commit.
C2 adapters share the same frozen fitting support, ordered controls and scoring
in `benchmarks.crosscell.study_data`. The native model retains its own entry
points. Shared low-level numeric and GPU lease utilities are model independent.

Record measured training plus evaluation time before admission. If a model
would take substantially longer than TxPert, reduce its training budget and
record changes to the learning-rate schedule. Preserve the frozen C2 evaluation
cohort and controls. If the reduced-budget run remains impractical, consider an
official open-source replacement from NeurIPS, ICML, ICLR or a Nature journal.
