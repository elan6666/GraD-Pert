# V2 shared infrastructure review

Reviewed against code `4707a8f479ce664c187e9154c548ec6333330593` and the user's
requirement to reuse v1's data and experiment infrastructure. This reviews reuse
boundaries, not overall experimental acceptance.

| Concern | Shared implementation | V2-specific boundary |
| --- | --- | --- |
| Condition splits and canonical axes | Existing DatasetLayout/manifests and CanonicalTrainingData | No split regeneration; v2 input/view assembly only |
| Frozen control populations | CanonicalEvaluationData and existing manifests | Query/context selection and model call |
| CLI/admission/source identity | execution.train_entry, execution.identity | Version dispatch and v2 runtime construction |
| Batch iteration/counting | training.epoch.execute_epoch used by both trainers | Model step, v2 rank slicing and microbatch accumulation |
| Best checkpoint selection | training.selection.EarlyStoppingState(min) | Fixed50 ignores early-stop signal; journal stores selected role |
| Random-state serialization | Native checkpoint RNG capture/restore | Per-rank lists and view-generator state; version-specific payload |
| Validation loss and metrics | mean_expression_mse, compute_condition_metrics, macro_summarize, existing reference state | Native v2 prediction adapter |
| Curves | training.curves.render_curves | Convert v2 committed epoch summaries; do not invent per-step data |
| Best/last loading | Shared v2 verified checkpoint loader for postfit/G1/D1 | Preserve training identity separately from evaluation source |
| Output policy | Existing metrics_only contract | V2 epoch journal, compact validation population receipt; no prediction persistence |

## Remaining version-specific orchestration

The historical v1 trainer/checkpoint interface directly owns its model, optimizer,
center state and legacy progress payload. V2 owns two distillation branches,
rank-specific RNG, accumulation and collective control operations. Its outer epoch
journal atomically commits a checkpoint before publishing best/last links, allowing
an interrupted epoch to replay. These state differences remain in a v2 lifecycle
adapter; they are not a second dataset preparation, split, evaluator or benchmark.

Do not retrofit v1 checkpoint keys or historical output/selection behavior merely
to force identical state containers. Further consolidation should preserve these
interfaces and be justified by a concrete duplicate behavior. V2 data views and
loss/model operations remain separately named and selected by configuration.

## Evidence and limits

The combined native step/resume, selection and v2 lifecycle/distributed regression
reported79 passed and1 local CUDA-only skip after the shared epoch extraction.
A final focused native trainer check passed6 tests. At4707a8f the v2 suite passed
145 tests, with3 curve tests separately passing in the matplotlib-enabled local
environment. These are software checks, not proof of real complete-validation,
50-epoch fitting, scientific performance or ready-to-launch experiment matrices.
Server acceptance remains tracked in `.byte-os/V2_ACCEPTANCE.md` and original
source-bound run receipts. Running capacity checkouts retain their original SHAs.
