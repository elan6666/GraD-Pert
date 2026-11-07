# Expression-only validation of the two-epoch speed comparison

The speed benchmark at published commit `1831d7d3c8617030b430b0f6827ed31a99758963`
completed two epochs per arm with deferred postfit. A later user request asks for
the final Pearson comparison; the user's current evaluation preference excludes DEG.

The new optional `metric_scope=expression_only` evaluates only the three existing
v2 expression Pearson definitions. It never reads per-condition DEG indices or
availability in the metric loop. The legacy default retains its six metrics and
unchanged inference arithmetic. The plan and result explicitly record the scope;
shards with different scopes cannot merge.

Use each retained epoch-2 last checkpoint and its original self-contained config.
Evaluate both on the same complete frozen validation conditions, ordered 300-control
manifest, original query/batch recipe, native precision and existing TRAIN-only
Systema reference. Report full-expression and the original checkpoint's seen/unseen
expression groups, finite support and NA. Do not compute DEG, test the test split,
change parameters, retrain, select another checkpoint or alter historical results.

The evaluation source differs from training only in this optional metric routing.
Before CUDA, verify clean published local/GitHub/server identity, unchanged modeling
and non-evaluation training blobs, original checkpoint/config/data bindings, fresh
output roots and the original selected-GPU1 admission policy and lock. Run the two
evaluations sequentially. Preserve failure outputs; no in-place retry. Keep raw
data, checkpoints, predictions and full condition IDs on the server; transfer only
reviewed, dry-run-checked small aggregates and receipts with SHA verification.

The two-epoch state mismatch is already recorded. Pearson validation describes
these checkpoints; it does not establish numerical equivalence or justify adopting
the speed switches as defaults. Waiting uses the designated supervisor chat with
native terminal return; the main chat independently accepts the final receipts.
