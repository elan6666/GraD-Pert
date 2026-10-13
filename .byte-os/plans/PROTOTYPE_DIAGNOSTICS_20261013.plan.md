# BR400/F1/N1/GM1 prototype diagnostics

User requests explain ~log(16384) cross-entropies versus N1 decline, inspect
Student/Teacher prototype distributions and entropy, and diagnose collapse.
This is read-only analysis; no retraining, architecture or loss changes.

Use each original clean training source, config, checkpoint and data identity.
Run four identically seeded training batches of 16 cells, full graph/query/view
recipe, BF16 eval mode without dropout or optimizer/EMA/center updates. Record
ordered input hashes to prove between-arm parity. Keep all tensors/checkpoints
server-only. Export small aggregates only; original run roots remain unchanged.

Measure per-sample and marginal entropy, effective marginal prototypes,
JS sample dependence H(mean p)-mean H(p), argmax occupancy/top probability,
temperature-scaled logit standard deviation, pre-head and bottleneck covariance
effective rank, relative centered variance and pair cosine. Decompose actual
cross-view cross-entropy into teacher entropy + KL(teacher||student).

Interpret uniform outputs separately from constant sharp outputs and backbone
collapse; low cross-entropy alone is not representation quality. This bounded
eval diagnostic is not all-training-distribution or causal proof. If needed,
add condition/identity-stratified or counterfactual checks based on results.

Publish diagnostic tooling after statistical invariant tests; use existing GPU
lease/admission and same-task Luna read-only short supervision. Main does not
duplicate polling. Final deliverable: compact Chinese comparison and linked
immutable aggregate/provenance report. No additional experiments authorized.
