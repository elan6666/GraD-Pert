# Sealed Jurkat engineering evaluation protocols

Created on published sourceffd94b9, without expression-based or test-score
selection. Server originals remain under
`/data/yilangliu/GraD-Pert/development/v2-jurkat-evaluation-protocols-ffd94b9`.
Checksums are in receipt.json; copies here contain IDs and protocol metadata only.

- validation.json covers all445 frozen validation conditions.
- context.json evaluates the same1000 genes at1000/2000/5000 context tokens,
  using an unrestricted training checkpoint.
- response.json chooses the first two ordered validation conditions and the
  same1000-gene axis for the three response diagnostics.

These are protocols, not executed evaluations. Separate seen/heldout1000-token
protocols remain under `/development/v2-g1-jurkat-71ee945`; they require the
expression-holdout training configuration. Never combine context scaling with
the heldout-column comparison as one causal factor.
