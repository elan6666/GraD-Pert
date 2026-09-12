# K1: KoLeo cell pool without condition deduplication

User-authorized 2026-09-12. New independent row, not a new representation.
Parent is the existing R50 E3 batch512 scientific baseline. Only the spread
sample pool changes: expand each global-view condition p into real batch-cell
order; nearest candidates exclude only the same cell index. Keep p, the
existing normalization/nearest-neighbor/clamp formula, weight0.1 and all other
loss/model/data/optimizer/schedule fields fixed. Do not add noise, a basal b
representation or a fusion representation. Global views remain separate and
their losses retain the existing averaging rule.

First run exactly one actual optimizer step, no full epoch or validation/test.
Record same-condition nearest-neighbor fraction, exact-zero nearest-distance
fraction, exact repeated-p equality, unweighted KoLeo and its gradient norm
with respect to unique p (separately label any model-parameter gradient norm).
The same normalization and argmax tie rule must be used by diagnostics and
the objective. Retain the complete configured schedule horizon.

Large clamped loss with zero gradient is an informative outcome, not a reason
to silently alter the experiment. This tests whether deduplication is needed;
it does not presume cell repetition is an improvement. Formal training, if
subsequently advanced, retains the authorized50-epoch best/true-last tests.

Implementation baseline is published main
`b563f405880f1ea7430e958416713ab050ffe887`. The helper was published as
`9850296b1f479b00a6b0e72379b43d9dc04972c6`, the pre-wiring baseline.
Native configuration/engine wiring now selects `spread_pool=batch_cell` only
in `configs/r50/k1_cell_pool/gradpert_b2/nadig_jurkat.yaml`. The default remains
`unique_condition` and follows its original branch unchanged. First-step
diagnostics are recorded separately for the two global views in the one-step
receipt. Gradient norm is unweighted with respect to unique p; the unchanged
0.1 objective weight is also recorded. The diagnostic uses a detached copy,
no RNG, and does not modify training gradients. A native-engine synthetic
repeated-cell test and complete config-diff test pass. Server real-data
one-step acceptance remains pending; no GPU measurement is claimed here.

The matching grad-pert-r50 monitor is paused during this bounded implementation
goal. Existing short-test results and roots stay immutable. Complete tests,
commit/push and new clean source/contract review before closing this goal;
then launch with no active goal and restore the same half-hour monitor,
including remaining registered short tests. No old run is restarted.
