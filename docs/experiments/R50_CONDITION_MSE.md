# R50-C1: batch-condition-balanced expression MSE

Status: authorized and queued, not implemented or launched. Date: 2026-09-12.

User requests expression-only GEARS-style aggregation, gamma=0, no direction
term. This is a native reduction ablation, not an official GEARS runner change
or a claim of full TriShift equivalence.

## Parent and single change

Use the frozen R50 batch512 E3 parent described in R50_GLM5_MUON.md: fixed
LR .001, unchanged AdamW, EMA, architecture, prior, data, sampling and losses
1/.8/.4/.1. Do not inherit G1/G2/G3 modifications. Current native reference
src/gradpert/training/step.py uses F.mse_loss(prediction, target_expression).
Before implementation audit the actual published parent source/config again.

Only replace prediction reduction by:

L_expr = mean over distinct conditions c present in this batch of
         mean over cells i with condition c and original expression genes g of
         (prediction[i,g] - target[i,g])**2.

Gamma is fixed to zero (the user's r=0). No L_dir is added. Keep the original
prediction gene set, not DE-only/top20 or a new condition-specific gene filter.
Keep condition-consistency, masked-node and spread terms and weights unchanged.
Use the full canonical perturbation condition ID (including combinations), not
a single anchor gene; include control as its own condition if present in the
parent's sampled batch. No absent conditions or dataset-wide frequencies enter
the denominator. Preserve parent batch composition and sampling.

This averages squared residuals, NOT expression vectors before squaring.
Cells in rarer batch conditions receive more relative weight than in global
cell-weighted MSE. Therefore this is intentionally not gradient-equivalent to
the parent, except for balanced group sizes or special data.

## Implementation acceptance

Add an explicit config-selected native reduction with backward-compatible
default. No TriShift/GEARS runtime imports. Audit frozen TriShift reference
before making any implementation-alignment claim; the formula above is the
authorized contract even if upstream behavior differs.

Test a hand-calculated unequal-condition-size example, single condition,
equal group counts, cell/condition ordering invariance within numerical
tolerance, combination IDs, optional control, correct gradients and unchanged
auxiliary losses. Add a counterexample proving this is not MSE of group means.
Validate checkpoint/config receipts and resume. Run full gates, clean main/server
publication and a fresh one-epoch smoke before full training. Never alter an
active checkout or reuse historical output roots.

## Execution and reporting

Exactly one new row: R50-C1 / condition_balanced_expr_mse. Run 50 epochs with
validation each epoch, no early stopping, save best.pt and actual last.pt,
automatically test both using identical canonical split/ordered controls/truth
and metric definitions, metrics_only and zero persistent PKL. Record actual
ordered steps from the frozen batch policy. Reuse parent control only with
verified receipts and scientific identity/implementation-equivalence evidence;
otherwise report the need for a control, do not silently add another run.

Queue after current runs, accepted performance work, GLM three rows and R50-R1
through R50-R4 reruns, before remaining R50 stages. The existing hourly
grad-pert-r50 monitor owns preparation and launch after those prerequisites.
No current run is interrupted by this addition.

2026-09-12 update: implementation/config preparation underway. The newer
R50_BUILD_ALL_GATE.md overrides full-run ordering: all authorized code and
smoke gates first. No smoke/full launch has occurred for this row.
