# Exploration: GraD-Pert design audit

## Scope and ownership

- Agent: `design_audit`
- Read-only audit of all 25 design Markdown files under `TxPert/grad-pert`, the
  three top-level study notes, `.byte-os/DISCUSSION.md`, and the frozen split
  helper.
- Files changed by subagent: none.

## Verified MVP contract

- `V_master = post-processing HVGs union known candidate targets`; candidates
  are the only non-HVG nodes forced into the graph.
- STRING and GO are independently Top-20-pruned. Edge weights select edges but
  are not passed as attention values in v1.
- The single backbone route is per-graph propagation followed by node-adaptive
  fusion. Teacher and student share the graph/projection architecture; Teacher
  contains no basal encoder, decoder, control input, or truth expression.
- Teacher receives two clean global views. Student receives the corresponding
  two globals plus eight 512-node Local-RingInduced views. One student global
  has node masking; exactly four local views mask all active anchors.
- The shared perturbation/node projection head is
  `d_pert -> 2048 -> 2048 -> 256 -> K_head`, with separate centers for the two
  objectives. `K_head` is chosen by server memory fit once, not performance.
- The prediction path is a deterministic full Top-20 graph view, with summed
  condition anchors and D0 additive basal-plus-perturbation decoding.
- Only B2 is active: prediction MSE plus weighted perturbation consistency,
  masked-node consistency, and embedding-spread regularization. B3 and all
  ablations are deferred.
- Four within-cell unseen-single protocols plus Norman `combo_seen2`; one
  canonical condition split per dataset/protocol, four paired run seeds, and a
  condition-equal shared evaluator.
- Inference is 300 shared controls with replacement per condition, while truth
  remains all real test cells; no row-level prediction/truth pairing exists.

## Conflicts to exclude

- Historical fanout/local-budget, backbone, graph-weight, prototype-capacity,
  temperature, B3, shuffled-graph, target-inductive, decoder, supervised-only,
  cross-cell, and extra-model adapter experiments are not part of v1.
- An older note proposed changing the upstream package and using asymmetric
  Top-100/Top-20 graphs. It conflicts with the independent package and current
  symmetric Top-20 decision.

## Open implementation gates

- The legacy design did not completely freeze graph-layer dimensions, exact
  fusion mechanics, basal/decoder operator order, optimizer, clipping, or AMP.
  These must be resolved from frozen source or explicitly designed and
  registered; they cannot be inferred from names.
- Split helper math was verified: its internal second-stage fraction is
  `val_size / (1 - test_size)`. Passing absolute `val_size=0.1875` yields the
  intended `0.5625/0.1875/0.25` condition proportions.
- Separate a truth-free runner `PredictionArtifact` from evaluator-created
  `EvaluationBundle`; this preserves the test-truth access boundary.
- Define nonlearned population outputs explicitly; do not duplicate a mean
  merely to claim distributional support.

