# GraD-Pert v1 Active Design

This is the authoritative implementation specification for v1. Historical
documents under `TxPert/` contain explored alternatives; they do not expand the
scope below.

## 1. Scope

- One native model only: B2, jointly trained from random initialization.
- Five datasets: Replogle K562 essential, Replogle RPE1 essential, Nadig
  Jurkat, Nadig HepG2, and Norman.
- Learned comparisons: GEARS and the frozen public TxPert implementation.
- Nonlearned comparisons: matched control, global train delta, and general
  train delta with additive single effects for unseen Norman doubles.
- No ablations, B3, cross-cell task, target-inductive variant, decoder variant,
  or extra learned model.

## 2. Canonical nodes and graphs

For dataset `d`:

```text
V_master(d) = ordered(HVG_d union known_candidate_targets_d)
```

- Ordinary context is restricted to the finalized dataset HVGs.
- A known candidate perturbation target is the only allowed forced non-HVG.
- The expression input/output axis remains the ordered HVGs; the graph axis is
  `V_master`. Forced non-HVG candidate targets do not become decoder outputs.
- Every node has one stable integer ID and one canonical gene ID.
- Source graphs are STRING and GO. Each source is independently filtered to
  nodes in `V_master`, deduplicated deterministically, and independently pruned
  to the top 20 incoming edges per target. Ties use source gene ID then target
  gene ID. Add exactly one self-loop per node after DropEdge.
- Source weights are used for ranking only. GAT receives no numeric edge value.

## 3. Native graph encoder

`AdaptiveGeneGraphEncoder` is the native GraD-Pert component name.

- Shared learned gene embedding: `[N,128]`, truncated-normal initialization
  with standard deviation 0.02.
- Each source graph has an independent four-layer GATv2 tower.
- Each layer: two heads, 128 output features per head, concatenation, learned
  skip concatenation, LeakyReLU, dropout 0.1. The next layer accepts the full
  concatenated main+skip width. Self-loops are already in the view and are not
  added inside the operator.
- Each tower projects its final state to `[N,64]`.
- Node-adaptive relation fusion follows a fully specified one-head channel
  attention: apply LeakyReLU to each source state; score each source/node with
  its learned `[1,64]` query; softmax across the two sources for each node; take
  the weighted sum. Output is `[N,64]`.
- Teacher and student are isomorphic. Teacher is initialized byte-identically
  from student and never receives gradients.

This fixes the previously underspecified GAT-Hybrid name. The numerical choice
is project design with the graph-tower shapes checked against frozen public
code; it is not represented as an upstream class.

## 4. Views and masks

### Globals

- Two independent global views keep all `V_master` nodes and both graphs.
- Apply independent DropEdge 0.1 per source/view to non-self edges.
- Preserve all active anchors and their self-loops.
- Teacher sees both clean-node-input global views.
- Student sees the same two graph topologies; exactly one randomly selected
  global replaces eligible node input embeddings with one shared learned mask
  token. Mask ratio is sampled uniformly from 0.1 through 0.5.
- Active anchors, nodes having only a self-loop, and held-out targets are not
  eligible for global node masking.

### Locals

- Build eight independent Local-RingInduced views per unique active condition.
- Multi-source neighborhood rings expand inward from all active anchors over
  the union of the pruned STRING and GO graphs.
- Include complete inner rings. If the next boundary would exceed 512 nodes,
  sample only that boundary with a stable view RNG; always retain all anchors.
- For each source, induce every base Top-20 edge whose endpoints are selected.
- Exactly four of eight locals replace all eligible active-anchor input
  embeddings with the shared mask token. Nodes/edges remain. An anchor having
  only self-loops is not masked and emits a structured warning.

## 5. Representations and heads

- A condition perturbation representation is the sum of its active anchor node
  states; no global mean pooling is added.
- Prediction path uses a separate deterministic full Top-20 graph view: no
  DropEdge, node mask, anchor mask, local sampling, or projection head.
- `ConsistencyProjector`: `64 -> 2048 -> 2048 -> 256 -> K_head` with GELU after
  the first two linears, no hidden BatchNorm, L2-normalized 256 bottleneck, and
  weight-normalized bias-free final linear.
- The condition and masked-node objectives share this projector but keep
  independent centers. `K_head` is not hard-coded until server fit; candidate
  order is 65536, 32768, 16384, 8192. Select the first whose 128 consecutive
  real train steps on every dataset keep peak reserved memory at or below 85%
  of initially observed usable memory, then freeze it for every dataset and
  seed. A one-step probe is insufficient because CUDA reservation can rise
  during sustained training.
- The completed development fit on the server RTX 5090 selected
  `K_head=16384`. The 65,536 candidate failed on K562 and 32,768 failed on
  Jurkat; 16,384 completed all 128 steps on all five datasets. The decision
  summary is `registry/capacity/prototype_head.development.json`; the exact
  per-step receipt remains server-side pending allowlisted synchronization.

## 6. Basal prediction

- `BasalStateEncoder`: `G -> 512 -> 64`, BatchNorm, LeakyReLU, dropout 0.2
  between the two linears.
- `ExpressionDecoder`: `64 -> 512 -> G`, BatchNorm, LeakyReLU, dropout 0.2
  between the two linears.
- For active anchors `P`:

```text
z_pert = sum(prediction_view_node_state[p] for p in P)
prediction = ExpressionDecoder(BasalStateEncoder(control) + z_pert)
```

- Each perturbed training cell is paired each epoch with one training control
  from the same cell line and experimental batch, sampled by a named RNG.

## 7. Losses

```text
L_ssl = L_condition_consistency + L_masked_node + 0.1 * L_spread
L_total = L_prediction + 0.1 * L_ssl
```

- `L_prediction`: mean squared error over cells and output genes.
- Teacher condition targets: center, divide by teacher temperature 0.04, softmax.
- Student condition log probabilities: divide by 0.1, log-softmax.
- For each of two teacher globals, compare to all ten student views except the
  student global with the same index: 9 + 9 = 18 equally weighted terms.
- `L_masked_node`: only nodes masked in the single masked global contribute;
  use the clean teacher state on the matching topology, separate node center,
  and average uniformly over the masked nodes in that one batch-shared global
  view. This follows the original graph design's `1 / |M|` objective; the
  condition axis belongs to the perturbation consistency loss, not this node
  loss.
- `L_spread`: unique-condition pre-projector student global representations are
  L2-normalized; find each off-diagonal nearest neighbor by maximum dot product
  and average negative log Euclidean distance. If fewer than two unique
  conditions exist, mark it unavailable and contribute zero.
- Center momentum is 0.9. Update centers once from detached teacher logits after
  loss targets for the step have been formed.

## 8. Update order and gradients

Each step uses this order:

1. Build deterministic prediction view and stochastic SSL views.
2. Teacher no-grad forward on two clean globals.
3. Student forward on prediction, two globals, and eight locals.
4. Compute all losses against pre-update centers.
5. Backward and update only student graph/projector, basal encoder, decoder, and
   mask token.
6. Update Teacher parameters by EMA from the post-optimizer student parameters.
7. Update the two centers from detached current-step teacher logits.
8. Log loss components, schedules, center entropy/norm, prototype usage, and
   gradient norms/ratios.

Teacher momentum follows a step-level cosine from 0.996 to 1 over the maximum
100-epoch schedule; resumed runs restore global step and all state exactly.

Gradient ownership:

- graph embedding/towers/fusion: prediction and SSL;
- basal encoder/decoder: prediction only;
- projector/mask token: SSL only;
- teacher/centers: no optimizer gradients.

## 9. Optimization and selection

- Default native optimizer: AdamW, LR 1e-3, weight decay 0. The user-locked
  native train/evaluation batch size is 256 in every dataset config.
- Native batches use a deterministic condition-limited order: at most eight
  unique perturbation conditions per 256-cell batch, with every condition's
  cells independently reshuffled each epoch. This keeps the graph objective at
  the condition statistical unit, preserves multi-condition spread terms, and
  bounds the eight condition-specific local-view forwards. Tail batches of
  2--255 cells are retained so conditions with few
  remaining cells are not discarded; only an unavoidable one-cell tail is
  dropped because both prediction MLPs contain BatchNorm. The cap, ordering and
  singleton policy are explicit `project_preregistered` values in every
  GraD-Pert dataset config; they are not tuned by dataset or test results.
- Maximum 100 epochs. Validate every epoch using the fixed validation
  300-control manifest and `txpert_macro_pearson_delta`.
- Stop after 10 consecutive validations without a strict improvement. Save best
  and last. Test the sealed best checkpoint once.
- Four run seeds: 1, 2, 3, 4. Split seed: 42. Evaluation seed: 20260824.
- If batch 256 cannot fit even with `K_head=8192`, this is a blocked capacity
  gate requiring a preregistered config change; do not silently reduce batch.

## 10. Required implementation evidence

- Shape and graph-view golden tests, 18-pair test, mask-only gradient test,
  teacher no-grad test, post-step EMA test, center-order test, resume equivalence,
  deterministic prediction-view test, and full synthetic train/predict/evaluate.
- Static test for forbidden native imports and upstream-named native classes.
- Server fit receipt selecting one `K_head` globally.
- One training and evaluation manifest per learned model/dataset/seed.
