# GraD-Pert B2-vNext and ablation contract

Status: active design for the 2026-08-28 config-driven ablation program.
Historical v1 runs and `docs/design/GRADPERT_V1.md` remain immutable evidence.

## 1. Product boundary

The native package remains `gradpert`. Every variant uses the existing native
CLI and one training/evaluation lifecycle. A variant is selected only through
one self-contained config; no ablation-specific main function is allowed.

Native source must not import or call TxPert, GEARS, TriShift, or another
upstream checkout. Public reference code is read-only evidence used to define
source-derived native behavior and frozen synthetic golden tests.

## 2. Default reference A0

- Dataset-independent expression/input/output/evaluation axis: frozen ordered
  Top-5000 genes.
- Runtime graph axis: ordered Top-512 HVGs union every representable canonical
  perturbation target. The actual graph size is therefore greater than 512.
- Graph sources: ordered STRING then GO, each independently pruned to Top-20
  incoming non-self edges before self-loop materialization.
- Graph encoder: native multi-source sparse graph Transformer aligned only to
  the public TxPert STRING+GO Exphormer-MG graph-encoder behavior.
- Global views: two complete runtime-graph views; source-specific DropEdge 0.1;
  exactly one Student global receives the existing eligible-node mask.
- Local views: four inbound four-layer RingInduced views. Their node cap is
  the exact integer floor of one half of the actual runtime graph size after
  target union; all active anchors are retained. The sealed 2,809-node H=512
  graph therefore resolves to a cap of 1,404 nodes.
- Local anchor masking: disabled (`0/4`). This does not disable the global
  masked-node objective.
- Prediction view: deterministic complete runtime graph, without DropEdge,
  view masks, local sampling, or projection head.
- Decoder: existing additive control/condition decoder.
- Direct loss weights: prediction `1.0`, condition consistency `0.8`, masked
  node `0.4`, spread `0.1`.
- Systems: all seven semantics-preserving groups.
- Artifacts: `metrics_only`, zero persistent PKL.

## 3. Strict native options

The config must resolve and receipt one immutable native architecture object:

```yaml
graph_axis_policy: recomputed_hvg_union_candidate_targets
graph_hvg_count: 512
graph_mode: multi
graph_sources: string_go
graph_encoder_family: multi_source_sparse_transformer
string_weight_mode: selection_only
local_view_builder: ring_induced
local_view_count: 4
local_view_node_budget_ratio: 1/2
local_view_fanout: 20_10_5_5
local_anchor_mask_view_ratio: 0/1
gene_feature_mode: learned_id
decoder_mode: additive
```

Unsupported combinations fail before model construction. Student and Teacher
must resolve the same encoder, graph sources, node features, and output width.

## 4. Graph-axis and GenePT coverage

HVG ranking mirrors the frozen TxPert within-cell preprocessing order. After
the registered weak-perturbation-signal filter, the complete cell-line data are
used before condition splitting, including conditions that will later belong to
train, validation, and test. Every cell is normalized with
`normalize_total(target_sum=4000)`, followed by `log1p` and Scanpy's Seurat
`highly_variable_genes(n_top_genes=H, subset=True)` for registered
`H in {512, 1024, 2048, 5000}`. The exact selected gene order and
normalized-dispersion ranking are frozen in a hash-bound receipt and verified
on every graph load. Every H row uses the same recomputed-HVG-plus-target
policy; H=5000 is not a shortcut to `canonical_full`. Expression inputs,
outputs, and evaluation remain on the frozen Top-5000 axis.

Every vNext GenePT E row is locked to the GenePT-Seed
`Protein+Reactome+SIGNOR` condition. GenePT-Seed's artifact filename and
historical condition label are `Seed-GO-ProteinPathway`; these are two names
for the same bytes, not two priors:

- server path
  `/data/yilangliu/GenePT-Seed/data/embeddings/seed-go-protein-pathway-master-aligned.npz`;
- exact SHA-256
  `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`;
- NPZ schema with exactly `genes`, `vectors`, and `model`, where `model` is
  exactly `doubao-embedding-vision`, the source axis has 17,730 unique
  exact-case labels, vector width is 2,048, all values are finite, and there
  are zero all-zero vectors;
- the upstream corpus profile is exactly `protein-pathway`: the Protein layer
  plus human Reactome pathway names and direct human SIGNOR causal relations
  with real partner symbols; no HPA, masked-SIGNOR, shuffled-SIGNOR,
  Reactome-only, or SIGNOR-only artifact is substituted;
- the ordered 2,809-node `hvg512_plus_targets` runtime axis is selected from
  the sealed superset by exact case-sensitive label and retained in runtime
  order; source genes outside that axis are ignored and their ordered count
  and hash are receipted;
- duplicate source labels, any missing train, validation, or test perturbation
  target, a wrong hash/model/width, or a non-finite/zero source row aborts
  before tensor or model construction;
- a missing non-perturbation runtime gene is omitted while preserving the
  requested canonical order; its ordered count/hash and the retained-axis
  order/matrix hashes are receipted. The frozen `Protein+Reactome+SIGNOR`
  artifact currently covers all 2,809 runtime genes, so no omission occurs in
  the registered coordinate;
- no aliasing, case folding, random fill, zero fill, unreceipted graph-gene
  removal, or silent condition removal is allowed; `ctrl` is not a graph node;
- receipts bind source artifact identity and order, selected runtime order and
  matrix hash, ignored-source order hash, complete target coverage, and the
  unchanged STRING/GO topology hashes.

Unlike the retired `emb_b` filtered-axis proposal, the registered E rows use
the same unfiltered `hvg512_plus_targets` graph as A0 because the selected prior
has full runtime coverage. The prior is a sealed superset; only a future,
receipted missing non-target could remove a runtime gene.

GenePT modes are distinct: frozen projected input, projected input plus learned
ID residual, trainable embedding initialized from a deterministic GenePT
projection, and shuffled-mapping negative control. They never share an
ambiguous boolean switch.

## 5. Public-code alignment boundary

The first supported encoder family is limited to public, auditable surfaces:

- single STRING native GATv2;
- single STRING native sparse graph Transformer;
- STRING+GO native multi-source sparse graph Transformer (A0);
- a separately identified native adaptive per-source GAT fusion only when its
  source contract and project-preregistered hyperparameters are explicit.

The public TxPert checkout exposes reproducible configs only for GAT,
Exphormer, and Exphormer-MG. GAT-MLG conflicts with the documented supra-graph
formula, Hybrid-BMP is absent, and GAT-Hybrid lacks a frozen public config.
Those names are blocked from claimed paper parity until exact official code,
configuration, license boundary, and golden outputs exist.

The public Exphormer-MG example uses STRING+GO. It must not be described as the
paper-best private four-KG model.

## 6. Decoder alignment

The control-conditioned Transformer ablation uses two ordered tokens:
`[control_latent, condition_latent]`, model width 64, four heads, one pre-norm
Transformer encoder layer, GELU, FFN width 256, dropout 0, concat readout, and
an explicit `128 -> 64` projection before the existing expression decoder.
It is a GraD-Pert shift-block alignment, not a complete TriShift generator
reproduction.

## 7. Frozen one-dataset protocol

- Dataset: `nadig_jurkat`.
- Split: the existing frozen canonical split; no new split is generated.
- Run seed: 1.
- Training: exactly 10 epochs, validation after every epoch, no early stop.
- Batch: 256; prototypes: 16,384; allocator:
  `expandable_segments:True`.
- Model selection: validation only. The complete matrix is frozen before any
  result is inspected. Test is opened once per preregistered completed run and
  never used to add or retune later variants.
- Evaluation: same ordered 300-control and truth-row IDs and exact three
  headline metrics.
- Artifact mode: `metrics_only`, zero persistent PKL.

## 8. Ablation matrix

Every row differs from A0 in only the named factor unless labeled interaction.

### Local construction, coverage, count, and masking

- A0: RingInduced, node ratio `1/2`, four locals, mask-view ratio `0/1`.
- L1: Fanout with all other A0 factors unchanged.
- L2: eight local views with all other A0 factors unchanged.
- L3: local node ratio `1/4` with all other A0 factors unchanged.
- L4: local anchor mask-view ratio `1/2` (`2/4`).
- L5: local anchor mask-view ratio `1/4` (`1/4`).

Every L row is constructed directly from A0. The node cap is
`floor(actual_graph_nodes * ratio)` and the mask count is
`local_view_count * ratio`; the latter must be integral or config resolution
fails. Fixed node budgets and fixed mask counts are not successor matrix
factors.

### Encoder and source

- M1: single STRING GATv2, W0.
- M2: single STRING sparse graph Transformer, W0.
- M3: multi-source sparse graph Transformer, STRING+GO (A0).
- M4: native adaptive per-source GAT fusion, STRING+GO, with explicit
  project-preregistered config and no paper-parity claim.

### STRING numerical weights, single STRING GATv2 only

- W0: selection only.
- W1: normalized weight as edge feature.
- W2: fixed normalized STRING prior.
- W3: prior logit plus learned attention residual, fixed lambda/tau 1.
- WS: W1 with weights shuffled over the retained topology.

### Decoder

- D0: additive (A0).
- D1: parameter-matched MLP control.
- D2: two-token control-conditioned Transformer.

### Node features

- E0: learned ID (A0).
- E1: frozen GenePT projection.
- E2: GenePT projection plus learned ID residual.
- E3: trainable embedding initialized from GenePT.
- ES: shuffled GenePT mapping.

GenePT rows are skipped as formally unavailable if the authoritative server
preflight finds any missing perturbation target.

### Objective

- O0: `1.0/0.8/0.4/0.1` (A0).
- O1: condition consistency weight 0.
- O2: masked-node weight 0.
- O3: spread weight 0.

### Shared global graph scale

- H0/A0: HVG512 plus all representable targets.
- H1: HVG1024 plus the same target universe.
- H2: HVG2048 plus the same target universe.
- H3: HVG5000 plus the same target universe.

Teacher and Student global views share the same ordered axis at each H. The
local node ratio remains `1/2`, so its effective integer cap changes only as a
declared derived consequence of H. H rows are not described as fixed node
counts because target union enlarges the runtime graph.

## 9. Required receipts

Each run records resolved native options, graph and feature hashes, actual local
node/edge counts, budget ratio, per-source edge counts, local Jaccard summary,
parameters, peak GPU memory, training/inference wall time, validation history,
checkpoint hash, no-test-fit proof, one-test-evaluation proof, zero-PKL scan,
and exact canonical/split/control/truth identities.

## 10. Non-goals

- No new main function or duplicated trainer.
- No full Cartesian sweep.
- No cross-cell task, extra dataset, B3 route, or full-run policy change.
- No upstream runtime dependency or source copying.
- No aliases to force GenePT target coverage.
- No performance or equivalence claim before completed evidence.
