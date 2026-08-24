# Reference Alignment Registry

This registry documents externally verified behavior. It is not an import map
and does not authorize copying/rebranding source. Native GraD-Pert code is an
independent implementation.

## Frozen references

| Reference ID | Repository or artifact | Commit / checksum | License boundary |
|---|---|---|---|
| `ref_txpert_public` | `https://github.com/valence-labs/TxPert` | `08d82eea86746b044cf7531f4ec8c5f60e1cb73f` | Recursion non-commercial EULA; isolated benchmark/evidence only |
| `ref_dino` | `https://github.com/facebookresearch/dino` | `7c446df5b9f45747937fb0d72314eb9f7b66930a` | Apache-2.0; behavioral reference |
| `ref_dinov2` | `https://github.com/facebookresearch/dinov2` | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache-2.0; behavioral reference |
| `ref_gears` | `https://github.com/snap-stanford/GEARS` | `f374e43e197b295016d80395d7a54ddb81cc6769` | preserve upstream license/citation; isolated benchmark |
| `ref_trishift_local` | `/Users/elan/code/trishift` | `87ac2c51c3c266391093f71a8bce2e6beaa81518` | internal workflow/metric reference |
| `ref_txpert_paper` | `/Users/elan/Documents/论文/02_TxPert_Nature_Biotechnology_2026.pdf` | SHA-256 `6eb12c8cf54a4d3d09e64e8b8516b759874dd36308d85a36115b7da41ec8120c` | cite paper; do not redistribute PDF |

TriShift experiment-architecture adoption/rejection decisions are recorded in
`docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md`. In particular, its thin
entry/core/result-adapter layering is retained, while global config merging and
runtime mutation of official learned models are rejected.

## Alignment entries

| Local behavior | Frozen source file/symbol | Verified contract | Intentional GraD-Pert difference | Planned verification |
|---|---|---|---|---|
| Basal encoder and additive decoder | `ref_txpert_public:gspp/models/basal_state_models/mlp.py:MLP`, `gspp/models/txpert.py:TxPert.forward` | `G→512→64` basal MLP; perturbation latents sum; decoder `64→512→G` | Native names; graph embedding comes from active B2 graph student; no upstream import | shape, additivity, gradient-route tests |
| Per-graph Top-K | `ref_txpert_public:gspp/data/graphmodule.py` | Top incoming edges selected separately per source graph | Node universe is HVG plus known candidate targets | deterministic graph golden test |
| Condition split | `ref_txpert_public:data/preprocessing_utils.py:define_splits_singles` plus paper Methods | condition grouping; paper target 0.5625/0.1875/0.25 | control excluded before split; one materialized manifest shared by all models | exact set/hash tests |
| Macro Pearson delta | `ref_txpert_public:gspp/metrics.py:compute_pert_metrics` | mean prediction/control/truth deltas per condition, then arithmetic macro | common 300-control input; undefined correlation remains unavailable instead of forced zero | numeric golden vectors |
| Teacher lifecycle and view routing | `ref_dino:main_dino.py:train_dino`, `train_one_epoch` | teacher copied from student, no grad, teacher globals only; optimizer step then EMA; cosine 0.996→1 | graph views and active-condition anchors replace image crops | init/grad/update-order tests |
| Centered consistency | `ref_dino:main_dino.py:DINOLoss` | student temp 0.1; teacher temp 0.04; center momentum 0.9; cross-view exclusion | perturbation condition representation replaces CLS token | 18-pair loss golden test |
| Shared projection head mechanics | `ref_dinov2:dinov2/layers/dino_head.py:DINOHead` | 3-layer 2048/2048/256 MLP, GELU, L2 bottleneck, weight-normalized output | GraD-Pert component naming; `K_head` selected by GPU fit | initialization/shape/norm tests |
| Masked-node consistency | `ref_dinov2:dinov2/loss/ibot_patch_loss.py:iBOTPatchLoss` | centered teacher targets; student CE on masked positions; per-sample mask normalization | graph nodes replace patch tokens; the single masked sample is the batch-shared global view and is normalized by its masked-node count | mask-only gradient and center tests |
| Embedding spread | `ref_dinov2:dinov2/loss/koleo_loss.py:KoLeoLoss` | L2 normalize, off-diagonal nearest neighbor, negative log distance | use unique-condition pre-head global student embeddings | duplicate/unique-condition golden tests |
| GEARS training defaults | `ref_gears:gears/gears.py:GEARS.train` and README | train/test batch 32/128; Adam 1e-3, WD 5e-4; best native checkpoint by val DE MSE | preserve official training behavior and cap the current smoke at exactly one epoch; no GEARS full run yet | isolated-runner config receipt |
| GEARS 300-control forward | `ref_gears:gears/gears.py:GEARS.predict`, `gears/utils.py:create_cell_graph_dataset_for_prediction` | upstream forms 300 graphs but averages predictions | use exact shared row IDs and retain `[300,G]` | adapter golden test on tiny data |
| TriShift Pearson/Systema | `ref_trishift_local:src/trishift/_external_metrics.py`, `TriShift.py:_systema_reference_from_train_val` | DE-masked delta correlation; train+val equal-condition centroid reference for Systema | stable metric IDs and evaluator-only truth join | independent numeric fixtures |
| Condition PKL workflow | `ref_trishift_local:scripts/trishift/analysis/recompute_metrics_from_pkl.py` | condition-keyed payloads enable offline metric recomputation | split prediction-only runner artifact from evaluator bundle; add row IDs/hashes | round-trip/recompute test |

## Unresolved evidence gates

- Public official training hyperparameters for all five TxPert datasets do not
  exist in the frozen repository. Every filled value must be marked
  `project_preregistered`.
- Independent within-cell source URLs, expected sizes, and checksums are frozen
  for RPE1, Jurkat, and HepG2. Their configs remain non-ready until downloads,
  checksums, canonical preprocessing, and QC receipts all pass. Norman still
  requires accepted official-source lineage rather than an unlabeled cache.
- Active GAT-Hybrid fusion is a GraD-Pert design choice, not a claim that the
  same class exists upstream. Its numerical contract belongs in the active
  design and tests.
