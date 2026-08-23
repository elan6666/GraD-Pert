# Benchmark landscape

Here "competitor" means a comparison surface or engineering precedent, not a
commercial product claim.

| Product | Positioning | Key features | Pricing | Strength | Weakness | Source |
|---|---|---|---|---|---|---|
| TxPert public | Graph-supported perturbation prediction | Basal + graph perturbation latents; STRING/GO public model; condition-level metrics | Research code; non-commercial license | Closest five-dataset method and metric/split precedent | Public two-graph model differs from paper-best; full training recipe absent; validation hook reads test | [Official repo at frozen commit](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f) |
| GEARS | Single-cell single/combinatorial perturbation prediction | GO/co-expression graphs, custom AnnData and split APIs | Open research code | Established Norman/Replogle interface; strong baseline recognition | No direct Nadig adapters; native prediction averages 300 controls; no cross-cell design | [Official repo at frozen commit](https://github.com/snap-stanford/GEARS/tree/f374e43e197b295016d80395d7a54ddb81cc6769) |
| DINO/DINOv2 code | Self-distillation implementation precedent | Multi-view teacher/student, centered targets, EMA, masked-token and KoLeo losses | Apache-2.0 code | Precise, testable update and loss behavior | Image/token assumptions require an independently designed graph mapping | [DINO](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a), [DINOv2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) |
| TriShift local | Reproducible perturbation experiment system precedent | Condition-keyed PKL, offline metric recomputation, notebooks, external adapters | Internal research code | Strong artifact/notebook workflow and multiple metric families | Existing dataset/runners differ; some adapter/payload behavior cannot be reused directly | Local commit `87ac2c51c3c266391093f71a8bce2e6beaa81518` |
| Nonlearned train-delta baselines | Performance floor | Matched control, global train delta, seen-specific/general composition | N/A | Transparent, cheap, detects leakage and weak learned models | Mean-effect outputs do not imply distributional fidelity | [TxPert baseline source](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/gspp/models/baselines.py) |

## GraD-Pert differentiation

- One standalone native package with a strict no-upstream-runtime boundary.
- One canonical split/control/evaluator contract applied to every model.
- Graph self-distillation and expression prediction jointly optimized in the
  single B2 route.
- Truth-free runner artifacts and evaluator-only truth joining.
- Self-contained model-by-dataset configs plus exact local/GitHub/server commit
  parity and server-only large-artifact retention.

