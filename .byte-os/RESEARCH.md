# Research

## Research date

2026-08-24 (Asia/Shanghai)

## Search scope

- GraD-Pert historical design and all 25 discussion documents in `TxPert/`.
- TxPert paper and frozen public repository commit
  [`08d82ee`](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f).
- Official GEARS repository at frozen commit
  [`f374e43`](https://github.com/snap-stanford/GEARS/tree/f374e43e197b295016d80395d7a54ddb81cc6769).
- Official DINO repository at frozen commit
  [`7c446df`](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a)
  and DINOv2 at
  [`7764ea0`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8).
- Local TriShift commit
  `87ac2c51c3c266391093f71a8bce2e6beaa81518` for artifact/notebook and metric
  behavior.
- Current project remote: [elan6666/GraD-Pert](https://github.com/elan6666/GraD-Pert).
- TxPert data publication: [Zenodo record 15420279](https://zenodo.org/records/15420279).

## Key findings

### Product and repository state

- The target GitHub repository exists, is public, and was empty when audited.
  The local project had design/reference files but was not yet a Git repository.
- The authorized server is reachable. It currently exposes two RTX 5090 GPUs,
  each with 32,607 MiB total and about 32,110 MiB free, plus about 1.6 TiB free
  under `/data/yilangliu`. These values are a point-in-time observation and
  must be rechecked before every formal job.
- Large data and model artifacts must remain server-side; source synchronization
  must use an exact Git commit, while local result sync is allowlist-only.

### TxPert evidence

- TxPert's core public code adds one or more graph-derived perturbation latents
  to a basal latent and decodes expression. Public unseen-single configs use
  batch size 64; the training module defaults to AdamW, LR 1e-3, and weight
  decay 0.
- The public two-graph model uses STRING and GO, but the paper-best model uses
  two additional private graphs. The public result cannot be described as an
  exact reproduction of the paper-best four-graph result.
- The paper specifies condition-level 0.5625/0.1875/0.25 train/validation/test
  proportions, while the public onboarding helper defaults to approximately
  0.65/0.10/0.25 and can include control in candidate conditions. A project-
  canonical manifest must override model-native splitting.
- The public validation hook also evaluates test each epoch. The isolated
  runner must remove test access from training and evaluate test once only.
- Public code supplies inference/checkpoint behavior but not a complete five-
  dataset training entry point or all paper hyperparameters. Missing values
  must be preregistered and labeled as project choices.
- TxPert's prediction outputs provide useful expression/control/truth fields
  but omit original row IDs, gene order, data/split/control/config hashes, and
  full seed/commit provenance.

### GEARS evidence

- Official GEARS directly exposes Replogle K562 essential, Replogle RPE1
  essential, and Norman. Nadig Jurkat/HepG2 require the official custom AnnData
  path with exact `condition`, `cell_type`, and `gene_name` semantics.
- The official repository's documented defaults are train/test batch 32/128,
  hidden size 64, 20 epochs, and its code defaults to Adam, LR 1e-3, weight
  decay 5e-4. This project overrides only the run budget to max 200 epochs plus
  common validation-only patience 10.
- Official `.predict()` forms control-derived graphs but averages outputs to a
  single population vector. The benchmark runner must forward the exact shared
  300 control row IDs and retain all 300 predicted rows.
- GEARS custom splits accept explicit train/val/test condition lists. Adapter
  output must prove those exact lists were consumed after graph coverage
  filtering.
- Existing TriShift GEARS code is useful evidence but is not a direct runner:
  it covers different datasets, monkeypatches behavior, mixes truth with model
  output, and contains a payload sampling argument that does not always subset.

### Teacher/student behavior evidence

- [DINO's official training loop](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py)
  initializes teacher from student, disables teacher gradients, sends only two
  global views to teacher, applies student optimizer step first, then performs
  the no-grad EMA update on the teacher. Teacher momentum follows a cosine
  schedule from 0.996 to 1.
- The official head is a three-layer MLP with hidden size 2048, bottleneck 256,
  GELU, bottleneck normalization, and a weight-normalized final prototype
  projection. The active GraD-Pert capacity remains subject to server fit.
- DINO's centered teacher targets use student temperature 0.1, teacher
  temperature 0.04 by current default, and center momentum 0.9.
- [DINOv2 masked-token loss](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/ibot_patch_loss.py)
  weights masked nodes per sample and keeps a separate center.
- [DINOv2 KoLeo](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/koleo_loss.py)
  L2-normalizes embeddings, finds nearest neighbors by off-diagonal inner
  product, and minimizes negative log distance.

### Five-dataset and evaluation evidence

- The five active datasets are Replogle K562 essential, Replogle RPE1
  essential, Nadig Jurkat, Nadig HepG2, and Norman.
- K562 has an upstream processed single-cell-line cache. RPE1/Jurkat/HepG2 are
  present in a cross-cell cache, but independent within-cell Top-5000-HVG views
  still require source-level reconstruction or separately verified assets.
- Norman uses the GEARS-targeted processed data and frozen doubles protocol.
- A runner must emit prediction-only condition bundles; the evaluator joins
  all truth cells after sealing. This prevents accidental test-truth access.
- Fairness requires one split manifest and one ordered 300-control manifest per
  dataset/condition. Equal seeds without equal materialized IDs are insufficient.
- Three Pearson metrics remain separate: TxPert macro Pearson delta, TriShift
  DE-masked Pearson delta, and Systema Top-DE Pearson against the train+val
  non-control centroid reference.

## User complaint patterns

- "Same seed" often hides different actual condition lists after loader-side
  filtering or model-native splitting.
- Model-specific evaluation silently changes the control baseline, DE mask,
  macro weighting, NaN behavior, or number of predictions.
- Published repositories frequently omit exact training entry points or use
  test during validation; wrappers can accidentally claim paper reproduction.
- A file named `processed` is often treated as canonical without checking
  scale, gene order, source, license, or task-specific HVG semantics.
- Notebook-only metrics and result files without hashes make later figure or
  downstream reproduction unreliable.
- Large artifacts copied to laptops waste storage and separate results from the
  exact server environment that generated them.

## Product opportunities

- Make fairness executable: schema validation fails on any split, gene, control,
  or evaluator hash mismatch before GPU work.
- Treat truth access as a lifecycle boundary between runner and evaluator.
- Use a self-contained model-by-dataset config matrix so every result can be
  traced without resolving a hidden global defaults tree.
- Make server-only artifacts still usable through small metric/receipt files
  and local server-artifact pointers.
- Provide one command for five-dataset preparation and a readiness report that
  says unavailable/blocked instead of guessing preprocessing semantics.
- Recompute metrics from sealed server artifacts without retraining, and keep
  notebooks as read-only analytical consumers.

## Risks and caveats

- Public data URLs/checksums for independent RPE1, Jurkat, and HepG2 within-cell
  artifacts are not yet fully resolved. Data preparation must block rather than
  substitute the cross-cell gene intersection.
- TxPert lacks a complete public five-dataset training recipe. A fair public-
  code benchmark is possible; an exact paper-best reproduction is not currently
  supportable.
- The projection head and ten graph views can exceed 32 GB depending on node/
  batch size. The one-time capacity fit gate is required.
- Static GO/STRING coverage may exclude perturbation targets, especially in
  Nadig datasets. Every requested/retained/dropped condition must be reported.
- GEARS, TxPert, and GraD-Pert have different native training-control semantics.
  Fairness applies to information access, split, prediction inputs, and common
  evaluation; it must not silently rewrite the learned method.
- PKL is unsafe for untrusted inputs. Only checksum-verified project artifacts
  may be loaded.

## Recommended impact on roadmap

1. Freeze the active v1 spec and provenance registry before product code.
2. Build schemas, config matrix, split/control manifests, artifacts, and metric
   golden tests before learned models.
3. Build and locally validate the standalone graph/self-distillation model on
   synthetic data.
4. Add server preflight/sync, then resolve/download and QC five datasets.
5. Fit projection capacity, run learned models with max 200/patience 10, seal
   prediction-only artifacts, evaluate once, and sync only small summaries.

