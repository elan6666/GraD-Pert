# Official external models on canonical Nadig Jurkat

Status: integration in progress; no new external training has started.

## Authorization and design

The 2026-09-07 user request authorizes Scouter, TxPert and GEARS for at most
100 epochs with validation-only early stopping after 10 non-improving epochs.
This supersedes the old external smoke-only execution restriction for these
three Jurkat runs only. Each new integration must pass an exactly one-epoch
server smoke before full training. Existing evidence is immutable.

Use the current canonical Nadig Jurkat condition split, GEARS representability
intersection, expression axis and ordered 300-control evaluation manifests.
No canonical test expression enters fitting, validation or model selection.
Evaluate the best checkpoint once with the common three metrics; metrics_only,
zero persistent PKL. No local scientific materialization.

The user's subsequent concurrency request is two experiments per GPU, four
total including the existing A3. Add processes incrementally after independent
integration checks. Measure per-process memory and throughput; do not change
official batch sizes or scientific settings to force concurrency. Preserve A3
and unrelated processes. Fall back to a queue if the paired capacity check
fails. The user-paused periodic monitor remains paused.

## Frozen official profiles

| Run | Upstream | Profile / optimizer | LR | Train batch | Scheduler |
| --- | --- | --- | --- | --- | --- |
| Scouter + GenePT-Seed | PancakeZoy/scouter @ 0cfddd000e19b72ff033ba67c8315f7bc3304932 | Replogle K562/RPE1; Adam | 0.001 | 256 | ExponentialLR gamma=0.9 |
| TxPert public | valence-labs/TxPert @ 08d82eea86746b044cf7531f4ec8c5f60e1cb73f | public Exphormer-MG; AdamW, wd=0 | 0.001 | 64 | none |
| GEARS | snap-stanford/GEARS @ f374e43e197b295016d80395d7a54ddb81cc6769 | official default; Adam, wd=0.0005 | 0.001 | 32 | StepLR step=1, gamma=0.5 |

Scouter profile is verified against the paper Methods and official
`scouter_misc/code/Scouter/run_scouter_k562.py`. Encoder 2048/512/64,
generator hidden 2048, SELU, BatchNorm, no LayerNorm/dropout, gradient norm
clip 1, loss gamma=0/lambda=0.5. Keep its official min_delta=0.001 while
overriding max epochs 40 -> 100 and patience 5 -> 10. Shared run seed1 is a
project pairing override (the reproduction script seeds model RNG at24).

Scouter's original frozen 1536-dimensional GenePT is replaced, as requested,
by the existing 17730 x 2048 GenePT-Seed Protein+Reactome+SIGNOR artifact:
SHA256 `34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.
Its official model derives input width from the embedding. This is a modified
prior baseline, not an unchanged Scouter result. Fail if any canonical
perturbation target is missing; unrelated missing gene embeddings do not
remove expression columns. Control embedding is zero, as in official script.

## Source-audited adapter requirements

- Scouter `train` stores a shallow `network.state_dict()` at improvements.
  The adapter must snapshot independently to prevent later training from
  mutating the selected best weights, with a regression test and receipt.
- TxPert `on_validation_epoch_end` calls both validation and test loaders.
  The full-run adapter must suppress the test call, retain official validation
  computation, and never pass canonical test truth to its fit data module.
- GEARS owns Adam/StepLR inside `train`; never call `train(epochs=1)` repeatedly
  to simulate a full run. Add a bounded validation-only stopping hook without
  resetting official optimizer/scheduler or modifying the frozen checkout.
- Official package execution stays in isolated `benchmarks/` processes; no
  upstream model imports in native GraD-Pert modeling. The common CLI dispatch
  is process-only, with explicit environment and checkout paths.

References: https://github.com/PancakeZoy/scouter ;
https://github.com/PancakeZoy/scouter_misc ;
https://www.nature.com/articles/s43588-025-00912-8 .
