# Exploration: frozen TxPert paper and code audit

## Scope and ownership

- Agent: `txpert_audit`
- Read-only audit of the local paper and frozen official checkout.
- Frozen official commit:
  `08d82eea86746b044cf7531f4ec8c5f60e1cb73f`.
- Paper SHA-256:
  `6eb12c8cf54a4d3d09e64e8b8516b759874dd36308d85a36115b7da41ec8120c`.
- Files changed by subagent: none.

## Verified behavior relevant to the benchmark

- The public architecture sums one or more perturbation latents with a basal
  latent and decodes expression. Its default raw-expression basal encoder is
  `G -> 512 -> 64`; its decoder is `64 -> 512 -> G`, with BatchNorm,
  LeakyReLU, and dropout 0.2 between linears.
- The public unseen-single two-graph config uses STRING and GO, independently
  Top-20-pruned, with a four-layer Exphormer-based encoder, hidden size 128,
  two heads, dropout 0.1, and latent size 64. It is not the paper-best four-
  graph model because two paper graphs are private.
- The paper covers the five selected datasets. The public onboarding path
  applies target-efficiency filtering, total-count normalization to 4000,
  `log1p`, Top-5000 HVG selection, and perturbation-target vocabulary filters.
- Paper splits are condition-level `0.5625/0.1875/0.25`, with GEARS' predefined
  Norman split as the exception. Public onboarding defaults instead produce
  roughly `0.65/0.10/0.25` and can include `ctrl` in the split candidates.
  Therefore the project canonical manifest must be authoritative.
- The public training module computes test metrics during validation epochs.
  An isolated benchmark runner must disable that behavior and enforce val-only
  checkpoint selection followed by one test pass.
- Native TxPert Pearson delta is computed per condition on mean
  `(prediction-control)` versus mean `(truth-control)`, then macro-averaged.
  Native controls and output counts differ from the shared 300-control
  protocol, so common-protocol predictions must be materialized separately.
- The published PKL omits original row IDs, gene order, split/control/config
  hashes, code/checkpoint provenance, and seeds. It cannot serve as the common
  sealed artifact without adaptation.

## Baseline and reproducibility implications

- Implement the common nonlearned floors as `matched_control_mean`,
  `global_train_delta`, and a general train-delta model whose Norman unseen
  doubles compose seen-single deltas.
- Do not include the split-half experimental-accuracy oracle as a deployable
  baseline because it reads test truth.
- The frozen public repository exposes checkpoint inference but no complete
  five-dataset training entry point or full training recipe. Its common-
  protocol result must be labeled as a frozen-code public benchmark with
  project-registered missing hyperparameters, not an exact paper-best
  reproduction.
- The frozen checkout is under a non-commercial license. Keep it isolated,
  preserve attribution, and never copy/rebrand its source into `gradpert`.

