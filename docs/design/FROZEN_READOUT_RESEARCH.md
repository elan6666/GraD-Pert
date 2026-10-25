# Frozen representation and nonlinear prediction readout adaptation

This research mode separates a fixed learned representation from its existing task readout. It does not change the GraD-Pert architecture or reinterpret earlier experiments. It is a separate two-epoch exploratory family, not the R50 benchmark protocol.

## Scientific question

A preceding staged run learned condition-sensitive self-distillation targets and improved differential-expression correlations, while complete-expression validation error remained higher than earlier references. Does the existing prediction MLP have recoverable supervised readout capacity on these fixed features?

Frozen-feature evaluation is established in self-supervised learning, including the linear classifiers evaluated by [DINO (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) and its [official evaluation code](https://github.com/facebookresearch/dino/blob/main/eval_linear.py). This experiment borrows only the separation between features and task readout. It uses an existing nonlinear MLP, direct perturbation conditioning, a previously supervised parent and additional supervised updates. It is not DINO linear evaluation or a pure unsupervised representation benchmark.

## Fixed and trainable state

Warm-start from the explicitly sealed final parent checkpoint, preserving both the learned encoder weights and the current prediction MLP. Train only the Student prediction MLP: concatenated response gene token and perturbation embedding, 512→256→1 with GELU and the original Softplus output. Both Linear biases remain enabled. No additional normalization, clipping, alternative output link or new head is introduced. For this architecture, four tensors contain 131,585 trainable parameters.

GenePT projection, graph encoder, Cell and Response encoders, condition injection, all SSL projectors/prototypes, Teacher and centers remain fixed. Their evaluation-mode behavior is used; backbone dropout and randomized relay order are disabled. Teacher EMA and center updates are disabled. The training objective contains only the existing row-mean expression MSE. Inactive SSL terms are not evidence of improved distillation loss.

The optimizer uses the native parameter routing over the trainable MLP only: Muon for its hidden matrix, AdamW for final scalar readout and biases, with fresh moments. Warm-started weights and newly initialized optimizer state are separate, explicit choices. Frozen parameters have no gradients or optimizer entries; all frozen parameter and buffer values must remain unchanged.

## Training and inference protocol

Run exactly two epochs, 300 successful readout updates, global batch320 (80 per GPU, two GPUs, accumulation2), seed1 and the original canonical training split, cap40 selection and permitted expression-gene/query recipe. The native two-epoch endpoint schedule uses maximum LR0.001, minimum LR0.0002, warmup16%, weight decay0. No early stopping, per-epoch validation or best-checkpoint selection is used. This is not a replay of the parent's training dropout/view realizations.

The parent used450 total updates, including300 prediction-supervised updates. The new readout adaptation adds300 MSE updates, yielding750 cumulative updates and600 prediction-supervised updates across the lineage. It is not an equal-compute comparison to the parent or earlier two-epoch references; improvement cannot uniquely establish harmful joint gradients or intrinsically superior features.

Only the final readout checkpoint is evaluated once on the frozen445-condition validation population with the same ordered300 controls/truth,5000 expression genes,18 grouped Pearson measures and prediction MSE. Retain seen4775/unseen225 expression axes and all/DEG finite counts/NA reasons. No test-set expressions, test metrics or validation-driven changes to learning rate, epoch budget or checkpoint are allowed.

## Identity and verification

Use a new run/config/source identity with explicit parent source, manifest, resolved configuration, training-stage receipt and checkpoint hashes. The parent need only satisfy its declared training/research terminal: its deferred tests must not be relabeled complete, and generic full-state continuation rules remain unchanged. Parent and child model architectures and immutable data/selection/gene/split identities must match.

A short native engineering probe starts from the parent independently of formal fitting. It verifies nonzero-LR prediction updates restricted to the four MLP tensors, unchanged frozen state and Teacher/centers, native output parity, exact child checkpoint/optimizer/RNG restore, and bound evaluation capacity on both devices. The probe is not part of the300 formal updates.

Record final frozen-state equality and original versus adapted prediction-head identities. Frozen encoder/Teacher diagnostics may reference the parent's same-population analyses only when this equality and the inference recipe are verified; identify reused observations explicitly. New prediction/output analyses use the same medium diagnostic population. Do not manufacture newly active Teacher losses from reused reference values. Keep large scientific artifacts on the server and preserve the successful-run zero-PKL contract.

## Interpretation

The primary question is final validation MSE relative to the sealed parent, with all18 Pearson measures retained as tradeoff checks. If training loss improves without validation improvement, this fixed-budget readout adaptation did not restore generalization; do not automatically extend it. If validation improves, report the additional-update/evaluation-feature-distribution confounds and any DEG/unseen tradeoff. Neither result alone settles the broader representation-learning problem.
