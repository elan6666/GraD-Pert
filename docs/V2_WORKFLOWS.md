# GraD-Pert v2 training and checkpoint workflows

All scientific data, graphs, checkpoints, and fitting stay on `/data/yilangliu`.
Run from a clean, published server checkout matching the publication receipt in
the runtime JSON. Each command creates a new run/output ID unless it is the
explicit same-run `resume-v2` command. A checkpoint file alone is insufficient:
the original run manifest, resolved config, frozen data, and control manifest
remain required.

## Fresh single- or multi-GPU training

The config is self-contained. `model.parameters.world_size` must match the
number of physical GPUs in `--gpu`; the global batch equals microbatch ×
accumulation × world size. Prepare a single-GPU config by explicitly setting
world size to 1 and adjusting the global train batch in that file, then run a
one-step capacity check on the server before formal use. Do not reuse a
two-GPU config with only `--gpu 0`.

```bash
python -m gradpert train --config CONFIG.yaml --runtime /data/yilangliu/GraD-Pert/runtime/train.json --gpu 0 --dry-run
python -m gradpert train --config CONFIG.yaml --runtime /data/yilangliu/GraD-Pert/runtime/train.json --gpu 0
python -m gradpert train --config TWO_GPU_CONFIG.yaml --runtime /data/yilangliu/GraD-Pert/runtime/train.json --gpu 0,1
```

The saved `RUN/launch.json` seals the input paths, source/config hashes, seed,
GPU selection, and new run ID. A stopped or interrupted run can only be resumed
with its original source, configuration, topology, and total epoch budget:

```bash
python -m gradpert resume-v2 --launch /data/yilangliu/GraD-Pert/RUN/launch.json
```

The regular run retains `fit/epoch_state.json`, `fit/history.json`, curves,
`best.pt`, `last.pt`, best/last test receipts, and `COMPLETE.json`. Every
committed epoch records joint-loss validation; population prediction metrics
are optional and controlled by `model.parameters.validation_mode`. An exit
or a live progress file alone does not prove scientific completion.

## Current fresh Jurkat B0 (2026-09-30)

Use `configs/v2/mhc_joint_only_jurkat/three_epoch_m68_a2/gradpert_v2/nadig_jurkat.yaml`
for the next fresh B0. It restores four mHC residual streams in Student/Teacher,
retains the complete prediction + SSL1 + SSL2 objective, and uses global batch272
(68 per GPU × accumulation2 × two GPUs), three epochs and seed1.
Each epoch runs only full joint-loss validation (`validation_mode: joint_only`)
and selects best by that loss. Joint and component loss curves remain available.
The frozen 300-control test still runs for both best and last at completion,
including the six all/DEG Pearson outputs and exposure groups.
The no-mHC three-epoch/continuation configs below are historical examples.
See [protocol and capacity boundary](experiments/GRADPERT_V2_MHC_JOINT_ONLY_B0.md).

## Continue the completed 3-epoch Jurkat B0

The two full configs are:

- `configs/v2/no_mhc_joint_eval_jurkat/continue_full_m74_a2/gradpert_v2/nadig_jurkat.yaml`
- `configs/v2/no_mhc_joint_eval_jurkat/finetune_lora_r8_m74_a2/gradpert_v2/nadig_jurkat.yaml`

Both bind the completed parent's run root, its `last.pt` SHA256, parent epoch
3, two additional epochs, and constant post-parent learning rate `0.0002`.
Inspect the dry-run and parent receipt before launching either:

```bash
python -m gradpert train --config configs/v2/no_mhc_joint_eval_jurkat/continue_full_m74_a2/gradpert_v2/nadig_jurkat.yaml --runtime /data/yilangliu/GraD-Pert/runtime/train.json --gpu 0,1 --dry-run
python -m gradpert train --config configs/v2/no_mhc_joint_eval_jurkat/continue_full_m74_a2/gradpert_v2/nadig_jurkat.yaml --runtime /data/yilangliu/GraD-Pert/runtime/train.json --gpu 0,1
```

Use the LoRA config in the same commands to launch the adapter branch. Do not
launch either until its published source, GPU capacity and parent checksum
are verified. A full-state child restores Student, Teacher, centers, Muon/AdamW
moments, global step, and rank-specific RNG. Its best checkpoint is selected
from all five observed validation epochs, including an immutable copy of the
parent best. The LoRA child initializes rank-8, alpha-16 adapters with zero
output, freezes every base parameter, restores Student/Teacher base weights
and centers, then starts a new adapter-only AdamW. Its best is selected from
epochs 4–5 because a parent checkpoint has no adapter parameters. Both retain
the parent's historical per-epoch expression-exposure record, so the three
Pearson families and seen/unseen expression groups remain comparable.

The original three epochs used a three-epoch endpoint cosine. The child does
not relabel them as a five-epoch cosine run: it records a `3+2` parent hash and
constant continuation LR. The parent run is never edited or overwritten.

For the 3→6 full-state continuation, use
`configs/v2/no_mhc_joint_eval_jurkat/continue_full_m74_a2_to6/gradpert_v2/nadig_jurkat.yaml`
with the same dry-run and launch commands above. It restores the same parent
`last.pt` and training state, retains the original three completed epochs,
then adds epochs 4–6 with constant LR `0.0002`. This is recorded as a `3+3`
stage, not retroactively as a six-epoch cosine run. Its best checkpoint is
selected by joint validation loss across all six observed epochs; best and last
are both tested at completion.

## Evaluate a saved v2 checkpoint without training

The input must be a checksum-verified `best` or `last` checkpoint in its
original run root. The evaluation config must reproduce that run's resolved
model/data/evaluation protocol. The current published source is recorded as
the **evaluation** source, separately from the checkpoint's training source.
If a full-state 3+2 child retains the parent's best checkpoint, pass the
parent's three-epoch config for that checkpoint; pass the child's config for
its new last checkpoint. The receipt records the selected checkpoint's actual
training identity in either case.

```bash
python -m gradpert evaluate-checkpoint --config CONFIG.yaml --training-run-root /data/yilangliu/GraD-Pert/RUN --checkpoint /data/yilangliu/GraD-Pert/RUN/fit/last.pt --checkpoint-sha256 CHECKPOINT_SHA256 --runtime /data/yilangliu/GraD-Pert/runtime/train.json --output-root /data/yilangliu/GraD-Pert/evaluations/NEW_ID --gpu 0 --split test --dry-run
python -m gradpert evaluate-checkpoint --config CONFIG.yaml --training-run-root /data/yilangliu/GraD-Pert/RUN --checkpoint /data/yilangliu/GraD-Pert/RUN/fit/last.pt --checkpoint-sha256 CHECKPOINT_SHA256 --runtime /data/yilangliu/GraD-Pert/runtime/train.json --output-root /data/yilangliu/GraD-Pert/evaluations/NEW_ID --gpu 0 --split test
```

Choose `--gpu 0,1` to evaluate disjoint condition shards on two GPUs. Each
worker uses the same frozen 300-control draws and its own single-GPU inference
runtime; the coordinator verifies complete, non-overlapping condition coverage
and recomputes the macro metrics in frozen manifest order. `--split val` is
also available, with the train-only Systema reference. The output contains a
small plan, per-worker condition progress and logs, evaluation environment
identity and `COMPLETE.json`, with zero prediction PKLs.
Reusing an output directory is rejected.

No new checkpoint, fine-tune run, or independent evaluation is launched by
adding these commands or example configs. Local CPU tests establish interface
and state-machine behavior; a formal server result still requires the usual
published-source and same-config GPU integration checks.
