# B0/B1: batch 1024 and step-indexed LR/EMA

Status: implemented and CPU-tested; no new CUDA training launched.
Supersedes the batch128 restart recipe for future B0/B1 runs only. Old configs,
capacity attempts and training evidence remain immutable. Use fresh run roots.

New standalone configs:
- `configs/combinations/b0_historical_b2_e3_step_batch1024/gradpert_b2/nadig_jurkat.yaml`
- `configs/combinations/b1_historical_b2_step_batch1024/gradpert_b2/nadig_jurkat.yaml`

Both use train batch1024, eval batch256, one process/GPU, no accumulation or DDP.
Independent experiments on two GPUs do not double either experiment's global batch.
Nominal global batch counts cells, never views, graph nodes or prototypes. Partial
batches keep the fixed nominal-batch LR, as in the reference configuration scaling.
This is a cell-batch adaptation: the SSL condition pool remains at most8 conditions,
so it is not equivalent to1024 independent DINO images or a1024-entry spread pool.

## Frozen reference and intentional scope

Official repository: https://github.com/facebookresearch/dinov2
Commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (Apache-2.0).
Inspected symbols: `dinov2/utils/config.py:apply_scaling_rules_to_cfg`,
`dinov2/utils/utils.py:CosineScheduler`, `dinov2/train/train.py:build_schedulers`
and optimizer-before-Teacher update order. Values come from
`configs/train/vitg14.yaml` and `configs/ssl_default_config.yaml`.
Native implementation imports no upstream project and independently evaluates
the audited scalar schedules. Tests compare the reference NumPy array definition.

- Peak LR = `base_lr * sqrt(global_batch_size / 1024)`; base2e-4 => peak2e-4.
- Budget T = actual sampler steps/epoch *100. No restart.
- W = floor(0.16*T): project adaptation of official80/500 warmup ratio.
- Warmup includes endpoints0 and peak at indices0 and W-1 (W=1 gives0).
  Cosine starts at peak again at indexW and approaches min1e-6.
- EMA uses0.994 ->1 with cosine over T steps, no EMA warmup.
  Like the audited code, indexT-1 is slightly short of the final value;
  indexT and later resolve exactly to the final value. Never use T-1 denominator.
- LR is applied before each optimizer step; EMA follows the Student update.
  Step0 has LR0 but still updates optimizer moments, centers and model buffers.
- Schedules use checkpoint global_step; no mutable scheduler clock to lose on resume.
  Actual LR and Teacher momentum are recorded in train_steps.csv.

Preserve B2 geometry (6506 nodes,8x512 locals), precision, optimizer/weight_decay0,
loss1/1/.1/.1, evaluation, centering, temperatures and100-epoch/patience10 policy.
No claim to reproduce the complete DINOv2 training recipe or improve accuracy.
Early stopping may end training before the schedule endpoint, even during warmup;
we have not silently disabled it or changed patience. B0 retains E3, B1 learned ID.

## Capacity boundary

Batch1024 passed16 full native steps on both B0/B1 under previous LR settings.
This is not a full-run or new-configuration integration pass. New source must pass
server gates and a fresh integration smoke before full runs. Capacity probe8192
was mostly underfilled, and must never be cited as sustained8192 capacity.

## Verification

Isolated server CPU snapshot `/data/yilangliu/GraD-Pert/development/b01-step-cpu-bLOhUi`:
full suite652passed/4honest skips (CUDA and unavailable frozen references).
After adding the step-schedule resume case, both legacy/new resume tests passed:
next-step LR, momentum, loss and all model state tensors agree after restore.
Strict mypy passed79source files. Local Ruff and format checks passed.
Server Ruff, format292files, isolated sdist and wheel build also passed.
This is development snapshot verification, not published exact-commit CUDA evidence.
