# A0 with the current step-based training recipe

**Cancelled before CUDA:** user corrected the target to A1 on2026-09-09.
This file records unlaunched preparation, not a result or launch authorization.
Use A1_STEP_BATCH1024.md instead.

User authorization: 2026-09-09. Independent single-seed run, not a replacement
of any sealed A0/B/C result. Identifier: A0-step1024-100.

Base: configs/ablations/nadig_jurkat/a0_ratio_ring_half/gradpert_b2/nadig_jurkat.yaml.
Preserve A0 architecture: learned-ID128, Exphormer-MG STRING+GO, HVG512 plus
targets (2809 nodes), four RingInduced locals at floor(N/2)=1404, no local
anchor masking, projector2048/256 and16384 prototypes, basal/decoder512,
loss1/.8/.4/.1. Do not substitute compact B/C architecture or E3.

Changes: training batch1024 (single GPU, no accumulation/DDP), LR peak2e-4
at nominal batch1024 with sqrt(B/1024) scaling,16% step warmup from zero and
cosine decay toward1e-6; Teacher EMA .994 toward1 by cosine. Keep AdamW,
weight decay0 and eval batch256. Max100 epochs, validation patience10,
seed1. Early stopping may terminate before the warmup ends; this experiment
retains that existing protocol rather than disabling early stopping.

Remove the old ten-epoch performance-pilot tag and use the explicit
vnext_combination_100 execution policy. Config is standalone. Exact data,
split and ordered300 controls/truth are unchanged. metrics_only, zero PKL,
best checkpoint only. One-epoch integration smoke precedes a fresh full run.
This joint LR/EMA/batch change is not a single-factor ablation.

## Execution handoff

Publish a clean source and pass pytest/Ruff/format/mypy/build before CUDA.
Use a fresh a0-step1024-<commit>-v1 namespace and one idle physical GPU.
Run scripts/server/run_compact200_pair.py --row a0 with the new publication
receipt; its one-epoch smoke must pass before full. Preserve failure evidence
and do not retry or reduce batch silently. Allocator expandable_segments:True.
After the preparation goal ends, launch without an active goal and register
an hourly quiet heartbeat for this row only. Inspect phase/RC/steps/validation,
source/config identities and zero PKL; two unchanged hourly snapshots require
read-only stall diagnosis. On terminal state, audit best.pt/test-once/common
ordered controls/truth, stop monitor, then report. No other experiment launch.
