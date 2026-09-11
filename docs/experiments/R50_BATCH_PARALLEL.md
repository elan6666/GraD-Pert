# R50 fixed-REF batch comparisons

2026-09-11 additional user-authorized coordinate: batch1024, fixed LR .001,
50 epochs, no scheduler, eval batch256, no accumulation/DDP. Only training
batch and artifact label differ from REF. This is not sched512 plus a batch
change. Preserve original E3 model/EMA/loss and canonical data. Run one-epoch
integration/capacity check in a fresh root before fresh50, then automatic
best/last tests. Do not infer safety from historical B0 batch1024 capacity.
Select GPU placement from live resources without interrupting active jobs.
Never lower batch or precision to recover OOM silently; preserve failure.
Network restoration, new clean publication/full server gates and hash-pinned
contract remain prerequisites; this entry does not claim training has started.

2026-09-10 user explicitly chooses LR=1e-3 and parallel batch128/batch512,
each50 epochs, without another full-epoch capacity experiment. This supersedes
waiting for LR-mid to select the parent for these two coordinates only. REF
is the fixed comparison parent, not a declared global LR winner. LR-mid continues.
Prior REF/LR-low test results were seen; acknowledge adaptive design history.
No test score will select these runs' checkpoints or change their configurations.

Self-contained configs: configs/r50/batch128 and configs/r50/batch512.
Only train_batch_size differs from REF; eval_batch_size stays256. LR.001,
AdamW/weight_decay0/no scheduler, original E3, graph/model, EMA, loss1/.8/.4/.1,
data/canonical split/ordered300 controls/seed1 and precision remain fixed.
The same50 epochs give equal exposure, not equal optimizer steps or compute.
Step budget derives from each frozen run_meta.steps_per_epoch, not29100 for all.

GPU1 runs both coordinates, each one-epoch native integration then fresh50
without early stopping; no additional capacity benchmark. Original GPU0 LR-mid
is untouched. Each allocation capped at40% of total GPU memory, also respecting
4GiB current-free reserve; cap persists in training and automatic postfit tests.
The short batch256 concurrency probe achieved1.9485x aggregate throughput,
but does not prove batch128/512 full-run memory safety. User accepts proceeding;
OOM/failure is preserved, never silently change batch/precision or auto-retry.

Retain best.pt and last.pt; after full training validate50 ordered epoch receipts
and all optimizer steps, then independently evaluate both on canonical test.
Identical checkpoints alias one evaluation. Zero persistent PKL after successful
test; temporary reconstructible metric inputs are managed only by the evaluator.
Historical LR-low last remains unavailable and must not be reconstructed.

Before launch: clean local/public/server commit, source/config/publication/prior
hashes, CPU tests/Ruff/format/mypy/isolated build, fresh roots, GPU1 idle and
host/disk capacity. Preparation goal ends before CUDA. Two new sessions bind
physical GPU1, worker and queue logs/RC are preserved. No active-source changes.
Update existing grad-pert-r50 hourly quiet monitor with exact runtime contract.
Inspect each process, progress, validations, source/input identities, memory and
terminal test receipts once; notify only meaningful changes/failure/completion.
On OOM/failure stop only this new pair if required to preserve resources, never
LR-mid or another user's process; do not auto-relaunch. On completion validate
small evidence, compare batch rows directly with REF; no new matrix without
the next bounded preparation stage. Waiting LR-mid test queue waits for GPU1 idle.
