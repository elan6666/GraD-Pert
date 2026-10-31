# Isolated native full-update performance diagnostics

The first implementation is a **partial arithmetic ledger plus actual complete
update timing**, not a measured MFU/HFU result. It always emits `mfu: null` and
`hfu: null`: unknown arithmetic, conversions, solve/special/custom kernels, GEMM
bias/scales and FP8 logical padding prevent claiming complete FLOP coverage.
Counted FLOPs are conventional software shape arithmetic, not hardware instruction
counters. Counts of unknown operator calls are not a FLOP coverage percentage.

`scripts/v2/mfu_probe.py` calls the existing native `optimizer_step`; it does not
implement another training loop mathematically. `mfu_ledger.py` executes every
original dispatch operator once without decomposition, replacement or profiler.
It counts declared matrix and elementary arithmetic formulas and inventories
unknown operations, with input/output shapes, dtypes, phase, owner and checkpoint
replay tags. It records actual Student/Teacher forward input shapes separately.
The counter is active only during native updates, excluding input preparation,
hashing and CPU snapshot copies. Nested checkpoint replay is one boolean category,
not an additive count of enclosing checkpoint regions.

## Immutable native runtime and parent identity

The native package must import from the **original training checkout**, first on
`PYTHONPATH`. The instrumentation checkout has its own clean published identity.
Both publication receipts are required; the original training SHA is never
replaced by the instrumentation SHA. The original run's terminal `TRAIN_COMPLETE`,
`run_manifest`, journal, raw/resolved config hashes, last checkpoint SHA,
architecture, runtime data/split/exclusion identity and rank topology are verified.
Native `load_checkpoint(identity=original_manifest)` retains all existing checks.
The native runtime and original publication must match the original training SHA.

This version supports only completed **joint** native training parents on the
approved two GPUs, global/microbatch settings inherited from the unchanged config.
It does not reinterpret staged/frozen-readout experiments. Unsupported parents
fail explicitly. Nothing writes into the original run or active source checkout.
Do not rerun a scientific baseline to obtain a timing parent.
Compiled/fused KDA backends and background I/O prefetch are explicitly unsupported
in this first transaction implementation; their settings are never silently changed.

The diagnostic LR and Teacher EMA are explicitly frozen at the parent's final
accepted update index (`completed_updates - 1`) using the saved native lifecycle
contract. A finished cosine is not extrapolated and the update clock is not reset
to zero. Temperature/prototype/auxiliary clocks follow the original native
optimizer updates inside the disposable transaction. This is a diagnostic policy,
never formal continuation or additional accepted training progress.

## Bounded protocol

1. Require a hash-sealed `mfu-resource-owner-binding-1` receipt, live same-user CPU
   guardian PID/start-ticks/command/parent identity and ancestor relationship. The
   guardian must hold **both original** `/data/yilangliu/GraD-Pert/runtime/v2-gpu-{0,1}.lock`
   FLOCK leases, verified from Linux `/proc/locks`, throughout the run. Use the
   unchanged original `shared_gpu.admit(policy, snapshot(owner_pid))` before GPU
   allocation, at every completed-update admission and after restoration; the
   original sealed policy's memory fraction is used. No wrapper can launch this
   CLI without the owner binding, leases and original policy. In addition require
   idle GPUs (capacity threshold of 512 MiB occupied per GPU) and
   `PYTORCH_ALLOC_CONF=expandable_segments:True`. No shared-preflight exception is
   repurposed for profiling. Main controls the CPU guardian, resource scheduling and parent terminal
   acceptance; this script neither interrupts nor queues experiments.
2. Preserve S0: objective/Student/Teacher/buffers, optimizer/moments/routes/steps,
   parameter gradients including None, Muon split-part gradients, pending stats,
   module/plain phase flags, temperature index, pipeline diagnostics, all native
   Python/NumPy/Torch CPU/CUDA and view RNGs, raw TRAIN epoch/batch cursor.
3. Perform three uncounted warmup updates, capture S3, then count the requested
   **N** complete native updates (`--measured-updates N`, integer 1..10; default
   10) with fixed TRAIN inputs. Raw row/condition identities, input/view
   hashes and forward shape counts are retained as small metadata. Raw data and
   all snapshots stay in the server process; no activation/matrix export.
4. Remove all counting contexts and restore S3 exactly. Seek the raw TRAIN cursor
   without consuming skipped view RNG draws. Replay the same **N** updates without
   any counter or profiler. Time from requesting the next batch through complete
   accumulation/communication/optimizer/Teacher/center and CUDA synchronization.
   Record preparation and update times too. Hashing/copying occurs after timers.
5. Require input/view identity and final full native state/RNG parity between
   count and timing. Always restore and fingerprint S0 in `finally`, including
   exceptions and partial windows. Failed restoration is a failed result.
6. A 900-second **soft** window includes warmup, counting, state copies, seeking,
   timing and restoration; startup/loading/hardware inspection is separate.
   Reserve at least 30 seconds (or four initial snapshot times) for restoration.
   Admission checks coordinate both ranks at completed-update boundaries. One
   indivisible native update or I/O call can exceed a soft deadline; report actual
   elapsed time and incomplete counts, never invent the requested samples. NCCL failure
   timeout is two minutes. Failure receipts remain per-rank if a collective aborts.

Every rank records `requested_measured_updates`, `counted_updates`,
`timed_updates`, and `actual_samples` (the actual timed sample count), including
partial/failure receipts. The TRAIN manifest seals exactly `3 + N` batches.
Aggregation requires one shared valid requested budget and exactly N counted and
N timed samples from each completed, restored rank. Mismatched budgets, invalid
integer types (including bool), missing samples, or fabricated completed counts
fail closed; incomplete runs produce no aggregate timing result. There is no
implicit legacy fallback for receipts missing the requested budget.

The first 900-second native attempt reached three warmups and eight counted
updates with zero timed samples; it is an incomplete diagnostic, not a timing
result. The next authorized attempt must explicitly use `--measured-updates 2`
after validation releases resources. Two paired samples are a short isolated
engineering diagnostic, not full MFU/HFU, sustained throughput, capacity evidence,
or a scientific baseline. This does not change the 900-second soft window, three
warmups, native optimizer arithmetic, resource policy or exact restoration checks.

The primary paired timing aggregates each step using the slower rank's direct
end-to-end wall time. Global cells are counted once, not once per rank. The report
includes median/p95, actual samples and covered TFLOP/s. Timing samples exclude
per-step resource admission, metadata and restore overhead, whose total remains in
the reported diagnostic window. Cells/s is the rate over the sum of these isolated
complete-update samples, **not continuous wall throughput** over the whole window.
These are short warmed diagnostic measurements, not sustained throughput/capacity
or scientific completion. Do not interpret kernel/GEMM TFLOP/s as hardware MFU.

## Server launch template

Run only after local tests, scoped commit/push, matching clean independent
instrumentation checkout and main resource acceptance. Paths below are explicit
placeholders, not an executable queue dispatch:

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/data/yilangliu/ORIGINAL_NATIVE/src:/data/yilangliu/INSTRUMENTATION/scripts/v2 \
OMP_NUM_THREADS=2 /path/to/original/python -m torch.distributed.run --nproc_per_node=2 \
  /data/yilangliu/INSTRUMENTATION/scripts/v2/mfu_probe.py \
  --config /data/yilangliu/ORIGINAL_NATIVE/configs/EXACT_PARENT.yaml \
  --data-root /data/yilangliu/EXACT_DATA_ROOT \
  --training-root /data/yilangliu/EXACT_COMPLETED_RUN \
  --checkpoint-sha256 EXACT_LAST_CHECKPOINT_SHA256 \
  --native-source /data/yilangliu/ORIGINAL_NATIVE \
  --native-publication /data/yilangliu/ORIGINAL_PUBLICATION.json \
  --native-publication-sha256 ORIGINAL_PUBLICATION_SHA256 \
  --resource-binding /data/yilangliu/EXACT_RESOURCE_OWNER_BINDING.json \
  --resource-binding-sha256 RESOURCE_OWNER_BINDING_SHA256 \
  --resource-owner-pid LIVE_GUARDIAN_PID \
  --publication /data/yilangliu/INSTRUMENTATION_PUBLICATION.json \
  --publication-sha256 INSTRUMENTATION_PUBLICATION_SHA256 \
  --measured-updates 2 \
  --gpu 0,1 --output /data/yilangliu/NEW_DIAGNOSTIC_RUN_ID
```

No checkpoint or PKL is written by this diagnostic. Success requires zero PKL
throughout the new run root and unchanged parent checkpoint hash. Main dry-reviews
the small receipt/metadata transfer list. `rank-*.json` and `receipt.json` retain
partial/failure evidence; a completed timing receipt still says
`scientific_complete: false` and MFU/HFU null. Original-native-runtime CPU tests
and eventual CUDA execution are separate required evidence.

The CPU guardian seals the following small resource binding after acquiring the
original leases and before spawning torchrun. `owner` is the exact live dictionary
from existing `shared_gpu.process_info(owner_pid)`. `policy` names the unchanged
previously authorized resource policy; this binding grants no new allowance.

```json
{
  "schema": "mfu-resource-owner-binding-1",
  "owner": {"pid": "exact live integer and all other process_info fields"},
  "policy": {"path": "/data/yilangliu/ORIGINAL_POLICY.json", "sha256": "EXACT_HASH"},
  "binding": {
    "output": "/data/yilangliu/NEW_DIAGNOSTIC_RUN_ID",
    "config_sha256": "EXACT_HASH",
    "native_source_commit": "ORIGINAL_TRAINING_SHA",
    "instrumentation_source_commit": "PUBLISHED_INSTRUMENTATION_SHA",
    "training_root": "/data/yilangliu/EXACT_COMPLETED_RUN",
    "training_manifest_sha256": "EXACT_HASH",
    "checkpoint_sha256": "EXACT_LAST_HASH"
  }
}
```

Output must be a new directory with no overlap **in either direction** with the
data, native source, instrumentation source or original run roots. The script
rechecks source identities and all bound config/publication/parent/checkpoint/
resource inputs after full restoration, including symlink target identity. A
closure failure invalidates the diagnostic; per-rank failure evidence remains.

For future complete counters, retain FMA=2 and the verified RTX5090 dense BF16 /
FP32 accumulation reference of 209.5 TFLOP/s/card (419 for two), separately from
dense FP8 419/card. Never substitute sparse or FP4 denominators. Source:
[NVIDIA Blackwell architecture v1.1, Table 3](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
Native hardware name/UUID/clock/power readback is required for a measured result;
this first slice records hardware facts but deliberately computes no peak-normalized
percentage. Mixed dtype counts cannot all be normalized as FP8 speed.
