# Private Hugging Face Trackio dashboards

## Purpose and authority

Formal GraD-Pert ablations may mirror their existing scalar training receipts
to a private Hugging Face Trackio Space so loss, validation and hardware curves
can be inspected while a run is active. Trackio is auxiliary telemetry only.
The server `train_steps.csv`, `validation.csv`, native receipts and sealed
evaluation evidence remain authoritative.

The integration is an out-of-process sidecar. It does not import Trackio from
the trainer, native executor or performance tools, and it does not change
forward order, RNG, CUDA work, receipt buffering or step timing. Capacity,
profile, exact-effect and ABBA timing lineages must not launch the sidecar.

## Dashboard contents

The scalar allowlist contains:

- total, prediction, condition-consistency, masked-node and spread losses;
- Teacher momentum, target entropies, prototype-use counts and center norms;
- prediction/auxiliary gradient norms and their ratio;
- batch and local-view scalar counts;
- data, transfer, view, Teacher, Student global/local, prediction,
  backward/update and end-to-end step times, plus derived throughput;
- validation TxPert macro Pearson delta, improvement state and patience count;
- one explicitly selected GPU's utilization/memory/power/temperature telemetry
  and host CPU telemetry.

The sidecar reads the four named scalar/identity receipts, but its explicit
remote allowlist never serializes or uploads `metrics_summary.*`, test gates,
per-epoch evaluation JSON, predictions, Truth, row/data/gene hashes or lists,
H5AD, PKL, checkpoints, data/config paths or artifacts. The exact source commit
is intentionally included as run identity. The sidecar never calls
`trackio.log_artifact()`.

The native writer buffers up to 64 step rows. Live curves therefore update in
small batches rather than forcing a per-step write into the training path.
The sidecar applies an owner-only `0077` file-creation mask for its full
lifetime and verifies the fresh local Trackio store as `0700`; its SQLite,
JSONL, state, receipt and lock files are therefore not group/world readable.

## One-time private authentication

Install the optional dependency set in the new clean server checkout:

```bash
uv sync --frozen --extra data --extra model --extra tracking
hf auth login
hf auth whoami
```

Authentication is performed privately on the server. Never put a Hugging Face
token in a command argument, repository file, log, receipt or chat message. The
token needs permission to create/write a private Space and its backing Bucket.

The registered dashboard target is the private Space
`elan68681/grad-pert-vnext-ablations` with the explicit private Bucket
`elan68681/grad-pert-vnext-ablations-bucket`. The sidecar checks both before
logging and again after Trackio initialization. An existing public destination
or a different Bucket already mounted at `/data` is rejected; a missing Bucket
is created private. A missing Space is created with `private=True` and rechecked
as private.

If private Space creation is externally blocked and the user explicitly
chooses to continue, the sidecar may run in
`--local-private-bucket-archive` mode. It skips Space creation and records the
same allowlisted metrics in an owner-only local Trackio store. The formal queue
then archives that tracking directory to the existing private Bucket with the
authenticated `hf buckets sync` command. Its receipt says
`delivery_mode=local_private_bucket_archive`, `space_sync_attempted=false` and
`remote_sync_verified=false`; it never claims a live web dashboard. Restoring
the archived store and running `trackio show` provides local loss curves, while
a future private Space requires a separate explicit sync.

## Formal sidecar launch

Use a fresh tracking directory, state file and receipt for every immutable
formal row. All tracking files stay outside the scientific run root. Start the
sidecar with the same one-GPU visibility as the training process:

```bash
export GRADPERT_TRACKIO_SPACE_ID="elan68681/grad-pert-vnext-ablations"
export GRADPERT_TRACKIO_BUCKET_ID="elan68681/grad-pert-vnext-ablations-bucket"
export CUDA_VISIBLE_DEVICES="0"

PYTHONPATH=src:. python scripts/tracking/sync_trackio.py \
  --run-root /data/yilangliu/GraD-Pert/runs/LINEAGE/ROW \
  --trackio-dir /data/yilangliu/GraD-Pert/tracking/LINEAGE/ROW/store \
  --state-path /data/yilangliu/GraD-Pert/tracking/LINEAGE/ROW/state.json \
  --receipt-path /data/yilangliu/GraD-Pert/contracts/LINEAGE/trackio/ROW.json \
  --run-name ROW-seed1 \
  --group nadig-jurkat-vnext-four-local-h \
  --variant-id ROW \
  --expected-run-id ablation/nadig_jurkat/ROW/seed-1 \
  --expected-source-commit EXACT_COMMIT \
  --expected-config-sha256 EXACT_CONFIG_SHA256 \
  --expected-optimizer-steps 5820 \
  --expected-validations 10 \
  --local-private-bucket-archive \
  --gpu-device 0
```

Trackio 0.37's automatic GPU monitor can inspect all physical GPUs, so this
integration disables it and requires `CUDA_VISIBLE_DEVICES` to contain exactly
one non-negative decimal physical index. UUID/MIG selectors are rejected because
Trackio 0.37 cannot safely map them. The sidecar then calls
`trackio.log_gpu(device=0)` every 30 seconds, where `0` is the logical index even
when the visible physical GPU is GPU 1. An empty GPU sample fails the sidecar.
CPU telemetry remains background-only in the separate process.

## Failure and completion semantics

The sidecar requires an exact formal clean-source `native-run-meta-v1`,
contiguous finite scalar CSV rows and the pre-test
`native-training-receipt-v1`. It uses an exclusive lineage lock,
`resume="never"`, explicit optimizer steps and an fsync-plus-atomic diagnostic
state journal. This is single-attempt provisional telemetry, not crash recovery:
a crash is not resumed because Trackio does not deduplicate replayed values by
`(run, step)`.

Trackio logging and delivery are best effort. The live dashboard and its local
receipt are therefore marked provisional, non-authoritative and
`remote_sync_verified=false`. A tracking failure never relabels or repairs the
scientific run. The formal CSV/JSON receipts remain complete even if the
dashboard is unavailable. The local receipt therefore says points were
`enqueued`, not remotely proven. The future hash-pinned formal A/H queue must
start and stop one sidecar per row; a manual dry run alone does not satisfy that
launch contract.

## Observed four-local A/H delivery

In `formal-vnext-ah-845c10a-v2`, A0 and H3 completed their owner-only local
Trackio stores with all 5,820 train points and ten validation points, but the
private Bucket archive failed after the reverse SOCKS endpoint disappeared.
H1 and H2 encountered the same refused endpoint during live Bucket preflight,
so their local Trackio clients did not start. All four native scientific runs
completed and passed strict validation. The tracking failure is retained
truthfully; no run was replayed, resumed or relabeled, and no public fallback
was created.
