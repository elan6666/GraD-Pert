# V2 dataset-specific ablation workflow

Default ablation dataset: `nadig_jurkat`. The same scripts support
`nadig_hepg2`, `replogle_k562_essential`, `replogle_rpe1_essential`, and `norman`.
This does not enqueue all five datasets. Jurkat is the default experimental
program; the others are available explicitly. Existing v1 canonical data,
condition splits, graph manifests and300-control evaluation populations are reused.

## Prepare and measure a dataset batch

From the repository root, choose one dataset and a candidate batch. This creates
a self-contained configuration, not a capacity claim:

```bash
DATASET=nadig_jurkat
BATCH=32
PYTHONPATH=src python scripts/v2/prepare_dataset.py \
  --dataset "$DATASET" --output "configs/v2/profiles/$DATASET/m$BATCH" \
  probe --batch "$BATCH"
```

Commit and publish the generated config with the code, then deploy the same clean
source to a new server checkout. In the commands below, `PUBLICATION`,
`PUBLICATION_SHA`, `PROBE_RUN`, `GPU` and `CONFIG` are explicit paths/hash/GPU for
that published checkout and a new server run. GPU work stays under/data/yilangliu:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
python scripts/v2/capacity_probe.py --config "$CONFIG" \
  --data-root /data/yilangliu/GraD-Pert/data-vnext-a942114 \
  --gpu "$GPU" --publication "$PUBLICATION" \
  --publication-sha256 "$PUBLICATION_SHA" --output "$PROBE_RUN"
```

Only a passed128-update full-chain receipt can choose the initial dataset batch.
One-step integration is insufficient. Compare measured profiles and reserve memory
headroom. Jurkat currently has repeated32/64 evidence; that evidence does not
establish another dataset's capacity.

## Generate initial groups and exact-row preflight

Use the chosen dataset's original capacity config and pinned receipt:

```bash
PYTHONPATH=src python scripts/v2/prepare_dataset.py \
  --dataset "$DATASET" --output "configs/v2/initial_$DATASET" initial \
  --capacity-config "$CONFIG" --receipt "$PROBE_RUN/receipt.json" \
  --receipt-sha256 "$PROBE_RECEIPT_SHA"
```

The command generates parent,B0,H1, verifies every row and records capacity
provenance. It rejects a receipt from a different dataset or a changed graph/model.
All generated files must be published before running. Run capacity_probe.py with
`--integration-only` for each exact generated config and seed on the intended
launch source, recording the receipts in the run_group preflight index. The same
run_group.py then validates and executes every dataset; no second trainer is used:

```bash
PYTHONPATH=src python scripts/v2/run_group.py \
  --manifest "$GROUP_ROOT/manifest.json" --parent "$PARENT_CONFIG" \
  --runtime "$RUNTIME" --gpu "$GPU" --preflight-index "$PREFLIGHT_INDEX" \
  --queue-root "$QUEUE_ROOT"
```

The default is dry-run. Add `--execute` only for the authorized formal launch.
To resume that same queue, add `--execute --resume`; do not change config/source
or overwrite its run IDs. The shared collection script validates50 epochs,
validation-selected best,epoch50last,both checkpoint tests and zeroPKL.

## Later groups and diagnostics

For every dataset the sequence is B0,H1(LR),H2(weight decay),H3(measured physical
batch),then model/loss/structure/sampling groups. Use select_parent.py and
prepare_followup.py with that dataset's actual upstream validation receipts.
Never import Jurkat's validation winner as another dataset's winner. H3 uses
that dataset's batch_probes list. Each later group gets exact-row preflight and
the same run_group/collect_results workflow.

G1 uses make_expression_holdout.py and evaluate_context.py; D1 uses
evaluate_diagnostics.py. Their protocols must contain IDs from the chosen dataset,
and their checkpoints/configs must pass the existing identity checks. Norman
engineering protocols are not reusable as Jurkat protocols. Engineering results
are labeled separately from formal best/last tests. All datasets retain small
metrics/curves/receipts locally and best/last checkpoints only on the server.
