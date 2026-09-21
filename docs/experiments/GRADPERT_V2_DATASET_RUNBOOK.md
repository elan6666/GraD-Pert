# V2 dataset-specific ablation workflow

Execution uses both RTX5090 cards in one distributed job. Global batch equals
per-rank microbatch ×2 with accumulation1. Initial batch is pending the dual-card
capacity sweep; single-card64 is no longer the default.

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
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/v2/capacity_probe.py --config "$CONFIG" \
  --data-root /data/yilangliu/GraD-Pert/data-vnext-a942114 \
  --gpu 0,1 --publication "$PUBLICATION" \
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

### G1 training configuration

The expression holdout is one additional training row, inherited from the actual
H3 validation winner. Context-size evaluation reuses a frozen checkpoint and
requires no separate training row. Store a small JSON descriptor with exactly
`path` and `sha256` for the server partition, then generate:

```bash
PYTHONPATH=src python scripts/v2/prepare_followup.py \
  --selection "$H3_SELECTION" --manifest "$H3_MANIFEST" --parent "$H3_PARENT" \
  --group G1 --holdout "$HOLDOUT_DESCRIPTOR" --output "$G1_GROUP_ROOT"
```

G1 uses the existing run_group.py preflight/run/resume path. Its generated row
changes only the expression-partition path/hash. Runtime checks the partition
against the exact canonical gene order; all training views exclude heldout
expression. Best-checkpoint selection uses only training-visible expression
columns. Seen and heldout evaluation axes use equal token budgets, with gene
selection fixed independently of expression values. The current Jurkat protocol
reserves1000 of5000 genes (seed1); this is a project setting. Control expression
is available at inference, so this measures training-column generalization,
not imputation of missing control measurements. Other datasets generate their
own partitions and protocols with the same scripts and their own canonical axis.


### Inference batch engineering check

`scripts/v2/inference_probe.py` loads a passed engineering checkpoint, including
checkpoints trained with two ranks. It uses one frozen validation condition and
its exact300controls, without reading truth expression. For a fixed query set,
`--cell-batches 2 8 16 32` records end-to-end time, peak memory and maximum
prediction difference against the first batch size. The explicit equivalence
tolerance is atol=rtol=2e-5. No prediction matrix is persisted. Run on an idle
GPU and use a new output path. Each point is one timed call, including graph
encoding/transfers; the first point may include cold overhead. This is a
throughput/parity probe, not a full validation result or a proven maximum.

The required config/training-run/engineering-receipt/publication arguments match
the independent engineering evaluation commands. Use `--query-count 1000` for
the standard inference context; test the full expression-axis count separately
for G1 larger contexts. An OOM or parity failure produces a failed receipt with
completed earlier points preserved. The formal inference batch remains pending
these real measurements and final execution-config preflight.
