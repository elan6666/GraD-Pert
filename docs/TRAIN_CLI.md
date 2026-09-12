# Simple training entry

From a clean, published server source checkout:

```bash
python -m gradpert train --gpu 0 --dry-run
python -m gradpert train --gpu 0
python -m gradpert train --config configs/your-experiment.yaml --gpu 1
```

Omitting `--config` currently selects the unchanged historical
`configs/r50/batch1024/gradpert_b2/nadig_jurkat.yaml`: E3, batch1024,
AdamW fixed LR 0.001, 50 epochs without early stopping, and historical
validation Pearson maximization for best. It is not a schedule ablation or
a claim of exact historical reproduction under a new source version.
Explicit configs preserve their own selection metric, architecture and losses.
This entry currently supports native R50 `metrics_only` configurations only;
unsupported protocols fail rather than silently substituting a configuration.

Every run computes validation prediction loss and the three canonical Pearson
metrics, saves per-epoch numeric records and training/validation PNG/PDF curves,
then evaluates best and actual last checkpoints using the existing postfit
runner. Validation references are prepared from train/validation only. Test
data is used after fitting, not for best selection. Both checkpoint roles are
recorded even when best is also last and their identical evaluation is aliased.

## One-time server runtime configuration

The server operator supplies a JSON file at
`/data/yilangliu/GraD-Pert/runtime/train.json`, or points `GRADPERT_RUNTIME` /
`--runtime` at a version-specific file. Paths are absolute. Receipts are
provenance documents, not passwords or access tokens.

```json
{
  "data_root": "/data/yilangliu/GraD-Pert/data-vnext-a942114",
  "runs_root": "/data/yilangliu/GraD-Pert/runs",
  "publication_receipt": "/data/yilangliu/GraD-Pert/contracts/THIS_VERSION/publication.json",
  "publication_sha256": "REPLACE_WITH_VERIFIED_SHA256",
  "genept_receipt": "/data/yilangliu/GraD-Pert/contracts/PRIOR/genept-preflight.json",
  "genept_sha256": "REPLACE_WITH_VERIFIED_SHA256"
}
```

Missing runtime inputs, changed hashes, dirty/unpublished source, and invalid
seeds are errors. Native graph/axis/GenePT compatibility checks still execute
before fitting. Publication receipts must match the executing checkout; update
the runtime binding for each release. No global credential search is performed.

`--seed` defaults to the first configured seed. An override is accepted only if
listed in `training.run_seeds`; default seed1-only configs reject `--seed 2`.
`--data-root` overrides only the data root, not split identities. `--gpu` accepts
one physical index or UUID; the process maps it to logical cuda:0 and sets the
required allocator before importing the training runtime.

Dry-run reads and verifies configuration/publication/hash inputs, prints the
resolved config and proposed unique output paths, but does not reserve a run,
create directories, prepare data, initialize CUDA, or launch training. It is
not a capacity or full data-compatibility gate. A subsequent invocation gets a
new UUID. Execution never resumes or overwrites an existing experiment.

Outputs are `<runs_root>/<unique-run-id>/fit/small_results` for curves/records,
`fit/checkpoints` for weights, and `<unique-run-id>-test` for best/last tests.
The entry runs in the foreground; scheduling, concurrency capacity checks and
monitoring remain the caller's responsibility. No experiment is auto-launched
by installing this feature.
