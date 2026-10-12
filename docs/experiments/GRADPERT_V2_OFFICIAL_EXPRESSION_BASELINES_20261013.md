# Official Scouter / scDFM / GEARS under the expression-exclusion protocol

This is a new experiment family, not an R50 runner modification. Native v1 and
active F1/N1/GM1 source/config/run identities remain unchanged. The user authorized
implementation, testing and automatic execution after the active queue finishes.

## Data and evaluation

All configs bind the same canonical Jurkat file, frozen condition split, cap40
train-row selection and expression axis. Fitting receives only selected train
rows, validation rows, and compatible controls; test perturbation rows are absent.
The sanitizer reads allowed expression columns only, then assembles a full-axis
placeholder matrix. No canonical raw/layers/uns/var expression summaries reach fit.
Excluded test-target genes retain identity/perturbation/prior eligibility; their
placeholder coordinates are excluded from losses, DE selection and numeric graph
statistics. Controls pair within the exact cell_type::batch context.

Official model-specific input organization is retained: Scouter/GEARS use their
fixed full output axis, while scDFM samples1000 queries from the allowed pool.
Thus information eligibility and data IDs match; the per-update token budget is
not identical across architectures. Batch/LR/epochs/optimizer are model-specific.
Pretrained textual gene embeddings in Scouter are prior inputs, not a pretrained
single-cell prediction network. "Unseen expression" refers to the current task's
training exposure, not an assertion about every external prior's training data.

After fit, evaluate the actual last checkpoint on the exact ordered300 controls
per condition and the frozen full truth population. The common v2 evaluator uses
exact shared DEG and Systema/TriShift reference arrays, then averages conditions
equally. Report all18 Pearsons (3 metrics x all/DEG x all/seen/unseen), effective
counts and unavailable reasons, train/eval source hashes, official/source/config/
checkpoint/environment/data/control/truth identities. No missing-gene zero/control
fill or condition deletion is permitted. Official Scouter/GEARS internal validation
uses allowed genes and official selection logic; the common test compares last,
not a test-selected checkpoint. scDFM's costly internal cell_eval reporting is
omitted and recorded, while optimizer/loss/update math is preserved.

## Frozen recipes

|Model|Official commit|Recipe|
|---|---|---|
|Scouter|0cfddd000e19b72ff033ba67c8315f7bc3304932|encoder2048,512 / bottleneck64 / decoder2048; BN on, dropout0; batch256; Adam LR.001; exponential gamma.9; up to40epochs, patience5; official gamma0/lambda.5|
|scDFM|8de47e7d3d443939f93f768f7902c66e0e82eda2|origin differential-perceiver; actual512/8heads/4blocks; batch48; Adam5e-5; cosine floor1e-6;200000updates; Gaussian/MMD.5; coexpressionTop30/negative edges; official RK4 on20gridpoints/tolerance1e-4|
|GEARS|f374e43e197b295016d80395d7a54ddb81cc6769|hidden64; GO/gene GNN1each; decoder16; Top20/threshold.4; direction_lambda.1; batch32; Adam.001/WD.0005; StepLR.5 each epoch;20epochs|

scDFM's CLI d_model128 is ignored by the original instantiate_model: report actual
512. The original model class is constructed directly only to size the vocabulary
for the canonical identity axis; attention/FFN/depth remain official. Its original
train_step, flow path, MMD, graph helper, vocabulary and generate_sample are imported
from a clean checkout. Training source/target are sliced before flow construction.
Full5000-gene inference uses the same fixed1000 query blocks as the native recipe,
with identical row-wise initial noise across inference cell batching.

Official sources:
- https://github.com/PancakeZoy/scouter/tree/0cfddd000e19b72ff033ba67c8315f7bc3304932
- https://github.com/PancakeZoy/scouter_misc/tree/6f2c83e5a32505038060155ca8257fa094732e35
- https://github.com/AI4Science-WestlakeU/scDFM/tree/8de47e7d3d443939f93f768f7902c66e0e82eda2
- https://github.com/snap-stanford/GEARS/tree/f374e43e197b295016d80395d7a54ddb81cc6769

## Adapter and launch gates

Configs are self-contained in configs/v2/official_expression_20261013. Shared
interfaces live only in benchmarks/expression_holdout; native modeling imports no
external learned model. Optional dependencies stay in an isolated server env.

```
python -m benchmarks.expression_holdout.cpu_integration --config CONFIG --runtime RUNTIME --output FRESH_CPU_ROOT
python -m benchmarks.expression_holdout.runner --config CONFIG --runtime RUNTIME --output FRESH_GATE_ROOT --mode preflight
python -m benchmarks.expression_holdout.runner --config CONFIG --runtime RUNTIME --output FRESH_RUN_ROOT --mode full --preflight FRESH_GATE_ROOT/receipt.json
python scripts/v2/run_official_expression_group.py --prepare --runtime FAMILY_RUNTIME --output FRESH_QUEUE_ROOT
python scripts/v2/run_official_expression_group.py --queue FRESH_QUEUE_ROOT/queue.json
```

The CPU gate runs10 real official Adam updates and restores/predicts synthetic
inputs. It does not prove GPU capacity or scientific performance. The deferred
GPU gate runs10 actual updates with the formal configuration and300-control full
inference, checkpoint restore and zeroPKL. Only a passed exact-source/config/runtime
gate permits formal fit. A dependency gate verifies the original queue seal and
all three true last-test completions before any new GPU work. Failed/interrupted
runs keep their IDs and evidence; no automatic retries/config/source changes.
Official single-device routines stay single-device; the existing two-GPU resource
policy/locks prevent overlap with the active native queue. Shared control IDs and
metric aggregation are unaffected by device count.

## Current verification and remaining work

Local30 focused tests pass: sanitization, context pools, official loss masks and
nonzeroLR updates, namespace restoration, full output order, prior coverage,
18-metric parity, config/source/10-step gate rejection. Real official server CPU
integration and deferred GPU gates are separate acceptance items, not yet claimed.

Scouter raw GenePT_V2 lacks17 targets. Official-supported scELMo covers most after
HGNC previous-symbol audit; one genuine missing vector is CCDC130/YJU2B. It may not
be silently replaced by zero/random vectors or removed from the shared dataset.
The user has been asked to select a legal completion path. Scouter remains blocked
until full target coverage is proved; the other two adapters remain independent.
The original prior files are not changed and remain server-only.

Review repairs bind the historical BR test controls
`5d4437766dcafcc096693b35fae57882720f418507e7783c358f3f9d544a2c20`
and v2 reference
`844b572a80ac06fe835827de9d942c02c7e06d3080bc85cf67892962bede1770`
against the original successful BR evaluation receipt at
`development/v2-br-last-only-7c40df4-20261012T123701Z/evaluation/COMPLETE.json`.
The runtime checks actual installed environment versions; CPU integration records
native source, resolved diagnostic configuration and runtime identities. scDFM
inference noise hashes are persisted after inference. Queue acceptance requires
all18 metrics/counts, source/config/runtime/prior/checkpoint/test identities and
whole-run zero-PKL, rather than merely process exit.

Scouter prior completeness is a prerequisite independent of synthetic integration.
CCDC130/YJU2B has no same-gene vector in the available official priors. No condition
will be dropped or assigned a zero vector to bypass this issue. The ready scDFM
and GEARS dependency chain can proceed while Scouter remains explicitly blocked.

First real CPU attempts (native063f26abe7a822e56ccbf6ee4fd4291a96577edb):
scDFM passed10updates/restored exact/zeroPKL; Scouter failed a conservative hygiene
check because actual AnnData0.13 exposes the primary X as layers[None]; GEARS
failed import of unversioned upstream dependency dcor. Preserve all attempt1 logs.
The repair accepts only an identical-object primaryX alias, still rejects raw,
named/auxiliary layers and nonzero excluded coordinates. GEARS uses a separate
isolated environment with dcor0.6; the passed scDFM environment is unchanged.
The project numerical stack differs from historical upstream environment pins;
actual versions are sealed and integration-tested, not claimed bit-identical to
an original author environment. The dependent queue additionally seals a real
CPU integration receipt for each model before GPU dispatch.

Second CPU attempts (native e5a041b189df3a616818d523f3ba37bded41ebe2):
Scouter and scDFM both pass10updates/restore/zeroPKL. GEARS fails before fit on
Pandas3 bool-to-int metadata assignment; preserve its failed scratch assets/logs.
A subsequent GEARS repair uses official0/1 control metadata and binds the existing
frozen GO CSV (`99622d9215462e7bbea99a5f6c0b1e86febfef9f57226269418053484fcb3f9f`),
avoiding an upstream network redownload. Its separate final environment restores
GEARS requirements' explicit numpy/pandas/scipy/scanpy/sklearn/networkx/tqdm pins;
Torch/PyG CUDA are retained and recorded. AnnData0.10.9/Zarr2 compatibility pins are
project runtime choices, not alleged explicit author requirements. No active
native or successful scDFM environment is changed.
