# GraD-Pert

GraD-Pert is a standalone research package for graph-conditioned prediction of
single-cell perturbation responses. The active v1 model jointly optimizes
expression prediction and graph-view self-distillation, then compares all
models under one condition split and population evaluation contract.

The repository is under active implementation. Current authoritative material:

- [active model design](docs/design/GRADPERT_V1.md)
- [data and evaluation contract](docs/design/DATA_AND_EVALUATION.md)
- [server execution contract](docs/design/SERVER_EXECUTION.md)
- [reference alignment and licenses](docs/provenance/REFERENCE_ALIGNMENT.md)
- [experiment architecture alignment](docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md)

## Scope

- Datasets: Replogle K562 essential, Replogle RPE1 essential, Nadig Jurkat,
  Nadig HepG2, and Norman.
- Models: GraD-Pert B2, isolated public GEARS and TxPert benchmarks, plus three
  nonlearned baselines.
- Formal training and large artifacts are server-only. Runs default to
  metrics-only receipts with no persistent PKL for every model and nonlearned
  baseline. An explicitly requested large export is one deduplicated
  `result.pkl`. The Mac receives only small summaries and artifact pointers.

## Checkpoints and reproducible inference

Learned runs retain one selected model checkpoint on the server. GraD-Pert and
GEARS use PyTorch `.pt` checkpoints; TxPert uses a Lightning `.ckpt` checkpoint.
The checkpoint contains learned parameters and, where required by the runner,
training state. It does **not** contain the canonical expression dataset, the
exact 300-control draws, test Truth, or a complete prediction export.

Reproducing a prediction therefore uses the checkpoint together with the
run's small evidence bundle:

- `config.resolved.yaml`, which freezes the self-contained model/dataset config;
- `inference_recipe.json`, which identifies the checkpoint and reconstruction
  inputs;
- `prediction_manifest.json` and `evaluation_manifest.json`, which preserve the
  exact ordered control and Truth row IDs for every condition;
- canonical data, split, control-manifest, gene-order, and graph hashes; and
- frozen official-checkout and runtime receipts for isolated external runners.

The canonical H5AD remains on the server. Ordered row IDs recover the exact
expression values from that file later, including repeated control draws and
their order, so those arrays do not need to be duplicated in a persistent PKL.
A checkpoint alone is not considered a reproducible result.

All GraD-Pert, GEARS, TxPert, and nonlearned-baseline configs default to
`metrics_only`. A successful metrics-only run retains its selected checkpoint
when applicable, the evidence bundle above, and the three metric summaries, but
leaves zero persistent `*.pkl` files anywhere in the run root. Nonlearned
baselines have no checkpoint; their deterministic rule, config, canonical data,
and ordered row IDs are sufficient to reconstruct them.

`single_pkl` is an explicit opt-in for a downstream workflow that needs frozen
per-cell matrices without rerunning inference. It writes exactly one
`artifacts/result.pkl`, with a shared deduplicated control-expression pool plus
ordered indices, predictions, Truth, metrics, and provenance. It is never the
default.

## Nadig Jurkat one-epoch speed pilot

The completed speed-only pilot kept the frozen 5,000-gene
expression/output/evaluation axes and selected the combined B3 performance
variant: a directly recomputed Top-500-HVG-plus-target graph together with all
seven semantics-preserving systems optimizations. A separate metrics-only B0
timing coordinate preserved the full graph, disabled all seven systems groups,
and left the historical B0 untouched. On the same server GPU and one-epoch
contract, actual training wall time was 2,951.487 seconds for B0, 844.180
seconds for graph-only B1, 718.681 seconds for systems-only B2, and 507.718
seconds for combined B3.

Reducing only the graph made B1 3.496x faster than B0; enabling only the seven
systems groups made B2 4.107x faster. On the reduced graph, the systems groups
made B3 1.663x faster than B1, while reducing the graph with the systems groups
active made B3 1.416x faster than B2. Combined B3 was 5.813x faster than the
metrics-only B0 timing baseline. The three prediction metrics are recorded as
non-decisional evidence only: one epoch does not establish unchanged
predictive effect. See
the [final pilot review](.byte-os/reviews/2026-08-26-nadig-jurkat-speed-pilots.md)
and [sealed small evidence](.byte-os/evidence/nadig-jurkat-speed-pilots/).

## Nadig Jurkat 10-epoch comparison

A separate fixed-duration follow-up trained systems-only B2 and combined B3
for exactly 10 epochs (5,820 optimizer steps) on separate GPUs. Both runs used
the same seed, batch size, 16,384 prototypes, canonical split, ordered
300-control draws, validation schedule, allocator, and `metrics_only` artifact
policy. Each produced 10 validation receipts, performed one final test
evaluation from its selected checkpoint, retained only `best.pt`, and left zero
persistent PKL.

| Variant | Graph nodes | Training wall | Actual cells/s | TxPert delta | TriShift delta | Systema |
|---|---:|---:|---:|---:|---:|---:|
| B2 systems-only | 6,506 | 7,221.773 s | 177.61 | 0.251625 | 0.253596 | 0.031636 |
| B3 combined | 2,798 | 5,223.127 s | 245.57 | 0.243824 | 0.251376 | 0.011911 |

B3 reduced training wall by 27.68% relative to B2, a 1.383x speedup. Its three
recorded metrics were lower by 0.007801, 0.002219, and 0.019724 respectively.
These metrics are reported as observations, not as proof of unchanged effect or
statistical equivalence. The runs were concurrent on separate GPUs but shared
host CPU, RAM, and storage, so their absolute timing should not be mixed with
the earlier sequential one-epoch pilot. Legacy receipt fields named
`one_epoch_training_wall_ms` and `one_epoch_fit_wall_ms` cover the complete
10-epoch interval in these fixed-duration runs.

See the [10-epoch review](.byte-os/reviews/2026-08-26-nadig-jurkat-ten-epoch.md)
and [sealed small evidence](.byte-os/evidence/nadig-jurkat-ten-epoch/).

## Development quick start

```bash
python -m pip install -e '.[dev]'
gradpert --version
python -m pytest -q
python -m ruff check .
python -m mypy src
```

Server experiment planning and result staging are dry-run-first:

```bash
PYTHONPATH=src:. python scripts/server/run_experiment_matrix.py --help
PYTHONPATH=src:. python scripts/server/stage_small_results.py --help
PYTHONPATH=src:. python scripts/results/build_final_catalog.py --help
```

Do not use historical design alternatives under `TxPert/` as active scope. See
root `AGENTS.md` before editing.

## Scientific provenance

External methods and code behavior are attributed in the provenance registry.
GraD-Pert native model code does not import or call their repositories. The
frozen TxPert checkout is not distributed by this repository.

## License

No project license has been granted yet. All rights are reserved unless and
until the repository owner adds an explicit license. External references and
datasets retain their own terms.
