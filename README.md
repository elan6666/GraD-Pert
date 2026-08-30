# GraD-Pert

GraD-Pert is a standalone research package for graph-conditioned prediction of
single-cell perturbation responses. The stable v1 model jointly optimizes
expression prediction and graph-view self-distillation. The active B2-vNext
ablation surface keeps that same trainer and evaluation lifecycle while making
graph axes, local views, graph encoders, node features, decoders, and objective
weights explicit self-contained config choices.

The repository is under active implementation. Current authoritative material:

- [active model design](docs/design/GRADPERT_V1.md)
- [B2-vNext ablation design](docs/design/GRADPERT_VNEXT_ABLATIONS.md)
- [data and evaluation contract](docs/design/DATA_AND_EVALUATION.md)
- [server execution contract](docs/design/SERVER_EXECUTION.md)
- [reference alignment and licenses](docs/provenance/REFERENCE_ALIGNMENT.md)
- [experiment architecture alignment](docs/provenance/TRISHIFT_ARCHITECTURE_ALIGNMENT.md)
- [private Trackio loss/utilization dashboards](docs/experiments/HUGGING_FACE_TRACKIO.md)

## Scope

- Datasets: Replogle K562 essential, Replogle RPE1 essential, Nadig Jurkat,
  Nadig HepG2, and Norman.
- Models: GraD-Pert B2, isolated public GEARS and TxPert benchmarks, plus three
  nonlearned baselines.
- Current default native architecture: the B2-vNext A0 graph/view/model contract
  described below. Historical full-graph B2 formal runs remain sealed evidence
  and are not silently reinterpreted as vNext.
- B2-vNext ablations: Nadig Jurkat only, one frozen canonical split, run seed 1,
  exactly 10 epochs, and one config per variant through the same native CLI.
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

## B2-vNext default and config-driven ablations

The preregistered vNext reference is `A0`. It preserves the frozen Top-5000
expression/input/output/evaluation axis. Its graph axis mirrors TxPert's
within-cell preprocessing before condition splitting: weak-signal filtering,
the complete cell line normalized to 4,000 counts per cell, `log1p`, and Scanpy
Seurat Top-512 HVGs with `subset=True`, followed by the union with every
representable perturbation target. The selected order and normalized-dispersion
ranking are retained as hash-bound receipts.

Dataset scale handling is fail-closed rather than inferred from a filename.
The three raw-count sources (RPE1, Jurkat, and HepG2) must pass a full-matrix
finite, nonnegative integer-count audit before the TxPert transform is applied.
The official processed K562 and GEARS Norman archives preserve their verified
`X` values and gene axes; in particular Norman is not sent through a second
`expm1`, normalization, or `log1p`. The vNext Nadig Jurkat HVG512 ranking uses
the full weak-signal-filtered cell line before the frozen condition split,
including future train, validation, and test conditions.

A0 uses ordered STRING+GO Top-20 graphs and a native sparse multi-source graph
Transformer aligned to the public TxPert Exphormer-MG graph-encoder surface.
It uses two global views and four four-hop RingInduced local views. Each local
node cap is the exact floor of 50% of the actual HVG512-plus-target runtime
graph (`1,404/2,809` on the sealed Jurkat axis). Local-anchor masking is off
(`0/4`); the existing global node mask remains enabled. The decoder is
additive, all seven systems groups remain enabled, and direct loss weights are
`1.0/0.8/0.4/0.1` for prediction, condition consistency, masked-node
consistency, and embedding spread.

The frozen matrix is generated under
`configs/ablations/nadig_jurkat/<variant>/gradpert_b2/nadig_jurkat.yaml`.
Each YAML is complete and is read by the same `gradpert model` entrypoint; there
is no ablation-specific main function. The matrix covers proportional
RingInduced/Fanout local construction, count/coverage/mask ratios, single
STRING GATv2, single/multi sparse graph Transformers, native adaptive source
fusion, five STRING weight routes, additive/MLP/
control-conditioned-Transformer decoders, four GenePT feature routes, and
three loss removals. The H module varies HVG512/1024/2048/5000 while retaining
50% local coverage and four locals. The L module is frozen as direct A0
single-factor rows: Fanout, eight locals, 25% local coverage, 50% mask ratio,
or 25% mask ratio. L execution is currently paused.

GenePT E rows use the frozen GenePT-Seed `Seed-GO-ProteinPathway` master
artifact (SHA-256
`34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`).
The 17,730-gene source covers the complete runtime axes by exact,
case-sensitive labels; extra non-runtime source genes are ignored and
receipted. A missing perturbation target aborts before model construction. A
missing non-perturbation runtime gene is omitted in preserved canonical order,
with its ordered count/hash receipted; aliases and silent zero/random filling
are forbidden. The selected `Seed-GO-ProteinPathway` artifact currently misses
no runtime gene, so this policy does not change the formal coordinate.

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
PYTHONPATH=src:. python scripts/tracking/sync_trackio.py --help
```

Formal ablations can install the optional `tracking` extra and mirror only
allowlisted training/validation scalars to a private Hugging Face Trackio
Space. The sidecar is excluded from performance timing and never uploads test
metrics, predictions, datasets, checkpoints or per-cell artifacts.

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
