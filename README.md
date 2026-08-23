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

## Scope

- Datasets: Replogle K562 essential, Replogle RPE1 essential, Nadig Jurkat,
  Nadig HepG2, and Norman.
- Models: GraD-Pert B2, isolated public GEARS and TxPert benchmarks, plus three
  nonlearned baselines.
- Formal training and large artifacts are server-only. The Mac receives only
  small summaries and artifact pointers.

## Development quick start

```bash
python -m pip install -e '.[dev]'
gradpert --version
python -m pytest -q
python -m ruff check .
python -m mypy src
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

