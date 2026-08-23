# Codebase Map

## Current top-level map

| Path | Purpose | Edit policy |
|---|---|---|
| `.byte-os/` | Product state, research, specs, decisions, plans, reviews, iterations, delivery | Primary agent owns state; plans govern implementation |
| `docs/design/` | Active, pruned GraD-Pert v1 specification | Editable; authoritative after migration |
| `docs/provenance/` | Frozen upstream commit/symbol/license alignment | Editable with evidence only |
| `TxPert/grad-pert/` | Local-only historical design discussion source | Ignored by the project Git repository after active-spec migration |
| `TxPert/official-repo/` | Frozen upstream TxPert checkout at `08d82ee...` | Never edit |
| `TxPert/official-repo.incomplete-archive/` | Incomplete/noisy earlier upstream archive | Avoid; not authoritative |
| `src/gradpert/` | Planned standalone product package | Active implementation surface |
| `configs/experiments/` | Planned self-contained model-by-dataset YAML matrix | Active configuration surface |
| `benchmarks/` | Planned isolated GEARS/TxPert runners and environment locks | No native-package imports from here |
| `tests/` | Planned unit/contract/integration tests with synthetic fixtures | No large real datasets |
| `notebooks/` | Planned read-only frozen-artifact analysis | Never source of split/training/metric truth |
| `scripts/server/` | Planned commit preflight, server launch, and result allowlist sync | No secrets or embedded credentials |

## Primary stacks and package managers

- Python 3.11/3.12 package, `pyproject.toml`, wheel/sdist build.
- PyTorch and PyTorch Geometric for server model/graph execution.
- NumPy/SciPy/Pandas/PyArrow and optional AnnData/Scanpy for data/evaluation.
- Pydantic and PyYAML for strict manifest/config validation.
- pytest, Ruff, mypy for local verification.
- Separate pinned environments for native GraD-Pert, GEARS, and TxPert.

## Command matrix

| Scope | Test | Lint | Typecheck | Build/verify |
|---|---|---|---|---|
| repository | `python -m pytest -q` | `python -m ruff check .` | `python -m mypy src` | `python -m build` |
| config/data contracts | `python -m pytest -q tests/contracts tests/data` | `python -m ruff check src/gradpert/config src/gradpert/data tests/contracts tests/data` | `python -m mypy src/gradpert/config src/gradpert/data` | `gradpert config verify --all` |
| model/loss/graph | `python -m pytest -q tests/modeling tests/graphs` | `python -m ruff check src/gradpert/modeling src/gradpert/graphs tests/modeling tests/graphs` | `python -m mypy src/gradpert/modeling src/gradpert/graphs` | `gradpert smoke model` |
| evaluation/artifacts | `python -m pytest -q tests/evaluation tests/artifacts` | `python -m ruff check src/gradpert/evaluation src/gradpert/artifacts tests/evaluation tests/artifacts` | `python -m mypy src/gradpert/evaluation src/gradpert/artifacts` | `gradpert smoke evaluation` |
| external runners | `python -m pytest -q tests/benchmarks` | `python -m ruff check benchmarks tests/benchmarks` | N/A across isolated envs | runner-specific `--preflight-only` |
| server topology | `python -m pytest -q tests/server` | `python -m ruff check scripts/server tests/server` | N/A | `scripts/server/preflight.sh` then dry-run sync |

Commands referencing planned paths become active in foundation/build. Until
then, use the Byte state and reference-integrity commands in `HARNESS.md`.

## Generated and noisy paths

- Ignored: Python caches/build outputs, virtualenvs, `data/`, `datasets/`,
  `graphs/`, `cache/`, `artifacts/`, `runs/`, `checkpoints/`, scientific binary
  formats, and `.server/`.
- Read-only/noisy and Git-ignored: all `TxPert/` local evidence, especially the
  incomplete archive.
- Server-only generated trees must not be copied into this repository.

## Navigation and LSP

- Use `rg --files --hidden` and `rg -n` before recursive viewers.
- Python: Pyright/Pylance or mypy-compatible LSP rooted at `src`.
- YAML: schema associations should point experiment configs to the strict
  experiment schema once generated.
- Begin implementation from the relevant `.byte-os/plans/*.plan.md`, load root
  `AGENTS.md`, then the nearest module `AGENTS.md`.

## Exploration candidates

- Data-source readiness and exact checksums on the server.
- Native GAT-Hybrid memory/shape profile after the synthetic full-step exists.
- Isolated runner compatibility against the exact pinned CUDA environments.
- Metric golden-output comparison against frozen source implementations.
