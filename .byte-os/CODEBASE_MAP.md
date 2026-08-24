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
| `src/gradpert/` | Standalone package: contracts, data, graphs, model, execution, artifacts, evaluation | Active implementation surface |
| `configs/experiments/` | Exact 30-file self-contained model-by-dataset YAML matrix | Active configuration surface |
| `benchmarks/` | Isolated official GEARS/TxPert adapters and environment locks | No native learned-model dependency |
| `tests/` | Unit/contract/integration tests with synthetic fixtures | No large real datasets |
| `notebooks/` | Read-only hash-pinned result analysis | Never source of split/training/metric truth |
| `scripts/server/` | Dry-run-first server launch, matrix orchestration and small-result staging | No secrets or embedded credentials |
| `scripts/results/` | Explicit-source final 45-run ResultCatalog planner/sealer | Never discover latest result directories |

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
| server topology | `python -m pytest -q tests/server tests/execution/test_matrix.py tests/execution/test_small_sync.py` | `python -m ruff check scripts/server tests/server src/gradpert/execution` | matrix/staging core only | matrix dry run, then small-result staging dry run |

Server commands remain dry-run-first and require exact explicit paths.

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

## Remaining execution evidence

- Fresh server regression after the latest source synchronization.
- Fifteen learned one-epoch gates and their fairness-hash audit.
- Seed-1 nonlearned matrix, then GraD-Pert formal full runs.
- Verified allowlisted small-result transfer and explicit ResultCatalog.
