# Codebase Harness

## Status

- Claude support: ready
- Codex support: ready
- AGENTS.md quality: ready, concise entry context with detailed rules linked
- Date reviewed: 2026-08-24

## Context files

- `AGENTS.md`: repository entrypoint and hard boundaries.
- `CLAUDE.md`: concise parity context.
- `.byte-os/ENGINEERING_RULES.md`: detailed persistent rules.
- `.byte-os/CODEBASE_MAP.md`: paths, commands, noise, navigation.
- `.byte-os/AGENTS_AUDIT.md`: freshness and coverage audit.
- `.byte-os/subagents/`: read-only exploration handoffs.

## Noise and safety filters

- `.gitignore` excludes data, model artifacts, caches, runs, checkpoints,
  virtual environments, secrets, and scientific binary formats.
- `.claude/settings.json` denies editing frozen upstream/data/artifact trees and
  reading common secret file types.
- Root rules make both upstream checkouts read-only evidence.

## Commands available now

```bash
python3 /Users/elan/.codex/skills/byte-do/scripts/byte_state.py validate --root /Users/elan/code/grad-pert
git -C TxPert/official-repo status --short
git -C TxPert/official-repo rev-parse HEAD
rg --files --hidden -g '!TxPert/official-repo.incomplete-archive/**'
```

## Active project commands

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m build
gradpert config verify --all
python -m pytest -q tests/server tests/execution/test_matrix.py tests/execution/test_small_sync.py
PYTHONPATH=src:. python scripts/server/run_experiment_matrix.py --help
PYTHONPATH=src:. python scripts/server/stage_small_results.py --help
```

## Known gaps and follow-up setup

- The latest local source still needs fresh server Torch/PyG/mypy regression.
- Fifteen learned one-epoch integrations have not yet all passed.
- Formal execution requires one clean commit shared by local, GitHub and server.
- Current datasets-v2 small receipts still need allowlisted synchronization.

## Subagent exploration

- Design audit: `.byte-os/subagents/exploration-design-audit.md`.
- TxPert paper/code audit: `.byte-os/subagents/exploration-txpert-audit.md`.
- GEARS/data audit: `.byte-os/subagents/exploration-gears-data-audit.md`.
