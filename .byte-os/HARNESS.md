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

## Commands activated by foundation

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m build
gradpert config verify --all
gradpert smoke all
```

## Known gaps and follow-up setup

- `pyproject.toml`, package code, tests, and module-local context files do not
  exist yet; plan 001 creates them before product implementation.
- Server Python/CUDA environments are not yet locked; formal jobs remain gated.
- RPE1/Jurkat/HepG2 independent within-cell source URLs/checksums remain open
  data gates.
- Current GitHub remote is empty; initialize and push only after repository
  source/spec validation, without force.

## Subagent exploration

- Design audit: `.byte-os/subagents/exploration-design-audit.md`.
- TxPert paper/code audit: `.byte-os/subagents/exploration-txpert-audit.md`.
- GEARS/data audit: `.byte-os/subagents/exploration-gears-data-audit.md`.

