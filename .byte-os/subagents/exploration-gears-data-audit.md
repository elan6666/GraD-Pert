# Exploration: GEARS and five-dataset integration audit

## Scope and ownership

- Agent: `gears_data_audit`
- Read-only audit of TriShift's GEARS/data/evaluation code, current design
  notes, and official GEARS repository interfaces.
- TriShift commit observed:
  `87ac2c51c3c266391093f71a8bce2e6beaa81518`.
- Files changed by subagent: none. No data or dependencies were downloaded.

## Dataset support

- GEARS directly names `replogle_k562_essential`,
  `replogle_rpe1_essential`, and `norman`, but the Replogle assets are marked
  filtered/incomplete and must be hash/content-checked against the canonical
  GraD-Pert data before reuse.
- Nadig Jurkat and HepG2 require a custom AnnData adapter and GEARS
  `new_data_process`; neither has a current local adapter or downloaded data.
- Train five independent GEARS models, one per dataset. The official method is
  not a cross-cell training architecture.
- Custom data must retain exact canonical `condition`, `cell_type`, and
  `gene_name` semantics, and must export requested/retained/dropped conditions
  after graph coverage filtering.

## Split and prediction interface

- GEARS accepts a custom split pickle containing exact `train`, `val`, and
  `test` condition lists. Adapter output must be compared back to the canonical
  condition/hash manifest; seed equality alone is insufficient.
- Some upstream versions can leave training-gene-set state incomplete after a
  custom-split early return. The runner must set and assert the state rather
  than silently trusting it.
- Native GEARS control graph creation samples 300 controls, but `.predict()`
  averages them to one vector. The common runner must instead take the exact
  control row IDs from the evaluation manifest and run the frozen best model
  per control to retain `Pred[300,G]`.
- External runners should emit prediction-only artifacts. The common evaluator
  joins canonical truth after the model phase and creates a sealed evaluation
  bundle and notebook-ready outputs.

## Risks and implementation boundary

- TriShift's current GEARS runner is not a drop-in solution: it only registers
  Adamson/Dixit/Norman, monkeypatches model/loss behavior, does not actually
  truncate one advertised 300-row payload, and mixes truth into runner output.
- Freeze an exact official GEARS commit and server-compatible Torch/PyG/CUDA
  lock; do not rely on an unpinned package name or TriShift's environment.
- Nadig perturbations can be silently lost through GO graph coverage. Make
  zero-unexplained-drop coverage a data gate.
- The local environment currently lacks Torch, PyG, AnnData, Scanpy, and
  GEARS. Formal dependency resolution and materialization belong on the
  server; local development will use lightweight schemas and synthetic tests.

