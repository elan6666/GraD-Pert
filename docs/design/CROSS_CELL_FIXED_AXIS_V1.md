# Four-line fixed-axis cross-cell benchmark

This is a new benchmark protocol using the verified official TxPert
`K562_cross_cell_lines.zip` cache from [Zenodo record 15420279](https://zenodo.org/records/15420279).
It is separate from the five within-cell datasets and does not replace their
canonical splits. The 3,352-gene axis is held fixed for every fold. Because the
official cache chose its feature axis with K562 as the held-out line, the other
three folds **do not reproduce** the paper's fold-specific feature selection.

## Audited source

- Archive: 1,750,726,245 bytes; MD5 `5b4081a5ee3a74c3e4e9a840da34c37c`.
- H5AD SHA256:
  `1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8`.
- H5AD: 632,488 rows, 3,352 genes. Cell-line labels: `K562`, `RPE1`,
  `jurkat`, `hepg2`, and auxiliary `K562_adamson`.
- Observation names are **not unique**. All membership manifests use global
  zero-based H5AD row positions, never observation names alone.
- The official `train_test_split.pkl` is K562-specific. Reusing its test list
  would cover only about 44% of RPE1 perturbation rows. It is retained as
  provenance, not applied as the four-fold partition.

## Fixed four-fold protocol

For each target line among K562, RPE1, Jurkat and HepG2:

1. Every non-control row from that line is test-only. None may enter training
   or validation, including through a condition shared with another line.
2. All control rows from the target line are available as basal inputs. Source
   controls and target controls are indexed separately so adapters cannot
   silently mix their evaluation pools.
3. Non-control rows from the other three lines supply training and validation.
   Within each `(source line, condition)` group, a deterministic seed-42
   sample of 10% (at least one row, leaving at least one train row) is
   validation. Singleton groups remain in training. This retains every source
   perturbation condition while giving model selection only source-line data.
4. `K562_adamson` is excluded from the four-line protocol, matching the
   official datamodule's `train_cell_types=all` exclusion of auxiliary lines.
5. Test conditions are reported as seen or unseen relative to perturbation
   conditions present in that fold's **training rows**. Both subsets use the
   same target-line test rows and gene axis.

`benchmarks/crosscell/prepare_fixed_axis.py` writes four compressed arrays of
row positions and a small JSON manifest with row, gene-axis, source and split
hashes. The H5AD and arrays stay on `/data/yilangliu`; only the small manifest
may be synced locally after transfer review. The preparation fails closed on a
different H5AD checksum and refuses to overwrite an existing fold artifact.

The existing condition-only `CanonicalTrainingData` and
`CanonicalEvaluationData` interfaces **must not** load these folds directly:
the same condition can occur in both source and target lines. Each model adapter
(GraD-Pert, GEARS, TxPert, Scouter and nonlearned baselines) must consume the
same row-position manifests and target-control pool before formal comparison.
No GPU result should be reported under this protocol until those adapters and
row-level leakage checks are validated.

## GraD-Pert v2 model contract

The GraD-Pert model in all four leave-one-cell-line-out folds uses the
2026-09-25 v2 method in
`docs/design/GRADPERT_V2_HAMILTONIAN_MLA_METHOD.md`: a trainable GenePT-only
deterministic PCA256 identity table; GO/STRING incoming Top20 plus three
fixed bidirectional Hamiltonian expander cycles and self edges; two graph
layers reading updated neighbor states; and, in each Cell/Response Encoder,
two KDA blocks followed by one noncausal full MLA block. There is no DSA
Top500 index in this parent. The older B0/B1 and Top500 encoder profiles
remain historical or separately named ablations, not this cross-cell default.

The four folds share this model design and the cache's exact ordered 3,352-gene
axis. A new GenePT-PCA256 seed artifact must be generated and hash-audited
against that axis; the Jurkat within-cell PCA256 artifact has a different gene
axis and must not be reused. Likewise, graph node IDs/neighbors and expression
columns must be aligned to the fixed axis. The row-level cross-cell split and
target-control pool above remain unchanged. Each fold requires a self-contained
config and explicit source/data/split/seed hashes before training. Model
comparisons must hold this GraD-Pert architecture fixed across folds; any
different width, depth, graph topology or prior is a separately labeled ablation.

`prepare_fixed_axis.py` remains a model-agnostic split generator. A cross-cell
training/evaluation adapter and the fixed-axis GenePT/graph artifacts are still
required; this section records the model choice, not a completed GPU run.
