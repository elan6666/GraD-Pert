# Prepared dataset receipts

The current benchmark protocol is `datasets-v2`, using the frozen GEARS
default-graph representability intersection for every model. All five canonical
datasets, their new split/control manifests, graphs, and evaluator states have
been rebuilt and verified on the compute server.

The reviewed small v2 receipt mirror is synchronized at source commit
`a9421142c086c4fe6b88cd48343a2cc03b1e408a`. The 79 transferred receipt files
are sealed by `small-receipts.SHA256SUMS.txt`; the five data, graph and
evaluation chains are checked by `tests/data/test_prepared_receipts.py`.

Expression preprocessing is source-state aware rather than uniformly replayed:

- Replogle K562 preserves the verified upstream log1p Top-5000 matrix.
- Replogle RPE1, Nadig Jurkat and Nadig HepG2 require raw integer counts, apply
  the official weak-signal filter, normalize each cell to 4,000, apply `log1p`,
  and select Scanpy Seurat Top-5000 genes before the condition split.
- Norman preserves the exact 5,045-gene GEARS `perturb_processed.h5ad`
  expression matrix. It is not normalized, logged or HVG-selected a second
  time; only canonical metadata and stale response caches are repaired.

The `_vnext/nadig_jurkat` receipts independently prove the new graph axis. It
fits Seurat HVG512 on all 238,977 weak-signal-filtered cells and all 2,393
conditions before the frozen condition split, then unions the 2,372 modeled
perturbation targets in canonical order. The resulting 2,809-gene topology is
hash-pinned. Frozen GenePT `emb_b` lacks 17 modeled perturbation targets, so the
four GenePT variants are marked unavailable before graph/model construction;
no missing target is silently removed or zero-filled.

The earlier `datasets-v1` receipts remain under
`superseded/datasets-v1/` for audit only. They are not current inputs and must
not be admitted into a result catalog.

H5AD files, NPZ graph arrays, evaluator arrays, PKL, checkpoints and large logs
remain server-only under `/data/yilangliu`. The local mirror contains only
allowlisted JSON, CSV, TXT and SHA-256 receipts needed to audit identities,
ordered row IDs and hashes.
