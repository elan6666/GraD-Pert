# Nadig Jurkat full-axis text-prior comparison

## Scope

This is a new, explicitly bounded experiment on `nadig_jurkat`. It does not
rewrite the frozen E1/E2/E3/ES availability receipt for the original 1,536-D
Ada `emb_b` artifact.

All conditions use the same 2,809-node `hvg512_plus_targets` graph axis with
SHA-256 `e14dc759fa3744c552ecc7be82ce84ea50bff773c6cfdc89b53418e5350ff389`.
No perturbation target or non-target graph node is removed. Prior NPZ files must
contain this exact case-sensitive axis in this exact order.

## Conditions

1. `genept_latest_id_residual`: latest official 3,072-D protein-aware GenePT,
   aligned by exact symbol plus six unambiguous HGNC mappings. Sixty unavailable
   rows use a zero prior and therefore reduce to the learned-ID residual. This
   condition is not described as fully official GenePT coverage.
2. `genept_seed_id_residual`: 2,048-D `doubao-embedding-vision` embeddings of the
   complete NCBI + UniProt + source-bounded identity corpus.
3. `genept_seed_goexp_id_residual`: the same model and full-axis corpus with
   bounded GO experimental annotations.

Every condition uses `genept_id_residual`, so the downstream route is
`learned gene ID + linear projection(prior)`. Projection width follows the
verified prior matrix width; all other architecture and optimization settings
remain fixed.

## Frozen evaluation

- dataset and split: existing Nadig Jurkat canonical train/validation/test;
- graph: identical `hvg512_plus_targets` topology;
- seed: 1;
- epochs: exactly 10 after one-epoch integration gates;
- test access: one sealed evaluation per completed run;
- primary comparison: the same metrics already emitted by the GraD-Pert native
  evaluator; no post-hoc condition filtering.

## Current artifact state

The official GenePT aligned NPZ exists on the server with SHA-256
`79a7033a3d5954a2450935b84e9661500e24fe2ab7e1ec5c4ce6c463b80d02e7`.
The two Doubao NPZ artifacts remain pending because the server has no
`ARK_API_KEY`. Training must not start until all three prior files pass exact
axis, finite-value, hash, and one-epoch integration checks.
