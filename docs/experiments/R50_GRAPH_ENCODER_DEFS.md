# R50 graph-encoder ablation definitions: gat_mlg and hybrid_bmp

Date: 2026-09-16. Purpose: unblock the two design-blocked rows in
`R50_NEW_ABLATIONS.md` with auditable paper definitions, so they can be
implemented natively without importing or copying any upstream code.

Authorization: user instruction 2026-09-16 — restore both rows, no upstream
implementation lookup; paper definitions are the authoritative source. Rows are
queued behind all existing lanes and the restored `r50n_source_gat`.

## Sources (frozen evidence)

- Published article: TxPert, Nature Biotechnology (2026),
  DOI `10.1038/s41587-026-03113-4`, open access (CC BY-NC-ND 4.0).
- Definitions: Supplementary Information PDF (Supplementary Notes 1-5),
  section S1.3 "GNN perturbation encoder model details", saved locally at
  `/tmp/txpert_supplement.pdf` (sha256 recorded at implementation time).
  The user's main-text PDF lacks the supplement; the preprint
  (arXiv 2505.14919, Appendix A) independently contains the GAT-MLG formula
  and was used as a cross-check. Hybrid-BMP appears only in the published
  version.
- Prior blocker text ("public implementation/documented formula conflict",
  "no verified complete official definition") is resolved by these paper
  definitions; no official code was consulted, per authorization.

## Hybrid-BMP (supplement S1.3.2, bi-directional message passing)

Every gene carries two learned input features: `H0_in` (target role) and
`H0_out` (source role). With the union-graph binarized adjacency
`(A^G)_ij = 1` iff edge `(v_i, v_j)` exists in any source graph, and
`A^G_out = (A^G_in)^T`:

```
Z := MLP( A^G_in @ H0_in + A^G_out @ H0_out )
```

One message-passing layer; "hybrid" = the union over K graphs. The paper
reports this one-layer model as its best backbone (Table S5: test Pearson-delta
0.639 vs Exphormer-MG 0.626).

## GAT-MLG (supplement S1.3, multi-layered graph attention)

Given K graphs over a common gene universe with weighted adjacency `A_k`:

- Supra graph `G! = (V!, E!)`: node `(w, k)` per gene w per layer k.
  Edges: intra-layer `(w1, w2) in Ek1, k1 = k2`; inter-layer `w1 = w2,
  k1 != k2` (identity coupling). Diagonal blocks of `A!` are the `A_k`,
  off-diagonal blocks are identity couplings.
- L GATv2 layers with multi-head attention propagate over `A!` (attention
  per Equations S4-S6, heads per S7).
- In parallel, a structural module per graph aggregates edge-weight-derived
  features into node-level structural embeddings `s_k,w` (not propagated
  across layers).
- Gated fusion per node: `z_k,w = GATE(s_k,w, h^L_k,w)`; the returned
  perturbation representation uses the reference graph `k = 1`:
  `Z = [z_1,w] for w in V1`.

## Native implementation mapping (GraD-Pert, no upstream copying)

To be finalized with the implementation commit; binding decisions:

- Both rows are native architecture packages inside `gradpert.modeling`,
  selected by config exactly like `r50n_source_gat`. They are not claims of
  official TxPert parity, and no upstream file is copied, imported, or
  rebranded.
- `hybrid_bmp`: the existing native union graph (STRING + GO, same node axis,
  same weights policy) supplies `A^G_in`; a native dual-feature one-layer
  encoder implements the equation above.
- `gat_mlg`: the native per-source adjacency tensors supply the K = 2 layers;
  the supra graph is built over the same shared gene axis; the structural
  module and gate are native choices documented next to the code.
- Hyperparameters (width, heads, dropout) follow the parent row's frozen
  values unless the row name names a change; any freedom the paper leaves
  unspecified is recorded in the config and in this file.
