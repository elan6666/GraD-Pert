# GenePT-Seed prior comparison

## vNext E-row binding

The current 25-row Nadig Jurkat vNext matrix does not compare the three prior
conditions below. Its E1/E2/E3/ES rows are all locked to the selected
GenePT-Seed `Seed-GO-ProteinPathway` condition using the sealed master artifact
`/data/yilangliu/GenePT-Seed/data/embeddings/seed-go-protein-pathway-master-aligned.npz`,
SHA-256
`34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.
The artifact contains 17,730 unique exact-case labels, width-2,048 vectors,
and zero all-zero vectors. Each E row selects the unchanged ordered 2,809-node
`hvg512_plus_targets` axis by exact label. Extra source labels are ignored and
receipted. Missing perturbation targets, duplicates, non-finite rows, wrong
identity metadata, and any zero-fill fallback abort before training. A missing
non-perturbation runtime gene is instead omitted in preserved canonical order
and its ordered count/hash is receipted. The selected artifact has complete
2,809-gene runtime coverage, so the current E-row axis remains unchanged.

Before any E-row launch, run the repository preflight
`scripts/ablations/preflight_genept_seed.py` against the unchanged
`vnext/graph_axes/nadig_jurkat/hvg512_plus_targets` root. The resulting sealed
receipt binds the parent topology, exact source artifact, 17,730-label source
axis, ordered 2,809-label selection, perturbation targets, ignored extras and
selected matrix bytes. It also binds any ignored missing non-perturbation
runtime labels; the sealed current receipt must record none. The native runner
independently repeats artifact and selection verification.

The comparison design below remains a separate future experiment and must not
be mixed with the vNext E-row evidence.

## Question

On the frozen Nadig Jurkat B2-vNext coordinate, does replacing the text
embedding prior improve perturbation prediction, and does adding bounded
high-quality GO experimental knowledge help beyond NCBI+UniProt?

## Conditions

1. Latest GenePT: official NCBI+UniProt `text-embedding-3-large`, aligned to the
   2,809-gene axis. Sixty absent rows are exact zeros plus the common learned-ID
   residual and are not represented as official GenePT vectors.
2. GenePT-Seed: the exact 2,809-gene NCBI+UniProt/HGNC identity corpus encoded by
   `doubao-embedding-vision`.
3. GenePT-Seed+GO: the same Seed corpus plus bounded GO-EXP text encoded by the
   same Doubao model.

## Frozen comparison contract

- Dataset and split: Nadig Jurkat canonical within-cell unseen-single split.
- Graph: the same ordered 2,809-node HVG512-plus-targets STRING+GO topology.
- Model mode: `genept_id_residual` for every condition.
- Training: seed 1, ten epochs, batch size 256, AdamW, and fixed loss weights
  1.0/0.8/0.4/0.1.
- Evaluation: one final test access, 300 controls per condition, and the same
  TxPert/TriShift/Systema metrics-only evaluator.
- Inputs: exact-case axis equality and artifact SHA-256 are checked before model
  construction.

Only prior artifact path/hash/model/width and condition/run labels may differ.
No result may be described as a multi-seed or multi-dataset conclusion.

## GO-EXP boundary

The GO condition keeps EXP, IDA, IMP, IEP, HTP, HDA, HMP, and HEP evidence;
excludes `NOT` and interaction-derived IPI/IGI/HGI; and retains at most eight
non-redundant terms per ontology aspect. The pinned corpus receipt records the
GO release and source hashes.

## Results

The three 2,048-dimensional Seed embedding conditions have been produced and
audited. The ProteinPathway condition led the separately frozen GGI diagnostic;
that evidence motivated prior selection only. No GraD-Pert prediction run has
yet established an E-row effect, so prediction results remain pending strict
training and evaluation receipts.
