# GenePT-Seed prior comparison

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

Pending the server embedding and training runs. This section must be populated
from strict receipts, never from console snippets alone.
