# Decoder fusion-width formal review

## Verdict

Pass for D3--D6 at exact source `75a2c2b`. All four rows satisfy the formal
ten-epoch, best-checkpoint and zero-PKL contract. No tested decoder replaces
the preregistered additive A0.

## Evidence checked

- Recomputed 5,820 contiguous steps and ten ordered validations per row.
- Confirmed test truth was absent during fitting and exactly one evaluation
  used the hash-matched retained `best.pt`.
- Confirmed the exact three finite metrics and identical canonical data,
  split, condition order, ordered 300-control rows and truth-row identities.
- Confirmed the intended 2-by-2 decoder mode and perturbation-width factors;
  all other frozen model, view, loss and system settings match.
- Confirmed zero PKL over the full lineage, no surviving evaluation work
  directory and exactly one `best.pt` per row.

## Scientific reading

At width 64, adding the Transformer raises TriShift Pearson delta but lowers
TxPert macro delta and Systema Pearson. At width 256, it again raises TriShift
slightly while lowering the other two metrics. Increasing width from 64 to 256
does not produce a consistent gain under either fusion route. These are
single-seed point estimates; no equivalence or general superiority claim is
made. A0 is cross-commit contextual evidence, and D5/D6 are not single-factor
comparisons to A0.

Reviewed compact evidence is in
`../evidence/vnext-performance/formal-decoder-75a2c2b.json`.
