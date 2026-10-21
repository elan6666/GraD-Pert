# U24 analysis metadata binding and remaining family audit

U24 run `nadig_jurkat-seed1-20261020T225459Z-79e604a8b2dd4865bcec120d64efb839`
completed four training epochs and last-role evaluation at training/evaluation SHA
`adfcc08a9417c040630d08876b52de30515ff044`. Medium sealing failed before
analysis: the only parent/runtime difference is
`data.unseen_gene_mechanisms.gene_conditioned_readout: false -> true`.

The controller derives an independent parent with exactly the registered U24
metadata. Full equality against expected data is required; no canonical, split,
control, gene order, mechanism or other protocol difference is ignored. Native
sealing continues requiring exact checkpoint/runtime/derived-parent equality.
The diagnostic schedule and its population hash are deep-copied unchanged.
Recovery uses a fresh server output; original failed output, training/evaluation
source, checkpoint, queue and metrics remain immutable. No training or test is
repeated. Record the repair controller SHA independently from the training SHA.

CPU-only runtime inspection of the five remaining configurations confirms MR1
adds precisely its preregistered masked-response supervision metadata; P1 adds
its population/MMD metadata, batch-order and population scope. Their derived
parents allow exactly those recorded fixed values. C1, O1 and K2-fixed-prefix
have no data identity differences and retain the existing parent path. This
changes analysis orchestration only; model, training, numerical analysis and
resource policy files are unchanged. Scope-specific runtime equality and parity
checks remain mandatory; this is not a declaration of strict metric comparability.

Validation: 98 targeted tests passed (no-SSL, PT2 and medium contracts); Ruff
passed. Tests reject changed canonical/control/protocol/mechanism identities,
wrong arm, dirty/numerically modified source and old output paths; population
and input immutability are checked. Full server analysis is still pending.
