# A1 exploratory combination

User authorization: 2026-09-07. Run A1 and the subsequently requested A2 only,
not other combinations or A0-long.
The combination is E3 (trainable GenePT-Seed initialization), L1 (Fanout), and
M1 (single STRING GAT, encoder dropout0.2). It is not a single-factor ablation.
Keep the ordered HVG512+targets axis, four half-budget locals, zero anchor-mask
views, additive decoder, batch256, prototypes16384, and losses1/.8/.4/.1.
The exact prior is Protein+Reactome+SIGNOR, SHA
`34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318`.

Use seed1, at most100epochs, validation-only early stopping with patience10,
min_delta0 and val/txpert_macro_pearson_delta maximized. Final test is once from
best.pt; metrics_only/zero persistent PKL. Early stopping is not a failure.
Preserve all earlier10epoch results. Selection used those test results, so this
is exploratory evidence and does not establish independent confirmatory gains.

Configuration is self-contained in
`configs/combinations/a1_e3_l1_m1/gradpert_b2/nadig_jurkat.yaml`; a separate explicit policy
keeps historical full and10epoch policies unchanged. Use the existing native
smoke/full entrypoints. Require clean synchronized source/full gates, exact
GenePT preflight, a fresh one-epoch integration/capacity check, then a fresh
long-run root. Never reuse smoke checkpoints as long-run initialization.
Do not launch CUDA with an active goal. Monitor hourly after launch.

## A2: A1 plus D5

A2 retains every A1 parameter except graph output width64 to256 and decoder
mode additive toconcat. The basal vector stays64 wide; the decoder consumes
concat(b64,p256), width320, without a Transformer. It uses the identical
100-epoch maximum, validation early stopping and seed1 protocol. Configuration:
`configs/combinations/a2_e3_l1_m1_d5/gradpert_b2/nadig_jurkat.yaml`.
Schedule A1 on GPU0 and A2 on GPU1 after separate integration gates. Use fresh
source/run lineages and never change a running checkout.

Status: A1 one-epoch smoke returned RC0 on source9277333; independent acceptance
and full launch remain required. A2 preparation requires new source gates and
its own smoke; no A2 CUDA launch is recorded yet.
