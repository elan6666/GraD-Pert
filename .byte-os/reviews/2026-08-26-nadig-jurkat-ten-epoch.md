# Nadig Jurkat 10-Epoch Comparison Review — 2026-08-26

## Verdict

`pass_with_limits`: B2 systems-only and B3 combined completed their exact
fixed 10-epoch contracts and passed all 71 strict validation checks. B3 is
faster under this concurrent two-GPU measurement. The metric observations do
not establish unchanged predictive effect or statistical equivalence.

## Frozen execution contract

- Source: clean local/GitHub/server commit
  `ddf40fd14db8c07da1e03ddf381508a2012ac632`.
- Dataset/model: Nadig Jurkat, native `gradpert_b2`, seed 1.
- Shared settings: batch 256, 16,384 prototypes,
  `expandable_segments:True`, 10 epochs, 5,820 ordered optimizer steps, 10
  validations, no early termination, one final test evaluation, and
  `metrics_only` with zero persistent PKL.
- B2: full graph, 6,506 nodes and 222,654 nonself edges, all seven systems
  groups, GPU0.
- B3: recomputed Top-500-HVG-plus-target graph, 2,798 nodes and 89,561 nonself
  edges, all seven systems groups, GPU1.
- The failed initial B3 launch used the wrong runtime graph root and stopped
  before optimizer step 1. Its immutable evidence remains under the server
  `superseded/20260826-b3-ten-ddf4-wrong-graph-root` path and is not mixed.

## Results

| Variant | Training wall | Steps/s | Cells/s | Peak allocated GPU | Peak CPU RAM |
|---|---:|---:|---:|---:|---:|
| B2 systems-only | 7,221.773 s | 0.8059 | 177.61 | 18.93 GiB | 26.44 GiB |
| B3 combined | 5,223.127 s | 1.1143 | 245.57 | 14.70 GiB | 26.41 GiB |

B3 is 1.3827x faster than B2 and reduces training wall by 27.68%. Both values
use the monotonic complete-training wall and all 1,282,660 processed cells.

| Variant | TxPert macro Pearson delta | TriShift Pearson delta | Systema Pearson |
|---|---:|---:|---:|
| B2 systems-only | 0.251625 | 0.253596 | 0.031636 |
| B3 combined | 0.243824 | 0.251376 | 0.011911 |
| B3 minus B2 | -0.007801 | -0.002219 | -0.019724 |

The three metrics are descriptive. No threshold test, statistical uncertainty
analysis, or effect-equivalence claim is applied.

## Validation evidence

For each run the verifier proves exact source/config/run identity, 10 completed
epochs, ordered global steps 0–5,819, 582 steps per epoch, 10 validation
receipts, no canonical test truth during fit, one test evaluation, exact
three-metric schema, finite metrics, matching checkpoint hashes, only
`checkpoints/best.pt`, and zero PKL or result-work directory.

It also proves 5,000 expression/output/evaluation genes, expected graph counts,
all seven systems groups requested and active, exact update order, and equality
with the corresponding sealed one-epoch variant for canonical data, split,
control manifest, gene/graph identities, and exact ordered control/truth row
IDs.

- B2 run-manifest SHA-256:
  `d5d286e7fb89a5e4f21926df3831081b8d773021ac972437f88be5e414b05389`.
- B2 checkpoint SHA-256:
  `4340c41b65a10d0a8899b4c3bf1a00ea1fa04355f630505d80084a3d3881b9cf`.
- B3 run-manifest SHA-256:
  `b4f757e5105ac7fa0509ed72ebf789d1bd1eb05b1750d4876988e2ce2420539d`.
- B3 checkpoint SHA-256:
  `685b37b7dd4516a66b44222c186225b757aaceff493744399c673fa6ce95ca5c`.
- Strict PASS receipt SHA-256:
  `be5294923d7bc2c9655d50a3c4f11b8b31de1799e1f7bbe8f60150aaeff1b05b`.
- Verifier SHA-256:
  `38e761df35264e3d41f50d8f7c5bd242537cf859a3c5eea3283d3006cc85e7a3`.

## Limitations

1. B2 and B3 ran concurrently on separate GPUs but shared host CPU, RAM, and
   storage. This comparison is useful within the paired run, but its absolute
   timing should not be mixed with the earlier sequential one-epoch pilot.
2. Receipt keys named `one_epoch_training_wall_ms` and
   `one_epoch_fit_wall_ms` are legacy schema names. For these fixed-duration
   runs they cover all 10 epochs; the verifier records this explicitly.
3. Ten epochs provide more training evidence than one epoch, but do not by
   themselves prove unchanged effect.

Large logs, checkpoints, H5AD, and scientific matrices remain server-side. The
tracked evidence bundle contains only contracts, the verifier, and its compact
PASS receipt.
