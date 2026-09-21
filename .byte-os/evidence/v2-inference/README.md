# V2 inference batch evidence

These are small server engineering receipts, not scientific model results.
Both load the same two-rank micro74 midpoint checkpoint, trained at
`f92ff5564afac35c48976469dd1d0b4a980bd982`; evaluation code is
`788ba3fcf25d46577683fd8c52b72e9a4fe2a328`. Training configuration and checkpoint
hashes, canonical identities, exact300controls and ordered gene IDs are in each
receipt. Predictions and checkpoints remain on the server.

| Recipe | batch2 seconds | batch64 seconds | batch128 seconds | Largest absolute difference | Receipt SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Full5000gene output in ordered1000gene blocks |129.478|4.995|4.102|8.654e-6|664cd6f2a4bfb9e8a1667a449675195133912d61192acd495911d19078cf42d9|
| Single5000gene context repeat |109.495|5.777|4.482|9.513e-6|a61ad3e8ae5b1be4ee4753528d3973e4db55e03e06c590c9b3523a69e4d56255|

All listed points passed fixed atol=rtol=2e-5 against their own batch2 reference.
There is no equivalence claim between the two gene-context recipes. Measurements
include graph computation/transfers; first points may include cold overhead.
The two recipes ran concurrently on separate otherwise idle GPUs. This is not a
multi-condition validation benchmark or a50epoch performance result.

Evaluation batch128 is a candidate supported by both recipes. The full-axis
partitioned recipe peaks at3811630592 allocated bytes; the full-context recipe
peaks at17509217792 bytes. Earlier exploration found full-context256 slower and
300 OOM; those failed-run files remain on the server. Do not replace a failure
receipt with a success or describe128 as the maximum capacity.

The `ddp_m74_eval128` and `ddp_m75_eval128` profiles change only evaluation batch
from their original dual-rank profiles. They are untested complete-path candidates
until new capacity receipts pass. Original training-capacity evidence does not
claim to have used evaluation batch128. Formal configs still require exact-source,
configuration and seed admission before launch.
