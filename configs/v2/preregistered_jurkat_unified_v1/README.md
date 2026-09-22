# GraD-Pert v2 Jurkat experiment matrix

This directory contains 46 self-contained training configurations in 15 groups.
The default parent uses two RTX 5090 GPUs, 64 cells per GPU, no accumulation,
`row_mean` loss reduction, and exclusion of frozen test-target expression genes
from every training expression input and target. Gene identities and graph
priors remain available. The S1 comparison switches all applicable losses to
`condition_mean` together.

`B0` and `H1` are the initial groups. Every other group is prospective and
requires a validation-selected parent before launch; its existing YAML files
are not a launch queue. `H3` contains only the two measured global batches,
64 and 128. `G1` adds a separately sealed expression holdout to the default
test-target exclusion.

The capacity evidence named in `matrix.json` was measured on clean published
source `1d8c7d7a2359d4388ecb93dbb8f052678306f063`. A subsequent fix to
the prediction-only path handles batches with no SSL cell views; it does not
change the joint-SSL path measured by those capacity probes. Every actual run
must use an exact configuration, source commit, seed and GPU topology preflight
from its final published source. Capacity evidence is engineering evidence, not
a scientific result or proof of 50-epoch stability.
