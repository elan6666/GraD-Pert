# CG1 medium analysis identity repair

CG1 training and last-role testing completed under source
`da35944228f9016332fe5e3805d62c6e715485b8`. Its same-medium seal failed
before any model analysis because the baseline parent did not contain
`data.loss_protocol.conditional_ssl1_scope`. The trained runtime records
`local_control_rows_global_condition_weights_masked_node_mean` there.
An independent recursive comparison found no other data identity difference.

The analysis controller now creates a derived parent with that exact recorded
field. Removing only this field must recover the complete original parent data
identity; every other difference fails closed. The original schedule, queries,
row/control selections, seeds and diagnostic population digest are preserved.
The native seal still requires full runtime/checkpoint/derived-parent equality.
The original parent, failed output, training source, configuration, checkpoint
and test result are preserved. Recovery uses a fresh analysis output directory.

CG1 is explicitly admitted to the existing analysis-only recovery path. The
source compatibility check additionally protects its family generator; model,
data, numerical observer and resource-policy implementations remain frozen.
Record the training SHA separately from the analysis controller SHA in recovery
receipts. No training or test inference is rerun by this repair.

Validation: 75 targeted tests passed across `test_no_ssl.py`, `test_pt2_flow.py`
and `test_cap40_posthoc_medium.py`; Ruff check and formatting passed. Tests cover
the exact allowed metadata addition, unchanged/deep-copied schedule, rejection
of canonical/control/protocol changes, dirty or numerically changed source,
wrong arms, and overwriting protected outputs. A completed server analysis and
its parity/state-preservation checks remain required before scientific delivery.
