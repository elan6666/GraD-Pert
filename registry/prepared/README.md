# Prepared dataset receipts

The current benchmark protocol is `datasets-v2`, using the frozen GEARS
default-graph representability intersection for every model. All five canonical
datasets, their new split/control manifests, graphs, and evaluator states have
been rebuilt and verified on the compute server.

The small v2 receipt mirror has not yet been synchronized back to this working
copy. `CURRENT_STATE.json` records that boundary and the five verified server
split hashes. No local file is relabeled or synthesized from partial terminal
output.

The earlier `datasets-v1` receipts remain under
`superseded/datasets-v1/` for audit only. They are not current inputs and must
not be admitted into a result catalog.

Once server access is restored, synchronize only the allowlisted JSON, CSV,
TXT, and Markdown receipts into the five dataset directories here, verify their
content hashes, and change `CURRENT_STATE.json` to `synchronized_verified`.
