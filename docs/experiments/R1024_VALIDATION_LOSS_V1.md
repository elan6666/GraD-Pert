# R1024 fixed-50 validation-loss selection

This continuation supersedes the proposed early-stop policy: all seven new
rows train exactly 50 epochs. Best is the strict minimum validation prediction
loss; last is the actual epoch-50 checkpoint. Both are tested automatically.
The completed S1/S2/U1/T1 and historical baseline remain unchanged.

Rows: T2, P1, P2, C1, L1, L2, L3 in self-contained
`configs/r50/r1024_loss_*_v1/gradpert_b2/nadig_jurkat.yaml` files.
Each retains its original independent scientific factor and batch1024.

Prediction validation loss is all-expression-gene MSE between each condition's
mean frozen-300-control prediction and mean observed validation expression,
then an equal average over validation conditions. It is not paired-cell MSE,
negative Pearson, nor the training auxiliary objective. Lower is better.

Every epoch also reports canonical TxPert delta, TriShift delta, and Systema
Pearson. The isolated validation reference state has only validation DE masks;
Systema uses training-condition means only. Test evaluation retains its existing
train-plus-validation reference. Thus validation and test scores have different
populations/reference scope and must not be mislabeled interchangeable.
Reference construction slices out test expression before materializing arrays.

Per-epoch JSON retains four values, metric finite counts/unavailable reasons,
reference SHA, epoch/global step, run ID and training source SHA. At training
completion, emit validation_curves.csv, training_loss.png/pdf,
validation_curves.png/pdf, and curves_manifest.json binding raw inputs/outputs.
Missing scores remain blank/plot gaps, never zero. Separate panels preserve scales.

The four-slot queue binds two processes to each physical GPU. A slot includes
training and best/last testing; it refills only after that complete child exits.
No automatic failed-row retry. Source/input drift halts dispatch and terminates
only owned process groups. Initial GPUs must be idle; per-process allocator
fraction remains 0.4 and the worker enforces free-memory reserve.

Before formal launch: clean published source and server parity, all CPU gates,
new single-step evidence, validation reference preparation/verification, and
all-training-anchor L3 reachability gate. Historical baseline reproduction is
a separate exact-source/protocol audit, not silently part of loss-selection.
