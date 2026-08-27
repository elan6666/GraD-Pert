# Graph Module Rules

- The expression/output/evaluation universe is the ordered canonical Top-5000
  axis. A separately sealed vNext runtime graph axis may be smaller while
  preserving exact canonical order and every perturbation target.
- STRING and GO are pruned independently to Top-20 incoming non-self edges.
- W0 numeric weights rank edges and never enter the encoder. Explicit W1/W2/W3
  single-STRING GAT ablations may consume receipted weights; no other route may.
- Distance rings follow incoming-message direction; source graphs are unioned
  only for ring membership and remain separate for encoding.
- RingInduced locals retain every induced base edge. Fanout locals retain only
  sampled message-passing edges. Both add one self-loop per selected node and
  neither applies DropEdge.
- View randomness uses named deterministic seed derivation; never use a module-
  global random generator.
