# Graph Module Rules

- The graph universe is the ordered canonical gene axis only.
- STRING and GO are pruned independently to Top-20 incoming non-self edges.
- Numeric weights rank edges and never enter the encoder.
- Distance rings follow incoming-message direction; source graphs are unioned
  only for ring membership and remain separate for encoding.
- Local views retain every induced base edge and add exactly one self-loop per
  selected node. They never apply DropEdge.
- View randomness uses named deterministic seed derivation; never use a module-
  global random generator.
