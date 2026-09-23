"""Seal an aligned GenePT PCA table on the server without expression access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gradpert.features.genept_reduction import pca_genept
from gradpert.features.text_prior import verify_text_prior_npz
from gradpert.graphs.materialization import load_dataset_graph_topology
from gradpert.hashing import sha256_file, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if not all(
        path.resolve().is_relative_to("/data/yilangliu")
        for path in (args.source, args.data_root, args.output)
    ):
        parser.error("GenePT scientific artifacts stay on the server")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("PCA artifact and receipt require new paths")
    topology = load_dataset_graph_topology(
        dataset_id=args.dataset_id, protocol_id=args.protocol_id, data_root=args.data_root
    )
    prior = verify_text_prior_npz(
        args.source,
        expected_sha256=args.source_sha256,
        expected_gene_ids=topology.gene_ids,
        perturbation_target_gene_ids=topology.gene_ids,
    )
    if prior.gene_ids != topology.gene_ids:
        raise ValueError("PCA requires complete aligned GenePT coverage")
    reduced, explained = pca_genept(prior.values, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        genes=np.asarray(prior.gene_ids),
        vectors=reduced,
        model=np.asarray(prior.model + "+pca" + str(args.width)),
    )
    receipt = {
        "method": "centered covariance eigenvectors, descending variance, canonical signs",
        "source_path": str(args.source.resolve()),
        "source_sha256": prior.source_sha256,
        "source_selected_matrix_sha256": prior.selected_matrix_sha256,
        "graph_gene_order_sha256": sha256_json(list(prior.gene_ids)),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "shape": list(reduced.shape),
        "explained_variance_ratio": explained,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
