from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gradpert.graphs.materialization import _atomic_graph, _load_pruned_graph
from gradpert.graphs.pruning import prune_incoming_edges
from gradpert.graphs.registry import GraphSourceRegistry, load_graph_source_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_public_graph_registry_locks_exact_official_files() -> None:
    registry = load_graph_source_registry(
        PROJECT_ROOT / "registry" / "graphs" / "public_string_go.yaml"
    )
    assert registry.commit == "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
    assert registry.sources["go"].sha256 == (
        "ad469c852ba9b8b5489749c7987467687aa566598fc0706fd7232aa27edce27b"
    )
    assert registry.sources["string"].sha256 == (
        "55d312f8d6186078eb00a7caf108ef731cd27aa08a5b48de59cd0333f0206404"
    )


def test_graph_registry_rejects_unsafe_relative_path() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "registry" / "graphs" / "public_string_go.yaml").read_text(encoding="utf-8")
    )
    payload["sources"]["go"]["relative_path"] = "../go.csv"
    with pytest.raises(ValueError, match="safe and relative"):
        GraphSourceRegistry.model_validate(payload)


def test_pruned_graph_npz_roundtrip_is_pickle_free(tmp_path: Path) -> None:
    graph = prune_incoming_edges(
        source_name="go",
        gene_ids=("A", "B", "C"),
        weighted_edges=(
            ("A", "B", 2.0),
            ("C", "B", 1.0),
            ("B", "B", 99.0),
        ),
        top_k=20,
    )
    path = tmp_path / "go.npz"
    _atomic_graph(path, graph)
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"edge_index", "edge_weight"}
    restored = _load_pruned_graph(
        path,
        source_name="go",
        gene_ids=("A", "B", "C"),
    )
    assert restored == graph
