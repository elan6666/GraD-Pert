from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
TXPERT = ROOT / "TxPert/official-repo"
FIXTURE = ROOT / "docs/provenance/fixtures/txpert_exphormer_mg_contract.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_frozen_public_exphormer_mg_sources_match_the_alignment_registry() -> None:
    fixture = _fixture()
    exphormer = TXPERT / str(fixture["official_exphormer_path"])
    multi_graph = TXPERT / str(fixture["official_multi_graph_path"])
    if not exphormer.is_file() or not multi_graph.is_file():
        pytest.skip("frozen official TxPert evidence checkout is not in this worktree")
    assert sha256_file(exphormer) == fixture["official_exphormer_sha256"]
    assert sha256_file(multi_graph) == fixture["official_multi_graph_sha256"]


def test_frozen_public_exphormer_mg_config_has_the_receipted_surface() -> None:
    fixture = _fixture()
    assert fixture["schema_version"] == "txpert-exphormer-mg-alignment-v2"
    assert fixture["official_commit"] == "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
    assert fixture["config"] == {
        "graph_sources": ["string", "go"],
        "graph_top_k": 20,
        "layer_type": "exphormer_w_mpnn",
        "num_layers": 4,
        "hidden_dim": 128,
        "dropout": 0.1,
        "num_heads": 2,
        "add_self_loops": True,
        "expander_degree": 3,
        "add_reverse_edges": True,
        "use_edge_weight": False,
        "union_edge_type": "multihot",
        "edge_feat_map_type": "linear",
    }


def test_frozen_public_exphormer_mg_fixture_records_official_synthetic_generation() -> None:
    golden = _fixture()["synthetic_union_golden"]
    assert golden["generation"] == {
        "device": "cpu",
        "expander_degree": 0,
        "official_class": "gspp.models.pert_models.exphormer.ExphormerModel",
        "official_commit": "08d82eea86746b044cf7531f4ec8c5f60e1cb73f",
        "union_edge_type": "multihot",
    }
