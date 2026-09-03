from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from gradpert.execution.ablation_matrix import (
    _command,
    build_ablation_launch_plan,
    load_ablation_matrix,
    main,
)
from gradpert.hashing import sha256_json
from gradpert.pilots import GenePTAvailabilityReceipt, GenePTSeedAvailabilityReceipt

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/ablations/nadig_jurkat/matrix.json"


def _write_matrix(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_launcher_validates_every_frozen_config_before_planning(tmp_path: Path) -> None:
    rows = load_ablation_matrix(MATRIX, repository_root=ROOT)
    assert len(rows) == 30
    plan = build_ablation_launch_plan(
        rows,
        selected_variants=("a0_ratio_ring_half", "d2_control_transformer"),
        runs_root=tmp_path / "runs",
        device="cuda:1",
        genept_availability_receipt=None,
    )
    assert [row.variant_id for row in plan] == [
        "a0_ratio_ring_half",
        "d2_control_transformer",
    ]
    assert all(row.disposition == "run" for row in plan)
    assert all(row.run_root.endswith("gradpert_b2/nadig_jurkat/seed-1") for row in plan)
    assert [row.matrix_schema_version for row in plan] == ["2", "2"]
    assert [row.semantic_factor for row in plan] == ["reference", "decoder_mode"]
    assert plan[0].declared_parameter_diffs == ()
    assert plan[1].declared_parameter_diffs == ("decoder_mode",)


def test_launcher_rejects_a_tampered_config_hash(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["rows"][0]["config_sha256"] = "0" * 64
    tampered = tmp_path / "matrix.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_ablation_matrix(tampered, repository_root=ROOT)


def test_schema_v2_rejects_a_forged_matrix_id(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["matrix_id"] = "forged"
    tampered = _write_matrix(tmp_path / "matrix.json", payload)
    with pytest.raises(ValueError, match="matrix id"):
        load_ablation_matrix(tampered, repository_root=ROOT)


def test_schema_v2_requires_the_exact_successor_variant_set(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["rows"][0]["variant_id"] = "legacy_or_unknown_coordinate"
    tampered = _write_matrix(tmp_path / "matrix.json", payload)
    with pytest.raises(ValueError, match="variant set"):
        load_ablation_matrix(tampered, repository_root=ROOT)


@pytest.mark.parametrize("missing_field", ["semantic_factor", "declared_parameter_diffs"])
def test_schema_v2_requires_semantic_fields(tmp_path: Path, missing_field: str) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["rows"][0].pop(missing_field)
    tampered = _write_matrix(tmp_path / "matrix.json", payload)
    with pytest.raises(ValueError, match="semantic declaration"):
        load_ablation_matrix(tampered, repository_root=ROOT)


def test_schema_v2_rejects_config_relabeling(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_id = {row["variant_id"]: row for row in payload["rows"]}
    by_id["l1_fanout_ratio_half"]["config_path"] = by_id["a0_ratio_ring_half"]["config_path"]
    by_id["l1_fanout_ratio_half"]["config_sha256"] = by_id["a0_ratio_ring_half"]["config_sha256"]
    tampered = _write_matrix(tmp_path / "matrix.json", payload)
    with pytest.raises(ValueError, match="config path does not match variant id"):
        load_ablation_matrix(tampered, repository_root=ROOT)


def test_schema_v2_binds_resolved_config_identity_to_variant_id(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    config_root = repository / "configs/ablations/nadig_jurkat"
    shutil.copytree(ROOT / "configs/ablations/nadig_jurkat", config_root)
    matrix = config_root / "matrix.json"
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    row = next(row for row in payload["rows"] if row["variant_id"] == "l1_fanout_ratio_half")
    config = repository / row["config_path"]
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["model"]["parameters"]["performance_pilot_variant"]["value"] = (
        "vnext_a0_ratio_ring_half"
    )
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    row["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    matrix.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config identity differs"):
        load_ablation_matrix(matrix, repository_root=repository)


def test_schema_v2_rejects_rehashed_multi_factor_config(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    config_root = repository / "configs/ablations/nadig_jurkat"
    shutil.copytree(ROOT / "configs/ablations/nadig_jurkat", config_root)
    matrix = config_root / "matrix.json"
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    row = next(row for row in payload["rows"] if row["variant_id"] == "l1_fanout_ratio_half")
    config = repository / row["config_path"]
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_payload["model"]["parameters"]["local_view_count"]["value"] = 8
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    row["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    matrix.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="resolved parameter diff differs"):
        load_ablation_matrix(matrix, repository_root=repository)


def test_genept_npz_variant_rejects_legacy_missing_target_receipt(
    tmp_path: Path,
) -> None:
    rows = load_ablation_matrix(MATRIX, repository_root=ROOT)
    receipt = GenePTAvailabilityReceipt(
        schema_version="genept-vnext-availability-v1",
        status="unavailable_missing_perturbation_targets",
        dataset_id="nadig_jurkat",
        identifier_matching="exact_case_sensitive",
        missing_non_target_policy="remove_preserving_canonical_order",
        missing_perturbation_target_policy="skip_variant_before_model_construction",
        parent_topology_content_sha256="1" * 64,
        candidate_target_order_sha256="2" * 64,
        genept_source_sha256="3" * 64,
        missing_perturbation_target_gene_ids=["MISSING"],
        missing_perturbation_target_gene_ids_sha256=sha256_json(["MISSING"]),
    )
    path = tmp_path / "genept.json"
    path.write_text(receipt.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="NPZ variant requires the Seed availability schema"):
        build_ablation_launch_plan(
            rows,
            selected_variants=("e1_frozen_genept",),
            runs_root=tmp_path / "runs",
            device="cuda:0",
            genept_availability_receipt=path,
        )


def test_genept_variant_requires_preflight_receipt(tmp_path: Path) -> None:
    rows = load_ablation_matrix(MATRIX, repository_root=ROOT)
    with pytest.raises(ValueError, match="sealed availability receipt"):
        build_ablation_launch_plan(
            rows,
            selected_variants=("e1_frozen_genept",),
            runs_root=tmp_path / "runs",
            device="cuda:0",
            genept_availability_receipt=None,
        )


def test_genept_seed_preflight_provenance_is_bound_to_each_e_row(tmp_path: Path) -> None:
    rows = load_ablation_matrix(MATRIX, repository_root=ROOT)
    genept_row = next(row for row in rows if row.variant_id == "e1_frozen_genept")
    assert genept_row.genept_artifact_path is not None
    assert genept_row.genept_expected_sha256 is not None
    assert genept_row.runtime_graph_root is not None
    data_root = tmp_path / "data"
    graph_manifest = data_root / genept_row.runtime_graph_root / "manifest.json"
    graph_manifest.parent.mkdir(parents=True)
    candidate_targets = [f"TARGET_{index}" for index in range(218)]
    graph_manifest.write_text(
        json.dumps(
            {
                "topology_content_sha256": "1" * 64,
                "graph_gene_order_sha256": "2" * 64,
                "graph_gene_count": 2809,
                "candidate_target_order_sha256": "3" * 64,
                "candidate_target_ids": candidate_targets,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt = GenePTSeedAvailabilityReceipt(
        schema_version="genept-seed-go-protein-pathway-availability-v2",
        status="available",
        dataset_id="nadig_jurkat",
        identifier_matching="exact_case_sensitive",
        extra_source_gene_policy="ignore_preserving_runtime_axis",
        missing_non_perturbation_gene_policy="omit_preserving_canonical_order",
        missing_perturbation_target_policy="fail_before_model_construction",
        parent_topology_content_sha256="1" * 64,
        parent_graph_gene_order_sha256="2" * 64,
        candidate_target_order_sha256="3" * 64,
        prior_contract_id="seed_go_protein_pathway_master_v1",
        runtime_graph_root=genept_row.runtime_graph_root,
        parent_graph_manifest_sha256=hashlib.sha256(graph_manifest.read_bytes()).hexdigest(),
        genept_source_path=genept_row.genept_artifact_path,
        genept_source_size_bytes=123,
        genept_source_sha256=genept_row.genept_expected_sha256,
        genept_model="doubao-embedding-vision",
        embedding_width=2048,
        source_gene_count=17730,
        source_gene_order_sha256="5" * 64,
        requested_runtime_gene_count=2809,
        requested_runtime_gene_order_sha256="2" * 64,
        selected_gene_count=2809,
        selected_gene_order_sha256="2" * 64,
        selected_matrix_sha256="6" * 64,
        extra_source_gene_count=14921,
        extra_source_gene_ids_sha256="7" * 64,
        ignored_missing_non_perturbation_gene_count=0,
        ignored_missing_non_perturbation_gene_ids_sha256=sha256_json([]),
        perturbation_target_gene_count=218,
        perturbation_target_gene_ids_sha256="3" * 64,
        zero_vector_gene_count=0,
        result_topology_content_sha256="1" * 64,
    )
    path = tmp_path / "genept-seed.json"
    path.write_text(receipt.model_dump_json(), encoding="utf-8")

    plan = build_ablation_launch_plan(
        rows,
        selected_variants=("a0_ratio_ring_half", "e1_frozen_genept"),
        runs_root=tmp_path / "runs",
        device="cuda:0",
        genept_availability_receipt=path,
        data_root=data_root,
    )

    assert plan[0].genept_preflight_receipt_path is None
    assert plan[1].disposition == "run"
    assert plan[1].genept_preflight_receipt_path == str(path.resolve())
    assert plan[1].genept_preflight_receipt_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        plan[1].genept_preflight_schema_version == "genept-seed-go-protein-pathway-availability-v2"
    )
    assert plan[1].genept_preflight_status == "available"
    assert plan[1].genept_preflight_artifact_sha256 == genept_row.genept_expected_sha256
    assert plan[1].genept_preflight_missing_target_ids_sha256 is None
    command = _command(
        plan[1],
        python_executable="python",
        data_root=data_root,
        repository_root=ROOT,
        source_publication_receipt=tmp_path / "publication.json",
        source_publication_receipt_sha256="a" * 64,
        source_publication_remote_ref="refs/heads/codex/vnext-performance",
    )
    assert command[command.index("--genept-preflight-receipt") + 1] == str(path.resolve())
    assert (
        command[command.index("--genept-preflight-receipt-sha256") + 1]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert command[command.index("--source-publication-remote-ref") + 1] == (
        "refs/heads/codex/vnext-performance"
    )

    live_payload = json.loads(graph_manifest.read_text(encoding="utf-8"))
    live_payload["topology_content_sha256"] = "9" * 64
    graph_manifest.write_text(json.dumps(live_payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="preflight differs from the live graph"):
        build_ablation_launch_plan(
            rows,
            selected_variants=("e1_frozen_genept",),
            runs_root=tmp_path / "runs-2",
            device="cuda:0",
            genept_availability_receipt=path,
            data_root=data_root,
        )


def test_dry_plan_binds_queue_publication_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    publication = tmp_path / "publication.json"
    publication.write_text("{}\n", encoding="utf-8")
    receipt_sha256 = "a" * 64

    assert (
        main(
            [
                "--matrix",
                str(MATRIX),
                "--repository-root",
                str(ROOT),
                "--data-root",
                str(tmp_path),
                "--runs-root",
                str(tmp_path / "runs"),
                "--receipt-root",
                str(tmp_path / "receipts"),
                "--expected-source-commit",
                "b" * 40,
                "--device",
                "cuda:0",
                "--variant",
                "a0_ratio_ring_half",
                "--source-publication-receipt",
                str(publication),
                "--source-publication-receipt-sha256",
                receipt_sha256,
                "--source-publication-remote-ref",
                "refs/heads/codex/vnext-performance",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_publication_receipt"] == str(publication)
    assert payload["source_publication_receipt_sha256"] == receipt_sha256
    assert payload["source_publication_remote_ref"] == ("refs/heads/codex/vnext-performance")
