from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradpert.execution.ablation_matrix import (
    build_ablation_launch_plan,
    load_ablation_matrix,
)
from gradpert.hashing import sha256_json
from gradpert.pilots import GenePTAvailabilityReceipt

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/ablations/nadig_jurkat/matrix.json"


def test_launcher_validates_every_frozen_config_before_planning(tmp_path: Path) -> None:
    rows = load_ablation_matrix(MATRIX, repository_root=ROOT)
    assert len(rows) == 22
    plan = build_ablation_launch_plan(
        rows,
        selected_variants=("a0_default", "d2_control_transformer"),
        runs_root=tmp_path / "runs",
        device="cuda:1",
        genept_availability_receipt=None,
    )
    assert [row.variant_id for row in plan] == [
        "a0_default",
        "d2_control_transformer",
    ]
    assert all(row.disposition == "run" for row in plan)
    assert all(row.run_root.endswith("gradpert_b2/nadig_jurkat/seed-1") for row in plan)


def test_launcher_rejects_a_tampered_config_hash(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["rows"][0]["config_sha256"] = "0" * 64
    tampered = tmp_path / "matrix.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_ablation_matrix(tampered, repository_root=ROOT)


def test_genept_missing_target_receipt_skips_before_model_construction(
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

    plan = build_ablation_launch_plan(
        rows,
        selected_variants=("e1_frozen_genept",),
        runs_root=tmp_path / "runs",
        device="cuda:0",
        genept_availability_receipt=path,
    )
    assert plan[0].disposition == "skip_genept_missing_target"
    assert not Path(plan[0].run_root).exists()


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
