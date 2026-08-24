from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from gradpert.artifacts import (
    PredictionConditionArrays,
    PredictionProvenance,
    load_prediction_artifact,
    seal_prediction_artifact,
)
from gradpert.hashing import sha256_json

SHA = "a" * 64
COMMIT = "b" * 40


def _provenance(gene_ids: tuple[str, ...]) -> PredictionProvenance:
    return PredictionProvenance(
        model_id="gradpert_b2",
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        run_id="fixture-run",
        run_seed=1,
        source_commit=COMMIT,
        source_dirty=False,
        formal_eligible=True,
        config_sha256=SHA,
        environment_sha256=SHA,
        canonical_data_sha256=SHA,
        gene_order_sha256=sha256_json(list(gene_ids)),
        split_content_sha256=SHA,
        control_manifest_sha256=SHA,
        checkpoint_sha256=SHA,
    )


def _condition(condition_id: str, offset: float = 0.0) -> PredictionConditionArrays:
    controls = np.arange(900, dtype=np.float32).reshape(300, 3)
    return PredictionConditionArrays(
        condition_id=condition_id,
        prediction=controls + offset,
        input_control=controls,
        input_control_row_ids=tuple(f"ctrl-{index % 17}" for index in range(300)),
    )


def test_prediction_artifact_round_trip_is_condition_keyed_and_truth_free(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prediction.pkl"
    artifact = seal_prediction_artifact(
        path,
        provenance=_provenance(("A", "B", "C")),
        gene_ids=("A", "B", "C"),
        conditions=[_condition("PERT_B", 2.0), _condition("PERT_A", 1.0)],
    )

    assert tuple(artifact.conditions) == ("PERT_A", "PERT_B")
    assert artifact.manifest.truth_included is False
    assert artifact.conditions["PERT_A"].prediction.shape == (300, 3)
    assert artifact.conditions["PERT_A"].input_control_row_ids[0] == "ctrl-0"
    assert b"Truth" not in path.read_bytes()


def test_loader_rejects_file_corruption_before_unpickling(tmp_path: Path) -> None:
    path = tmp_path / "prediction.pkl"
    artifact = seal_prediction_artifact(
        path,
        provenance=_provenance(("A", "B", "C")),
        gene_ids=("A", "B", "C"),
        conditions=[_condition("PERT_A")],
    )
    with path.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_prediction_artifact(
            path,
            expected_file_sha256=artifact.file_sha256,
            trusted_root=tmp_path,
        )


def test_loader_rejects_path_outside_trusted_root(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside.pkl"
    artifact = seal_prediction_artifact(
        outside,
        provenance=_provenance(("A", "B", "C")),
        gene_ids=("A", "B", "C"),
        conditions=[_condition("PERT_A")],
    )

    with pytest.raises(ValueError, match="outside trusted_root"):
        load_prediction_artifact(
            outside,
            expected_file_sha256=artifact.file_sha256,
            trusted_root=trusted,
        )


def test_loader_rejects_truth_like_payload_even_with_matching_file_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prediction.pkl"
    artifact = seal_prediction_artifact(
        path,
        provenance=_provenance(("A", "B", "C")),
        gene_ids=("A", "B", "C"),
        conditions=[_condition("PERT_A")],
    )
    with path.open("rb") as handle:
        package = pickle.load(handle)
    package["conditions"]["PERT_A"]["Truth"] = np.zeros((1, 3))
    with path.open("wb") as handle:
        pickle.dump(package, handle)
    import hashlib

    matching_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert matching_hash != artifact.file_sha256

    with pytest.raises(ValueError, match="truth-like field"):
        load_prediction_artifact(
            path,
            expected_file_sha256=matching_hash,
            trusted_root=tmp_path,
        )


def test_sealer_rejects_wrong_population_shape_and_gene_hash(tmp_path: Path) -> None:
    bad = _condition("PERT_A")
    bad = PredictionConditionArrays(
        condition_id=bad.condition_id,
        prediction=bad.prediction[:299],
        input_control=bad.input_control,
        input_control_row_ids=bad.input_control_row_ids,
    )
    with pytest.raises(ValueError, match="shape"):
        seal_prediction_artifact(
            tmp_path / "bad.pkl",
            provenance=_provenance(("A", "B", "C")),
            gene_ids=("A", "B", "C"),
            conditions=[bad],
        )

    with pytest.raises(ValueError, match="gene_order_sha256"):
        seal_prediction_artifact(
            tmp_path / "bad-genes.pkl",
            provenance=_provenance(("A", "B", "C")),
            gene_ids=("A", "B", "D"),
            conditions=[_condition("PERT_A")],
        )
