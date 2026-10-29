"""Synthetic immutable sidecar contracts; no server or model execution."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gradpert.hashing import sha256_json
from gradpert.training.v2 import population_anchor as pa


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=digest(path))


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "SERVER_ROOT", tmp_path)
    counts = dict(conditions=2, treated_rows=9, allowed=3, forbidden=2)
    monkeypatch.setattr(pa, "COUNTS", counts)
    output = tmp_path / "output"
    private = output / "server-only"
    private.mkdir(parents=True)
    allowed = np.array([0, 2, 4], dtype=np.int64)
    means = np.array([[0, 1, 3], [3, 8, 0]], dtype=np.float32)
    identity = dict(
        source={"historical": True},
        data_manifest="a" * 64,
        split_content_sha256="b" * 64,
        selection={"selected": True},
        ordered_used_row_ids_sha256="c" * 64,
        allowed_policy={"excluded": 2},
        counts=counts,
        metadata_sha256="d" * 64,
    )
    bankdir = tmp_path / "bank"
    bankdir.mkdir()
    np.savez(bankdir / "bank.npz", allowed_gene_indices=allowed)
    bankseal = dump(
        bankdir / "manifest.json",
        dict(
            identity=identity,
            condition_ids=["p", "q"],
            arrays=dict(path="bank.npz", sha256=digest(bankdir / "bank.npz")),
        ),
    )
    bank = SimpleNamespace(
        manifest_path=bankdir / "manifest.json",
        manifest_sha256=bankseal["sha256"],
        arrays_path=bankdir / "bank.npz",
        arrays_sha256=digest(bankdir / "bank.npz"),
        identity=identity,
        condition_ids=("p", "q"),
        allowed_gene_indices=allowed,
        control_slot_offsets=np.array([0, 4, 9], np.int64),
    )
    member_entries = [
        dict(
            eligible=True,
            condition_id=p,
            rows=dict(
                rows=[str(i) for i in rows], ordered_sha256=sha256_json([str(i) for i in rows])
            ),
        )
        for p, rows in [("p", range(4)), ("q", range(4, 9))]
    ]
    membership_seal = dump(
        tmp_path / "repeat/server-only/memberships.json",
        dict(splits=dict(selected_train=member_entries)),
    )
    repeat = dump(
        tmp_path / "repeat/COMPLETE.json",
        dict(
            source=dict(commit=pa.REPEAT_SOURCE),
            closure_verified=True,
            zero_PKL_within_output=True,
            outputs_sha256={"server-only/memberships.json": membership_seal["sha256"]},
        ),
    )
    census = dump(tmp_path / "census.json", {})
    preprocessing = dump(tmp_path / "preprocessing.json", {})
    monkeypatch.setattr(
        pa,
        "ACCEPTED_INPUT_SHAS",
        {
            "census_complete": census["sha256"],
            "repeatability_complete": repeat["sha256"],
            "preprocessing": preprocessing["sha256"],
        },
    )
    publication = dump(tmp_path / "publication.json", {"verification_method": "git_ls_remote_live"})
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = dict(
        repository_root=str(source_root),
        commit="e" * 40,
        dirty=False,
        formal_eligible=True,
        published_commit="e" * 40,
        publication_receipt_sha256=publication["sha256"],
    )
    bound = dict(**bankseal, arrays_sha256=bank.arrays_sha256)
    spec = dump(
        tmp_path / "SPEC.json",
        dict(
            schema="population-prediction-preparation-spec-v1",
            sampling_bank=bound,
            analysis_source=dict(commit=source["commit"], publication=publication),
            census_complete=census,
            repeatability_complete=repeat,
            preprocessing=preprocessing,
            expected_counts=counts,
        ),
    )
    np.savez(private / "population_targets.npz", allowed_gene_indices=allowed, treated_mean=means)
    array_sha = digest(private / "population_targets.npz")
    metadata = dict(
        schema=pa.SCHEMA,
        mode="population_prediction",
        condition_ids=["p", "q"],
        sampling_bank=bound,
        identity=identity,
        arrays=dict(path="population_targets.npz", sha256=array_sha),
        target_rule=pa.TARGET_RULE.copy(),
        provenance=dict(
            spec=spec,
            source=source,
            publication=publication,
            census_complete=census,
            repeatability_complete=repeat,
            preprocessing=preprocessing,
            memberships=membership_seal,
        ),
        membership=dict(
            condition_ids_sha256=sha256_json(["p", "q"]),
            ordered_treated_row_ids_sha256=sha256_json([str(i) for i in range(9)]),
            selected_membership_sha256=sha256_json(member_entries),
            group_sizes=[4, 5],
        ),
        numeric_read_audit=dict(
            rows=9,
            columns=3,
            logical_values=27,
            calls=1,
            row_ids_sha256=sha256_json([str(i) for i in range(9)]),
            read_row_ids=[str(i) for i in range(9)],
            metadata_sha256=identity["metadata_sha256"],
            forbidden_calls={"expression_loader_calls": 0},
        ),
    )
    dump(output / "PRE_NUMERICAL_METADATA.json", {})
    dump(output / "summary.json", {})

    def seal():
        metadata["arrays"]["sha256"] = digest(private / "population_targets.npz")
        manifest = dump(private / "manifest.json", metadata)
        dump(
            output / "COMPLETE.json",
            dict(
                schema=pa.COMPLETE_SCHEMA,
                source=source,
                spec_sha256=spec["sha256"],
                closure_verified=True,
                zero_PKL_within_output=True,
                outputs_sha256={
                    str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*")
                    if p.is_file() and p.name != "COMPLETE.json"
                },
            ),
        )
        return manifest["path"], manifest["sha256"], metadata["arrays"]["sha256"], bank

    return SimpleNamespace(
        output=output,
        private=private,
        metadata=metadata,
        bank=bank,
        allowed=allowed,
        means=means,
        seal=seal,
    )


def test_success_is_readonly_detached_exact_target(sidecar):
    original = copy.deepcopy(sidecar.bank.identity)
    value = pa.load_population_targets(*sidecar.seal())
    np.testing.assert_array_equal(value, sidecar.means)
    assert value.dtype == np.float32 and not value.flags.writeable
    assert sidecar.bank.identity == original
    with pytest.raises(ValueError):
        value[0, 0] = 4


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema="old"),
        lambda m: m.update(mode="mse"),
        lambda m: m.update(extra=True),
        lambda m: m.update(condition_ids=["q", "p"]),
        lambda m: m["sampling_bank"].update(arrays_sha256="a" * 64),
        lambda m: m.update(identity={}),
        lambda m: m["membership"].update(group_sizes=[5, 4]),
        lambda m: m["target_rule"].update(context_weighting="equal_context"),
        lambda m: m["target_rule"].update(own_target="zero"),
        lambda m: m["provenance"]["source"].update(dirty=True),
        lambda m: m["provenance"]["source"].update(published_commit="f" * 40),
        lambda m: m["numeric_read_audit"].update(rows=10),
        lambda m: m["numeric_read_audit"].update(logical_values=28),
        lambda m: m["numeric_read_audit"].update(calls=True),
        lambda m: m["numeric_read_audit"]["forbidden_calls"].update(expression_loader_calls=1),
    ],
)
def test_resealed_semantic_tampering_rejected(sidecar, change):
    change(sidecar.metadata)
    with pytest.raises(ValueError):
        pa.load_population_targets(*sidecar.seal())


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(treated_mean=np.full((2, 3), np.nan, np.float32)),
        lambda a: a.update(treated_mean=np.full((2, 3), -1, np.float32)),
        lambda a: a.update(treated_mean=np.ones((2, 3), np.float64)),
        lambda a: a.update(treated_mean=np.ones((3, 2), np.float32)),
        lambda a: a.update(allowed_gene_indices=np.array([0, 2, 3], np.int64)),
        lambda a: a.update(allowed_gene_indices=np.array([0, 2, 4], np.int32)),
        lambda a: a.update(unapproved=np.ones(3)),
    ],
)
def test_npz_contract_rejects_invalid_targets_axes_and_extra_arrays(sidecar, change):
    arrays = dict(allowed_gene_indices=sidecar.allowed, treated_mean=sidecar.means)
    change(arrays)
    np.savez(sidecar.private / "population_targets.npz", **arrays)
    with pytest.raises(ValueError):
        pa.load_population_targets(*sidecar.seal())


def test_raw_hash_tamper_missing_complete_and_failed_output(sidecar):
    args = sidecar.seal()
    with pytest.raises(ValueError, match="SHA256"):
        pa.load_population_targets(args[0], "0" * 64, *args[2:])
    (sidecar.output / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        pa.load_population_targets(*args)
    args = sidecar.seal()
    (sidecar.output / "FAILURE.json").write_text("{}")
    with pytest.raises(ValueError, match="closure"):
        pa.load_population_targets(*args)


def test_external_input_tampering_and_symlink_are_rejected(sidecar):
    args = sidecar.seal()
    p = sidecar.bank.arrays_path
    p.write_bytes(p.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        pa.load_population_targets(*args)


def test_manifest_array_and_ancestor_symlink_rejected(sidecar, tmp_path):
    args = sidecar.seal()
    alias = tmp_path / "alias"
    alias.symlink_to(sidecar.output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        pa.load_population_targets(alias / "server-only/manifest.json", *args[1:])
    array = sidecar.private / "population_targets.npz"
    moved = tmp_path / "outside.npz"
    array.rename(moved)
    array.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        pa.load_population_targets(*args)


def test_escape_and_relative_server_paths_rejected(sidecar):
    for value in [
        "relative",
        "/tmp/not-the-server.json",
        str(sidecar.output / "../output/summary.json"),
    ]:
        with pytest.raises((ValueError, FileNotFoundError)):
            pa.server_path(value)


def test_resealed_alternative_membership_lineage_is_not_accepted(sidecar):
    """Self-consistent replacement hashes cannot replace the accepted TRAIN lineage."""
    metadata = sidecar.metadata
    provenance = metadata["provenance"]
    member_path = pa.Path(provenance["memberships"]["path"])
    members = json.loads(member_path.read_text())
    entries = members["splits"]["selected_train"]
    entries[0]["rows"]["rows"][0] = "alternative-treated-row"
    entries[0]["rows"]["ordered_sha256"] = sha256_json(entries[0]["rows"]["rows"])
    provenance["memberships"].update(dump(member_path, members))
    repeat_path = pa.Path(provenance["repeatability_complete"]["path"])
    repeat = json.loads(repeat_path.read_text())
    repeat["outputs_sha256"]["server-only/memberships.json"] = provenance["memberships"]["sha256"]
    provenance["repeatability_complete"].update(dump(repeat_path, repeat))
    spec_path = pa.Path(provenance["spec"]["path"])
    spec = json.loads(spec_path.read_text())
    spec["repeatability_complete"] = provenance["repeatability_complete"]
    provenance["spec"].update(dump(spec_path, spec))
    metadata["membership"].update(
        selected_membership_sha256=sha256_json(entries),
        ordered_treated_row_ids_sha256=sha256_json([r for e in entries for r in e["rows"]["rows"]]),
    )
    metadata["numeric_read_audit"]["row_ids_sha256"] = metadata["membership"][
        "ordered_treated_row_ids_sha256"
    ]
    metadata["numeric_read_audit"]["read_row_ids"] = [r for e in entries for r in e["rows"]["rows"]]
    with pytest.raises(ValueError, match=r"accepted.*lineage"):
        pa.load_population_targets(*sidecar.seal())
