"""Strict external TRAIN treated-mean targets for the population prediction mode.

No model buffers, optimizer state, expression reader, or historical-bank mutation.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gradpert.hashing import sha256_json

if TYPE_CHECKING:
    from .response_anchor import FixedResponseBank

SERVER_ROOT = Path("/data/yilangliu")
COUNTS = dict(conditions=1322, treated_rows=47807, allowed=4775, forbidden=225)
SCHEMA = "fixed-population-prediction-targets-v1"
COMPLETE_SCHEMA = "population-prediction-preparation-complete-v1"
ACCEPTED_INPUT_SHAS = {
    "census_complete": "f1e5fffff87d0edea95c5dd814865c250e10b146addfb72b363d78c3fc9e70be",
    "repeatability_complete": "f7777481138e2dbd16c4d86e48919b57d5dc6a5e8eee7f786001e19ef7088a95",
    "preprocessing": "89c06418c180a91d0eda26f936540c8709ed318facab2bef4652ca84bfb65cb8",
}
REPEAT_SOURCE = "809346f55f9025e84dfd1a3b2ca1643eda9182f4"
TARGET_RULE = dict(
    target="selected_train_treated_mean",
    accumulation="float64_then_float32",
    context_weighting="selected_treated_row_count",
    own_target="retain_allowed_values",
    control_values_read=False,
    excluded_columns_read=False,
)
_IDENTITY_FIELDS = {
    "source",
    "data_manifest",
    "split_content_sha256",
    "selection",
    "ordered_used_row_ids_sha256",
    "allowed_policy",
    "counts",
    "metadata_sha256",
}


def server_path(value: str | Path, *, exists: bool = True) -> Path:
    """Reject aliases, symlink ancestors and paths outside the scientific server."""
    if not isinstance(value, (str, Path)):
        raise ValueError("server path required")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve(strict=exists) != path.absolute()
        or not path.is_relative_to(SERVER_ROOT)
        or path == SERVER_ROOT
    ):
        raise ValueError("population targets require a non-symlink server path")
    return path


def checked_seal(binding: dict[str, Any]) -> Path:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("exact path/SHA256 binding required")
    digest = binding["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid sealed SHA256")
    path = server_path(binding["path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("sealed population input SHA256 mismatch")
    return path


def validate_arrays(
    allowed: NDArray, treated: NDArray, bank: FixedResponseBank
) -> NDArray[np.float32]:
    if (
        allowed.dtype != np.int64
        or allowed.shape != (COUNTS["allowed"],)
        or np.any(allowed < 0)
        or np.any(allowed >= COUNTS["allowed"] + COUNTS["forbidden"])
        or np.any(np.diff(allowed) <= 0)
        or not np.array_equal(allowed, bank.allowed_gene_indices)
        or treated.dtype != np.float32
        or treated.shape != (COUNTS["conditions"], COUNTS["allowed"])
        or not np.isfinite(treated).all()
        or np.any(treated < 0)
    ):
        raise ValueError(
            "finite nonnegative float32 population targets / exact allowed axis required"
        )
    treated.flags.writeable = False
    return treated


def validate_manifest(metadata: dict, bank: FixedResponseBank, arrays_sha256: str) -> None:
    """Bind targets to the original immutable sampling bank and accepted membership."""
    if (
        not isinstance(metadata, dict)
        or set(metadata)
        != {
            "schema",
            "mode",
            "condition_ids",
            "sampling_bank",
            "identity",
            "arrays",
            "target_rule",
            "provenance",
            "membership",
            "numeric_read_audit",
        }
        or metadata["schema"] != SCHEMA
        or metadata["mode"] != "population_prediction"
        or metadata["condition_ids"] != list(bank.condition_ids)
        or len(bank.condition_ids) != COUNTS["conditions"]
        or len(set(bank.condition_ids)) != len(bank.condition_ids)
        or metadata["identity"] != bank.identity
        or set(bank.identity) != _IDENTITY_FIELDS
        or bank.identity["counts"] != COUNTS
        or metadata["target_rule"] != TARGET_RULE
    ):
        raise ValueError("population schema/condition/bank identity/target rule mismatch")
    bound = metadata["sampling_bank"]
    if bound != dict(
        path=str(bank.manifest_path),
        sha256=bank.manifest_sha256,
        arrays_sha256=bank.arrays_sha256,
    ):
        raise ValueError("sampling bank binding mismatch")
    checked_seal(dict(path=bound["path"], sha256=bound["sha256"]))
    checked_seal(dict(path=str(bank.arrays_path), sha256=bound["arrays_sha256"]))
    actual_bank = json.loads(bank.manifest_path.read_text())
    if (
        actual_bank.get("identity") != bank.identity
        or actual_bank.get("condition_ids") != list(bank.condition_ids)
        or actual_bank.get("arrays") != dict(path=bank.arrays_path.name, sha256=bank.arrays_sha256)
    ):
        raise ValueError("in-memory sampling bank differs from its immutable manifest")
    arrays = metadata["arrays"]
    if not isinstance(arrays, dict) or arrays != dict(
        path="population_targets.npz", sha256=arrays_sha256
    ):
        raise ValueError("population array path/hash mismatch")
    provenance = metadata["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "spec",
        "source",
        "publication",
        "census_complete",
        "repeatability_complete",
        "preprocessing",
        "memberships",
    }:
        raise ValueError("complete population provenance required")
    for key in provenance.keys() - {"source"}:
        checked_seal(provenance[key])
    if any(provenance[key]["sha256"] != digest for key, digest in ACCEPTED_INPUT_SHAS.items()):
        raise ValueError("accepted census/repeatability/preprocessing lineage mismatch")
    source = provenance["source"]
    if (
        not isinstance(source, dict)
        or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", "")))
        or source.get("dirty") is not False
        or source.get("formal_eligible") is not True
        or source.get("published_commit") != source["commit"]
        or source.get("publication_receipt_sha256") != provenance["publication"]["sha256"]
    ):
        raise ValueError("clean published population builder source required")
    server_path(source["repository_root"])
    spec = json.loads(Path(provenance["spec"]["path"]).read_text())
    if (
        spec.get("schema") != "population-prediction-preparation-spec-v1"
        or spec.get("sampling_bank") != bound
        or spec.get("analysis_source", {}).get("commit") != source["commit"]
        or spec["analysis_source"].get("publication") != provenance["publication"]
        or any(
            spec.get(k) != provenance[k]
            for k in ("census_complete", "repeatability_complete", "preprocessing")
        )
        or spec.get("expected_counts") != COUNTS
    ):
        raise ValueError("population spec/provenance binding mismatch")
    repeat = json.loads(Path(provenance["repeatability_complete"]["path"]).read_text())
    if (
        repeat.get("closure_verified") is not True
        or repeat.get("zero_PKL_within_output") is not True
        or repeat.get("source", {}).get("commit") != REPEAT_SOURCE
        or repeat.get("output_sha256", {}).get("server-only/memberships.json")
        != provenance["memberships"]["sha256"]
        or Path(provenance["memberships"]["path"])
        != Path(provenance["repeatability_complete"]["path"]).parent
        / "server-only/memberships.json"
    ):
        raise ValueError("accepted TRAIN membership closure mismatch")
    private = json.loads(Path(provenance["memberships"]["path"]).read_text())
    entries = [e for e in private["splits"]["selected_train"] if e["eligible"] is True]
    rows = [r for e in entries for r in e["rows"]["rows"]]
    lengths = [len(e["rows"]["rows"]) for e in entries]
    membership = dict(
        condition_ids_sha256=sha256_json([e["condition_id"] for e in entries]),
        ordered_treated_row_ids_sha256=sha256_json(rows),
        selected_membership_sha256=sha256_json(entries),
        group_sizes=lengths,
    )
    if (
        metadata["membership"] != membership
        or [e["condition_id"] for e in entries] != list(bank.condition_ids)
        or len(rows) != COUNTS["treated_rows"]
        or len(set(rows)) != len(rows)
        or any(not isinstance(r, str) or not r for r in rows)
        or any(not 4 <= n <= 40 for n in lengths)
        or not np.array_equal(lengths, np.diff(bank.control_slot_offsets))
        or any(e["rows"]["ordered_sha256"] != sha256_json(e["rows"]["rows"]) for e in entries)
    ):
        raise ValueError("ordered TRAIN treated membership/group sizes mismatch")
    audit = metadata["numeric_read_audit"]
    if (
        not isinstance(audit, dict)
        or set(audit)
        != {
            "rows",
            "columns",
            "logical_values",
            "calls",
            "row_ids_sha256",
            "read_row_ids",
            "metadata_sha256",
            "forbidden_calls",
        }
        or audit["rows"] != COUNTS["treated_rows"]
        or audit["columns"] != COUNTS["allowed"]
        or audit["logical_values"] != COUNTS["treated_rows"] * COUNTS["allowed"]
        or type(audit["calls"]) is not int
        or audit["calls"] < 1
        or not isinstance(audit["read_row_ids"], list)
        or len(audit["read_row_ids"]) != len(rows)
        or any(not isinstance(r, str) or not r for r in audit["read_row_ids"])
        or set(audit["read_row_ids"]) != set(rows)
        or audit["row_ids_sha256"] != sha256_json(audit["read_row_ids"])
        or audit["metadata_sha256"] != bank.identity["metadata_sha256"]
        or not isinstance(audit["forbidden_calls"], dict)
        or not audit["forbidden_calls"]
        or any(type(n) is not int or n != 0 for n in audit["forbidden_calls"].values())
    ):
        raise ValueError("treated-only allowed-first numerical audit mismatch")


def load_population_targets(
    path: str | Path, manifest_sha256: str, arrays_sha256: str, bank: FixedResponseBank
) -> NDArray[np.float32]:
    """Load sealed server-only [P,G] TRAIN means, independent of model state_dict."""
    manifest = checked_seal(dict(path=str(path), sha256=manifest_sha256))
    if manifest.parent.name != "server-only" or manifest.name != "manifest.json":
        raise ValueError("population targets must reside in the fresh server-only output")
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("population manifest bytes changed")
    metadata = json.loads(raw)
    validate_manifest(metadata, bank, arrays_sha256)
    arrays = checked_seal(
        dict(path=str(manifest.parent / "population_targets.npz"), sha256=arrays_sha256)
    )
    output = manifest.parent.parent
    complete = json.loads(server_path(output / "COMPLETE.json").read_text())
    expected = {
        "PRE_NUMERICAL_METADATA.json",
        "summary.json",
        "server-only/manifest.json",
        "server-only/population_targets.npz",
    }
    if (
        complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("closure_verified") is not True
        or complete.get("zero_PKL_within_output") is not True
        or complete.get("source") != metadata["provenance"]["source"]
        or complete.get("spec_sha256") != metadata["provenance"]["spec"]["sha256"]
        or set(complete.get("outputs_sha256", {})) != expected
        or {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
        != expected | {"COMPLETE.json"}
        or any(p.is_symlink() for p in output.rglob("*"))
        or {str(p.relative_to(output)) for p in output.rglob("*") if p.is_dir()} != {"server-only"}
    ):
        raise ValueError("population preparation lacks exact successful output closure")
    for name, digest in complete["outputs_sha256"].items():
        checked_seal(dict(path=str(output / name), sha256=digest))
    if (
        complete["outputs_sha256"]["server-only/manifest.json"] != manifest_sha256
        or complete["outputs_sha256"]["server-only/population_targets.npz"] != arrays_sha256
    ):
        raise ValueError("population COMPLETE target hashes mismatch")
    payload = arrays.read_bytes()
    if hashlib.sha256(payload).hexdigest() != arrays_sha256:
        raise ValueError("population array bytes changed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != 2 or set(archive.files) != {
            "allowed_gene_indices",
            "treated_mean",
        }:
            raise ValueError("population NPZ requires exactly two arrays")
        allowed, treated = archive["allowed_gene_indices"].copy(), archive["treated_mean"].copy()
    return validate_arrays(allowed, treated, bank)
