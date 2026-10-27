from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.hashing import sha256_file, sha256_json

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/v2/response_grounding_metadata_census.py"
spec = importlib.util.spec_from_file_location("metadata_census", SCRIPT)
assert spec is not None and spec.loader is not None
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def metadata(groups=None, *, controls=4):
    groups = groups or [("A", "x", 4), ("B", "y", 4), ("V", "x", 2)]
    rows, conditions, contexts = [], [], []
    for condition, context, count in groups:
        for _ in range(count):
            rows.append(f"row{len(rows)}")
            conditions.append(condition)
            contexts.append(context)
    train = [i for i, p in enumerate(conditions) if p in {"A", "B", "C"}]
    pools = {}
    for context in sorted(set(contexts)):
        pool = []
        for _ in range(controls):
            row = f"row{len(rows)}"
            rows.append(row)
            conditions.append("ctrl")
            contexts.append(context)
            pool.append(row)
        if pool:
            pools[context] = pool
    return dict(
        row_ids=rows,
        condition_ids=conditions,
        context_ids=contexts,
        train_row_indices=train,
        train_conditions=["A", "B"],
        val_conditions=["V"],
        control_condition_id="ctrl",
        control_pools=pools,
    )


def test_cartesian_zero_bins_and_nonadditive_deduplicated_coverage():
    result = census.summarize_metadata(metadata())
    s = result["summary"]["splits"]["selected_train"]
    assert (s["treated_rows"], s["cartesian_groups"], s["observed_groups"]) == (8, 4, 2)
    assert s["structural_n0_groups"] == 2
    assert s["control_rows_unique_in_split_contexts"] == 8  # not 4 controls x4 groups
    assert s["eligible_condition_count_by_context_histogram"] == {"0": 0, "1": 2, "2": 0, "ge3": 0}
    assert sum(r["groups"] for r in result["histogram"] if r["split"] == "selected_train") == 4
    zero = next(
        r
        for r in result["coverage"]
        if r["split"] == "selected_train" and r["scope"] == "cartesian" and r["treated_bin"] == "0"
    )
    assert (zero["conditions"], zero["contexts"]) == (2, 2)
    assert all("condition_id" not in r and "context_id" not in r for r in result["coverage"])


def test_shared_control_is_counted_once_and_context_hist_ge3():
    m = metadata([("A", "x", 4), ("B", "x", 4), ("C", "x", 4), ("V", "x", 1)])
    m["train_conditions"] = ["A", "B", "C"]
    s = census.summarize_metadata(m)["summary"]["splits"]["selected_train"]
    assert s["control_rows_unique_in_split_contexts"] == 4
    assert s["eligible_groups"] == 3 and s["eligible_contexts"] == 1
    assert s["eligible_condition_count_by_context_histogram"]["ge3"] == 1


def test_four_both_required_not_two_or_three_and_not_reliability():
    result = census.summarize_metadata(
        metadata([("A", "x", 2), ("B", "x", 3), ("V", "x", 4)], controls=3)
    )
    a, v = (result["summary"]["splits"][x] for x in ("selected_train", "validation"))
    assert a["eligible_groups"] == v["eligible_groups"] == 0
    assert a["ineligible_reasons"] == {"both_lt4": 2}
    assert v["ineligible_reasons"] == {"control_lt4": 1}
    assert a["eligible_condition_count_by_context_histogram"]["0"] == 1
    assert "not reliability" in result["summary"]["interpretation"]


def test_validation_is_full_canonical_membership_not_selected_train_cap():
    m = metadata([("A", "x", 4), ("B", "x", 4), ("V", "x", 7)])
    m["train_row_indices"] = [0, 4]
    result = census.summarize_metadata(m)
    assert result["summary"]["splits"]["selected_train"]["treated_rows"] == 2
    assert result["summary"]["splits"]["validation"]["treated_rows"] == 7
    assert result["summary"]["splits"]["validation"][
        "ordered_treated_membership_sha256"
    ] == sha256_json(m["row_ids"][8:15])


def test_missing_split_condition_fails_instead_of_dropping_axis():
    m = metadata()
    m["val_conditions"].append("absent")
    with pytest.raises(census.MetadataIntegrityError, match="zero-row"):
        census.summarize_metadata(m)


def test_duplicate_ids_fail_before_control_dedup():
    m = metadata()
    m["row_ids"][-1] = m["row_ids"][-2]
    with pytest.raises(census.MetadataIntegrityError, match="duplicate canonical row IDs"):
        census.summarize_metadata(m)


def test_missing_controls_for_validation_fails_honestly():
    m = metadata([("A", "x", 1), ("B", "x", 1), ("V", "y", 1)])
    control_y = set(m["control_pools"].pop("y"))
    keep = [i for i, r in enumerate(m["row_ids"]) if r not in control_y]
    for key in ("row_ids", "condition_ids", "context_ids"):
        m[key] = [m[key][i] for i in keep]
    with pytest.raises(census.MetadataIntegrityError, match="validation contexts lack controls"):
        census.summarize_metadata(m)


def test_pool_membership_is_checked_not_just_equal_counts():
    m = metadata()
    m["control_pools"]["x"][0] = m["row_ids"][0]
    with pytest.raises(census.MetadataIntegrityError, match="control pools differ"):
        census.summarize_metadata(m)


def test_selection_cannot_include_validation_row():
    m = metadata()
    m["train_row_indices"].append(8)
    with pytest.raises(census.MetadataIntegrityError, match="non-TRAIN"):
        census.summarize_metadata(m)


class FakeBacked:
    def __init__(self):
        self.closed = False
        self.file = SimpleNamespace(close=self.close)

    @property
    def X(self):
        raise AssertionError("expression must never be accessed")

    def close(self):
        self.closed = True


class FakeNative:
    last = None
    read_on_init = False

    def __init__(self, **kwargs):
        FakeNative.last = self
        self._adata = FakeBacked()
        self.row_ids = ("a0", "a1", "b0", "b1", "control", "val0", "test")
        self.condition_ids = ("A", "A", "B", "B", "ctrl", "V", "T")
        self.context_ids = ("x",) * 7
        self.train_row_indices = (0, 1, 2, 3)
        self.control_pools = {"x": ("control",)}
        self.split = SimpleNamespace(
            train_conditions=("A", "B"),
            val_conditions=("V",),
            control_condition_id="ctrl",
            split_content_sha256="b" * 64,
        )
        self.manifest = SimpleNamespace(
            dataset_id="nadig_jurkat",
            protocol_id="within_cell_unseen_single",
            canonical_adata_sha256="a" * 64,
            observation_order_sha256=sha256_json(list(self.row_ids)),
            expression_gene_order_sha256="c" * 64,
            graph_gene_order_sha256="d" * 64,
        )
        if self.read_on_init:
            self._read_expression_indices([0])

    def _read_expression_indices(self, indices):
        return self._adata.X

    def iter_train_epoch(self):
        raise AssertionError("batch path should be intercepted")

    def close(self):
        self._adata.file.close()


def sealed_selection(tmp_path):
    original = FakeNative()
    source = dict(
        parent_h5ad_sha256="f" * 64,
        selected_train_row_ids=["a1", "b0"],
        selected_train_row_ids_sha256=sha256_json(["a1", "b0"]),
        train_condition_ids=["A", "B"],
        unmodified_row_ids_sha256=sha256_json(list(original.row_ids[4:])),
        cap=1,
        seed=42,
        sampling="condition_batch_largest_remainder_uniform_without_replacement",
        by_condition={"A": ["a1"], "B": ["b0"]},
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(source))
    proof = tmp_path / "equivalence.json"
    proof.write_text(
        json.dumps(
            dict(
                schema_version="gradpert-canonical-equivalence-1",
                files=[{"sha256": "a" * 64}, {"sha256": "f" * 64}],
                comparison=dict(X_values_equal=True, obs_equal=True, var_equal=True),
            )
        )
    )
    payload = {
        **source,
        "schema_version": "gradpert-v2-training-selection-1",
        "state": "frozen_training_row_selection",
        "dataset_id": original.manifest.dataset_id,
        "protocol_id": original.manifest.protocol_id,
        "parent_h5ad_sha256": "a" * 64,
        "split_content_sha256": "b" * 64,
        "observation_order_sha256": original.manifest.observation_order_sha256,
        "expression_gene_order_sha256": "c" * 64,
        "graph_gene_order_sha256": "d" * 64,
        "analysis_selection_path": str(analysis),
        "analysis_selection_sha256": sha256_file(analysis),
        "analysis_source_commit": "e" * 40,
        "equivalence_path": str(proof),
        "equivalence_sha256": sha256_file(proof),
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload))
    return path


def test_native_selection_and_fake_backed_X_never_read_and_closed(tmp_path, monkeypatch):
    path = sealed_selection(tmp_path)
    fixture = FakeNative()
    genes = ["A", "B", "V", "T"]
    canonical = tmp_path / "nadig_jurkat/within_cell_unseen_single/canonical"
    manifests = canonical.parent / "manifests"
    canonical.mkdir(parents=True)
    manifests.mkdir()
    for name in ("expression_gene_ids.txt", "graph_gene_ids.txt"):
        (canonical / name).write_text("\n".join(genes) + "\n")
    split = dict(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        policy_id="synthetic",
        split_seed=1,
        control_condition_id="ctrl",
        train_conditions=["A", "B"],
        val_conditions=["V"],
        test_conditions=["T"],
    )
    split_sha = sha256_json(split)
    (manifests / "split.json").write_text(
        json.dumps(
            {**split, "schema_version": "split-manifest-v1", "split_content_sha256": split_sha}
        )
    )
    manifest = dict(
        schema_version="canonical-data-v1",
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        state="canonical_ready",
        canonical_adata_path="synthetic-not-materialized.h5ad",
        canonical_adata_sha256="a" * 64,
        n_cells=7,
        n_expression_genes=4,
        n_graph_genes=4,
        n_conditions=4,
        n_controls=1,
        split_content_sha256=split_sha,
        observation_order_sha256=sha256_json(list(fixture.row_ids)),
        expression_gene_order_sha256=sha256_json(genes),
        graph_gene_order_sha256=sha256_json(genes),
    )
    for field in (
        "source_manifest_sha256",
        "preprocessing_manifest_sha256",
        "qc_manifest_sha256",
        "split_manifest_sha256",
        "evaluation_controls_sha256",
    ):
        manifest[field] = "a" * 64
    (manifests / "canonical.json").write_text(json.dumps(manifest))
    payload = json.loads(path.read_text())
    for field in (
        "split_content_sha256",
        "observation_order_sha256",
        "expression_gene_order_sha256",
        "graph_gene_order_sha256",
    ):
        payload[field] = manifest[field]
    path.write_text(json.dumps(payload))

    class Obs(dict):
        @property
        def columns(self):
            return self.keys()

    backed = FakeBacked()
    backed.shape = (7, 4)
    backed.obs_names = fixture.row_ids
    backed.var = {"gene_name": genes}
    backed.obs = Obs(
        condition=fixture.condition_ids,
        cell_type=["cell"] * 7,
        batch=["batch"] * 7,
        control=[p == "ctrl" for p in fixture.condition_ids],
    )
    monkeypatch.setitem(
        sys.modules, "anndata", SimpleNamespace(read_h5ad=lambda path, backed: fake)
    )
    fake = backed
    original = census.CanonicalTrainingData._read_expression_indices
    audit = census.new_audit()
    result, receipt = census.collect_metadata(
        dict(
            dataset_id="nadig_jurkat",
            protocol_id="within_cell_unseen_single",
            data_root=tmp_path,
            run_seed=1,
        ),
        dict(path=str(path), sha256=sha256_file(path)),
        audit,
    )
    assert result["train_row_indices"] == [1, 2] and receipt["selected_train_rows"] == 2
    assert backed.closed and census.CanonicalTrainingData._read_expression_indices is original
    assert not any(audit.values())
    assert census.summarize_metadata(result)["summary"]["splits"]["validation"]["treated_rows"] == 1


def test_expression_guard_installed_before_init_and_restored_on_failure(tmp_path, monkeypatch):
    path = sealed_selection(tmp_path)
    original = FakeNative._read_expression_indices
    monkeypatch.setattr(FakeNative, "read_on_init", True)
    audit = census.new_audit()
    with pytest.raises(RuntimeError, match="forbids _read_expression_indices"):
        census.collect_metadata(
            {}, dict(path=str(path), sha256=sha256_file(path)), audit, data_type=FakeNative
        )
    assert FakeNative.last._adata.closed and FakeNative._read_expression_indices is original
    assert audit["expression_loader_calls"] == 1


def test_native_selection_hash_failure_restores_guard_and_keeps_population(tmp_path):
    path = sealed_selection(tmp_path)
    original = FakeNative._read_expression_indices
    with pytest.raises(ValueError, match="file/hash mismatch"):
        census.collect_metadata(
            {}, dict(path=str(path), sha256="0" * 64), census.new_audit(), data_type=FakeNative
        )
    assert FakeNative.last.train_row_indices == (0, 1, 2, 3)
    assert FakeNative.last._adata.closed and FakeNative._read_expression_indices is original


def test_batch_guard_restoration_including_inherited_attribute():
    class Child(FakeNative):
        pass

    assert "iter_train_epoch" not in Child.__dict__
    audit = census.new_audit()
    with (
        pytest.raises(RuntimeError, match="forbids iter_train_epoch"),
        census.metadata_only_guard(Child, audit),
    ):
        Child().iter_train_epoch()
    assert "iter_train_epoch" not in Child.__dict__ and audit["batch_calls"] == 1


def test_no_COMPLETE_on_failed_closure_and_no_rewrite_after_completion(tmp_path):
    result = census.summarize_metadata(metadata())
    receipt = dict(audit_counts=census.new_audit())
    failed = tmp_path / "failed"
    failed.mkdir()

    def changed_input():
        assert (failed / "summary.json").is_file()
        raise ValueError("closure input changed")

    with pytest.raises(ValueError, match="closure"):
        census.write_results(failed, result, receipt, changed_input)
    assert not (failed / "COMPLETE.json").exists()
    assert (failed / "server-only/groups.csv").is_file()
    completed = tmp_path / "completed"
    completed.mkdir()
    census.write_results(completed, result, receipt, lambda: None)
    before = (completed / "COMPLETE.json").read_bytes()
    with pytest.raises(FileExistsError):
        census.write_results(completed, result, receipt, lambda: None)
    assert (completed / "COMPLETE.json").read_bytes() == before
    assert "condition_id" not in (completed / "summary.json").read_text()
    assert "condition_id" in (completed / "server-only/groups.csv").read_text()


def test_CLI_materialization_paths_fail_closed_outside_server(tmp_path):
    with pytest.raises(ValueError, match="under /data/yilangliu"):
        census._server_path(str(tmp_path / "census"))
    with pytest.raises(ValueError, match="under /data/yilangliu"):
        census._server_path("relative-output")
    assert census._server_path("/data/yilangliu/new-census") == Path("/data/yilangliu/new-census")
