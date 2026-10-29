"""Allowed-first canonical slices and selected-treated-only mean preparation."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gradpert.training.v2 import population_anchor as pa
from scripts.v2 import prepare_population_anchor as prep


def fixture():
    entries = [
        dict(condition_id="p", rows=[0, 1, 3, 4]),
        dict(condition_id="q", rows=[6, 7, 8, 9, 10]),
    ]
    metadata = dict(
        row_ids=[str(i) for i in range(14)],
        condition_ids=[
            "p",
            "p",
            "ctrl",
            "p",
            "p",
            "val",
            "q",
            "q",
            "q",
            "q",
            "q",
            "ctrl",
            "test",
            "val",
        ],
        context_ids=["a", "a", "a", "b", "b", "a", "a", "b", "b", "b", "b", "b", "a", "b"],
        train_row_indices=[0, 1, 3, 4, 6, 7, 8, 9, 10],
        train_conditions=["p", "q"],
        control_condition_id="ctrl",
    )
    return entries, metadata, dict(splits=dict(selected_train=dict(eligible=entries)))


class SliceSentinel:
    def __init__(self, values, rows, columns):
        self.values, self.rows, self.columns = values, set(rows), columns
        self.calls = []

    @property
    def X(self):
        raise AssertionError("full-width X exposure forbidden")

    def __getitem__(self, key):
        rows, columns = key
        assert isinstance(rows, slice) and rows.step is None
        indices = list(range(rows.start, rows.stop))
        assert set(indices) <= self.rows
        np.testing.assert_array_equal(columns, self.columns)
        self.calls.append((indices, list(columns)))
        return SimpleNamespace(X=self.values[np.ix_(indices, columns)])


def test_actual_reader_slices_allowed_before_x_and_only_selected_treated():
    entries, metadata, plan = fixture()
    _, rows = prep.treated_boundary(plan, metadata)
    allowed = np.array([0, 2, 4], np.int64)
    values = np.full((14, 5), np.nan, np.float32)
    values[np.ix_(rows, allowed)] = np.arange(27).reshape(9, 3)
    adata = SliceSentinel(values, rows, allowed)
    reader = prep.rr.AllowedReader(
        adata, rows, allowed.tolist(), row_block=3, expected_dtype=np.dtype("float32")
    )
    cache = reader.read()
    means = prep.build_treated_means(cache, rows, entries, plan, metadata, allowed)
    np.testing.assert_array_equal(
        means[0], values[np.ix_([0, 1, 3, 4], allowed)].mean(0, dtype=np.float64).astype(np.float32)
    )
    np.testing.assert_array_equal(
        means[1],
        values[np.ix_([6, 7, 8, 9, 10], allowed)].mean(0, dtype=np.float64).astype(np.float32),
    )
    assert reader.logical_values == 27 and reader.calls == len(adata.calls)
    assert set(i for group, _ in adata.calls for i in group) == set(rows)
    with pytest.raises(RuntimeError, match="once"):
        reader.read()


def test_context_weights_are_selected_row_counts_not_equal_contexts_and_own_kept():
    entries, metadata, plan = fixture()
    _, rows = prep.treated_boundary(plan, metadata)
    cache = np.zeros((9, 3), np.float32)
    cache[4] = [10, 15, 20]  # q: one context a and four b; mean is 1/5 not 1/2.
    out = prep.build_treated_means(
        cache, rows, entries, plan, metadata, np.array([0, 2, 4], np.int64)
    )
    np.testing.assert_array_equal(out[1], [2, 3, 4])
    assert out[1, 0] == 2  # Allowed own-target value is not masked.


@pytest.mark.parametrize(
    "mutation",
    ["control", "validation", "test", "wrong_condition", "duplicate", "bool", "short", "oversize"],
)
def test_nontreated_or_inexact_group_membership_rejected(mutation):
    entries, metadata, plan = fixture()
    if mutation in ["control", "validation", "test"]:
        row = dict(control=2, validation=5, test=12)[mutation]
        entries[0]["rows"][-1] = row
    elif mutation == "wrong_condition":
        metadata["condition_ids"][0] = "q"
    elif mutation == "duplicate":
        entries[0]["rows"] = [0, 0, 3, 4]
    elif mutation == "bool":
        entries[0]["rows"][0] = False
    elif mutation == "short":
        entries[0]["rows"] = [0, 1, 3]
    else:
        entries[0]["rows"] = list(range(41))
    with pytest.raises(ValueError):
        prep.treated_boundary(plan, metadata)


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "reorder", "nan", "negative", "dtype", "axis"]
)
def test_cache_and_whitelist_fail_closed(kind):
    entries, metadata, plan = fixture()
    _, rows = prep.treated_boundary(plan, metadata)
    cache = np.ones((9, 3), np.float32)
    allowed = np.array([0, 2, 4], np.int64)
    if kind == "missing":
        rows = rows[:-1]
    elif kind == "extra":
        rows = [*rows, 12]
    elif kind == "reorder":
        rows = rows[::-1]
    elif kind == "nan":
        cache[0, 0] = np.nan
    elif kind == "negative":
        cache[0, 0] = -1
    elif kind == "dtype":
        cache = cache.astype(np.float64)
    else:
        allowed = np.array([0, 2, 2], np.int64)
    with pytest.raises(ValueError):
        prep.build_treated_means(cache, rows, entries, plan, metadata, allowed)


def test_sampling_metadata_never_accesses_old_float_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "SERVER_ROOT", tmp_path)
    allowed = np.array([0, 2, 4], np.int64)
    arrays = dict(
        allowed_gene_indices=allowed,
        control_slot_offsets=np.array([0, 4, 9], np.int64),
        control_slot_indices=np.arange(9, dtype=np.int64),
        condition_anchor_gene_indices=np.array([0, 1], np.int64),
        control_expression=np.full((9, 3), np.nan, np.float32),
        condition_targets=np.full((2, 256), np.nan, np.float32),
    )
    path = tmp_path / "bank.npz"
    np.savez(path, **arrays)
    metadata = dict(
        schema="fixed-response-group-anchor-bank-v1",
        arrays=dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        condition_ids=["p", "q"],
        identity={"original": True},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    bound = dict(
        path=str(manifest),
        sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        arrays_sha256=metadata["arrays"]["sha256"],
    )
    original = np.load
    accesses = []

    class ArchiveSpy:
        def __init__(self, archive):
            self.archive = archive
            self.files = archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key not in {"control_expression", "condition_targets"}
            accesses.append(key)
            return self.archive[key]

    monkeypatch.setattr(prep.np, "load", lambda *a, **k: ArchiveSpy(original(*a, **k)))
    bank = prep.sampling_metadata(bound, metadata, allowed)
    assert set(accesses) == {
        "allowed_gene_indices",
        "control_slot_offsets",
        "control_slot_indices",
        "condition_anchor_gene_indices",
    }
    assert not hasattr(bank, "control_expression") and not hasattr(bank, "condition_targets")
    np.testing.assert_array_equal(bank.control_slot_offsets, [0, 4, 9])
    assert path.read_bytes()  # NaNs in unread float payloads never entered target mean arithmetic.


def test_spec_schema_and_fixed_counts_rejected_before_any_reader(tmp_path, monkeypatch):
    spec = dict(
        schema="population-prediction-preparation-spec-v1",
        analysis_source={},
        census_complete={"sha256": prep.rr.CENSUS_SHA},
        repeatability_complete={"sha256": prep.REPEAT_SHA},
        preprocessing={"sha256": prep.rr.SCALE_SHA},
        sampling_bank={"path": "/not-used", "sha256": "0" * 64, "arrays_sha256": "0" * 64},
        output="/not-used",
        expected_counts=dict(prep.COUNTS),
        cpu_budget=dict(prep.CPU_BUDGET),
    )
    spec["expected_counts"]["forbidden"] = 224
    monkeypatch.setattr(
        prep, "checked_seal", lambda *_: pytest.fail("invalid spec must fail before input reads")
    )
    with pytest.raises(ValueError, match="exact R16"):
        prep._check_spec(spec)
    spec["expected_counts"] = dict(prep.COUNTS)
    spec["fallback"] = True
    with pytest.raises(ValueError, match="exact R16"):
        prep._check_spec(spec)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    """Exercise real run/AllowedReader/loader closure with synthetic published bindings."""
    from pathlib import Path

    from gradpert.hashing import sha256_json
    from gradpert.training.expression_policy import expression_policy

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    counts = dict(conditions=2, treated_rows=9, allowed=3, forbidden=2)
    monkeypatch.setattr(prep, "COUNTS", counts)
    monkeypatch.setattr(pa, "COUNTS", counts)
    # The immutable source files live in the checkout; all synthetic data/output stay tmp.
    monkeypatch.setattr(pa, "SERVER_ROOT", Path("/"))
    c = prep.rr.census
    monkeypatch.setattr(c, "SERVER", tmp_path)
    data_root = tmp_path / "canonical"
    data_root.mkdir()
    monkeypatch.setattr(c, "DATA_ROOT", data_root)
    metadata = dict(
        row_ids=[str(i) for i in range(21)],
        condition_ids=["p"] * 4 + ["q"] * 5 + ["ctrl"] * 8 + ["v"] * 4,
        context_ids=["a"] * 21,
        control_pools={"a": [str(i) for i in range(9, 17)]},
        train_row_indices=list(range(9)),
        train_conditions=["p", "q"],
        val_conditions=["v"],
        control_condition_id="ctrl",
    )
    plan = prep.rr.make_plan(metadata)
    member, _ = prep.rr.membership_receipts(plan, metadata)
    member = prep.rr._clean(member)
    genes = ("g0", "g1", "g2", "g3", "g4")
    tests = ("g1+ctrl", "g3+ctrl")
    allowed, policy = expression_policy(genes, tests, "ctrl", enabled=True)
    monkeypatch.setattr(prep.rr, "ALLOWED_SHA", policy["allowed_expression_gene_ids_sha256"])
    monkeypatch.setattr(prep.rr, "FORBIDDEN_SHA", policy["excluded_expression_gene_ids_sha256"])
    selected = {"synthetic_selection": True}
    historical = write(
        tmp_path / "historical.json", {"data": {"training_expression_policy": policy}}
    )
    old_spec = write(tmp_path / "historical/old-spec.json", {"historical_run_manifest": historical})
    summary = write(tmp_path / "census/summary.json", plan["census_summary"])
    census = write(
        tmp_path / "census/COMPLETE.json",
        dict(
            selection_receipt=selected,
            source={"repository_root": str(Path.cwd())},
            historical_training_source={"repository_root": str(Path.cwd())},
            input_seals=[old_spec],
            outputs_sha256={"summary.json": summary["sha256"]},
        ),
    )
    membership = write(tmp_path / "repeat/server-only/memberships.json", member)
    repeat = write(
        tmp_path / "repeat/COMPLETE.json",
        dict(
            source={"commit": pa.REPEAT_SOURCE},
            closure_verified=True,
            zero_PKL_within_output=True,
            input_seals=[],
            outputs_sha256={"server-only/memberships.json": membership["sha256"]},
        ),
    )
    preprocessing = write(
        tmp_path / "preprocessing.json",
        {"report": {"expression_scale_action": "normalize_total_4000_then_log1p"}},
    )
    monkeypatch.setattr(prep.rr, "CENSUS_SHA", census["sha256"])
    monkeypatch.setattr(prep, "REPEAT_SHA", repeat["sha256"])
    monkeypatch.setattr(prep.rr, "SCALE_SHA", preprocessing["sha256"])
    monkeypatch.setattr(
        pa,
        "ACCEPTED_INPUT_SHAS",
        dict(
            census_complete=census["sha256"],
            repeatability_complete=repeat["sha256"],
            preprocessing=preprocessing["sha256"],
        ),
    )
    canonical = write(
        data_root / "manifest.json", {"preprocessing_manifest_sha256": preprocessing["sha256"]}
    )
    adata_file = data_root / "fake.h5ad"
    adata_file.write_bytes(b"synthetic metadata stat only")
    monkeypatch.setattr(
        prep,
        "DatasetLayout",
        lambda *_: SimpleNamespace(
            canonical_manifest=Path(canonical["path"]), canonical_adata=adata_file
        ),
    )
    context_controls = {i for f in ("TA", "TB") for i in plan["controls"]["a"][f]}
    identity = dict(
        source={"original": "bank source retained"},
        data_manifest=sha256_json({"synthetic": True}),
        split_content_sha256="a" * 64,
        selection=selected,
        ordered_used_row_ids_sha256=sha256_json(
            [metadata["row_ids"][i] for i in sorted(set(range(9)) | context_controls)]
        ),
        allowed_policy=policy,
        counts=counts,
        metadata_sha256=sha256_json(metadata),
    )
    bankdir = tmp_path / "bank/server-only"
    bankdir.mkdir(parents=True)
    arrays = bankdir / "bank.npz"
    np.savez(
        arrays,
        allowed_gene_indices=np.asarray(allowed, dtype=np.int64),
        control_slot_offsets=np.array([0, 4, 9], np.int64),
        control_slot_indices=np.arange(9, dtype=np.int64) % 8,
        condition_anchor_gene_indices=np.array([0, 1], np.int64),
        control_expression=np.full((8, 3), np.nan, np.float32),
        condition_targets=np.full((2, 256), np.nan, np.float32),
    )
    array_sha = hashlib.sha256(arrays.read_bytes()).hexdigest()
    bankseal = write(
        bankdir / "manifest.json",
        dict(
            schema="fixed-response-group-anchor-bank-v1",
            arrays={"path": "bank.npz", "sha256": array_sha},
            condition_ids=["p", "q"],
            identity=identity,
        ),
    )
    publication = write(
        tmp_path / "PUBLICATION.json", {"verification_method": "git_ls_remote_live"}
    )
    source = dict(
        repository_root=str(Path.cwd()),
        commit="e" * 40,
        dirty=False,
        formal_eligible=True,
        published_commit="e" * 40,
        publication_receipt_sha256=publication["sha256"],
    )
    monkeypatch.setattr(prep.rr, "_extra_source", lambda *_: source)
    monkeypatch.setattr(
        prep.subprocess,
        "check_output",
        lambda argv, **_: (Path.cwd() / argv[-1].split(":", 1)[1]).read_bytes(),
    )
    monkeypatch.setattr(prep.rr.importlib.metadata, "version", lambda _: "0.13.2")
    monkeypatch.setattr(c, "apply_train_selection", lambda *_: selected)
    monkeypatch.setattr(c, "OBS_ORDER_SHA", sha256_json(metadata["row_ids"]))
    monkeypatch.setattr(c, "SELECTED_IDS_SHA", sha256_json(metadata["row_ids"][:9]))
    values = np.full((21, 5), np.nan, np.float32)
    values[np.ix_(range(9), allowed)] = np.arange(27, dtype=np.float32).reshape(9, 3)
    sentinel = SliceSentinel(values, range(9), allowed)
    closed = []

    class Data:
        def __init__(self, **_):
            self.__dict__.update(
                {
                    k: v
                    for k, v in metadata.items()
                    if k not in ["train_conditions", "val_conditions", "control_condition_id"]
                }
            )
            self.split = SimpleNamespace(
                train_conditions=["p", "q"],
                val_conditions=["v"],
                test_conditions=tests,
                control_condition_id="ctrl",
                split_content_sha256="a" * 64,
            )
            self.expression_gene_ids = genes
            self.manifest = SimpleNamespace(payload=lambda: {"synthetic": True})
            self._adata = sentinel

        def _read_expression_indices(self, *_):
            raise AssertionError("full width reader")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(c, "CanonicalTrainingData", Data)
    for key, value in dict(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    ).items():
        monkeypatch.setenv(key, value)
    output = tmp_path / "new-output"
    spec = write(
        tmp_path / "SPEC.json",
        dict(
            schema="population-prediction-preparation-spec-v1",
            analysis_source=dict(
                root=str(Path.cwd()),
                commit=source["commit"],
                repository=c.REPOSITORY,
                publication=publication,
            ),
            census_complete=census,
            repeatability_complete=repeat,
            preprocessing=preprocessing,
            sampling_bank=dict(**bankseal, arrays_sha256=array_sha),
            output=str(output),
            expected_counts=counts,
            cpu_budget=dict(prep.CPU_BUDGET),
        ),
    )
    return SimpleNamespace(
        spec=spec,
        output=output,
        sentinel=sentinel,
        closed=closed,
        bankmeta=json.loads(Path(bankseal["path"]).read_text()),
        binding=dict(**bankseal, arrays_sha256=array_sha),
        allowed=allowed,
        values=values,
    )


def test_real_preparation_run_complete_loader_and_immutable_output(preparation):
    from pathlib import Path

    f = preparation
    before_bank = Path(f.binding["path"]).read_bytes()
    prep.run(Path(f.spec["path"]), f.spec["sha256"])
    manifest = f.output / "server-only/manifest.json"
    meta = json.loads(manifest.read_text())
    bank = prep.sampling_metadata(f.binding, f.bankmeta, f.allowed)
    result = pa.load_population_targets(
        manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), meta["arrays"]["sha256"], bank
    )
    np.testing.assert_array_equal(
        result[0],
        f.values[np.ix_(range(4), f.allowed)].mean(0, dtype=np.float64).astype(np.float32),
    )
    np.testing.assert_array_equal(
        result[1],
        f.values[np.ix_(range(4, 9), f.allowed)].mean(0, dtype=np.float64).astype(np.float32),
    )
    assert Path(f.binding["path"]).read_bytes() == before_bank
    assert f.closed == [True]
    calls = len(f.sentinel.calls)
    with pytest.raises(FileExistsError):
        prep.run(Path(f.spec["path"]), f.spec["sha256"])
    assert len(f.sentinel.calls) == calls


def test_preparation_failure_retains_preflight_no_complete_and_closes(preparation, monkeypatch):
    from pathlib import Path

    f = preparation
    monkeypatch.setattr(
        prep,
        "build_treated_means",
        lambda *_: (_ for _ in ()).throw(ValueError("synthetic mean failure")),
    )
    with pytest.raises(ValueError, match="synthetic mean failure"):
        prep.run(Path(f.spec["path"]), f.spec["sha256"])
    assert (f.output / "PRE_NUMERICAL_METADATA.json").exists()
    assert (f.output / "FAILURE.json").exists()
    assert not (f.output / "COMPLETE.json").exists()
    assert json.loads((f.output / "FAILURE.json").read_text())["scientific_complete"] is False
    assert f.closed == [True]
