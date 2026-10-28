from __future__ import annotations

import itertools
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.v2 import response_repeatability as rr


def fixture(*, n=6, p=4, genes=9, seed=17):
    ids, conditions, contexts = [], [], []
    train = [f"p{i}" for i in range(p)]
    val = [f"v{i}" for i in range(p)]
    for name in train + val:
        for j in range(n):
            ids.append(f"{name}:row{j}")
            conditions.append(name)
            contexts.append("c")
    train_indices = [i for i, c in enumerate(conditions) if c in train]
    controls = []
    for j in range(12):
        ids.append(f"control{j}")
        conditions.append("ctrl")
        contexts.append("c")
        controls.append(ids[-1])
    ids.append("test-row-never-read")
    conditions.append("TEST")
    contexts.append("c")
    meta = dict(
        row_ids=ids,
        condition_ids=conditions,
        context_ids=contexts,
        train_row_indices=train_indices,
        train_conditions=train,
        val_conditions=val,
        control_condition_id="ctrl",
        control_pools={"c": controls},
    )
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(ids), genes))
    return meta, values, [*train, *val, *[f"gene{i}" for i in range(max(0, genes - 2 * p))]][:genes]


def execute(meta, values, genes, block=512):
    plan = rr.make_plan(meta)
    matrix = values[plan["used_rows"]]
    result = rr.analyze_blocks(
        plan, meta["context_ids"], genes, lambda cols: matrix[:, list(cols)], block_size=block
    )
    return plan, result


def oracle(plan, meta, values, genes):
    def response(rows, fold):
        return np.mean(
            [
                values[i] - values[plan["controls"][meta["context_ids"][i]][fold]].mean(axis=0)
                for i in rows
            ],
            axis=0,
        )

    da, db = {}, {}
    for split, p in plan["splits"].items():
        fa, fb = ("TA", "TB") if split == "selected_train" else ("VA", "VB")
        da[split] = np.array([response(e["A"], fa) for e in p["eligible"]])
        db[split] = np.array([response(e["B"], fb) for e in p["eligible"]])
    ma, mb = da["selected_train"].mean(axis=0), db["selected_train"].mean(axis=0)

    def residual(x):
        return np.array([x[i] - np.delete(x, i, axis=0).mean(axis=0) for i in range(len(x))])

    def scores(a, b, entries, mask):
        results = []
        for i, e in enumerate(entries):
            keep = (
                np.array([g not in e["condition_id"].split("+") for g in genes])
                if mask
                else np.ones(len(genes), dtype=bool)
            )
            aa, bb = a[i, keep], b[i, keep]
            results.append(np.mean(aa * bb) if len(aa) else np.nan)
        return float(np.nanmean(results))

    result = {}
    for split, p in plan["splits"].items():
        aa = residual(da[split]) if split == "selected_train" else da[split] - ma
        observed_b = residual(db[split]) if split == "selected_train" else db[split] - mb
        result[split] = dict(
            primary=[scores(aa, observed_b, p["eligible"], False)],
            direct_target_excluded=[scores(aa, observed_b, p["eligible"], True)],
            raw_direct_target_excluded=[scores(da[split], db[split], p["eligible"], True)],
        )
        for owners in p["null"]["owners"]:
            null = np.array(
                [
                    response(
                        [row for j, row in enumerate(p["null"]["source_rows"]) if owners[j] == i],
                        "TB" if split == "selected_train" else "VB",
                    )
                    for i in range(len(p["eligible"]))
                ]
            )
            bb = residual(null) if split == "selected_train" else null - mb
            result[split]["primary"].append(scores(aa, bb, p["eligible"], False))
            result[split]["direct_target_excluded"].append(scores(aa, bb, p["eligible"], True))
            result[split]["raw_direct_target_excluded"].append(
                scores(da[split], null, p["eligible"], True)
            )
    return result


def test_odd_half_constant_response_not_old_two_over_N_artifact():
    for n in (4, 5):
        meta, values, genes = fixture(n=n)
        values[:] = 1
        values[[i for i, p in enumerate(meta["condition_ids"]) if p == "ctrl"]] = 0
        _, result = execute(meta, values, genes, block=3)
        raw = result["splits"]["selected_train"]["raw"]["observed"]
        assert (
            raw["S"] == pytest.approx(1)
            and raw["E"] == pytest.approx(1)
            and raw["Q"] == pytest.approx(0)
        )
        assert result["splits"]["selected_train"]["primary"]["observed"]["S"] == pytest.approx(
            0, abs=1e-28
        )
    old_HT = [(2 / 5) * 2, (2 / 5) * 3]
    assert old_HT == pytest.approx([0.8, 1.2])


def test_enumerated_SRS_both_half_means_unbiased_for_finite_odd_population():
    y = np.array([1.0, 2.0, 7.0, -3.0, 9.0])
    a, b = [], []
    for selected in itertools.combinations(range(5), 2):
        a.append(y[list(selected)].mean())
        b.append(y[[i for i in range(5) if i not in selected]].mean())
    assert np.mean(a) == pytest.approx(y.mean()) and np.mean(b) == pytest.approx(y.mean())
    aa, bb = rr.fixed_halves(range(5), [str(i) for i in range(5)], ["fixed"])
    assert len(aa) == 2 and len(bb) == 3 and sorted(aa + bb) == list(range(5))


def test_streamed_FP64_all199_scores_match_independent_full_oracle():
    meta, values, genes = fixture()
    plan, result = execute(meta, values, genes, block=2)
    truth = oracle(plan, meta, values, genes)
    for split in truth:
        for key, scores in truth[split].items():
            bank = result["splits"][split][key]
            actual = [bank["observed"]["S"], *[s["S"] for s in bank["nulls"]]]
            np.testing.assert_allclose(actual, scores, rtol=2e-13, atol=2e-14)
    _, all_genes = execute(meta, values, genes, block=len(genes))
    for split in truth:
        for key in truth[split]:
            a, b = result["splits"][split][key], all_genes["splits"][split][key]
            for metric in ("S", "E", "Q", "Pearson_mean"):
                assert a["observed"][metric] == pytest.approx(
                    b["observed"][metric], rel=1e-12, abs=1e-13
                )
            assert a["observed"]["Q"] == pytest.approx(
                2 * a["observed"]["E"] - 2 * a["observed"]["S"]
            )


def test_LOO_self_exclusion_and_equivalent_formula():
    x = np.arange(24.0, dtype=np.float64).reshape(4, 6)
    r, mean = rr.loo(x)
    np.testing.assert_allclose(r, 4 / 3 * (x - mean))
    previous = x[0] - r[0]
    x[0] += np.array([3, -5, 9, 2, 7, 1])
    r2, _ = rr.loo(x)
    np.testing.assert_allclose(x[0] - r2[0], previous)


def test_val_changes_never_refit_train_center_or_train_null():
    meta, values, genes = fixture()
    plan, a = execute(meta, values, genes)
    changed = values.copy()
    for i, name in enumerate(meta["condition_ids"]):
        if name in meta["val_conditions"]:
            changed[i] += np.arange(len(genes)) * 10
    _, b = execute(meta, changed, genes)
    assert a["splits"]["selected_train"] == b["splits"]["selected_train"]
    assert (
        a["splits"]["validation"]["primary"]["observed"]["S"]
        != b["splits"]["validation"]["primary"]["observed"]["S"]
    )
    assert all(set(e["A"]).isdisjoint(e["B"]) for s in plan["splits"].values() for e in s["axis"])


def test_null_slots_context_membership_and_A_never_move():
    meta, _values, _genes = fixture()
    plan = rr.make_plan(meta)
    p = plan["splits"]["selected_train"]
    before = [e["A"][:] for e in p["eligible"]]
    for owners in p["null"]["owners"]:
        for owner, entry in enumerate(p["eligible"]):
            assigned = [row for j, row in enumerate(p["null"]["source_rows"]) if owners[j] == owner]
            assert len(assigned) == len(entry["B"])
            assert Counter(meta["context_ids"][i] for i in assigned) == Counter(
                meta["context_ids"][i] for i in entry["B"]
            )
    assert [e["A"] for e in p["eligible"]] == before and len(p["null"]["owners"]) == 199
    assert any(
        owners.tolist() != p["null"]["owners"][0].tolist() for owners in p["null"]["owners"][1:]
    )


def test_four_control_folds_no_overlap_and_shared_error_is_preserved_in_null():
    meta, values, genes = fixture()
    values[:] = 0
    controls = [i for i, p in enumerate(meta["condition_ids"]) if p == "ctrl"]
    values[controls] = np.arange(len(controls))[:, None] + np.arange(len(genes))[None, :]
    plan, result = execute(meta, values, genes)
    folds = list(plan["controls"]["c"].values())
    assert all(len(x) >= 2 for x in folds)
    assert all(set(a).isdisjoint(b) for i, a in enumerate(folds) for b in folds[i + 1 :])
    for key in ("primary", "raw_direct_target_excluded"):
        bank = result["splits"]["selected_train"][key]
        assert all(x["S"] == pytest.approx(bank["observed"]["S"], abs=1e-26) for x in bank["nulls"])
        assert bank["inference"]["p"] == 1


def test_context_only_response_and_composition_contribution_cannot_beat_slots():
    meta, values, genes = fixture()
    # Two p per context: context composition creates a positive observed association.
    for i, p in enumerate(meta["condition_ids"]):
        if p in ("p2", "p3", "v2", "v3"):
            meta["context_ids"][i] = "d"
    for j in range(12):
        meta["row_ids"].append(f"dcontrol{j}")
        meta["condition_ids"].append("ctrl")
        meta["context_ids"].append("d")
    meta["control_pools"]["d"] = [f"dcontrol{j}" for j in range(12)]
    values = np.zeros((len(meta["row_ids"]), len(genes)))
    for i, (p, k) in enumerate(zip(meta["condition_ids"], meta["context_ids"], strict=True)):
        values[i] = 20 if k == "d" else 7
        if p != "ctrl":
            values[i] += 2 if k == "d" else -2
    _, result = execute(meta, values, genes)
    bank = result["splits"]["selected_train"]["primary"]
    assert bank["observed"]["S"] > 0
    assert all(x["S"] == pytest.approx(bank["observed"]["S"]) for x in bank["nulls"])
    assert bank["inference"]["p"] == 1


def test_immobile_slots_coverage_NA_not_fake199_effective_comparisons():
    entries = [dict(B=[0, 1]), dict(B=[2, 3])]
    null = rr.slot_plan(entries, ["a", "a", "b", "b"], ["0", "1", "2", "3"], "selected_train")
    assert null["mobility"]["macro"] == 0 and null["mobility"]["log_label_assignment_count"] == 0
    assert null["mobility"]["unique_label_assignments"] == 1
    inference = rr.null_inference(dict(S=1.0, E=1.0), [1.0] * 199, False)
    assert inference["p"] is None and inference["route_support"] is None
    assert len(null["draws"]) == 199


def test_99_to1_metadata_boundary_reports_low_micro_without_changing_frozen_gates():
    null = rr.slot_plan(
        [dict(B=list(range(99))), dict(B=[99])],
        ["c"] * 100,
        [str(i) for i in range(100)],
        "selected_train",
    )
    assert null["mobility"]["micro"] == pytest.approx(0.0198)
    assert null["mobility"]["macro"] == pytest.approx(0.5)
    assert null["mobility"]["coverage_gate"] is True


def test_true_condition_signal_beats_null_and_common_background_is_removed():
    meta, values, genes = fixture(n=8)
    for i, p in enumerate(meta["condition_ids"]):
        values[i] = (
            0
            if p == "ctrl"
            else np.arange(len(genes))
            * (
                meta["train_conditions"].index(p) + 1
                if p in meta["train_conditions"]
                else meta["val_conditions"].index(p) + 1
                if p in meta["val_conditions"]
                else 0
            )
        )
    _, result = execute(meta, values, genes)
    bank = result["splits"]["selected_train"]["primary"]
    assert bank["observed"]["S"] > 0 and bank["inference"]["p"] <= 0.05
    for i, p in enumerate(meta["condition_ids"]):
        values[i] = 0 if p == "ctrl" else np.arange(len(genes))
    _, common = execute(meta, values, genes)
    assert common["splits"]["selected_train"]["primary"]["observed"]["S"] == pytest.approx(
        0, abs=1e-26
    )


def test_target_only_center_shadow_retained_but_raw_mask_null_all_zero():
    meta, values, genes = fixture(n=8, genes=4)
    values[:] = 0
    for i, p in enumerate(meta["condition_ids"]):
        if p in genes:
            values[i, genes.index(p)] = 1
    _, result = execute(meta, values, genes)
    centered = result["splits"]["selected_train"]["direct_target_excluded"]
    assert centered["observed"]["S"] == pytest.approx(1 / 9)  # fixed LOO center-shadow oracle
    raw = result["splits"]["selected_train"]["raw_direct_target_excluded"]
    assert raw["observed"]["S"] == 0 and all(x["S"] == 0 for x in raw["nulls"])
    assert raw["inference"]["p"] == 1 and result["next_anchor_design_eligible"] is False
    assert raw["observed"]["E"] == 0 and any(x["E"] > 0 for x in raw["nulls"])


def test_constant_Pearson_empty_mask_and_zero_energy_are_NA_not_zero():
    moment = rr.Moments(1, 2)
    x = np.ones((2, 3), dtype=np.float64)
    keep = np.array([[False] * 3, [True] * 3])
    moment.add(0, x, x, keep)
    result = moment.finish(0)
    assert result["scored_conditions"] == 1 and result["NA_conditions"] == 1
    assert result["Pearson_finite_conditions"] == 0 and np.isnan(result["per_condition"]["S"][0])
    zero = rr.Moments(1, 1)
    zero.add(0, np.zeros((1, 3)), np.zeros((1, 3)))
    assert zero.finish(0)["ratio"] is None
    assert rr.null_inference(zero.finish(0), [0.0] * 199, True)["p"] == 1


def test_NA_full_axis_and_bad_fold_support_not_reweighted_away():
    meta, _values, _genes = fixture(n=4)
    # One condition has N3, another sees a context with only7 controls.
    index = meta["condition_ids"].index("p0")
    meta["train_row_indices"].remove(index)
    meta["context_ids"][meta["condition_ids"].index("p1")] = "bad"
    for j in range(7):
        meta["row_ids"].append(f"bad{j}")
        meta["condition_ids"].append("ctrl")
        meta["context_ids"].append("bad")
    meta["control_pools"]["bad"] = [f"bad{j}" for j in range(7)]
    plan = rr.make_plan(meta)
    axis = plan["splits"]["selected_train"]["axis"]
    assert len(axis) == 4 and len(plan["splits"]["selected_train"]["eligible"]) == 2
    assert axis[0]["reasons"] == ["treated_n_lt4"] and axis[1]["lost_context_mass"] == 0.25
    assert not set(axis[1]["rows"]) & set(plan["used_rows"])


def test_insufficient_train_center_ends_without_reader_call():
    meta, _values, genes = fixture(p=1, genes=3)
    plan = rr.make_plan(meta)

    def forbidden(_):
        raise AssertionError("numerical reader must not run")

    result = rr.analyze_blocks(plan, meta["context_ids"], genes, forbidden)
    assert result["numeric_blocks"] == 0 and result["next_anchor_design_eligible"] is False


class SliceOnlyBacked:
    def __init__(self, values, whitelist, allowed):
        self.values, self.whitelist, self.allowed = values, set(whitelist), list(allowed)
        self.calls = []

    def __getitem__(self, index):
        rows, columns = index
        assert isinstance(rows, slice) and rows.step is None
        assert set(range(rows.start, rows.stop)) <= self.whitelist  # never span a gap/test row
        assert list(columns) == self.allowed  # constrained before .X exposure
        self.calls.append((rows.start, rows.stop))
        return SimpleNamespace(X=self.values[rows][:, columns])


def test_reader_slice_only_allowed_before_array_sentinel_and_no_test_rows():
    values = np.arange(72.0, dtype=np.float64).reshape(12, 6)
    rows, columns = [0, 1, 4, 5, 6, 10], [0, 2, 4]
    values[:, [1, 3, 5]] = np.nan
    backed = SliceOnlyBacked(values, rows, columns)
    reader = rr.AllowedReader(backed, rows, columns, row_block=2)
    expected = values[np.ix_(rows, columns)].copy()
    np.testing.assert_array_equal(reader.read(), expected)
    assert backed.calls == [(0, 2), (4, 6), (6, 7), (10, 11)]
    assert reader.logical_values == len(rows) * len(columns)
    values[:, [1, 3, 5]] = 1e300
    np.testing.assert_array_equal(
        rr.AllowedReader(SliceOnlyBacked(values, rows, columns), rows, columns).read(), expected
    )
    with pytest.raises(RuntimeError, match="once"):
        reader.read()


def test_nonfinite_allowed_values_and_FP64_overflow_fail_closed():
    values = np.ones((4, 3))
    values[0, 0] = np.nan
    reader = rr.AllowedReader(SliceOnlyBacked(values, range(4), range(3)), range(4), range(3))
    with pytest.raises(ValueError, match="nonfinite"):
        reader.read()
    moment = rr.Moments(1, 1)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="nonfinite"):
        moment.add(0, np.full((1, 2), 1e300), np.zeros((1, 2)))


def test_Wilson_boundary_and_tie_rule_fixed199_no_adaptation():
    observed = dict(S=1.0, E=2.0)
    nulls = [1.0] * 8 + [0.0] * 191
    r = rr.null_inference(observed, nulls, True)
    assert r["p"] == 0.045 and r["positive_rule"] and r["MC_borderline"]
    assert r["Wilson95"][0] < 0.05 < r["Wilson95"][1]
    assert rr.null_inference(observed, [1 - 1e-11] * 199, True)["p"] == 1
    with pytest.raises(ValueError, match="exactly199"):
        rr.null_inference(observed, [0.0] * 50, True)


def test_within_context_subset_uses_main_LOO_center_and_fixed_group_weights():
    meta, values, genes = fixture(n=8)
    plan, result = execute(meta, values, genes)
    # A single available context per p makes condition/group sensitivity coincide;
    # its independent hash-SRS differs from the primary partition but same LOO center is used.
    assert len(plan["within"]) == 4
    assert result["within"]["group"]["S"] == pytest.approx(result["within"]["condition"]["S"])
    assert result["within"]["condition"]["scored_conditions"] == 4
    assert result["within"]["condition"]["Q"] == pytest.approx(
        2 * result["within"]["condition"]["E"] - 2 * result["within"]["condition"]["S"]
    )


def test_negative_S_retained_with_exact_energy_identity():
    m = rr.Moments(1, 1)
    a = np.array([[1.0, -1.0]])
    m.add(0, a, -a)
    score = m.finish(0)
    assert score["S"] == -1 and score["E"] == 1 and score["Q"] == 4
    assert score["ratio"] == -1
    assert score["Pearson_mean"] == pytest.approx(-1, abs=2e-15, rel=0)


def test_publishing_full_axis_NA_preserves_evidence_if_final_closure_fails(tmp_path):
    meta, _values, genes = fixture(p=1, genes=3)
    plan = rr.make_plan(meta)
    result = rr.analyze_blocks(plan, meta["context_ids"], genes, lambda _: None)
    _, coverage = rr.membership_receipts(plan, meta)
    (tmp_path / "server-only").mkdir()

    def changed():
        assert (tmp_path / "summary.json").exists()
        raise ValueError("input changed before COMPLETE")

    with pytest.raises(ValueError, match="input changed"):
        rr.publish_result(tmp_path, result, plan, coverage, {}, changed)
    assert not (tmp_path / "COMPLETE.json").exists()
    content = (tmp_path / "server-only/conditions.csv").read_text()
    assert "p0" in content and "v0" in content and "insufficient_TRAIN_for_center" in content
    assert len((tmp_path / "null_199_scalars.csv").read_text().splitlines()) == 2 * 3 * 199 + 1


def test_fresh_output_cannot_mutate_historical_or_census_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(rr.census, "_server_path", lambda value: __import__("pathlib").Path(value))
    original = dict(
        source={"repository_root": str(tmp_path / "old-source")},
        historical_training_source={"repository_root": str(tmp_path / "training-source")},
        input_seals=[
            {"path": str(tmp_path / "old-job/INPUT.json")},
            {"path": str(tmp_path / "old-run/run_manifest.json")},
        ],
    )
    spec = dict(
        output=str(tmp_path / "old-run/new-output"),
        analysis_source={"root": str(tmp_path / "new-source")},
        census_complete={"path": str(tmp_path / "old-census/COMPLETE.json")},
    )
    with pytest.raises(ValueError, match="independent"):
        rr._claim_output(spec, original)
    assert not (tmp_path / "old-run").exists()
    spec["output"] = str(tmp_path / "fresh-output")
    rr._claim_output(spec, original)
    with pytest.raises(FileExistsError):
        rr._claim_output(spec, original)


def test_within_context_unequal_group_weights_and_main_center_full_oracle():
    meta, _values, genes = fixture(n=12)
    for p, c_count in (("p0", 4), ("p2", 0), ("p3", 3)):
        selected = [i for i, c in enumerate(meta["condition_ids"]) if c == p]
        for i in selected[c_count:]:
            meta["context_ids"][i] = "d"
    for j in range(12):
        meta["row_ids"].append(f"dctrl{j}")
        meta["condition_ids"].append("ctrl")
        meta["context_ids"].append("d")
    meta["control_pools"]["d"] = [f"dctrl{j}" for j in range(12)]
    values = np.random.default_rng(420).normal(size=(len(meta["row_ids"]), len(genes)))
    plan, result = execute(meta, values, genes, block=2)

    def response(rows, fold):
        return np.mean(
            [
                values[i] - values[plan["controls"][meta["context_ids"][i]][fold]].mean(axis=0)
                for i in rows
            ],
            axis=0,
        )

    main = plan["splits"]["selected_train"]["eligible"]
    da = np.array([response(e["A"], "TA") for e in main])
    db = np.array([response(e["B"], "TB") for e in main])
    ca = np.array([np.delete(da, i, axis=0).mean(axis=0) for i in range(len(main))])
    cb = np.array([np.delete(db, i, axis=0).mean(axis=0) for i in range(len(main))])
    groups_a, groups_b, by_p = [], [], {}
    for g in plan["within"]:
        a, b = response(g["A"], "TA"), response(g["B"], "TB")
        groups_a.append(a - ca[g["main_index"]])
        groups_b.append(b - cb[g["main_index"]])
        by_p.setdefault(g["main_index"], []).append((len(g["rows"]), a, b))
    expected = []
    for p, groups in by_p.items():
        total = sum(n for n, _, _ in groups)
        a = sum(n * a for n, a, _ in groups) / total - ca[p]
        b = sum(n * b for n, _, b in groups) / total - cb[p]
        expected.append(np.mean(a * b))
    assert len(plan["within"]) == 5  # p3 c-singletons excluded, not the whole p
    assert result["within"]["condition"]["S"] == pytest.approx(np.mean(expected), abs=1e-14)
    assert result["within"]["group"]["S"] == pytest.approx(
        np.mean(np.array(groups_a) * np.array(groups_b)), abs=1e-14
    )


@pytest.mark.parametrize("constant_b", [0.2, -0.2])
def test_nonbinary_exact_constants_have_undefined_Pearson(constant_b):
    m = rr.Moments(1, 1)
    m.add(0, np.full((1, 3), 0.1), np.full((1, 3), constant_b))
    score = m.finish(0)
    assert score["Pearson_mean"] is None and score["Pearson_finite_conditions"] == 0
    assert np.isnan(score["per_condition"]["Pearson"][0])
    assert score["S"] == pytest.approx(0.1 * constant_b)


@pytest.mark.parametrize("constant_side", ["A", "B"])
def test_one_sided_nonbinary_constant_is_not_a_correlation(constant_side):
    variable = np.array([[0.1, 0.2, 0.4]])
    constant = np.full((1, 3), 0.1)
    a, b = (constant, variable) if constant_side == "A" else (variable, constant)
    m = rr.Moments(1, 1)
    m.add(0, a, b)
    assert m.finish(0)["Pearson_mean"] is None


def test_masked_constant_ranges_merge_across_unequal_and_empty_blocks_per_draw():
    m = rr.Moments(2, 2)
    a = np.array([[0.1] * 7, [0.1, 5.0, 0.1, 2.0, 0.1, 4.0, 0.1]])
    b = np.array([[0.2] * 7, [0.2, -1.0, 0.2, -2.0, 0.2, -3.0, 0.2]])
    keep = np.array(
        [
            [True, False, False, True, True, True, False],
            [True, False, True, False, True, False, True],
        ]
    )
    other_a = np.array([np.arange(7.0), np.arange(7.0) + 1])
    other_b = other_a * np.array([[2.0], [-3.0]])
    for start, stop in ((0, 1), (1, 3), (3, 4), (4, 7)):
        m.add(0, a[:, start:stop], b[:, start:stop], keep[:, start:stop])
        m.add(1, other_a[:, start:stop], other_b[:, start:stop])
    assert m.finish(0)["Pearson_finite_conditions"] == 0
    assert m.finish(0)["per_condition"]["gene_count"] == [4, 4]
    np.testing.assert_allclose(m.finish(1)["per_condition"]["Pearson"], [1.0, -1.0])
    empty = rr.Moments(1, 1)
    empty.add(0, np.empty((1, 0)), np.empty((1, 0)))
    empty.add(0, np.ones((1, 3)), np.ones((1, 3)), np.zeros((1, 3), dtype=bool))
    assert empty.finish(0)["Pearson_mean"] is None and empty.finish(0)["S"] is None


def test_equal_block_means_do_not_hide_actual_within_block_variation():
    m = rr.Moments(1, 1)
    a = np.array([[0.0, 2.0, 1.0, 1.0]])
    for start, stop in ((0, 2), (2, 4)):
        m.add(0, a[:, start:stop], 2 * a[:, start:stop])
    assert m.finish(0)["Pearson_mean"] == pytest.approx(1)


def test_distinct_values_one_ULP_apart_are_not_classified_by_an_epsilon():
    first = 0.1
    second = np.nextafter(first, np.inf)
    a = np.array([[first, second, np.nextafter(second, np.inf)]])
    m = rr.Moments(1, 1)
    m.add(0, a, 2 * a)
    assert m.finish(0)["Pearson_finite_conditions"] == 1
    assert m.finish(0)["Pearson_mean"] == pytest.approx(1)


def test_finite_large_moments_do_not_overflow_Pearson_denominator_product():
    m = rr.Moments(1, 1)
    a = np.array([[1.0, 2.0, 3.0]]) * 1e150
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        m.add(0, a, a)
        result = m.finish(0)
    assert result["Pearson_mean"] == pytest.approx(1)
    assert result["S"] == pytest.approx(float(np.mean(a * a)))
    assert result["Q"] == 0
