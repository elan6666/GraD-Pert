import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
m = importlib.import_module("cap40_posthoc_accel")
stats = importlib.import_module("cap40_posthoc_stats")


@pytest.mark.parametrize("shape", [(1, 256), (4, 256), (128, 256), (300, 256)])
@pytest.mark.parametrize("kind", ["random", "constant", "rank2", "offset"])
def test_smaller_gram_preserves_every_feature_statistic(shape, kind):
    rng = np.random.default_rng(11)
    x = rng.normal(size=shape)
    if kind == "constant":
        x[:] = 1
    elif kind == "rank2":
        x = rng.normal(size=(shape[0], 2)) @ rng.normal(size=(2, shape[1]))
    elif kind == "offset":
        x += 1e8
    ref = stats.Moments(shape[1])
    ref.add(x)
    reference, fast = ref.result(), m.sample_moments(x)
    assert fast.keys() == reference.keys()
    for k, v in reference.items():
        if isinstance(v, float):
            assert fast[k] == pytest.approx(v, abs=1e-9, rel=1e-10), k
        else:
            assert fast[k] == v


@pytest.mark.skipif(torch is None, reason="PyTorch required; executed in server runtime")
@pytest.mark.parametrize("width", [4, 257, 16384])
@pytest.mark.parametrize("kind", ["uniform", "peaked", "different", "random"])
def test_reduced_distribution_matches_numpy_float64(width, kind):
    x = torch.randn(17, width, generator=torch.Generator().manual_seed(18))
    if kind == "uniform":
        x.zero_()
    elif kind == "peaked":
        x.zero_()
        x[:, 0] = 1000
    elif kind == "different":
        x.zero_()
        x[torch.arange(17), torch.arange(17) % width] = 1000
    before = x.clone()
    ref = stats.Distribution(width)
    for block in x.split(7):
        ref.add(block.numpy())
    fast = m.reduced_distribution(x, chunk_rows=7)
    np.testing.assert_array_equal(fast.hits, ref.hits)
    np.testing.assert_allclose(fast.sum, ref.sum, rtol=1e-12, atol=1e-12)
    assert fast.n == ref.n
    for k, v in ref.result().items():
        if isinstance(v, float):
            assert fast.result()[k] == pytest.approx(v, abs=1e-10, rel=1e-10), k
        else:
            assert fast.result()[k] == v
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.skipif(torch is None, reason="PyTorch required; executed in server runtime")
def test_reduced_statistics_reject_nonfinite_and_preserve_no_grad():
    for bad in [float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="nonfinite"):
            m.sample_moments(np.full((4, 9), bad))
        with pytest.raises(ValueError, match="nonfinite"):
            m.reduced_distribution(torch.full((4, 9), bad))
    x = torch.randn(3, 7, requires_grad=True)
    m.reduced_distribution(x)
    assert x.grad is None
    assert m.reduced_distribution(torch.zeros(0, 7)).n == 0


@pytest.mark.skipif(torch is None, reason="PyTorch required; executed in server runtime")
def test_feature_selection_and_gram_preserve_gene_ids_and_grouping():
    from types import SimpleNamespace

    worker = importlib.import_module("cap40_posthoc")
    batch = SimpleNamespace(
        graph=SimpleNamespace(ids=torch.arange(1000)), graph_views=[], cell_views=[]
    )
    records = [
        {
            "truth_row_id": str(i),
            "control_row_id": str(i),
            "context_id": "a",
            "cell_type": "x",
            "batch": "b",
        }
        for i in range(3)
    ]
    x = torch.randn(3, 1000, 256, generator=torch.Generator().manual_seed(5))
    a = worker.Collector(batch, records, 42)
    b = worker.Collector(batch, records, 42, stats_backend="fast")
    ids = np.random.default_rng(6).permutation(1000)
    a.current = b.current = ("student", "main", ids)
    a.add_features("genes", x)
    b.add_features("genes", x)
    assert len(a.per_cell) == len(b.per_cell) == 3
    for ref, fast in zip(a.per_cell, b.per_cell, strict=True):
        assert ref.keys() == fast.keys()
        for k, v in ref.items():
            if isinstance(v, float):
                assert fast[k] == pytest.approx(v, rel=1e-10, abs=1e-9)
            else:
                assert fast[k] == v
    np.testing.assert_array_equal(a.features["genes"].m2, b.features["genes"].m2)
    assert a.same_gene == b.same_gene


def test_acceptance_rejects_changed_native_rows_and_model_state(tmp_path):
    import json

    driver = importlib.import_module("profile_medium_observer")
    worker = importlib.import_module("cap40_posthoc")
    left, right = tmp_path / "reference", tmp_path / "candidate"
    fields = (
        "manifest_sha256",
        "checkpoint_sha256",
        "conditions",
        "rows",
        "deep_conditions",
        "deep_rows",
        "gradient_conditions",
        "gradient_rows",
        "updates",
        "global_heads",
        "global_features",
        "cls_decomposition",
        "metrics",
    )
    for folder in (left, right):
        folder.mkdir()
        summary = dict.fromkeys(fields, 0)
        summary.update(state_before={"model": "original"}, state_after={"model": "original"})
        (folder / "summary.json").write_text(json.dumps(summary))
        for name in ("deep.json", "gradients.json"):
            (folder / name).write_text('[{"condition_id":"a", "value":1.0}]')
        for name in ("outputs_full.csv", "cls_within_condition.csv"):
            (folder / name).write_text("row_id,value\na,1.0\n")
        worker.write(folder / "COMPLETE.json", {"state_unchanged": True, "outputs": {}})
    assert driver.compare(left, right)["passed"]
    (right / "outputs_full.csv").write_text("row_id,value\nb,1.0\n")
    with pytest.raises(ValueError, match="native predictions or basic identities"):
        driver.compare(left, right)
    (right / "outputs_full.csv").write_text("row_id,value\na,1.0\n")
    summary["state_after"]["model"] = "mutated"
    (right / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="state changed"):
        driver.compare(left, right)
