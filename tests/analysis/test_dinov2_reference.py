"""Meaningful sampling/statistic invariants without local scientific execution."""

import importlib.util
import io
import itertools
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PATH = Path(__file__).resolve().parents[2] / "scripts/analysis/dinov2_reference.py"
SPEC = importlib.util.spec_from_file_location("dinov2_reference", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_selection_is_identity_stable_and_has_no_replacement():
    rows = [{"image_id": f"{c}/{i}", "class_id": c} for c in "abc" for i in range(9)]
    selected = MODULE.balanced(rows, 11, 42)
    assert selected == MODULE.balanced(list(reversed(rows)), 11, 42)
    assert len({r["image_id"] for r in selected}) == 11
    counts = [sum(r["class_id"] == c for r in selected) for c in "abc"]
    assert max(counts) - min(counts) == 1
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.balanced(rows + rows[:1], 11, 42)


def test_cosine_groups_match_explicit_nonself_pairs():
    x = np.random.default_rng(1).normal(size=(7, 5))
    labels = np.array(["a", "a", "a", "b", "b", "c", "c"])
    result = MODULE.pair_summary(x, labels)
    for row, same in zip(result, [True, False], strict=True):
        values = [
            MODULE.paired_cosine(x[i], x[j])
            for i, j in itertools.permutations(range(7), 2)
            if (labels[i] == labels[j]) == same
        ]
        assert row["ordered_pairs"] == len(values)
        assert row["mean_cosine"] == pytest.approx(np.mean(values), abs=1e-12)


def test_geometry_is_repeatable_and_keeps_input_unmodified():
    x = np.random.default_rng(2).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    image = Image.fromarray(x)
    for view, size in [("canonical", 224), ("G1", 224), ("L1", 98)]:
        a, box = MODULE.view_image(image, "image-a", view, 42)
        b, other = MODULE.view_image(image, "image-a", view, 42)
        assert a.size == (size, size) and box == other
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(image), x)


def test_same_moments_definition_and_distinct_mixed_within_units():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(16, 8)) * 0.01
    b = a + np.arange(8)[None] * 30
    within = MODULE.sample_moments(a)
    m = MODULE.Moments(8)
    m.add(a)
    m.add(b)
    mixed = m.result()
    full = MODULE.sample_moments(np.concatenate((a, b)))
    assert mixed["effective_rank"] == pytest.approx(full["effective_rank"], abs=1e-8)
    assert mixed["relative_variance"] == pytest.approx(full["relative_variance"], abs=1e-10)
    assert within["effective_rank"] > mixed["effective_rank"]


def test_download_ranges_cover_exact_bytes_without_overlap():
    path = PATH.with_name("fetch_imagenette_reference.py")
    spec = importlib.util.spec_from_file_location("fetch_imagenette_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for size in [12, 101, module.SIZE]:
        ranges = module.spans(size)
        assert ranges[0][0] == 0 and ranges[-1][1] == size - 1
        assert sum(b - a + 1 for a, b in ranges) == size
        assert all(a[1] + 1 == b[0] for a, b in itertools.pairwise(ranges))


def test_cpu_guard_rejects_ambient_cuda_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="protect active jobs"):
        MODULE.run(None)


def test_prefix_staging_stops_after_complete_validation_and_preserves_bytes(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stage_imagenette_validation", PATH.with_name("stage_imagenette_validation.py")
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["dinov2_reference"] = MODULE
    spec.loader.exec_module(module)
    raw = io.BytesIO()
    image = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(image, format="JPEG")
    payload = image.getvalue()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        values = [
            (
                "imagenette2-320/noisy_imagenette.csv",
                b"path,is_valid\nval/a/a.jpg,True\nval/b/b.jpg,True\ntrain/c/c.jpg,False\n",
            ),
            ("imagenette2-320/val/a/a.jpg", payload),
            ("imagenette2-320/val/b/b.jpg", payload),
            ("imagenette2-320/train/c/c.jpg", b"not-needed"),
        ]
        for name, data in values:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = raw.getvalue()
    parts = [tmp_path / "part0", tmp_path / "part1"]
    parts[0].write_bytes(data[: len(data) // 2])
    parts[1].write_bytes(data[len(data) // 2 :])
    receipt = module.stage(parts, tmp_path / "output", expected_count=2, expected_classes=2)
    assert receipt["validation_images"] == 2 and receipt["training_images_extracted"] == 0
    assert not receipt["whole_archive_crc_verified"]
    assert not (tmp_path / "output/train").exists()
    assert (tmp_path / "output/val/a/a.jpg").read_bytes() == payload
