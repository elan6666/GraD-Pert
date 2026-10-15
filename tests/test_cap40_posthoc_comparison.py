"""Comparison joins actual IDs and verifies every per-arm artifact before rendering."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
from cap40_posthoc import csv_write, read, sha, write
from cap40_posthoc_compare import compare


def fixture(tmp_path, mismatch=False):
    rows = []
    for arm in ("DP", "FN", "FG", "FB", "PO"):
        p = tmp_path / arm
        p.mkdir()
        (p / "historical").mkdir()
        s = {
            "arm": arm,
            "method_sha": "method",
            "analysis": {"commit": "analysis"},
            "checkpoint_epoch": 4,
            "checkpoint_sha256": arm,
            "rows": 2,
            "conditions": 2,
            "deep_rows": 2,
            "scope": "medium_cap40_stratified",
            "global_heads": {"teacher.ssl1_cls.view0": {"H_sample": 1.0, "JS": 0.2}},
            "global_features": {"student.main.response_cls": {"effective_rank": 2.0}},
            "metrics": {"prediction_mse": {"condition_mean": 0.1}},
        }
        write(p / "analysis_summary.json", s)
        csv_write(
            p / "outputs_full.csv",
            [
                {
                    "condition_id": c,
                    "truth_row_id": ("different" if mismatch and arm == "FN" else "truth") + c,
                    "control_row_id": "control" + c,
                    "prediction_mse": v,
                }
                for c, v in (("p", 0.1), ("q", 0.2))
            ],
        )
        csv_write(
            p / "historical/metrics.csv",
            [
                {
                    "group": g,
                    "metric_id": m,
                    "macro_mean": 0.1,
                    "finite_condition_count": 10,
                    "total_condition_count": 10,
                }
                for g in ("all", "seen", "unseen")
                for m in ("t-all", "t-deg", "r-all", "r-deg", "s-all", "s-deg")
            ],
        )
        for i in range(9):
            (p / f"{i:02d}.png").write_bytes(b"synthetic test png index")
        write(p / "report.md", {"synthetic": True})
        write(
            p / "receipt.json",
            {
                "manifest_sha256": "common",
                "checkpoint_sha256": arm,
                "outputs": {
                    str(f.relative_to(p)): {"sha256": sha(f), "bytes": f.stat().st_size}
                    for f in p.rglob("*")
                    if f.is_file()
                },
            },
        )
        rows.append({"arm": arm, "report": str(p), "checkpoint_sha256": arm})
    q = tmp_path / "queue.json"
    write(q, {"manifest_sha256": "common", "rows": rows})
    return q


def test_complete_shared_comparison_indexes_45_arm_figures_and_18_scores(tmp_path):
    q = fixture(tmp_path)
    out = tmp_path / "comparison"
    compare(SimpleNamespace(queue=q, output=out))
    content = (out / "report.md").read_text()
    assert content.count("![") == 49
    with (out / "test18.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 90
    rec = read(out / "receipt.json")
    assert len(rec["sources"]) == 5
    for name, entry in rec["outputs"].items():
        assert sha(out / name) == entry["sha256"]


def test_reject_misaligned_truth_control_pairs_before_any_report(tmp_path):
    q = fixture(tmp_path, mismatch=True)
    out = tmp_path / "comparison"
    with pytest.raises(ValueError, match="alignment"):
        compare(SimpleNamespace(queue=q, output=out))
    assert not out.exists()
