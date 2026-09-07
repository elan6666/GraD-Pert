import csv
import hashlib
import json
from pathlib import Path

import yaml

from gradpert.config import NativeArchitectureOptions, load_experiment_config

base = Path("/data/yilangliu/GraD-Pert/runs")
ref = (
    base
    / "formal-vnext-m-1bb0068-v1/m1_single_string_gat/gradpert_b2/nadig_jurkat/seed-1/small_results"
)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(s):
    m = read(s / "run_manifest.json")
    p = read(s / "prediction_manifest.json")["conditions"]
    e = read(s / "evaluation_manifest.json")["conditions"]
    assert len(p) == len(e) == 592
    assert all(len(x["input_control_row_ids"]) == 300 for x in p)
    return {
        "canonical": [
            m[k]
            for k in ["canonical_data_sha256", "split_content_sha256", "control_manifest_sha256"]
        ],
        "controls": [
            (x["condition_id"], x["input_control_row_ids"], x["input_control"]["content_sha256"])
            for x in p
        ],
        "truth": [
            (
                x["condition_id"],
                x["truth_row_ids"],
                x["truth"]["content_sha256"],
                x["top_de_gene_indices_sha256"],
            )
            for x in e
        ],
    }


reference = identity(ref)
out = []
for lineage in [
    "formal-vnext-o-51e922f-v1",
    "formal-vnext-w-51e922f-v1",
    "formal-vnext-m-51e922f-v1",
    "formal-vnext-h4-2ca755f-v1",
]:
    assert not list((base / lineage).rglob("*.pkl"))
    for s in sorted((base / lineage).glob("*/gradpert_b2/nadig_jurkat/seed-1/small_results")):
        run = s.parent
        m = read(s / "run_manifest.json")
        t = read(s / "training_receipt.json")
        meta = read(s / "run_meta.json")
        sys = read(s / "systems_runtime.json")
        config = Path(meta["config_path"])
        assert sha(config) == m["config_sha256"] == meta["config_sha256"]
        params = yaml.safe_load(config.read_text())["model"]["parameters"]
        arch = meta["native_architecture"]
        resolved = NativeArchitectureOptions.from_parameters(
            load_experiment_config(config).model.parameters
        )
        assert resolved.payload() == arch, (s, "complete architecture payload")
        assert resolved.payload_sha256 == meta["native_architecture_sha256"]
        checked = []
        for k, v in arch.items():
            if k in params and isinstance(v, (int, float, str, bool)):
                assert v == params[k]["value"], (s, k, v, params[k])
                checked.append(k)
        assert m["status"] == "evaluated" and m["formal_eligible"] and m["source_dirty"] is False
        assert m["source_commit"] == meta["source"]["commit"] and m["run_seed"] == 1
        assert m["test_evaluations"] == 1 and t["canonical_test_truth_present_during_fit"] is False
        assert t["epochs_requested"] == t["epochs_completed"] == 10 and t["optimizer_steps"] == 5820
        steps = list(csv.DictReader((s / "train_steps.csv").open()))
        weights = [
            params[k]["value"]
            for k in [
                "prediction_loss_weight",
                "condition_consistency_loss_weight",
                "masked_node_loss_weight",
                "spread_loss_weight",
            ]
        ]
        for step in steps:
            expected = sum(
                float(step[k]) * float(w)
                for k, w in zip(
                    [
                        "prediction_loss",
                        "condition_consistency_loss",
                        "masked_node_loss",
                        "spread_loss",
                    ],
                    weights,
                    strict=True,
                )
            )
            assert abs(expected - float(step["total_loss"])) < 1e-4, (s, "loss decomposition")
        assert [int(x["global_step"]) for x in steps] == list(range(5820))
        assert [int(x["epoch"]) for x in steps] == [i // 582 for i in range(5820)]
        assert len(list(csv.DictReader((s / "validation.csv").open()))) == 10
        assert sys["training_pipeline"]["yielded_batches"] == 5820
        assert meta["gradient_schedule_implementation"] == "staged_auxiliary"
        assert meta["global_activation_checkpointing"] is True
        recipe = read(s / "inference_recipe.json")
        assert (
            recipe["result_mode"] == "metrics_only"
            and recipe.get("result_pkl_path") is None
            and recipe.get("result_pkl_sha256") is None
        )
        pts = list(run.rglob("*.pt"))
        assert len(pts) == 1 and pts[0].name == "best.pt"
        assert sha(pts[0]) == t["checkpoint_sha256"] == m["best_checkpoint_sha256"]
        assert not any(p.is_dir() and p.name.startswith(".result-work-") for p in run.rglob("*"))
        assert identity(s) == reference
        metrics = read(s / "metrics_summary.json")["metrics"]
        assert [x["metric_id"] for x in metrics] == [
            "txpert_macro_pearson_delta",
            "trishift_pearson_delta",
            "systema_pearson",
        ]
        assert all(x["available"] for x in metrics)
        out.append(
            {
                "variant": s.parts[-5],
                "lineage": lineage,
                "source_commit": m["source_commit"],
                "config_sha256": m["config_sha256"],
                "metrics": metrics,
                "architecture_config_keys_checked": checked,
                "architecture": arch,
                "resolved_locals": meta["resolved_local_view_contract"],
                "file_sha256": {
                    n: sha(s / n)
                    for n in [
                        "run_manifest.json",
                        "training_receipt.json",
                        "run_meta.json",
                        "systems_runtime.json",
                        "prediction_manifest.json",
                        "evaluation_manifest.json",
                        "metrics_summary.json",
                    ]
                },
                "core_and_fairness_checks": "passed",
                "wrapper_status": "failed_rc80_preserved"
                if "h4-" in lineage
                else "reported_complete",
            }
        )
assert len(out) == 10
print(
    json.dumps(
        {
            "schema_version": "remaining-ablation-audit-v1",
            "reference": str(ref),
            "rows": out,
            "limitations": [
                "single seed; cross-commit retained references; no equivalence claims",
                "H4 native completion does not relabel failed wrapper",
                "architecture resolved under source51e; retained rows remain cross-commit",
            ],
        },
        indent=2,
    )
)
