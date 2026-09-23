from pathlib import Path

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options

ROOT = Path(__file__).resolve().parents[2] / "configs/v2/b01_five_epoch_jurkat"


def test_b0_b1_match_except_genept_table_and_are_five_epoch_dual_gpu_runs():
    paths = [ROOT / name / "gradpert_v2/nadig_jurkat.yaml" for name in ("B0", "B1")]
    configs = [load_experiment_config(path) for path in paths]
    assert all(c.training.formal_run_policy == "v2_fixed_5" for c in configs)
    assert all(c.training.max_epochs.value == 5 for c in configs)
    assert all(c.training.train_batch_size.value == 128 for c in configs)
    assert all(c.model.excludes_test_target_expression for c in configs)
    a, b = [V2Options.parse_parameters(c.model.parameters) for c in configs]
    assert a[0] == b[0]
    left = vars(a[1]).copy()
    right = vars(b[1]).copy()
    for name in ("genept_artifact_path", "genept_sha256"):
        assert left.pop(name) != right.pop(name)
    assert left == right
    assert a[1].world_size == b[1].world_size == 2
