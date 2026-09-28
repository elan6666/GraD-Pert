"""The optional live viewer reports the current stage without model imports."""

from scripts.v2.show_progress import render


def test_training_progress_includes_step_and_loss():
    line = render(
        {
            "phase": "training",
            "epoch": 2,
            "epochs_total": 3,
            "epoch_step": 25,
            "epoch_steps_total": 100,
            "latest_training_terms": {"joint_loss": 0.125},
            "throughput": {"cells_per_second_this_epoch": 12.5},
        }
    )
    assert "25.0%" in line
    assert "epoch 2/3 step 25/100" in line
    assert "joint_loss=0.125000" in line
    assert "12.50 cells/s" in line


def test_evaluation_progress_identifies_checkpoint_role():
    line = render(
        {
            "phase": "test",
            "checkpoint_role": "best",
            "conditions_completed": 20,
            "conditions_total": 40,
        }
    )
    assert "50.0%" in line
    assert "best 20/40 conditions" in line
