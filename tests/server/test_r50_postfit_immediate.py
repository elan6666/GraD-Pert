import sys
from types import SimpleNamespace

from scripts.server import run_r50_postfit_queue as queue


def test_completed_training_tests_without_idle_gpu_poll(tmp_path, monkeypatch):
    training = tmp_path / "training"
    output = tmp_path / "output"
    for row in ("ref", "lr_low", "lr_mid"):
        (training / row).mkdir(parents=True)
        (training / row / "COMPLETE.json").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "queue",
            "--source",
            str(tmp_path),
            "--training-root",
            str(training),
            "--output-root",
            str(output),
            "--data-root",
            str(tmp_path),
            "--publication",
            str(tmp_path / "pub"),
            "--publication-sha",
            "sha",
            "--gpu-uuid",
            "GPU-test",
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("must not wait for an idle GPU")

    monkeypatch.setattr(queue.time, "sleep", forbidden)
    monkeypatch.setattr(queue.subprocess, "check_output", forbidden)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[-1])
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(queue.subprocess, "run", run)
    queue.main()
    assert calls == ["ref", "lr_low", "lr_mid"]
