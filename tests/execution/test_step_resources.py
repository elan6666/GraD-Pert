import pytest
import torch

from gradpert.execution.step_resources import capture_step_resources


def test_cpu_fixture_is_not_cuda_acceptance(tmp_path):
    result = capture_step_resources("cpu", tmp_path)
    assert not result["cuda_acceptance"]
    assert result["disk_free_bytes"] > 0


@pytest.mark.parametrize(
    "free_gib,peak_gib,retries,ooms,passed",
    [
        (25, 7, 0, 0, True),
        (4, 7, 0, 0, False),
        (25, 29, 0, 0, False),
        (25, 7, 1, 0, False),
        (25, 7, 0, 1, False),
    ],
)
def test_cuda_gate_covers_free_peak_and_allocator_history(
    tmp_path, monkeypatch, free_gib, peak_gib, retries, ooms, passed
):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (free_gib * 1024**3, 32 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 6 * 1024**3)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda device: {
            "reserved_bytes.all.peak": peak_gib * 1024**3,
            "num_alloc_retries": retries,
            "num_ooms": ooms,
        },
    )
    if passed:
        assert capture_step_resources("cuda:0", tmp_path)["cuda_acceptance"]
    else:
        with pytest.raises(RuntimeError, match="resource gate failed"):
            capture_step_resources("cuda:0", tmp_path)


def test_cuda_allocator_required_before_query(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    with pytest.raises(RuntimeError, match="expandable_segments"):
        capture_step_resources("cuda:0", tmp_path)
