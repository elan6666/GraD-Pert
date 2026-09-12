import os
from pathlib import Path

from scripts.server.run_r1024_steps import process_environment


def test_worker_imports_published_src_before_installed_package(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/old/checkout")
    env = process_environment(Path("/new/checkout"), "GPU-physical")
    assert env["PYTHONPATH"].split(os.pathsep) == ["/new/checkout/src", "/new/checkout"]
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-physical"
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
