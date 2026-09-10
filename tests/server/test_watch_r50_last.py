import os

import pytest

from scripts.server.watch_r50_last import preserve


def test_atomic_replacement_and_deletion_do_not_mutate_archive(tmp_path):
    source = tmp_path / "last.pt"
    archive = tmp_path / "archive.pt"
    source.write_bytes(b"epoch49")
    assert preserve(source, archive)
    staged = tmp_path / "new.pt"
    staged.write_bytes(b"epoch50")
    os.replace(staged, source)
    assert archive.read_bytes() == b"epoch49"
    assert preserve(source, archive)
    source.unlink()
    assert archive.read_bytes() == b"epoch50"
    assert not preserve(source, archive)


def test_symlink_rejected(tmp_path):
    real = tmp_path / "real.pt"
    real.write_bytes(b"checkpoint")
    link = tmp_path / "last.pt"
    link.symlink_to(real)
    with pytest.raises(ValueError):
        preserve(link, tmp_path / "archive.pt")
