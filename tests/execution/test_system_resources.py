from __future__ import annotations

from pathlib import Path

import pytest

from gradpert.execution import system_resources


def test_linux_mem_available_uses_reclaimable_kernel_estimate(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       263000000 kB\n"
        "MemFree:          3900000 kB\n"
        "MemAvailable:   253129896 kB\n"
        "Cached:         239400596 kB\n",
        encoding="utf-8",
    )

    assert system_resources._linux_mem_available_bytes(meminfo) == 253129896 * 1024


@pytest.mark.parametrize(
    "content",
    (
        "MemFree: 10 kB\n",
        "MemAvailable: nope kB\n",
        "MemAvailable: -1 kB\n",
        "MemAvailable: 1 MB\n",
        "MemAvailable: 1 kB\nMemAvailable: 2 kB\n",
    ),
)
def test_linux_mem_available_rejects_missing_or_malformed_values(
    tmp_path: Path,
    content: str,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(content, encoding="utf-8")

    assert system_resources._linux_mem_available_bytes(meminfo) is None


def test_host_available_memory_prefers_linux_memavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 20 kB\n", encoding="utf-8")
    monkeypatch.setattr(system_resources.sys, "platform", "linux")

    def fail_sysconf(_name: str) -> int:
        raise AssertionError("SC_AVPHYS_PAGES must not override Linux MemAvailable")

    monkeypatch.setattr(system_resources.os, "sysconf", fail_sysconf)
    assert system_resources.host_available_memory_bytes(meminfo_path=meminfo) == 20 * 1024


def test_host_available_memory_falls_back_when_meminfo_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_resources.sys, "platform", "linux")
    values = {"SC_AVPHYS_PAGES": 11, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(system_resources.os, "sysconf", values.__getitem__)

    assert (
        system_resources.host_available_memory_bytes(meminfo_path=tmp_path / "missing") == 11 * 4096
    )


def test_host_available_memory_rejects_invalid_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_resources.sys, "platform", "linux")
    values = {"SC_AVPHYS_PAGES": -1, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(system_resources.os, "sysconf", values.__getitem__)

    assert system_resources.host_available_memory_bytes(meminfo_path=tmp_path / "missing") is None
