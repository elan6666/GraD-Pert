import copy

import pytest

from scripts.server.probe_r50_concurrency import compare, save


def record(duration=1.0, offset=0.0):
    return dict(
        status="complete",
        commit="a",
        config_sha="b",
        gpu_uuid="gpu1",
        steps=32,
        warmup=4,
        evaluation_access=[],
        persistent_pkl=0,
        samples=[
            dict(
                global_step=i,
                seconds=duration,
                start=offset + i * duration,
                end=offset + (i + 1) * duration,
                free_bytes=5 * 1024**3,
                oom=0,
                retry=0,
            )
            for i in range(32)
        ],
    )


def test_overlap_and_aggregate_rate():
    result = compare(record(), [record(1.5), record(1.5)])
    assert result["aggregate_throughput_ratio"] == pytest.approx(4 / 3)
    assert result["concurrency_candidate"]
    assert not result["formal_launch_authorized_by_this_receipt"]


def test_serial_pair_or_memory_pressure_not_accepted():
    assert not compare(record(), [record(), record(offset=40)])["concurrency_candidate"]
    pair = [record(), record()]
    pair[0]["samples"][3]["retry"] = 1
    assert not compare(record(), pair)["concurrency_candidate"]


@pytest.mark.parametrize(
    "field,value", [("commit", "c"), ("evaluation_access", ["test"]), ("persistent_pkl", 1)]
)
def test_identity_and_test_guard(field, value):
    altered = copy.deepcopy(record())
    altered[field] = value
    with pytest.raises(ValueError):
        compare(record(), [record(), altered])


def test_no_overwrite(tmp_path):
    target = tmp_path / "receipt.json"
    save(target, {"status": "failed"})
    with pytest.raises(FileExistsError):
        save(target, {"status": "complete"})
