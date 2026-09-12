import os

import pytest

from scripts.performance.profile_native_a0 import ProfileGateError, _abba_isolation


def snapshot():
    return {
        "load_average": [1.0, 1.0, 1.0],
        "nvidia_smi": {
            "gpus": {"returncode": 0, "rows": [{"uuid": "other", "utilization.gpu": "0"}]},
            "compute_apps": {"returncode": 0, "rows": [{"pid": str(os.getpid())}]},
        },
    }


def test_own_compute_allowed():
    _abba_isolation(snapshot(), "selected")


@pytest.mark.parametrize(
    "case", ["other_pid", "busy_gpu", "busy_host", "missing_load", "query_failure"]
)
def test_isolation_fails_closed(case):
    s = snapshot()
    if case == "other_pid":
        s["nvidia_smi"]["compute_apps"]["rows"].append({"pid": "-1"})
    elif case == "busy_gpu":
        s["nvidia_smi"]["gpus"]["rows"][0]["utilization.gpu"] = "6"
    elif case == "busy_host":
        s["load_average"][0] = 2.01
    elif case == "missing_load":
        s["load_average"] = None
    else:
        s["nvidia_smi"]["gpus"]["returncode"] = 1
    with pytest.raises(ProfileGateError):
        _abba_isolation(s, "selected")
