import io
import json
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/v2/trace_costs.py"))


def test_stream_across_small_chunks_and_escaped_brackets():
    rows = [{"ph": "X", "cat": "kernel", "name": 'a]"中', "dur": i} for i in range(15)]
    source = json.dumps({"meta": 1, "traceEvents": rows, "tail": {}})
    assert list(MODULE["events"](io.StringIO(source), 7)) == rows
    result = MODULE["aggregate"](iter(rows))
    assert result["categories"]["kernel"]["calls"] == 15
    assert result["categories"]["kernel"]["duration_sum_us"] == sum(range(15))


@pytest.mark.parametrize("text", ['{"traceEvents":[{"a":', '{"traceEvents":[{}', "{}"])
def test_truncated_trace_fails(text):
    with pytest.raises(ValueError):
        list(MODULE["events"](io.StringIO(text), 3))
