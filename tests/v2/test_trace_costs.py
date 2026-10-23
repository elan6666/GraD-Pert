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


@pytest.fixture(scope="module")
def baseline():
    import subprocess

    source = subprocess.check_output(
        ["git", "show", "02cb70ca05e0b8285a94cae2ef307f36f5261942:scripts/v2/trace_costs.py"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )
    namespace = {"__name__": "baseline"}
    exec(compile(source, "baseline_trace_costs.py", "exec"), namespace)
    return namespace


def outcome(parser, source, chunk):
    rows = []
    try:
        rows.extend(parser(io.StringIO(source), chunk))
    except ValueError as error:
        cause = error.__cause__
        decoder_error = (
            (cause.msg, cause.doc, cause.pos) if isinstance(cause, json.JSONDecodeError) else None
        )
        return rows, str(error), decoder_error
    return rows, None, None


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 31, 65536])
@pytest.mark.parametrize(
    "source",
    [
        '{"traceEvents":[]}',
        '{"traceEvents":[{},]}trailing content ignored',
        '{"traceEvents":[{} {}]}',
        '{"traceEvents":[1]}',
        '{"traceEvents":[null]}',
        '{"traceEvents":[{}',
        '{"traceEvents":[{},',
        '{"traceEvents":[{"a":"unfinished',
        '{"traceEvents":[{}\u2003,\u001c{}]}',
        '{"traceEvents"\u2003:\n[\u00a0{},\t{}\r]}',
        '{"other": "traceEvents", "traceEvents": [{"name":"a]\\"中"}]}',
        "{}",
    ],
)
def test_exact_baseline_outcomes(baseline, source, chunk):
    assert outcome(MODULE["events"], source, chunk) == outcome(baseline["events"], source, chunk)


def test_large_event_and_every_truncation(baseline):
    source = json.dumps({"traceEvents": [{"name": '中\\"[]' * 2000}, {"dur": 3}]})
    assert outcome(MODULE["events"], source, 127) == outcome(baseline["events"], source, 127)
    small = '{"traceEvents":[{"a":1},{"b":"x"}]}'
    for end in range(len(small) + 1):
        assert outcome(MODULE["events"], small[:end], 5) == outcome(
            baseline["events"], small[:end], 5
        )


def test_bounded_reads_and_lazy_streaming():
    class BoundedStream(io.StringIO):
        def read(self, size=-1):
            assert size == 64
            return super().read(size)

    source = json.dumps({"traceEvents": [{"i": i} for i in range(10000)]})
    stream = BoundedStream(source)
    iterator = MODULE["events"](stream, 64)
    assert next(iterator) == {"i": 0}
    assert stream.tell() <= 64
    assert len(list(iterator)) == 9999


def test_header_bound_matches_baseline(baseline):
    source = " " * (1048576 + 32)
    assert outcome(MODULE["events"], source, 65536) == outcome(baseline["events"], source, 65536)
