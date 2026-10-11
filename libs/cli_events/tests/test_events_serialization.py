import io
import json
from pathlib import Path

import numpy as np
import pytest

from cli_events import EventEmitter, error_event


def _reject_constant(token):
    raise AssertionError(f"JSON に {token} トークンが出力された")


def _emit_line(method="result", **fields):
    buf = io.BytesIO()
    getattr(EventEmitter(buf), method)(**fields)
    return buf.getvalue().decode("utf-8")


def _emit_object(**fields):
    return json.loads(_emit_line(**fields), parse_constant=_reject_constant)


@pytest.mark.parametrize("value, expected", [
    pytest.param(np.float32(0.25), 0.25, id="float32"),
    pytest.param(np.float64(1.5), 1.5, id="float64"),
    pytest.param(np.int64(3), 3, id="int64"),
    pytest.param(np.int32(-2), -2, id="int32"),
    pytest.param(np.uint8(200), 200, id="uint8"),
])
def test_numpy_numeric_scalars_become_json_numbers(value, expected):
    obj = _emit_object(v=value)
    assert obj["v"] == expected
    assert type(obj["v"]) is type(expected)


def test_numpy_bool_becomes_json_boolean():
    assert _emit_object(v=np.bool_(True))["v"] is True
    assert _emit_object(v=np.bool_(False))["v"] is False


def test_python_bool_stays_boolean_not_number():
    assert _emit_object(v=True)["v"] is True


def test_numpy_scalars_nested_in_containers_are_converted():
    obj = _emit_object(stats={"keys": np.int64(7), "ratios": [np.float32(0.5), (np.float64(2.0),)]})
    assert obj["stats"] == {"keys": 7, "ratios": [0.5, [2.0]]}


def test_path_becomes_its_string_form():
    path = Path("out") / "a.vmd"
    assert _emit_object(output=path)["output"] == str(path)


def test_path_nested_in_a_list_becomes_string():
    path = Path("out") / "a.vmd"
    assert _emit_object(outputs=[path])["outputs"] == [str(path)]


def test_non_ascii_path_stays_utf8_unescaped():
    line = _emit_line(output=Path("出力") / "曲.vmd")
    assert "曲.vmd" in line


def test_error_event_with_path_object_is_serialized_as_string():
    path = Path("out") / "a.vmd"
    buf = io.BytesIO()
    EventEmitter(buf).error(**error_event(code="write_failed", message="m", exit_code=3, path=path))
    assert json.loads(buf.getvalue().decode("utf-8"))["path"] == str(path)


@pytest.mark.parametrize("value", [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="neg_inf"),
    pytest.param(np.float64("nan"), id="numpy_float64_nan"),
    pytest.param(np.float64("inf"), id="numpy_float64_inf"),
    pytest.param(np.float32("nan"), id="numpy_float32_nan"),
    pytest.param(np.float32("-inf"), id="numpy_float32_neg_inf"),
])
def test_non_finite_float_becomes_null_without_nan_or_infinity_tokens(value):
    line = _emit_line(v=value)
    assert "NaN" not in line and "Infinity" not in line
    assert json.loads(line, parse_constant=_reject_constant)["v"] is None


def test_non_finite_values_nested_in_containers_become_null():
    obj = _emit_object(
        max_amplitude=float("nan"),
        frame_range=[0, float("inf")],
        stats={"err": (np.float64("nan"), 1.0), "ratio": np.float32("inf")})
    assert obj["max_amplitude"] is None
    assert obj["frame_range"] == [0, None]
    assert obj["stats"] == {"err": [None, 1.0], "ratio": None}


def test_non_finite_value_in_progress_and_warning_events_becomes_null():
    for method in ("progress", "warning"):
        line = _emit_line(method, done=float("nan"), total=None)
        assert json.loads(line, parse_constant=_reject_constant)["done"] is None


def test_finite_floats_are_unchanged():
    obj = _emit_object(a=0.0, b=-1.5, c=1e300, d=5e-324)
    assert (obj["a"], obj["b"], obj["c"], obj["d"]) == (0.0, -1.5, 1e300, 5e-324)


def test_event_with_non_finite_value_is_still_a_single_terminated_line():
    buf = io.BytesIO()
    emitter = EventEmitter(buf)
    emitter.result(mode="inspect", max_amplitude=float("nan"))
    assert emitter.terminated
    assert buf.getvalue().count(b"\n") == 1


def test_unserializable_object_still_raises_type_error_and_writes_nothing():
    buf = io.BytesIO()
    with pytest.raises(TypeError):
        EventEmitter(buf).result(v=object())
    assert buf.getvalue() == b""
