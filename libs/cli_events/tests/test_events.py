import io
import json

import pytest

from cli_events import (
    EVENT_TYPES,
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    StreamTerminatedError,
    argparse_error_event,
    argparse_error_field,
    error_event,
)


def _lines(buf):
    return [ln for ln in buf.getvalue().decode("utf-8").split("\n") if ln]


def test_event_types_are_the_four_kinds_in_order():
    assert EVENT_TYPES == ("progress", "warning", "result", "error")


def test_each_event_is_one_json_object_per_line_with_type():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.progress(stage="bake", done=0, total=None)
    em.result(mode="bake", output="x.vmd")
    raw = buf.getvalue().decode("utf-8")
    assert raw.endswith("\n")
    parts = raw.split("\n")
    assert parts[-1] == ""
    obj_lines = parts[:-1]
    assert len(obj_lines) == 2
    assert all(ln != "" for ln in obj_lines)
    objs = [json.loads(ln) for ln in obj_lines]
    assert objs[0]["type"] == "progress"
    assert objs[0]["stage"] == "bake"
    assert objs[0]["total"] is None
    assert objs[1]["type"] == "result"
    assert objs[1]["mode"] == "bake"


def test_progress_and_warning_are_non_terminal():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.progress(stage="bake", done=1, total=10)
    em.warning(code="fade_shortened", message="...", section=None)
    em.progress(stage="bake", done=2, total=10)
    assert not em.terminated
    assert len(_lines(buf)) == 3


def test_result_terminates_stream():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.result(mode="bake")
    assert em.terminated
    with pytest.raises(StreamTerminatedError):
        em.warning(code="x", message="y", section=None)


def test_error_terminates_stream():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.error(code="not_vmd", exit_code=1, field="input", path=None, message="bad")
    assert em.terminated
    with pytest.raises(StreamTerminatedError):
        em.result(mode="bake")


def test_error_after_error_is_rejected():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.error(code="x", exit_code=1, field=None, path=None, message="m")
    with pytest.raises(StreamTerminatedError):
        em.error(code="y", exit_code=1, field=None, path=None, message="n")


def test_result_after_result_is_rejected():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.result(mode="bake")
    with pytest.raises(StreamTerminatedError):
        em.result(mode="bake")


def test_non_ascii_is_written_as_utf8_without_escaping():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.warning(code="non_camera_sections_passthrough", message="カメラ以外", section=["bone"])
    raw = buf.getvalue()
    assert "カメラ以外".encode("utf-8") in raw
    obj = json.loads(raw.decode("utf-8"))
    assert obj["message"] == "カメラ以外"
    assert obj["section"] == ["bone"]


def test_newline_in_payload_is_escaped_within_one_line():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.warning(code="x", message="line1\nline2", section=None)
    text = buf.getvalue().decode("utf-8")
    assert text.count("\n") == 1
    assert json.loads(text)["message"] == "line1\nline2"


def test_line_separator_is_lf_only():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.progress(stage="bake", done=0, total=None)
    em.warning(code="octave_clamped", message="日本語メッセージ", section=None)
    em.result(mode="bake", output="out.vmd")
    raw = buf.getvalue()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 3


def test_empty_payload_emits_type_only():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.result()
    assert json.loads(buf.getvalue().decode("utf-8")) == {"type": "result"}


def test_stream_without_flush_is_accepted():
    class _WriteOnly:
        def __init__(self):
            self.data = b""

        def write(self, b):
            self.data += b

    s = _WriteOnly()
    EventEmitter(s).result(mode="bake")
    assert json.loads(s.data.decode("utf-8"))["type"] == "result"


def test_error_event_has_fixed_key_set():
    ev = error_event(code="bad_argument", message="unknown", exit_code=2, field="--foo")
    assert ev == {
        "type": "error",
        "code": "bad_argument",
        "exit_code": 2,
        "field": "--foo",
        "path": None,
        "message": "unknown",
    }


def test_error_event_field_and_path_default_to_none():
    ev = error_event(code="internal_error", message="m", exit_code=1)
    assert ev["field"] is None
    assert ev["path"] is None


def test_machine_parser_raises_parse_error_instead_of_exiting():
    p = MachineArgumentParser(prog="x")
    p.add_argument("--n", type=int)
    assert p.parse_args(["--n", "3"]).n == 3
    with pytest.raises(ArgumentParseError) as exc:
        p.parse_args(["--n", "notint"])
    assert exc.value.message


def test_help_and_version_exit_instead_of_raising_parse_error():
    p = MachineArgumentParser(prog="x")
    p.add_argument("--version", action="version", version="x 1.0")
    with pytest.raises(SystemExit):
        p.parse_args(["--help"])
    with pytest.raises(SystemExit):
        p.parse_args(["--version"])


def test_argparse_error_event_carries_argparse_message_and_exit_code_2():
    p = MachineArgumentParser(prog="x")
    p.add_argument("--n", type=int)
    with pytest.raises(ArgumentParseError) as exc:
        p.parse_args(["--n", "notint"])
    ev = argparse_error_event(exc.value, code="bad_argument", field="--n")
    assert ev == {
        "type": "error",
        "code": "bad_argument",
        "exit_code": 2,
        "field": "--n",
        "path": None,
        "message": exc.value.message,
    }


def test_argparse_error_event_field_defaults_to_none():
    ev = argparse_error_event(ArgumentParseError("m"), code="bad_argument")
    assert ev["field"] is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param("argument --foo: invalid int value: 'x'", "--foo", id="argument_option"),
        pytest.param("argument -o/--output: expected one argument", "--output", id="argument_takes_last_option_string"),
        pytest.param("argument input: invalid choice: 'z'", "input", id="argument_positional_as_is"),
        pytest.param("unrecognized arguments: --bar baz", "--bar", id="unrecognized_first_token"),
        pytest.param("the following arguments are required: input", "input", id="required_single"),
        pytest.param("the following arguments are required: input, --other", "input", id="required_first_of_list"),
        pytest.param("argument --foo invalid", None, id="argument_without_colon"),
        pytest.param("argument ", None, id="argument_prefix_only"),
        pytest.param("some other unexpected message", None, id="unknown_form"),
        pytest.param("", None, id="empty"),
    ],
)
def test_argparse_error_field_extraction(message, expected):
    assert argparse_error_field(message) == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param([], "input", id="missing_positional"),
        pytest.param(["in", "--n", "notint"], "--n", id="invalid_type"),
        pytest.param(["in", "--nope"], "--nope", id="unknown_option"),
    ],
)
def test_argparse_error_field_from_real_parser_message(argv, expected):
    p = MachineArgumentParser(prog="x", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("--n", type=int)
    with pytest.raises(ArgumentParseError) as exc:
        p.parse_args(argv)
    assert argparse_error_field(exc.value.message) == expected
