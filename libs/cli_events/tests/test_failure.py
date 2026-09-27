import io
import json
import sys

import pytest

from cli_events import EventEmitter, emit_failure


class _RaisingStream:
    def __init__(self, exc):
        self._exc = exc

    def write(self, _data):
        raise self._exc


class _TextSink:
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s
        return len(s)


class _RaisingTextSink:
    def __init__(self, exc):
        self._exc = exc

    def write(self, _s):
        raise self._exc


_SINK_EXCEPTIONS_OF_UNRELATED_TYPES = [
    pytest.param(OSError("stderr closed"), id="os_error"),
    pytest.param(ValueError("closed file"), id="value_error"),
    pytest.param(RuntimeError("壊れた出力先"), id="runtime_error"),
]


def test_emits_error_event_without_error_line_when_emitter_usable():
    buf = io.BytesIO()
    sink = _TextSink()
    rc = emit_failure(EventEmitter(buf), code="not_vmd", message="読めない", exit_code=1,
                      field="input", path="in.vmd", stderr=sink)
    assert rc == 1
    assert sink.text == ""
    ev = json.loads(buf.getvalue().decode("utf-8"))
    assert ev == {
        "type": "error",
        "code": "not_vmd",
        "exit_code": 1,
        "field": "input",
        "path": "in.vmd",
        "message": "読めない",
    }


def test_extra_keys_are_added_to_error_event_keeping_base_keys():
    buf = io.BytesIO()
    sink = _TextSink()
    rc = emit_failure(EventEmitter(buf), code="stage_failed", message="分離に失敗", exit_code=4,
                      field="input", path="in.wav", stderr=sink, stage="separate")
    assert rc == 4
    assert json.loads(buf.getvalue().decode("utf-8")) == {
        "type": "error",
        "code": "stage_failed",
        "exit_code": 4,
        "field": "input",
        "path": "in.wav",
        "message": "分離に失敗",
        "stage": "separate",
    }
    assert sink.text == ""


def test_error_line_when_emitter_is_none():
    sink = _TextSink()
    rc = emit_failure(None, code="internal_error", message="RuntimeError: 想定外", exit_code=1,
                      stderr=sink)
    assert rc == 1
    assert sink.text == "error: RuntimeError: 想定外\n"


def test_error_line_carries_message_only_not_field_path_or_extra_keys():
    sink = _TextSink()
    rc = emit_failure(None, code="stage_failed", message="分離に失敗", exit_code=4,
                      field="input", path="in.wav", stderr=sink, stage="separate")
    assert rc == 4
    assert sink.text == "error: 分離に失敗\n"


def test_terminated_emitter_gets_error_line_and_given_exit_code_leaving_stream_unchanged():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.result(mode="convert")
    before = buf.getvalue()
    sink = _TextSink()
    rc = emit_failure(em, code="cancelled", message="中断された", exit_code=130, stderr=sink)
    assert rc == 130
    assert buf.getvalue() == before
    assert sink.text == "error: 中断された\n"


@pytest.mark.parametrize("exc", [
    pytest.param(OSError("broken pipe"), id="broken_pipe"),
    pytest.param(ValueError("closed file"), id="closed_stdout"),
])
def test_falls_back_to_error_line_with_given_exit_code_when_event_write_fails(exc):
    sink = _TextSink()
    rc = emit_failure(EventEmitter(_RaisingStream(exc)), code="output_exists",
                      message="出力先に既存ファイルがあります", exit_code=2, field="--output",
                      stderr=sink, stage="write")
    assert rc == 2
    assert sink.text == "error: 出力先に既存ファイルがあります\n"


def test_other_event_write_exceptions_propagate_without_error_line():
    sink = _TextSink()
    with pytest.raises(RuntimeError):
        emit_failure(EventEmitter(_RaisingStream(RuntimeError("想定外"))), code="internal_error",
                     message="漏らす", exit_code=1, stderr=sink)
    assert sink.text == ""


@pytest.mark.parametrize("sink_exc", _SINK_EXCEPTIONS_OF_UNRELATED_TYPES)
def test_swallows_failure_of_the_error_line_after_event_write_failed(sink_exc):
    rc = emit_failure(EventEmitter(_RaisingStream(OSError("broken pipe"))), code="internal_error",
                      message="どこにも書けない", exit_code=1, stderr=_RaisingTextSink(sink_exc))
    assert rc == 1


@pytest.mark.parametrize("sink_exc", _SINK_EXCEPTIONS_OF_UNRELATED_TYPES)
def test_swallows_failure_of_the_error_line_without_emitter(sink_exc):
    rc = emit_failure(None, code="internal_error", message="どこにも書けない", exit_code=1,
                      stderr=_RaisingTextSink(sink_exc))
    assert rc == 1


def test_stderr_default_resolves_at_call_time(monkeypatch):
    sink = _TextSink()
    monkeypatch.setattr(sys, "stderr", sink)
    rc = emit_failure(None, code="bad_argument", message="引数が不正", exit_code=2)
    assert rc == 2
    assert sink.text == "error: 引数が不正\n"
