import io
import time
from unittest.mock import MagicMock

import pytest

from cli_progress_router import ProgressEmitError, ProgressRouter
from cli_progress_router import router as router_module

_LABELS = {"load": "音声読み込み", "write": "書き出し"}


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


class _NonTTYStream(io.StringIO):
    def isatty(self):
        return False


class _FakeEmitter:
    def __init__(self):
        self.calls = []

    def progress(self, **fields):
        self.calls.append(fields)


class _NoTouchEmitter:
    def __getattr__(self, name):
        raise AssertionError(f"non-machine mode must not touch emitter (accessed {name!r})")


@pytest.fixture
def spy_display(monkeypatch):
    spy_cls = MagicMock(name="ProgressReporterClass")
    monkeypatch.setattr(router_module._display_module, "ProgressReporter", spy_cls)
    return spy_cls


def _router(**kwargs):
    kwargs.setdefault("machine", False)
    kwargs.setdefault("quiet", False)
    kwargs.setdefault("emitter", None)
    kwargs.setdefault("stream", _TTYStream())
    kwargs.setdefault("labels", _LABELS)
    return ProgressRouter(**kwargs)


def test_machine_mode_forwards_stage_to_emitter_and_builds_no_display(spy_display):
    emitter = _FakeEmitter()
    router = _router(machine=True, emitter=emitter, stream=_NonTTYStream())
    spy_display.assert_not_called()
    router.stage("load", done=1, total=2, note="n", elapsed=0.5)
    assert emitter.calls == [{"stage": "load", "done": 1, "total": 2, "note": "n", "elapsed": 0.5}]


def test_machine_mode_stage_defaults_done_zero_total_none_note_empty_elapsed_zero(spy_display):
    emitter = _FakeEmitter()
    router = _router(machine=True, emitter=emitter, stream=_NonTTYStream())
    router.stage("load")
    assert emitter.calls == [{"stage": "load", "done": 0, "total": None, "note": "", "elapsed": 0.0}]


@pytest.mark.parametrize("isatty,quiet", [(True, False), (True, True), (False, False), (False, True)])
def test_machine_mode_emits_regardless_of_tty_and_quiet(spy_display, isatty, quiet):
    emitter = _FakeEmitter()
    stream = _TTYStream() if isatty else _NonTTYStream()
    router = _router(machine=True, quiet=quiet, emitter=emitter, stream=stream)
    router.stage("load")
    assert len(emitter.calls) == 1
    assert stream.getvalue() == ""
    spy_display.assert_not_called()


def test_machine_mode_forwards_every_call_in_order(spy_display):
    emitter = _FakeEmitter()
    router = _router(machine=True, emitter=emitter, stream=_NonTTYStream())
    for i, stage in enumerate(("load", "load", "write")):
        router.stage(stage, done=i, total=3, note=f"n{i}", elapsed=float(i))
    assert [c["stage"] for c in emitter.calls] == ["load", "load", "write"]
    assert [c["done"] for c in emitter.calls] == [0, 1, 2]


@pytest.mark.parametrize("isatty,quiet,expect_enabled", [
    (True, False, True),
    (True, True, False),
    (False, False, False),
])
def test_non_machine_display_gets_all_kwargs_and_is_enabled_only_on_tty_without_quiet(
        spy_display, isatty, quiet, expect_enabled):
    stream = _TTYStream() if isatty else _NonTTYStream()

    def clock():
        return 0.0

    lock = object()
    _router(quiet=quiet, emitter=_NoTouchEmitter(), stream=stream,
            now=clock, interval=0.2, write_lock=lock)
    spy_display.assert_called_once_with(
        stream=stream, enabled=expect_enabled, now=clock, interval=0.2, write_lock=lock)


def test_interval_is_not_passed_when_unspecified(spy_display):
    _router()
    assert "interval" not in spy_display.call_args.kwargs


def test_stage_id_maps_to_its_label(spy_display):
    display = spy_display.return_value
    router = _router()
    router.stage("load")
    display.stage.assert_called_once_with("音声読み込み")


def test_unknown_stage_id_used_as_label_as_is(spy_display):
    display = spy_display.return_value
    router = _router()
    router.stage("未知の工程")
    display.stage.assert_called_once_with("未知の工程")


def _live_router(clock):
    stream = _TTYStream()
    router = _router(stream=stream, now=lambda: clock[0], interval=0.01)
    return router, stream


def _wait_for(stream, text, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if text in stream.getvalue():
            return True
        time.sleep(0.005)
    return False


def test_same_stage_id_keeps_the_elapsed_time_and_new_stage_id_restarts_it():
    clock = [0.0]
    router, stream = _live_router(clock)
    try:
        router.stage("load", done=0, total=3)
        clock[0] = 5.0
        router.stage("load", done=1, total=3)
        assert _wait_for(stream, "[音声読み込み] 1/3 経過 0:05")
        router.stage("write")
        assert _wait_for(stream, "[書き出し] 経過 0:00")
    finally:
        router.close()


def test_stage_after_close_restarts_even_with_same_stage_id():
    clock = [0.0]
    router, stream = _live_router(clock)
    try:
        router.stage("load")
        router.close()
        clock[0] = 5.0
        router.stage("load", done=2, total=4)
        assert _wait_for(stream, "[音声読み込み] 2/4 経過 0:00")
    finally:
        router.close()


class _RaisingEmitter:
    def __init__(self, error):
        self._error = error

    def progress(self, **fields):
        raise self._error


class _NonIOEmitError(Exception):
    pass


def test_emit_failure_raises_dedicated_error_keeping_cause_type_and_message():
    error_type = _NonIOEmitError
    error = error_type("標準出力へ書けません")
    router = _router(machine=True, emitter=_RaisingEmitter(error), stream=_NonTTYStream())
    with pytest.raises(ProgressEmitError) as exc:
        router.stage("load", note="10%")
    assert exc.value.__cause__ is error
    assert error_type.__name__ in str(exc.value)
    assert "標準出力へ書けません" in str(exc.value)


def test_emit_does_not_wrap_keyboard_interrupt():
    error = KeyboardInterrupt()
    router = _router(machine=True, emitter=_RaisingEmitter(error), stream=_NonTTYStream())
    with pytest.raises(KeyboardInterrupt) as exc:
        router.stage("load", note="10%")
    assert exc.value is error
