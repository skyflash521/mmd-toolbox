import io
from unittest.mock import MagicMock, call

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


def test_machine_mode_close_and_summary_do_not_touch_display_or_emitter(spy_display):
    emitter = _FakeEmitter()
    router = _router(machine=True, emitter=emitter, stream=_NonTTYStream())
    router.close()
    router.summary("完了")
    spy_display.return_value.close.assert_not_called()
    spy_display.return_value.summary.assert_not_called()
    assert emitter.calls == []


@pytest.mark.parametrize("isatty,quiet,expect_enabled", [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
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


@pytest.mark.parametrize("option", ["write_lock", "now"])
def test_unspecified_display_option_is_passed_as_none(spy_display, option):
    _router()
    assert spy_display.call_args.kwargs[option] is None


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


def test_same_stage_id_updates_only_and_new_stage_id_switches(spy_display):
    display = spy_display.return_value
    router = _router()
    router.stage("load", done=0, total=3)
    router.stage("load", done=1, total=3, note="n")
    display.stage.assert_called_once_with("音声読み込み")
    assert display.update.call_args_list == [call(0, 3, ""), call(1, 3, "n")]
    router.stage("write")
    assert display.stage.call_args_list == [call("音声読み込み"), call("書き出し")]


def test_stage_after_close_restarts_even_with_same_stage_id(spy_display):
    display = spy_display.return_value
    router = _router()
    router.stage("load")
    router.close()
    router.stage("load")
    assert display.stage.call_args_list == [call("音声読み込み"), call("音声読み込み")]


def test_close_is_idempotent(spy_display):
    display = spy_display.return_value
    router = _router()
    router.close()
    router.close()
    assert display.close.call_count == 2


def test_close_and_summary_delegate_to_the_display(spy_display):
    display = spy_display.return_value
    router = _router()
    router.close()
    router.summary("完了 out.vmd")
    display.close.assert_called_once_with()
    display.summary.assert_called_once_with("完了 out.vmd")


class _RaisingEmitter:
    def __init__(self, error):
        self._error = error

    def progress(self, **fields):
        raise self._error


class _NonIOEmitError(Exception):
    pass


@pytest.mark.parametrize("error_type", [OSError, BrokenPipeError, ValueError, _NonIOEmitError])
def test_emit_failure_raises_dedicated_error_keeping_cause_type_and_message(error_type):
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
