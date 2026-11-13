import io
from unittest.mock import MagicMock

import pytest

from cli_progress_router import router as router_module
from vocal_analysis.quiet import STDERR_WRITE_LOCK
from vocal_analysis_cli import FRONT_STAGE_LABELS, build_router, stage_label


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


class _FakeEmitter:
    def __init__(self):
        self.calls = []

    def progress(self, **fields):
        self.calls.append(fields)


@pytest.fixture
def spy_display(monkeypatch):
    spy_cls = MagicMock(name="ProgressReporterClass")
    monkeypatch.setattr(router_module._display_module, "ProgressReporter", spy_cls)
    return spy_cls


def _build(**kwargs):
    kwargs.setdefault("machine", False)
    kwargs.setdefault("quiet", False)
    kwargs.setdefault("emitter", None)
    kwargs.setdefault("stream", _TTYStream())
    return build_router(**kwargs)


@pytest.mark.parametrize("stage_id, label", [
    ("load", "音声読み込み"),
    ("separate", "ボーカル分離"),
    ("recognize", "音素認識"),
    ("rms", "音量解析"),
    ("write", "書き出し"),
])
def test_stage_label_returns_shared_label(stage_id, label):
    assert stage_label(stage_id) == label


def test_stage_label_returns_unknown_stage_id_as_is():
    assert stage_label("unknown_stage") == "unknown_stage"


def test_stage_label_uses_extra_labels_for_tool_specific_stage():
    assert stage_label("notes", {"notes": "音符化"}) == "音符化"
    assert stage_label("load", {"notes": "音符化"}) == "音声読み込み"


def test_build_router_passes_stream_to_display_with_stderr_write_lock(spy_display):
    stream = _TTYStream()
    _build(stream=stream)

    kwargs = spy_display.call_args.kwargs
    assert kwargs["stream"] is stream
    assert kwargs["enabled"] is True
    assert kwargs["write_lock"] is STDERR_WRITE_LOCK


def test_build_router_quiet_disables_display(spy_display):
    _build(quiet=True)

    assert spy_display.call_args.kwargs["enabled"] is False


def test_build_router_shows_shared_stage_as_japanese_label(spy_display):
    router = _build()

    router.stage("rms")

    spy_display.return_value.stage.assert_called_with("音量解析")


def test_build_router_shows_extra_stage_label(spy_display):
    router = _build(extra_labels={"events": "口形イベント確定"})

    router.stage("events")

    spy_display.return_value.stage.assert_called_with("口形イベント確定")


def test_build_router_machine_mode_forwards_progress_events_to_emitter():
    emitter = _FakeEmitter()
    router = _build(machine=True, emitter=emitter)

    router.stage("load", done=1, total=2, note="n", elapsed=0.5)

    assert emitter.calls == [
        {"stage": "load", "done": 1, "total": 2, "note": "n", "elapsed": 0.5}]


def test_front_stage_labels_are_not_mutated_by_extra_labels():
    before = dict(FRONT_STAGE_LABELS)
    stage_label("notes", {"notes": "音符化"})
    _build(extra_labels={"notes": "音符化"})
    assert FRONT_STAGE_LABELS == before
