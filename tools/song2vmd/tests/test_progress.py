"""song2vmd 進捗表示ファサードの配線テスト。

ProgressReporter は machine/quiet/stream から表示先を1箇所で出し分ける薄いファサードで、非機械モードの
実体は共有 cli_progress.progress.ProgressReporter に委譲する。実描画・TTY判定・ハートビート・書き込み
失敗保護の検証は libs/cli_progress/tests/test_progress.py が担うため、本テストではファサードの配線責務
——機械/非機械の分岐、stage_id から日本語ラベルへの変換、共有表示器への委譲——だけを検証する。

検証は共有 ProgressReporter クラス(`progress._display_module.ProgressReporter`)をモック/スパイに
差し替えて、構築時のキーワード引数・stage/close/summary の呼び出し回数と引数を確認する形で行う。
"""

import io
from unittest.mock import MagicMock, call

import pytest

from song2vmd import progress
from vocal_analysis.quiet import STDERR_WRITE_LOCK

ProgressReporter = progress.ProgressReporter


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
    """属性アクセスがあれば即座に失敗するセンチネル(非機械モードでemitterに一切触れないことの検証用)。"""

    def __getattr__(self, name):
        raise AssertionError(f"non-machine mode must not touch emitter (accessed {name!r})")


@pytest.fixture
def spy_display(monkeypatch):
    """共有 ProgressReporter クラス自体をスパイへ差し替えて返す(構築呼び出しの検証用。生成
    インスタンスは各テストが spy_cls.return_value で取り出す)。"""
    spy_cls = MagicMock(name="SharedProgressReporterClass")
    monkeypatch.setattr(progress._display_module, "ProgressReporter", spy_cls)
    return spy_cls


# --- 機械モード: emitter へ送出し、共有表示器は一切生成しない --------------------------


def test_machine_mode_forwards_stage_to_emitter_and_builds_no_display(spy_display):
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_NonTTYStream())
    assert reporter._display is None
    spy_display.assert_not_called()
    reporter.stage("load", done=1, total=2, note="n", elapsed=0.5)
    assert emitter.calls == [{"stage": "load", "done": 1, "total": 2, "note": "n", "elapsed": 0.5}]


def test_machine_mode_stage_defaults_done_zero_total_none_note_empty_elapsed_zero(spy_display):
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_NonTTYStream())
    reporter.stage("load")
    assert emitter.calls == [{"stage": "load", "done": 0, "total": None, "note": "", "elapsed": 0.0}]


def test_machine_mode_multiple_stage_calls_are_all_forwarded_in_order(spy_display):
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_NonTTYStream())
    stages = ("load", "separate", "recognize", "rms", "events", "generate", "write")
    for i, stage in enumerate(stages):
        reporter.stage(stage, done=i, total=len(stages), note=f"n{i}", elapsed=float(i))
    assert emitter.calls == [
        {"stage": s, "done": i, "total": len(stages), "note": f"n{i}", "elapsed": float(i)}
        for i, s in enumerate(stages)
    ]


def test_machine_mode_close_and_summary_do_not_touch_display_or_emitter(spy_display):
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_NonTTYStream())
    reporter.close()
    reporter.summary("完了 out.vmd")
    spy_display.return_value.close.assert_not_called()
    spy_display.return_value.summary.assert_not_called()
    assert emitter.calls == []


# --- 非機械モード: 共有表示器を構築して委譲する。emitter には一切触れない -----------------


@pytest.mark.parametrize("isatty,quiet,expect_enabled", [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
])
def test_non_machine_mode_constructs_shared_reporter_with_all_kwargs(
        spy_display, isatty, quiet, expect_enabled):
    stream = _TTYStream() if isatty else _NonTTYStream()
    clock = lambda: 0.0  # noqa: E731 - テスト用の単純な注入クロック
    ProgressReporter(
        machine=False, quiet=quiet, emitter=_NoTouchEmitter(), stream=stream, now=clock, interval=0.2)
    spy_display.assert_called_once_with(
        stream=stream, enabled=expect_enabled, now=clock, interval=0.2, write_lock=STDERR_WRITE_LOCK)


def test_non_machine_mode_works_with_emitter_none(spy_display):
    ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    spy_display.assert_called_once()


def test_no_enabled_property_on_facade(spy_display):
    # ファサードは enabled 属性を公開しない(実体を持つ共有表示器の enabled と紛れさせないため)。
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    assert not hasattr(reporter, "enabled")


# --- 表示書式: stage id を日本語の工程名へ変換して共有表示器へ渡す -----------------------


@pytest.mark.parametrize("stage_id,label", [
    ("load", "音声読み込み"),
    ("separate", "ボーカル分離"),
    ("recognize", "音素認識"),
    ("rms", "音量解析"),
    ("events", "口形イベント確定"),
    ("generate", "モーフ生成"),
    ("write", "書き出し"),
])
def test_stage_id_maps_to_japanese_label(spy_display, stage_id, label):
    display = spy_display.return_value
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    reporter.stage(stage_id)
    display.stage.assert_called_once_with(label)


def test_unknown_stage_id_used_as_label_as_is(spy_display):
    display = spy_display.return_value
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    reporter.stage("未知の工程")
    display.stage.assert_called_once_with("未知の工程")


def test_same_stage_id_repeat_call_updates_only_new_stage_id_starts_new_stage(spy_display):
    # 同一 stage_id の連続呼び出しは display.stage() を再呼び出しせず display.update() のみ呼ぶ。
    # stage_id が変わったら display.stage() を呼ぶ(_STAGE_LABELS 変換込み)。
    display = spy_display.return_value
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    reporter.stage("separate", done=0, total=3)
    reporter.stage("separate", done=1, total=3, note="n")
    display.stage.assert_called_once_with("ボーカル分離")
    assert display.update.call_args_list == [call(0, 3, ""), call(1, 3, "n")]
    reporter.stage("recognize")
    assert display.stage.call_args_list == [call("ボーカル分離"), call("音素認識")]


def test_stage_after_close_restarts_display_stage_even_with_same_stage_id(spy_display):
    # stage("recognize") → close() → 同じ stage("recognize") の順で呼んだとき、2回目の
    # stage("recognize") が(継続とみなされて update() のみになるのではなく)display.stage(...) を
    # 呼び直す(close() が _current_stage をリセットするため)。
    display = spy_display.return_value
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    reporter.stage("recognize")
    reporter.close()
    reporter.stage("recognize")
    assert display.stage.call_args_list == [call("音素認識"), call("音素認識")]


# --- close/summary: 共有表示器への委譲(非機械モードのみ) ------------------------------


def test_close_delegates_to_display_close_exactly_once(spy_display):
    display = spy_display.return_value
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    reporter.close()
    display.close.assert_called_once_with()


def test_summary_delegates_message_to_display_summary_exactly_once(spy_display):
    display = spy_display.return_value
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=_TTYStream())
    reporter.summary("完了 out.vmd")
    display.summary.assert_called_once_with("完了 out.vmd")
