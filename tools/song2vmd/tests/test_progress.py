"""song2vmd 進捗表示のテスト。

進捗表示は、非機械モードでは標準エラーが端末(TTY)かつ --quiet 未指定のときだけ人間向け
1行を出し、機械モードでは TTY 判定に関係なく progress イベントを標準出力へ送出する。ProgressReporter
はこの分岐を1箇所にまとめる。実際のパイプライン処理・外部呼び出しは対象外(呼び出しの検証に限る)。
"""

import pytest

from song2vmd import progress


class _FakeEmitter:
    def __init__(self):
        self.calls = []

    def progress(self, **fields):
        self.calls.append(fields)


class _FakeStream:
    def __init__(self, *, isatty):
        self._isatty = isatty
        self.written = []

    def isatty(self):
        return self._isatty

    def write(self, text):
        self.written.append(text)


class _NoTouchEmitter:
    """属性アクセスがあれば即座に失敗するセンチネル(非機械モードでemitterに一切触れないことの検証用)。"""

    def __getattr__(self, name):
        raise AssertionError(f"non-machine mode must not touch emitter (accessed {name!r})")


# --- 機械モード: TTY×quietのどの組み合わせでも emitter へ送出し、streamには書かない ----------


@pytest.mark.parametrize("isatty", [True, False])
@pytest.mark.parametrize("quiet", [True, False])
def test_machine_mode_always_emits_to_emitter_and_never_writes_to_stream(isatty, quiet):
    emitter = _FakeEmitter()
    stream = _FakeStream(isatty=isatty)
    reporter = progress.ProgressReporter(machine=True, quiet=quiet, emitter=emitter, stream=stream)
    reporter.stage("load", done=0, total=None, note="", elapsed=0.5)
    assert emitter.calls == [{"stage": "load", "done": 0, "total": None, "note": "", "elapsed": 0.5}]
    assert stream.written == []


# --- 非機械モード: TTYかつ--quiet未指定のときだけ人間向け表示、emitterは一切参照しない --------


@pytest.mark.parametrize("isatty,quiet,expect_written", [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
])
def test_non_machine_mode_writes_only_when_tty_and_not_quiet(isatty, quiet, expect_written):
    # emitterに一切触れないことを、参照した瞬間に失敗するセンチネルで直接検証する
    # (「呼ばれない」だけでなく「一切参照しない」ことの保証)。
    stream = _FakeStream(isatty=isatty)
    reporter = progress.ProgressReporter(machine=False, quiet=quiet, emitter=_NoTouchEmitter(), stream=stream)
    reporter.stage("recognize", done=0, total=None, note="", elapsed=1.0)
    if expect_written:
        assert len(stream.written) == 1
        line = stream.written[0]
        assert "recognize" in line
        # emitterのrepr等が人間向け表示に紛れ込んでいないこと(_NoTouchEmitterに触れていれば
        # __getattr__経由の呼び出しで検出できるが、repr()はそれを経由しないため別途確認する)。
        assert "_NoTouchEmitter" not in line
        assert "object at 0x" not in line
    else:
        assert stream.written == []


def test_non_machine_mode_works_with_emitter_none():
    # 非機械モードではemitterを参照しないため、Noneを渡しても例外にならない。
    stream = _FakeStream(isatty=True)
    reporter = progress.ProgressReporter(machine=False, quiet=False, emitter=None, stream=stream)
    reporter.stage("write", done=0, total=None, note="", elapsed=1.0)
    assert len(stream.written) == 1


# --- 段id・複数回呼び出し -----------------------------------------------------


def test_multiple_stage_calls_are_all_forwarded_in_order_with_full_fields():
    emitter = _FakeEmitter()
    reporter = progress.ProgressReporter(
        machine=True, quiet=False, emitter=emitter, stream=_FakeStream(isatty=False))
    stages = ("load", "separate", "recognize", "rms", "events", "generate", "write")
    for i, stage in enumerate(stages):
        reporter.stage(stage, done=i, total=len(stages), note=f"n{i}", elapsed=float(i))
    assert emitter.calls == [
        {"stage": s, "done": i, "total": len(stages), "note": f"n{i}", "elapsed": float(i)}
        for i, s in enumerate(stages)
    ]


def test_stage_defaults_done_zero_total_none_note_empty_elapsed_zero():
    emitter = _FakeEmitter()
    reporter = progress.ProgressReporter(
        machine=True, quiet=False, emitter=emitter, stream=_FakeStream(isatty=False))
    reporter.stage("load")
    assert emitter.calls == [{"stage": "load", "done": 0, "total": None, "note": "", "elapsed": 0.0}]
