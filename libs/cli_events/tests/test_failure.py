"""cli_events の失敗報告ヘルパの単体テスト。

構造化出力モードの error イベントと人間向けのエラー行の切り替え、標準出力へ書けない場合の後退、
返り値の固定を検証する。決定論的に実行し、外部依存・ネットワークを使わない。
"""
import io
import json
import sys

import pytest

from cli_events import EventEmitter

PENDING = "impl pending: cli_events の失敗報告ヘルパ emit_failure が未実装"


class _RaisingStream:
    """write が常に指定の例外を送出するバイナリストリーム(標準出力が使えない状況)。"""

    def __init__(self, exc):
        self._exc = exc

    def write(self, _data):
        raise self._exc


class _TextSink:
    """エラー行の出力先として注入するテキストストリーム。"""

    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s
        return len(s)


class _RaisingTextSink:
    """エラー行の書き込み自体が失敗する出力先。"""

    def __init__(self, exc):
        self._exc = exc

    def write(self, _s):
        raise self._exc


# エラー行の書き込み失敗は例外の種類を問わず握り潰す。特定の型だけを捕まえる実装で通らないよう、
# 異なる種類で同じ契約を固定する。
SINK_EXCEPTIONS = [OSError("stderr closed"), ValueError("closed file"), RuntimeError("壊れた出力先")]


@pytest.mark.xfail(reason=PENDING)
def test_emits_error_event_when_emitter_usable():
    from cli_events import emit_failure

    buf = io.BytesIO()
    sink = _TextSink()
    rc = emit_failure(EventEmitter(buf), code="not_vmd", message="読めない", exit_code=1,
                      field="input", path="in.vmd", stderr=sink)
    assert rc == 1
    assert sink.text == ""  # イベントを出せたので人間向けのエラー行は出さない
    ev = json.loads(buf.getvalue().decode("utf-8"))
    assert ev == {
        "type": "error",
        "code": "not_vmd",
        "exit_code": 1,
        "field": "input",
        "path": "in.vmd",
        "message": "読めない",
    }


@pytest.mark.xfail(reason=PENDING)
def test_extra_keys_ride_on_error_event_only():
    from cli_events import emit_failure

    buf = io.BytesIO()
    sink = _TextSink()
    rc = emit_failure(EventEmitter(buf), code="stage_failed", message="分離に失敗", exit_code=4,
                      field="input", path="in.wav", stderr=sink, stage="separate")
    assert rc == 4
    # 追加キーはイベントへ載り、基底のイベントのキーはどれも欠けない。
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


@pytest.mark.xfail(reason=PENDING)
def test_error_line_when_emitter_is_none():
    from cli_events import emit_failure

    sink = _TextSink()
    rc = emit_failure(None, code="internal_error", message="RuntimeError: 想定外", exit_code=1,
                      stderr=sink)
    assert rc == 1
    assert sink.text == "error: RuntimeError: 想定外\n"


@pytest.mark.xfail(reason=PENDING)
def test_error_line_carries_message_only_not_extra_keys():
    from cli_events import emit_failure

    # 追加キー・field・path はイベント専用。人間向けのエラー行は本文だけを載せる。
    sink = _TextSink()
    rc = emit_failure(None, code="stage_failed", message="分離に失敗", exit_code=4,
                      field="input", path="in.wav", stderr=sink, stage="separate")
    assert rc == 4
    assert sink.text == "error: 分離に失敗\n"


@pytest.mark.xfail(reason=PENDING)
def test_error_line_when_emitter_already_terminated():
    from cli_events import emit_failure

    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.result(mode="convert")
    before = buf.getvalue()
    sink = _TextSink()
    rc = emit_failure(em, code="cancelled", message="中断された", exit_code=130, stderr=sink)
    assert rc == 130  # 終端済みでも今回受け取った終了コードを返す(二次的な失敗を握り潰さない)
    assert buf.getvalue() == before  # 既に出した終端イベントは書き換えない
    assert sink.text == "error: 中断された\n"


@pytest.mark.xfail(reason=PENDING)
@pytest.mark.parametrize("exc", [OSError("broken pipe"), ValueError("closed file")])
def test_falls_back_to_error_line_when_event_write_fails(exc):
    from cli_events import emit_failure

    # 送出失敗でも受け取った終了コードをそのまま返す(特定の値へ正規化しない)。追加キーは
    # 人間向けのエラー行へ載せない。
    sink = _TextSink()
    rc = emit_failure(EventEmitter(_RaisingStream(exc)), code="output_exists",
                      message="出力先に既存ファイルがあります", exit_code=2, field="--output",
                      stderr=sink, stage="write")
    assert rc == 2
    assert sink.text == "error: 出力先に既存ファイルがあります\n"


@pytest.mark.xfail(reason=PENDING)
def test_other_event_write_exceptions_are_not_swallowed():
    from cli_events import emit_failure

    # 人間向けのエラー行へ後退するのは、書き込み先が使えないことを示す OSError と ValueError だけ。
    # それ以外の例外(イベントの中身の不備など、標準出力が使えることと両立する失敗)は握り潰さず
    # 呼び出し側へ渡す。握り潰すと、終端イベントを出せる状況なのに黙って途切れさせてしまう。
    sink = _TextSink()
    with pytest.raises(RuntimeError):
        emit_failure(EventEmitter(_RaisingStream(RuntimeError("想定外"))), code="internal_error",
                     message="漏らす", exit_code=1, stderr=sink)
    assert sink.text == ""


@pytest.mark.xfail(reason=PENDING)
@pytest.mark.parametrize("sink_exc", SINK_EXCEPTIONS)
def test_swallows_failure_of_the_error_line_after_event_write_failed(sink_exc):
    from cli_events import emit_failure

    # イベント送出も人間向けのエラー行も書けない場合。例外を漏らさず終了コードだけを返す。
    rc = emit_failure(EventEmitter(_RaisingStream(OSError("broken pipe"))), code="internal_error",
                      message="どこにも書けない", exit_code=1, stderr=_RaisingTextSink(sink_exc))
    assert rc == 1


@pytest.mark.xfail(reason=PENDING)
@pytest.mark.parametrize("sink_exc", SINK_EXCEPTIONS)
def test_swallows_failure_of_the_error_line_without_emitter(sink_exc):
    from cli_events import emit_failure

    # 直接エラー行へ進む分岐(エミッタ無し)でも、エラー行の書き込み失敗を握り潰す。
    rc = emit_failure(None, code="internal_error", message="どこにも書けない", exit_code=1,
                      stderr=_RaisingTextSink(sink_exc))
    assert rc == 1


@pytest.mark.xfail(reason=PENDING)
def test_stderr_default_resolves_at_call_time(monkeypatch):
    from cli_events import emit_failure

    # 既定値 None は呼び出しの時点で標準エラーへ解決する(定義時に束縛しない)。
    sink = _TextSink()
    monkeypatch.setattr(sys, "stderr", sink)
    rc = emit_failure(None, code="bad_argument", message="引数が不正", exit_code=2)
    assert rc == 2
    assert sink.text == "error: 引数が不正\n"
