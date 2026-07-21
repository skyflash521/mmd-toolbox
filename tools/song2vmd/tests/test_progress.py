"""song2vmd 進捗表示のテスト。

ProgressReporter は machine/quiet/stream から表示先を1箇所で出し分ける:
機械モードでは TTY 判定に関係なく progress イベントを emitter へ送出し(--quiet は影響しない)、
非機械モードでは標準エラーが端末(TTY)かつ --quiet 未指定のときだけ、stage id を日本語の
工程名へ変換した1本のライブ行で進行を表示する(段が替わっても改行しない)。完了イベント(stage)と
再描画(ハートビートスレッド)を分離し、stage が来ない待ち時間でも経過時間(now() - 段開始)を
進め続けることで、重い処理(モデルの初回ダウンロード等)で表示が停滞して見えるのを防ぐ。

検証は stream/clock/有効化/再描画間隔を注入で差し替えて確定的に行い、デーモンスレッドの実時間
スケジューリングには依存しない(描画経路 _draw を直接駆動するか、小さい interval で発火させ待つ)。
"""

import io
import threading
import time

import pytest

from song2vmd import progress

ProgressReporter = progress.ProgressReporter


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


class _NonTTYStream(io.StringIO):
    def isatty(self):
        return False


class _RaisingStream(io.StringIO):
    """write が常に例外を送出する端末ストリーム(cp932 端末で全角がエンコード不能なケースの代理)。"""

    def __init__(self, exc=None):
        super().__init__()
        self._exc = exc if exc is not None else OSError("write failed")

    def isatty(self):
        return True

    def write(self, _s):
        raise self._exc


class _FakeEmitter:
    def __init__(self):
        self.calls = []

    def progress(self, **fields):
        self.calls.append(fields)


class _NoTouchEmitter:
    """属性アクセスがあれば即座に失敗するセンチネル(非機械モードでemitterに一切触れないことの検証用)。"""

    def __getattr__(self, name):
        raise AssertionError(f"non-machine mode must not touch emitter (accessed {name!r})")


def _wait_until(pred, timeout=2.0):
    # 条件成立を寛容なタイムアウトで待つ。成功は確定的(成立し次第すぐ返る)、実装が誤りなら時間切れで fail。
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


# --- 機械モード: TTY×quietのどの組み合わせでも emitter へ送出し、streamには書かない ----------


@pytest.mark.parametrize("isatty", [True, False])
@pytest.mark.parametrize("quiet", [True, False])
def test_machine_mode_always_emits_to_emitter_and_never_writes_to_stream(isatty, quiet):
    emitter = _FakeEmitter()
    stream = _TTYStream() if isatty else _NonTTYStream()
    reporter = ProgressReporter(machine=True, quiet=quiet, emitter=emitter, stream=stream)
    reporter.stage("load", done=0, total=None, note="", elapsed=0.5)
    assert emitter.calls == [{"stage": "load", "done": 0, "total": None, "note": "", "elapsed": 0.5}]
    assert stream.getvalue() == ""


def test_multiple_stage_calls_are_all_forwarded_in_order_with_full_fields():
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_NonTTYStream())
    stages = ("load", "separate", "recognize", "rms", "events", "generate", "write")
    for i, stage in enumerate(stages):
        reporter.stage(stage, done=i, total=len(stages), note=f"n{i}", elapsed=float(i))
    assert emitter.calls == [
        {"stage": s, "done": i, "total": len(stages), "note": f"n{i}", "elapsed": float(i)}
        for i, s in enumerate(stages)
    ]


def test_stage_defaults_done_zero_total_none_note_empty_elapsed_zero():
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_NonTTYStream())
    reporter.stage("load")
    assert emitter.calls == [{"stage": "load", "done": 0, "total": None, "note": "", "elapsed": 0.0}]


def test_machine_mode_never_starts_heartbeat_thread_even_with_tty_stream():
    # 機械モードでは stream が TTY でも人間向けライブ表示機構(ハートビートスレッド)を
    # 一切起動しない。決定的な待機駆動で十分な時間が経過しても stream に何も書かれないことも
    # あわせて固定する(タイミング依存で偶然通る実装を弾く)。
    emitter = _FakeEmitter()
    stream = _TTYStream()
    reporter = ProgressReporter(
        machine=True, quiet=False, emitter=emitter, stream=stream, now=lambda: 0.0, interval=0.01)
    reporter.stage("load")
    assert reporter.enabled is False
    assert reporter._thread is None
    time.sleep(0.05)  # ハートビート間隔の複数倍待っても再描画されないことを確認する
    assert stream.getvalue() == ""


def test_machine_mode_never_constructs_heartbeat_thread(monkeypatch):
    # 機械モードはライブ表示の描画方式(改行型かハートビート方式か)に依存せず、常に
    # emitter.progress の送出だけで完結すべき不変条件なので、xfail を付けず通常のテストとして
    # 固定する。スレッドを起動して即座に終了させ _thread を None へ戻す実装でも「start されない」だけの
    # 検証は通ってしまう。threading.Thread の生成(__init__)・start それぞれの呼び出し回数を
    # 個別に記録し、両方とも一度も呼ばれないことを直接確認する。
    constructed = []
    started = []

    class _SpyThread:
        def __init__(self, *args, **kwargs):
            constructed.append(True)

        def start(self):
            started.append(True)

    monkeypatch.setattr(threading, "Thread", _SpyThread)
    emitter = _FakeEmitter()
    reporter = ProgressReporter(machine=True, quiet=False, emitter=emitter, stream=_TTYStream())
    reporter.stage("load")
    reporter.stage("separate", done=1, total=2)
    assert constructed == []
    assert started == []


# --- 非機械モード: 有効条件(TTY×not quiet)のときだけライブ表示、emitterは一切参照しない -----


@pytest.mark.parametrize("isatty,quiet,expect_enabled", [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
])
def test_enabled_only_when_tty_and_not_quiet(isatty, quiet, expect_enabled):
    stream = _TTYStream() if isatty else _NonTTYStream()
    reporter = ProgressReporter(machine=False, quiet=quiet, emitter=_NoTouchEmitter(), stream=stream)
    assert reporter.enabled is expect_enabled


def test_disabled_reporter_stage_is_noop_no_thread_no_output():
    # 非TTY/quiet 相当(enabled=False)は stage を呼んでもスレッドを起こさず何も書かない。
    stream = _NonTTYStream()
    reporter = ProgressReporter(machine=False, quiet=False, emitter=_NoTouchEmitter(), stream=stream)
    reporter.stage("load")
    reporter.stage("separate", done=1, total=2)
    reporter.close()
    reporter.summary("完了 out.vmd")
    assert stream.getvalue() == ""
    assert reporter._thread is None


def test_non_machine_mode_never_touches_emitter():
    # emitterに一切触れないことを、参照した瞬間に失敗するセンチネルで直接検証する。
    stream = _TTYStream()
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=_NoTouchEmitter(), stream=stream, interval=3600.0)
    reporter.stage("recognize")
    reporter._draw()
    reporter.close()


def test_non_machine_mode_works_with_emitter_none():
    # 非機械モードではemitterを参照しないため、Noneを渡しても例外にならない。
    stream = _TTYStream()
    reporter = ProgressReporter(machine=False, quiet=False, emitter=None, stream=stream, interval=3600.0)
    reporter.stage("write")
    reporter._draw()
    assert "[書き出し]" in stream.getvalue()
    reporter.close()


# --- 表示書式: stage id を日本語の工程名へ変換して表示する -----------------------------


@pytest.mark.parametrize("stage_id,label", [
    ("load", "音声読み込み"),
    ("separate", "ボーカル分離"),
    ("recognize", "音素認識"),
    ("rms", "音量解析"),
    ("events", "口形イベント確定"),
    ("generate", "モーフ生成"),
    ("write", "書き出し"),
])
def test_stage_id_maps_to_japanese_label(stage_id, label):
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage(stage_id)
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == f"[{label}] 経過 0:00"
    reporter.close()


def test_total_none_hides_count_shows_elapsed_only():
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("load", done=0, total=None)
    clock[0] = 3.0
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[音声読み込み] 経過 0:03"
    reporter.close()


def test_total_set_shows_done_over_total():
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("separate", done=0, total=3)
    reporter.stage("separate", done=2, total=3)  # 同一段内の進行更新(段は変わらない)
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[ボーカル分離] 2/3 経過 0:00"
    reporter.close()


def test_stages_share_one_line_without_newline():
    # 段を切り替えても改行せず、同じライブ行を上書きする(終わった段の行を残さない)。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("load")
    reporter._draw()
    reporter.stage("separate", done=3, total=10)
    reporter._draw()
    out = stream.getvalue()
    assert "\n" not in out  # 段間で改行しない=1行に集約
    assert out.count("\r") == 2  # 各 draw は同じ行を CR で上書き
    assert out.rsplit("\r", 1)[1] == "[ボーカル分離] 3/10 経過 0:00"  # 後段へ切り替わる
    reporter.close()


def test_shorter_line_pads_over_previous_longer_line():
    # 長い行の後に短い行を描くとき、残像(前の行の余分な文字)を空白で埋めて消す。
    # 期待パディング数は本番の _display_width を使わず、表示幅を手計算した固定値で検証する
    # (全角を1桁と誤って数える実装があっても、描画と期待値が同じ誤りを共有して見逃さないため)。
    # "ボーカル分離"・"音声読み込み" はどちらも全角6文字(表示幅12)で工程名の幅を揃えてあるので、
    # 差分は total 付き書式が追加する "1/1 "(表示幅4)の分だけになる。
    # 長い行 "[ボーカル分離] 1/1 経過 0:00" の表示幅: [(1)+ボーカル分離(12)+](1)+ (1)+1/1(3)+ (1)+経過(4)+ (1)+0:00(4) = 28
    # 短い行 "[音声読み込み] 経過 0:00" の表示幅: [(1)+音声読み込み(12)+](1)+ (1)+経過(4)+ (1)+0:00(4) = 24
    # pad = 28 - 24 = 4
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("separate", done=1, total=1)
    reporter._draw()
    reporter.stage("load")  # 短い行の段へ切り替え
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[音声読み込み] 経過 0:00" + " " * 4
    reporter.close()


def test_heartbeat_advances_elapsed_without_new_stage_call():
    # 停滞回避の核: stage が来なくてもハートビートが now() - 段開始 を再評価して再描画し、
    # 経過時間が進む(モデルの初回ダウンロード等、次の段報告までに長い待ちがあっても固まらない)。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=0.01)
    reporter.stage("separate", done=1, total=5)
    clock[0] = 5.0
    _wait_until(lambda: "経過 0:05" in stream.getvalue())
    clock[0] = 60.0
    _wait_until(lambda: "経過 1:00" in stream.getvalue())
    reporter.close()
    out = stream.getvalue()
    assert "[ボーカル分離] 1/5 経過 0:05" in out
    assert "[ボーカル分離] 1/5 経過 1:00" in out


def test_stage_change_resets_elapsed_and_stops_old_heartbeat():
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=0.01)
    reporter.stage("load")
    first = reporter._thread
    assert first is not None and first.is_alive()
    clock[0] = 50.0
    reporter.stage("separate")  # 新しい段(経過はリセット)
    assert not first.is_alive()  # 旧ハートビートは停止・join された
    assert reporter._thread is not None and reporter._thread is not first
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[ボーカル分離] 経過 0:00"
    reporter.close()


def test_same_stage_repeat_call_does_not_reset_elapsed():
    # 同じ stage id での再呼び出し(長尺分割のチャンク進行など)は経過をリセットしない。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("separate", done=0, total=3)
    clock[0] = 10.0
    reporter.stage("separate", done=1, total=3)  # 同一段の進行更新
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[ボーカル分離] 1/3 経過 0:10"
    reporter.close()


def test_note_appears_in_drawn_line():
    # note(現在対象などの補足。測定可能な外部処理の進捗を含む)は行末へ併記する。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("separate", done=1, total=2, note="モデルをダウンロード中")
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[ボーカル分離] 1/2 経過 0:00 モデルをダウンロード中"
    reporter.close()


def test_note_updates_on_same_stage_repeat_call():
    # 同一段の再呼び出しでnoteが更新される(ダウンロード進捗のバイト数更新等)。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("separate", done=0, total=1, note="1MB/10MB")
    reporter.stage("separate", done=0, total=1, note="5MB/10MB")
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[ボーカル分離] 0/1 経過 0:00 5MB/10MB"
    reporter.close()


def test_note_resets_on_stage_change():
    # 段を切り替えると note は据え置かれずリセットされる。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("separate", done=1, total=2, note="モデルをダウンロード中")
    reporter.stage("recognize")  # note を渡さない新しい段
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[音素認識] 経過 0:00"
    reporter.close()


def test_empty_note_appends_no_extra_space():
    # note が空文字("")のときは行末に余分な空白を付けない。
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("load", note="")
    reporter._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[音声読み込み] 経過 0:00"
    reporter.close()


def test_close_clears_line_leaves_nothing():
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("recognize", done=2, total=4)
    reporter._draw()
    drawn = stream.getvalue()
    width = progress._display_width("[音素認識] 2/4 経過 0:00")
    reporter.close()
    assert stream.getvalue() == drawn + "\r" + " " * width + "\r"
    assert "\n" not in stream.getvalue()
    assert reporter._thread is None


def test_close_stops_live_heartbeat_thread():
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=0.01)
    reporter.stage("recognize")
    clock[0] = 3.0
    _wait_until(lambda: "経過 0:03" in stream.getvalue())
    live = reporter._thread
    assert live is not None and live.is_alive()
    reporter.close()
    assert not live.is_alive()
    assert reporter._thread is None


def test_close_is_noop_when_nothing_active():
    stream = _TTYStream()
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: 0.0, interval=3600.0)
    reporter.close()
    reporter.close()
    assert stream.getvalue() == ""
    assert reporter._thread is None


def test_summary_writes_line_only_when_enabled():
    stream = _TTYStream()
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: 0.0, interval=3600.0)
    reporter.summary("完了 out.vmd")
    assert stream.getvalue() == "完了 out.vmd\n"

    stream2 = _NonTTYStream()
    reporter2 = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream2, now=lambda: 0.0)
    reporter2.summary("完了 out.vmd")
    assert stream2.getvalue() == ""


# 書き込み失敗保護が捕捉すべき例外種別。端末のエンコード不能(全角ラベルが端末コーデックで表せない等)は
# UnicodeError 系。OSError だけ捕捉する実装を弾くため、3種すべてで保護が効くことを固定する。
_WRITE_FAILURES = [OSError("io failed"), ValueError("bad stream"), UnicodeError("encode failed")]


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_draw_write_failure_disables_without_raising_and_close_stops_thread(exc):
    stream = _RaisingStream(exc)
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("generate")
    live = reporter._thread
    assert live is not None and live.is_alive()
    reporter._draw()  # 内部 write が例外送出 → 捕捉して enabled=False(例外は外へ出ない)
    assert reporter.enabled is False
    reporter.close()  # enabled が False でもハートビートを停止・join する
    assert not live.is_alive()
    assert reporter._thread is None
    # 以後の呼び出しも no-op で例外を出さない。
    reporter.stage("write")
    reporter._draw()
    reporter.summary("完了 out.vmd")


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_summary_write_failure_swallowed(exc):
    stream = _RaisingStream(exc)
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: 0.0, interval=3600.0)
    reporter.summary("完了 out.vmd")
    assert reporter.enabled is False


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_close_write_failure_swallowed(exc):
    stream = _TTYStream()
    clock = [0.0]
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: clock[0], interval=3600.0)
    reporter.stage("generate", done=2, total=4)
    reporter._draw()  # 成功描画で _last_width を立てる
    live = reporter._thread
    reporter._stream = _RaisingStream(exc)  # close の行消去 write を失敗させる
    reporter.close()
    assert reporter.enabled is False
    assert not live.is_alive()
    assert reporter._thread is None


# vocal_analysis.quiet.suppress_native_stderr と同じ STDERR_WRITE_LOCK を取ってから書くことを
# 固定する(取らずに書くと、fd 差し替え中の別スレッド書き込みが実コンソール環境で失敗しうる)。
# 検証は書き込み中に別スレッド(テスト本体)から非ブロッキングでロック取得を試み、取れない
# (=書き込み側が保持している)ことで行う。


class _BlockingStream:
    """write 呼び出し中、release イベントが立つまで保持し続ける疑似ストリーム。"""

    def __init__(self):
        self.write_started = threading.Event()
        self.release = threading.Event()

    def write(self, s):
        self.write_started.set()
        self.release.wait(2.0)

    def flush(self):
        pass

    def isatty(self):
        return True


def test_draw_holds_stderr_write_lock_during_write():
    from vocal_analysis.quiet import STDERR_WRITE_LOCK

    stream = _BlockingStream()
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: 0.0, interval=3600.0)
    reporter.stage("load")

    t = threading.Thread(target=reporter._draw)
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert STDERR_WRITE_LOCK.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)
        reporter._stop_heartbeat()  # stage() が起こしたハートビートスレッドを回収する


def test_close_holds_stderr_write_lock_during_write():
    from vocal_analysis.quiet import STDERR_WRITE_LOCK

    stream = _BlockingStream()
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=_TTYStream(), now=lambda: 0.0, interval=3600.0)
    reporter.stage("load")
    reporter._draw()  # _last_width を立てて close の行消去 write を発生させる
    reporter._stop_heartbeat()
    reporter._stream = stream

    t = threading.Thread(target=reporter.close)
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert STDERR_WRITE_LOCK.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)


def test_summary_holds_stderr_write_lock_during_write():
    from vocal_analysis.quiet import STDERR_WRITE_LOCK

    stream = _BlockingStream()
    reporter = ProgressReporter(
        machine=False, quiet=False, emitter=None, stream=stream, now=lambda: 0.0, interval=3600.0)

    t = threading.Thread(target=reporter.summary, args=("完了 out.vmd",))
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert STDERR_WRITE_LOCK.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)
