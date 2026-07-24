"""ProgressReporter(進捗のライブ表示)の単体テスト。

ProgressReporter は重い処理の進行を stderr へ1本のライブ行で表示する(TTY時のみ)。
全段を通して同じ行を上書きし(段間で改行しない)、終わった段の行を画面に残さない。close で行を消し、
summary で完了行を1行残す。完了イベント(update)と再描画(ハートビートスレッド)を分離し、update が
来ない待ち時間でも経過時間(now() - 段開始)を進め続けることで、重い処理で表示が停滞して見えるのを防ぐ。

update に note スロットを持たせ、描画行へ note を併記する(note の用途は呼び出し側が定める)。stderr
書き込みが失敗(エンコード不能等)しても例外を外へ漏らさず enabled=False に落ち、以後 no-op。書き込み
失敗で表示が無効化された後も、close はハートビートスレッドを必ず停止・join する。

write_lock を注入すると、_draw/close/summary の各書き込みをそのロックを保持して行う(呼び出し側が
同じ標準エラーへ別の仕組みでOSレベルのfd操作を行う場合に競合を避けるため)。未指定時はロックなしで動作する。

検証は stream/clock/有効化/再描画間隔を注入で差し替えて確定的に行い、デーモンスレッドの実時間
スケジューリングには依存しない(描画経路 _draw を直接駆動するか、小さい interval で発火させ join で確定)。
"""

import io
import threading
import time

import pytest

from cli_progress import progress

ProgressReporter = progress.ProgressReporter

pytestmark = pytest.mark.xfail(reason="impl pending: libs/cli_progress 新設", strict=False)


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


class _RaisingStream(io.StringIO):
    """write が常に例外を送出する端末ストリーム(cp932 端末で全角がエンコード不能なケースの代理)。"""

    def __init__(self, exc=None):
        super().__init__()
        self._exc = exc if exc is not None else OSError("write failed")

    def isatty(self):
        return True

    def write(self, _s):
        raise self._exc


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


def _wait_until(pred, timeout=2.0):
    # 条件成立を寛容なタイムアウトで待つ。成功は確定的(成立し次第すぐ返る)、実装が誤りなら時間切れで fail。
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def test_format_line_shows_label_count_elapsed():
    assert progress._format_line("キーフレーム圧縮", 5, 88, 73.0) == "[キーフレーム圧縮] 5/88 経過 1:13"


def test_format_line_hides_count_when_total_unset():
    # total 未確定(stage 直後・最初の update 前)はカウントを出さず経過のみ。
    assert progress._format_line("キーフレーム圧縮", 0, None, 2.0) == "[キーフレーム圧縮] 経過 0:02"


def test_format_line_appends_note_when_present():
    # note があれば行末へ併記する(現在処理中の対象の注記)。note 既定("")では併記しない。
    assert (progress._format_line("キーフレーム圧縮", 60, 60, 5.0, "ボーン: センター")
            == "[キーフレーム圧縮] 60/60 経過 0:05 ボーン: センター")
    assert progress._format_line("キーフレーム圧縮", 60, 60, 5.0, "") == "[キーフレーム圧縮] 60/60 経過 0:05"


def test_display_width_counts_fullwidth_as_two():
    assert progress._display_width("キーフレーム圧縮") == 16  # 全角8文字=16
    assert progress._display_width("[X] 0/5") == 7  # 半角はそのまま


def test_disabled_is_noop_no_thread_no_output():
    # 非TTY/quiet 相当(enabled=False)は全メソッド no-op。stream へ何も書かず、スレッドも起こさない。
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=False, now=lambda: 0.0)
    r.stage("キーフレーム圧縮")
    r.update(1, 2)
    r.update(1, 2, note="ボーン: センター")
    r.close()
    r.summary("完了 out.vmd")
    assert stream.getvalue() == ""
    assert r._thread is None


def test_auto_enabled_follows_isatty():
    # enabled 省略時は stream.isatty() で自動判定し、enabled は解決済みの真偽値。
    assert ProgressReporter(_TTYStream(), now=lambda: 0.0).enabled is True
    assert ProgressReporter(io.StringIO(), now=lambda: 0.0).enabled is False  # StringIO.isatty()==False


def test_update_does_not_write_heartbeat_is_sole_writer():
    # 進捗行を書くのはハートビートだけ。interval を十分大きくしてハートビートを発火させなければ、
    # update を呼んでも何も書かれない(update が直接 stream へ書く実装を弾く)。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    clock[0] = 5.0
    r.update(3, 10)
    assert stream.getvalue() == ""  # update 自身は描かない
    r.close()


def test_stages_share_one_line_without_newline():
    # 段を切り替えても改行せず、同じライブ行を上書きする(終わった段の行を残さない)。ハートビート未発火
    # (interval 大)で _draw を手動で呼び、改行が入らないこと・行が後段へ切り替わることを決定的に固定する。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(3, 10, note="カメラ")
    r._draw()
    r.stage("キーフレーム圧縮")  # 同じラベルでも stage() 呼び出しで新しい段として経過をリセットする
    r.update(1, 5, note="ボーン: センター")
    r._draw()
    out = stream.getvalue()
    assert "\n" not in out  # 段間で改行しない=1行に集約
    assert out.count("\r") == 2  # 各 draw は同じ行を CR で上書き
    assert out.rsplit("\r", 1)[1] == "[キーフレーム圧縮] 1/5 経過 0:00 ボーン: センター"
    r.close()


def test_update_note_appears_in_drawn_line():
    # update の note が描画行に併記される(対象の表示確認)。stage() は note をリセットする。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(60, 60, note="ボーン: センター")
    r._draw()
    assert "ボーン: センター" in stream.getvalue().rsplit("\r", 1)[1]
    # 段を切り替えると note は据え置かれずリセットされる。
    r.stage("キーフレーム圧縮")
    r._draw()
    assert "ボーン: センター" not in stream.getvalue().rsplit("\r", 1)[1]
    r.close()


def test_heartbeat_advances_elapsed_without_update():
    # 停滞回避の核: update は最初の1回だけ。以後 update が来なくてもハートビートが now() - 段開始 を
    # 再評価して再描画し、経過時間が進む。重い処理で表示が固まらないことの担保。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.stage("キーフレーム圧縮")
    r.update(2, 10)  # 完了通知はこの1回だけ
    clock[0] = 5.0
    _wait_until(lambda: "経過 0:05" in stream.getvalue())  # ハートビートが経過 0:05 で再描画
    clock[0] = 60.0
    _wait_until(lambda: "経過 1:00" in stream.getvalue())  # update 無しでも経過 1:00 へ進む
    r.close()
    out = stream.getvalue()
    assert "\r[キーフレーム圧縮] 2/10 経過 0:05" in out
    assert "[キーフレーム圧縮] 2/10 経過 1:00" in out


def test_stage_uses_begin_time_and_resets_per_stage():
    # 経過は段開始(stage)時刻基準。構築時刻と stage 時刻を変え、構築時刻でなく stage 時刻からの差分が
    # 出ることを固定する。さらに段ごとに開始時刻がリセットされる。
    stream = io.StringIO()
    clock = [50.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)  # 構築時刻 50.0
    clock[0] = 100.0
    r.stage("キーフレーム圧縮")  # 段開始 100.0
    clock[0] = 108.0
    r._draw()
    clock[0] = 200.0
    r.stage("キーフレーム圧縮")  # 段開始 200.0(前段を引きずらない)
    clock[0] = 205.0
    r._draw()
    segments = stream.getvalue().split("\r")[1:]  # 先頭の空要素を除く
    assert segments[0].startswith("[キーフレーム圧縮] 経過 0:08")  # 108-100
    assert segments[1].startswith("[キーフレーム圧縮] 経過 0:05")  # 205-200
    r.close()


def test_close_clears_line_leaves_nothing():
    # close は行を空白で上書きし行頭へ戻して消す(改行しないので画面に残らない)。ハートビート未発火で
    # 出力を厳密に固定する。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(2, 4)
    r._draw()
    drawn = stream.getvalue()  # "\r[キーフレーム圧縮] 2/4 経過 0:00"
    width = progress._display_width("[キーフレーム圧縮] 2/4 経過 0:00")
    r.close()
    assert stream.getvalue() == drawn + "\r" + " " * width + "\r"  # 空白で消して行頭へ
    assert "\n" not in stream.getvalue()
    assert r._thread is None


def test_close_stops_live_heartbeat_thread():
    # close は活動中のハートビートを停止イベントで止め join する(止め損ねた残存スレッドが後段の表示を
    # 乱す穴を弾く)。捕捉した実スレッド参照が close 後に is_alive()==False になることで固定する。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.stage("キーフレーム圧縮")
    r.update(1, 2)
    clock[0] = 3.0
    _wait_until(lambda: "経過 0:03" in stream.getvalue())  # ハートビートが実際に動いている
    live = r._thread
    assert live is not None and live.is_alive()
    r.close()
    assert not live.is_alive()  # close が実スレッドを停止・join した
    assert r._thread is None


def test_close_is_noop_when_nothing_active():
    # 段を開始していないときの close は no-op(無出力・スレッドなし)。多重 close でも害がない。
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.close()
    r.close()
    assert stream.getvalue() == ""
    assert r._thread is None


def test_stage_while_active_stops_old_heartbeat():
    # 段を切り替えると旧ハートビートを止めてから新段を起こす。旧スレッドが残って後から描画し混線する
    # 漏れを弾く(単一描画所有者)。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.stage("キーフレーム圧縮")
    first = r._thread
    assert first is not None and first.is_alive()
    r.stage("キーフレーム圧縮")
    assert not first.is_alive()  # 旧ハートビートは停止・join された
    assert r._thread is not None and r._thread is not first and r._thread.is_alive()
    r.close()


def test_summary_writes_line_only_when_enabled():
    # summary は完了行を1行(改行付き)残す。有効時のみ書き、無効時は no-op。
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.summary("完了 out.vmd")
    assert stream.getvalue() == "完了 out.vmd\n"
    stream2 = io.StringIO()
    r2 = ProgressReporter(stream2, enabled=False, now=lambda: 0.0)
    r2.summary("完了 out.vmd")
    assert stream2.getvalue() == ""


def test_shorter_line_pads_over_previous_longer_line():
    # 長い行の後に短い行を描くとき、残像(前の行の余分な文字)を空白で埋めて消す。
    # 期待パディング数は本番の _display_width を使わず、表示幅を手計算した固定値で検証する
    # (全角を1桁と誤って数える実装があっても、描画と期待値が同じ誤りを共有して見逃さないため)。
    # ラベルはどちらも全角5文字(表示幅10)で揃えてあるので、差分は total 付き書式が追加する
    # " 1/1"(表示幅4)の分だけになる。
    # 長い行 "[処理その一] 1/1 経過 0:00" の表示幅:
    #   [(1)+処理その一(10)+](1)+ (1)+1/1(3)+ (1)+経過(4)+ (1)+0:00(4) = 26
    # 短い行 "[処理その二] 経過 0:00" の表示幅:
    #   [(1)+処理その二(10)+](1)+ (1)+経過(4)+ (1)+0:00(4) = 22
    # pad = 26 - 22 = 4
    stream = _TTYStream()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("処理その一")
    r.update(1, 1)
    r._draw()
    r.stage("処理その二")  # 短い行の段へ切り替え
    r._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[処理その二] 経過 0:00" + " " * 4
    r.close()


# 書き込み失敗保護が捕捉すべき例外種別。端末のエンコード不能(全角ラベルが端末コーデックで表せない等)は UnicodeError 系。
# OSError だけ捕捉する実装を弾くため、3種すべてで保護が効くことを固定する。
_WRITE_FAILURES = [OSError("io failed"), ValueError("bad stream"), UnicodeError("encode failed")]


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_draw_write_failure_disables_without_raising_and_close_stops_thread(exc):
    # stderr 書き込みが失敗しても例外を外へ漏らさず enabled=False に落ち、以後は描画 no-op。
    # 表示が無効化された後も、close はハートビートスレッドを必ず停止・join する(残存スレッド防止)。
    stream = _RaisingStream(exc)
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    live = r._thread
    assert live is not None and live.is_alive()
    r.update(1, 2)
    r._draw()  # 内部 write が例外送出 → 捕捉して enabled=False(例外は外へ出ない)
    assert r.enabled is False
    r.close()  # enabled が False でもハートビートを停止・join する
    assert not live.is_alive()
    assert r._thread is None
    # 以後の呼び出しも no-op で例外を出さない。
    r.update(3, 4, note="ボーン: センター")
    r._draw()
    r.summary("完了 out.vmd")


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_summary_write_failure_swallowed(exc):
    # summary の書き込み失敗も握りつぶし、例外を外へ漏らさず enabled=False に落ちる。
    stream = _RaisingStream(exc)
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.summary("完了 out.vmd")  # write が例外 → 捕捉、例外は外へ出ない
    assert r.enabled is False


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_close_write_failure_swallowed(exc):
    # close の行消去書き込みが失敗しても例外を外へ漏らさず、スレッドは停止・join される。
    # 成功描画で _last_width を立ててからストリームを失敗版へ差し替え、close の消去 write を失敗させる。
    stream = _TTYStream()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(2, 4)
    r._draw()  # 成功描画で _last_width を立てる
    live = r._thread
    r._stream = _RaisingStream(exc)  # close の行消去 write を失敗させる
    r.close()  # 例外を外へ漏らさず enabled=False、スレッドは停止・join
    assert r.enabled is False
    assert not live.is_alive()
    assert r._thread is None


# write_lock: 呼び出し元が同じ標準エラーへ別の仕組み(OSレベルのfd操作等)で書く場合に備え、構築時に
# 排他ロックを注入できる。_draw/close/summary の書き込みがそのロックを保持している間に行われることを、
# 別スレッドから非ブロッキングでロック取得を試み、取れない(=書き込み側が保持している)ことで固定する。


def test_draw_holds_write_lock_during_write():
    lock = threading.Lock()
    stream = _BlockingStream()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0, write_lock=lock)
    r.stage("キーフレーム圧縮")

    t = threading.Thread(target=r._draw)
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert lock.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)
        r._stop_heartbeat()  # stage() が起こしたハートビートスレッドを回収する


def test_close_holds_write_lock_during_write():
    lock = threading.Lock()
    stream = _BlockingStream()
    r = ProgressReporter(_TTYStream(), enabled=True, now=lambda: 0.0, interval=3600.0, write_lock=lock)
    r.stage("キーフレーム圧縮")
    r._draw()  # _last_width を立てて close の行消去 write を発生させる
    r._stop_heartbeat()
    r._stream = stream

    t = threading.Thread(target=r.close)
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert lock.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)


def test_summary_holds_write_lock_during_write():
    lock = threading.Lock()
    stream = _BlockingStream()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0, write_lock=lock)

    t = threading.Thread(target=r.summary, args=("完了 out.vmd",))
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert lock.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)


def test_write_lock_defaults_to_noop_when_unspecified():
    # write_lock 未指定時は contextlib.nullcontext() で no-op となり、ロックなしで動作する
    # (既存呼び出し側が write_lock を渡さなければ現状と完全に同じ挙動になる)。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(1, 2)
    r._draw()
    r.summary("完了 out.vmd")
    r.close()
    assert "[キーフレーム圧縮] 1/2 経過 0:00" in stream.getvalue()
