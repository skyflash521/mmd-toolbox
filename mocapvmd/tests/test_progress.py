"""ProgressReporter(進捗のライブ表示)の単体テスト(mocapvmd.md 進捗表示、実装計画 §2-§4)。

ProgressReporter は疎化など重い処理の進行を stderr へ1行ライブ表示する(TTY時のみ)。完了イベント
(update)と再描画(ハートビートスレッド)を分離し、update が来ない待ち時間でも経過時間を進め続ける
ことで、重いボーン1本で表示が停滞して見えるのを防ぐ。経過時間は now() - 段階開始時刻で測る。

検証は stream/clock/有効化/再描画間隔を注入で差し替えて行う:
  - 行整形(_format_line)・no-op化(非TTY)・isatty 自動判定は注入クロックで確定的に。
  - 「update 自身は描かず、描くのはハートビートだけ」は interval を十分大きくしてハートビートを
    発火させず、update 後に無出力であることで固定する。
  - 「join してから最終行＋改行で確定」は同じくハートビート未発火にし、end_stage の出力が厳密に
    最終行＋改行になることで固定する。
  - 「update なしでも経過が進む」停滞回避の核は、小さい interval でハートビートを発火させ、クロックを
    進めて複数の経過表記が現れることで固定する(結末は join 同期で確定)。
"""

import io
import time

from mocapvmd import progress

ProgressReporter = progress.ProgressReporter


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


def _wait_until(pred, timeout=2.0):
    # 条件成立を寛容なタイムアウトで待つ。成功は確定的(成立し次第すぐ返る)、実装が誤りなら時間切れで fail。
    # 負荷時にスケジューリングが遅れても、固定 sleep のように「その瞬間に描画済み」を仮定せず取りこぼさない。
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def test_format_line_shows_label_count_elapsed():
    assert progress._format_line("疎化", 5, 88, 73.0) == "[疎化] 5/88 経過 1:13"


def test_format_line_hides_count_when_total_unset():
    # total 未確定(begin_stage 直後・最初の update 前)はカウントを出さず経過のみ。
    assert progress._format_line("疎化", 0, None, 2.0) == "[疎化] 経過 0:02"


def test_format_line_width_counts_fullwidth_as_two():
    assert progress._display_width("疎化") == 4  # 全角2文字=4
    assert progress._display_width("[X] 0/5") == 7  # 半角はそのまま


def test_draw_pads_to_clear_leftover_from_longer_line():
    # 長い行のあと短い行で同じ行を上書きするとき、差ぶんを空白で埋めて旧行の残像を消す。
    # interval 大でハートビート未発火にし、_draw を2回直接呼んで決定的に確かめる。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.begin_stage("X")
    r.update(0, 1000000)  # 長い行
    r._draw()
    r.update(0, 5)  # 短い行
    r._draw()
    last = stream.getvalue().rsplit("\r", 1)[1]  # 最後に描いた行ぶん
    assert last.startswith("[X] 0/5 経過 0:00")
    assert last.endswith(" ")  # 旧行(0/1000000)を覆う空白が続く
    r.close()


def test_disabled_is_noop_no_thread_no_output():
    # 非TTY/quiet 相当(enabled=False)は全メソッド no-op。stream へ何も書かず、スレッドも起こさない。
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=False, now=lambda: 0.0)
    r.begin_stage("疎化")
    r.update(1, 2)
    r.end_stage()
    r.close()
    assert stream.getvalue() == ""
    assert r._thread is None


def test_auto_enabled_follows_isatty():
    # enabled 省略時は stream.isatty() で自動判定する。
    assert ProgressReporter(_TTYStream(), now=lambda: 0.0)._enabled is True
    assert ProgressReporter(io.StringIO(), now=lambda: 0.0)._enabled is False  # StringIO.isatty()==False


def test_update_does_not_write_heartbeat_is_sole_writer():
    # 進捗行を書くのはハートビートだけ(計画 §2)。interval を十分大きくしてハートビートを発火させなければ、
    # update を呼んでも何も書かれない(update が直接 stream へ書く実装を弾く)。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.begin_stage("疎化")
    clock[0] = 5.0
    r.update(3, 10)
    assert stream.getvalue() == ""  # update 自身は描かない
    r.close()


def test_end_stage_draws_final_line_then_newline_after_join():
    # ハートビート未発火(interval 大)にして、end_stage がハートビートを止め join した後にメインが
    # 最終行を1度描き、改行で確定する順序を厳密な出力で固定する(計画 §2 単一描画所有者)。
    # 経過は段階開始(begin_stage)時刻基準。構築時刻(50.0)と begin_stage 時刻(100.0)を変え、構築時刻でなく
    # begin_stage 時刻からの差分(108-100=8 → 0:08)が出ることで開始基準を固定する(構築時刻基準なら 0:58 で落ちる)。
    stream = io.StringIO()
    clock = [50.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)  # 構築時刻 50.0
    clock[0] = 100.0  # begin_stage 前に進める
    r.begin_stage("疎化")  # 段階開始時刻 = 100.0
    r.update(10, 10)
    clock[0] = 108.0
    r.end_stage()
    assert stream.getvalue() == "\r[疎化] 10/10 経過 0:08\n"
    assert r._thread is None  # ハートビートは停止・join 済み


def test_stages_reused_reset_label_and_start():
    # 複数段階の再利用(CLI 主要経路): end_stage 後の begin_stage で新しいラベル・開始時刻にリセットし、
    # ハートビートを再起動する。各段の経過はその段の begin_stage 基準。ハートビート未発火(interval 大)で
    # 出力を厳密に固定し、前段の開始時刻やラベルを引きずる実装を弾く。
    stream = io.StringIO()
    clock = [10.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.begin_stage("クリーニング")  # 開始 10.0
    r.update(4, 4)
    clock[0] = 12.0
    r.end_stage()  # 経過 12-10 = 0:02
    clock[0] = 100.0
    r.begin_stage("疎化")  # 開始 100.0(前段を引きずらない)
    r.update(7, 7)
    clock[0] = 105.0
    r.end_stage()  # 経過 105-100 = 0:05
    assert stream.getvalue() == "\r[クリーニング] 4/4 経過 0:02\n\r[疎化] 7/7 経過 0:05\n"
    assert r._thread is None


def test_end_stage_stops_live_heartbeat_thread():
    # end_stage は活動中のハートビートを停止イベントで止め join する(_thread を None にするだけでなく
    # 実スレッドを終わらせる)。捕捉した実スレッド参照が end_stage 後に is_alive()==False になることで、
    # 止め損ねた残存スレッドが後段の表示を乱す穴を弾く(計画 §2 単一描画所有者)。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.begin_stage("疎化")
    r.update(1, 2)
    clock[0] = 3.0
    _wait_until(lambda: "経過 0:03" in stream.getvalue())  # ハートビートが実際に動いている
    live = r._thread
    assert live is not None and live.is_alive()
    r.end_stage()
    assert not live.is_alive()  # end_stage が実スレッドを停止・join した
    assert r._thread is None


def test_begin_stage_while_active_stops_old_heartbeat():
    # end_stage を挟まずに begin_stage を再呼びしても、旧ハートビートを止めてから新段を起こす。
    # 旧スレッドが残って後から描画し混線する漏れを弾く(計画 §2 単一描画所有者)。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.begin_stage("クリーニング")
    first = r._thread
    assert first is not None and first.is_alive()
    r.begin_stage("疎化")  # end_stage を経ずに再開始
    assert not first.is_alive()  # 旧ハートビートは停止・join された
    assert r._thread is not None and r._thread is not first and r._thread.is_alive()
    r.close()


def test_heartbeat_runs_after_stage_reuse():
    # 段階を再利用しても、後段でハートビートが再起動して動く(計画 §2 全段で一様に停滞回避)。2段目で
    # update を最初の1回だけにし、クロックを進めて経過が進む描画が出ることを待つ。begin_stage が
    # 2回目以降スレッドを再起動しない実装なら時間切れで fail する。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.begin_stage("クリーニング")  # 1段目
    r.update(1, 1)
    r.end_stage()
    clock[0] = 200.0
    r.begin_stage("疎化")  # 2段目: ここでハートビートが再起動するはず
    r.update(0, 5)  # 完了通知はこの1回だけ
    clock[0] = 205.0
    _wait_until(lambda: "[疎化] 0/5 経過 0:05" in stream.getvalue())  # 再利用後もハートビートが経過を進める
    r.end_stage()
    assert r._thread is None


def test_heartbeat_advances_elapsed_without_update():
    # 停滞回避の核(計画 §2): update は最初の1回だけ。以後 update が来なくてもハートビートが
    # now() - 段階開始 を再評価して再描画し、経過時間が進む。重いボーン1本で表示が固まらないことの担保。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.begin_stage("疎化")
    r.update(2, 10)  # 完了通知はこの1回だけ
    clock[0] = 5.0
    _wait_until(lambda: "経過 0:05" in stream.getvalue())  # ハートビートが経過 0:05 で再描画するのを待つ
    clock[0] = 60.0
    _wait_until(lambda: "経過 1:00" in stream.getvalue())  # update 無しでも経過 1:00 へ進む
    r.end_stage()
    out = stream.getvalue()
    assert "\r[疎化] 2/10 経過 0:05" in out
    assert "[疎化] 2/10 経過 1:00" in out


def test_close_is_noop_when_no_active_stage():
    # 活動段が無いとき close は no-op。CLI は正常終了後に finally: close を呼ぶので、end_stage 済み(または
    # begin_stage 前)の close が重複改行・重複最終行を出さないことを固定する。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.close()  # begin_stage 前: 無出力
    assert stream.getvalue() == ""
    r.begin_stage("疎化")
    r.update(2, 2)
    clock[0] = 4.0
    r.end_stage()
    after_end = stream.getvalue()
    r.close()  # end_stage 後(活動段なし): 追加出力なし
    assert stream.getvalue() == after_end
    assert r._thread is None


def test_close_draws_final_line_then_newline_after_join():
    # 例外時の経路でも close は end_stage と対称: 活動段があれば、停止 join 後にメインが最終行を1度描き
    # 改行で確定する(計画 §2/§3)。ハートビート未発火(interval 大)にして、出力が厳密に最終行＋改行に
    # なることで、既存行へ改行だけ足す/最終行を残さない実装を弾く。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.begin_stage("疎化")
    r.update(1, 4)
    clock[0] = 3.0
    r.close()
    assert stream.getvalue() == "\r[疎化] 1/4 経過 0:03\n"
    assert r._thread is None


def test_close_finalizes_active_line_without_end_stage():
    # 例外時の経路: end_stage を経ずに close しても、描画済みの行を改行確定しスレッドを止める。
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.begin_stage("疎化")
    r.update(1, 4)
    clock[0] = 3.0
    _wait_until(lambda: "\r[疎化] 1/4 経過 0:03" in stream.getvalue())  # ハートビートに1行描かせてから close
    live = r._thread
    assert live is not None and live.is_alive()
    r.close()
    out = stream.getvalue()
    assert "\r[疎化] 1/4 経過 0:03" in out
    assert out.endswith("\n")
    assert not live.is_alive()  # close も活動中ハートビートを停止・join する(例外時経路の混線防止)
    assert r._thread is None
