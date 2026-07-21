"""進捗のライブ表示。

重い処理(主に疎化)の進行を stderr へ1行ライブ表示する。既定は出力先が端末(TTY)のときだけ有効で、
リダイレクト・パイプ時や無効化時は完全な no-op(オーバーヘッドなし)。

表示は全段を通して1本のライブ行を上書きする。段を切り替えても改行せず同じ行を書き換えるので、終わった
段の行が画面に残らない。処理が終わると行を消し、必要なら呼び出し側が完了行を1行残す(summary)。

停滞回避の要は「完了イベントと再描画の分離」。完了通知 update(done, total) は進行カウンタを更新するだけで
描画しない。実際に行を描くのは一定間隔で回るハートビート用デーモンスレッドで、update が来ない待ち時間
(重いボーン1本の疎化中など)でも経過時間を進め続けるため、表示が固まって見えない。

単一描画所有者: 進捗行を stderr へ書くのはハートビートスレッドだけ。メインが書く場面(段切り替え stage /
終了 close)は、書く前にハートビートを停止イベントで止めて join し(描画所有権を回収)てから書く。これにより
同時に stderr へ書くスレッドが常に1つになり、行の混線を防ぐ。

副作用専用の保証: 表示の stream 書き込みが失敗(端末のエンコード不能・閉じたストリームへの書き込み等)
しても例外を外へ漏らさず、以後の表示を無効化(enabled=False)するだけで処理は止めない(終了コード不変)。
書き込み失敗で無効化された後でも、残存スレッドを防ぐため close はハートビートの停止・join を必ず行う。

経過時間は now() - 段階開始時刻で測る。表示器は表示の判断だけを持ち、何を1段とするか・各段の総数の
決め方・完了行の文言は呼び出し側に委ねる(このモジュールは他モジュールを参照しない)。
"""

import sys
import threading
import time
import unicodedata

# 表示書き込みで握りつぶす例外。端末のエンコード不能(全角ラベルが端末コーデックで表せない等)・
# 閉じたストリームへの書き込みなどを捕捉し、表示を無効化するだけで処理は止めない(副作用専用)。
_WRITE_ERRORS = (OSError, ValueError, UnicodeError)


def _format_line(label, done, total, elapsed):
    """1行ぶんの表示文字列を組む。total が None(未確定)のときはカウントを出さず経過のみ。"""
    secs = int(elapsed)
    count = "" if total is None else f" {done}/{total}"
    return f"[{label}]{count} 経過 {secs // 60}:{secs % 60:02d}"


def _display_width(text):
    """端末表示幅。全角(東アジア幅 W/F)を2、その他を1として数える(残像消去の埋め幅算出に使う)。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


class ProgressReporter:
    """段単位の進捗を1本のライブ行で表示する。stage(label) で段を開始/切り替え、update(done, total) で
    進行を通知、close() で行を消して終える。summary(message) は完了行を1行残す。enabled 省略時は
    stream.isatty() で自動判定し、無効時は全メソッド no-op。

    stream 書き込みが失敗しても例外を外へ漏らさず、以後の表示を無効化(enabled=False)するだけで
    処理は止めない。無効化された後でも close はハートビートスレッドを必ず停止・join する。
    詳細はモジュール docstring を参照。
    """

    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15):
        self._stream = stream if stream is not None else sys.stderr
        if enabled is None:
            isatty = getattr(self._stream, "isatty", None)
            enabled = bool(isatty()) if callable(isatty) else False
        self.enabled = enabled
        self._now = now if now is not None else time.monotonic
        self._interval = interval
        self._snap = None  # (label, done, total, start) または None。単一参照代入で原子的に差し替える
        self._thread = None
        self._stop = None
        self._last_width = 0  # 直前に同じ行へ描いた表示幅。短い行で上書きするとき残像を埋めるのに使う

    def stage(self, label):
        """段を開始/切り替える。同じライブ行を使い、改行しない。総数は未確定(最初の update で確定)。

        直前段のハートビートを止めてから新しい段のハートビートを起こす(旧スレッドが残らないように)。
        """
        if not self.enabled:
            return
        self._stop_heartbeat()
        self._snap = (label, 0, None, self._now())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def update(self, done, total):
        """進行を通知する(描画はしない)。総数は毎回 total で更新する。段が無いときは無視する。"""
        snap = self._snap
        if snap is None:
            return
        self._snap = (snap[0], done, total, snap[3])

    def close(self):
        """進捗を終える。ハートビートを止め join してから、ライブ行を消して何も残さない。

        書き込み失敗で表示が無効化された後でも、残存スレッドを防ぐためハートビートの停止・join は必ず行う。
        """
        self._stop_heartbeat()
        if self.enabled and self._last_width:
            try:
                # 行を空白で上書きし、行頭へ戻して消す(改行しないので画面に残らない)。
                self._stream.write("\r" + " " * self._last_width + "\r")
                self._stream.flush()
            except _WRITE_ERRORS:
                self.enabled = False
        self._snap = None
        self._last_width = 0

    def summary(self, message):
        """完了行を1行残す(有効時のみ)。close で行を消した後に呼び、結果を1行で示すのに使う。"""
        if not self.enabled:
            return
        try:
            self._stream.write(message + "\n")
            self._stream.flush()
        except _WRITE_ERRORS:
            self.enabled = False

    def _heartbeat(self):
        # 停止イベントが立つまで一定間隔で再描画する。wait は間隔経過で False、停止で True を返す。
        while not self._stop.wait(self._interval):
            self._draw()

    def _stop_heartbeat(self):
        # ハートビートが動いていれば停止イベントで止めて join し、描画所有権を回収する。
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _draw(self):
        if not self.enabled:
            return
        snap = self._snap
        if snap is None:
            return
        label, done, total, start = snap
        line = _format_line(label, done, total, self._now() - start)
        width = _display_width(line)
        # 直前の行が今より長ければ、その差ぶん空白で埋めて残像を消す(次の \r で行頭へ戻る)。
        pad = " " * max(0, self._last_width - width)
        try:
            self._stream.write("\r" + line + pad)
            self._stream.flush()
        except _WRITE_ERRORS:
            self.enabled = False
            return
        self._last_width = width
