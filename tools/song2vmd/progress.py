"""song2vmd 進捗表示。

ProgressReporter は machine/quiet/stream から表示先を1箇所で出し分ける。機械モードでは TTY 判定に
関係なく progress イベントを emitter へ送出する(--quiet は機械モードの送出に影響しない)。非機械
モードでは標準エラーが端末(TTY)かつ --quiet 未指定のときだけ、stage id を日本語の工程名へ変換した
1本のライブ行で進行を表示する(全工程を通して同じ行を上書きし、工程が替わっても改行しない)。

停滞回避の要は「完了通知(stage)と再描画の分離」。stage は進行スナップショットを更新するだけで
描画しない。実際に行を描くのは一定間隔で回るハートビート用デーモンスレッドで、次の stage が来ない
待ち時間(モデルの初回ダウンロード等)でも経過時間を進め続けるため、表示が固まって見えない。

単一描画所有者: 進捗行を stderr へ書くのはハートビートスレッドだけ。段の切り替え・終了はハートビートを
停止イベントで止めて join し(描画所有権を回収)てから書くため、同時に stderr へ書くスレッドが常に1つに
なり、行の混線を防ぐ。

fd 2 の排他: vocal_analysis.quiet.suppress_native_stderr は G2P 変換中、標準エラー(fd 2)を
os.dup2 で一時的に devnull へ差し替える(pyopenjtalk のネイティブ拡張が fd へ直接書く警告を
抑えるため)。この差し替え中に別スレッドが fd 2 へ書き込むと、Windows の実コンソールハンドルでは
OSError([WinError 1])が発生しうるため、stream への書き込みは同じ
vocal_analysis.quiet.STDERR_WRITE_LOCK を取ってから行い、suppress_native_stderr の fd 差し替えと
重ならないようにする。

副作用専用の保証: 表示の stream 書き込みが失敗(端末のエンコード不能・閉じたストリームへの書き込み等)
しても例外を外へ漏らさず、以後の表示を無効化(enabled=False)するだけで処理は止めない(終了コード不変)。
書き込み失敗で無効化された後でも、残存スレッドを防ぐため close はハートビートの停止・join を必ず行う。
"""

import sys
import threading
import time
import unicodedata

from vocal_analysis.quiet import STDERR_WRITE_LOCK

# 表示書き込みで握りつぶす例外。端末のエンコード不能(全角ラベルが端末コーデックで表せない等)・
# 閉じたストリームへの書き込みなどを捕捉し、表示を無効化するだけで処理は止めない(副作用専用)。
_WRITE_ERRORS = (OSError, ValueError, UnicodeError)

# stage id(機械モードの progress イベントと同じ安定 id)から、利用者向けの日本語工程名への変換表。
# load/separate/recognize/write は docs/conventions/terminology.md の「処理工程の利用者向け名称」
# 表が定義する共有工程、rms/events/generate は song2vmd 固有工程(song2vmd.md 12.1)。
_STAGE_LABELS = {
    "load": "音声読み込み",
    "separate": "ボーカル分離",
    "recognize": "音素認識",
    "rms": "音量解析",
    "events": "口形イベント確定",
    "generate": "モーフ生成",
    "write": "書き出し",
}


def _format_line(label, done, total, elapsed, note=""):
    """1行ぶんの表示文字列を組む。total が None(未確定)のときはカウントを出さず経過のみ。
    note があれば行末へ併記する。"""
    secs = int(elapsed)
    count = "" if total is None else f" {done}/{total}"
    suffix = f" {note}" if note else ""
    return f"[{label}]{count} 経過 {secs // 60}:{secs % 60:02d}{suffix}"


def _display_width(text):
    """端末表示幅。全角(東アジア幅 W/F)を2、その他を1として数える(残像消去の埋め幅算出に使う)。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


class ProgressReporter:
    """song2vmd 進捗表示の出し分けを1箇所にまとめる。

    stage(stage_id, *, done=0, total=None, note="", elapsed=0.0) で工程の開始/進行を報告する。
    機械モードは emitter.progress へそのまま送出する。非機械モードは stage_id が変わるたびに
    新しい段としてライブ行を再起動し(旧ハートビートを止めてから新ハートビートを起こす)、同じ
    stage_id での再呼び出しは進行スナップショット(done/total/note)を更新するだけで段は継続する。

    close() でライブ行を消し、summary(message) は完了行を1行残す。副作用専用の保証・書き込み失敗時の
    無効化・ハートビートの単一描画所有者の詳細はモジュール docstring を参照。
    """

    def __init__(self, *, machine, quiet, emitter, stream, now=None, interval=0.15):
        self._machine = machine
        self._emitter = emitter if machine else None
        self.enabled = (not machine) and (not quiet) and stream.isatty()
        self._stream = stream
        self._now = now if now is not None else time.monotonic
        self._interval = interval
        self._current_stage = None
        self._snap = None  # (label, done, total, start, note) または None。単一参照代入で原子的に差し替える
        self._thread = None
        self._stop = None
        self._last_width = 0  # 直前に同じ行へ描いた表示幅。短い行で上書きするとき残像を埋めるのに使う

    def stage(self, stage_id, *, done=0, total=None, note="", elapsed=0.0):
        """工程の開始/進行を報告する。machine モードは emitter.progress へそのまま送出し、非 machine
        モードはこのメソッド内で表示の出し分けを完結させる(呼び出し側は machine/non-machine を
        意識しない)。"""
        if self._machine:
            self._emitter.progress(stage=stage_id, done=done, total=total, note=note, elapsed=elapsed)
            return
        if not self.enabled:
            return
        if stage_id != self._current_stage:
            self._stop_heartbeat()
            self._current_stage = stage_id
            label = _STAGE_LABELS.get(stage_id, stage_id)
            self._snap = (label, done, total, self._now(), note)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
        else:
            snap = self._snap
            if snap is not None:
                self._snap = (snap[0], done, total, snap[3], note)

    def close(self):
        """進捗を終える。ハートビートを止め join してから、ライブ行を消して何も残さない。

        書き込み失敗で表示が無効化された後でも、残存スレッドを防ぐためハートビートの停止・join は必ず行う。
        """
        self._stop_heartbeat()
        if self.enabled and self._last_width:
            try:
                with STDERR_WRITE_LOCK:
                    self._stream.write("\r" + " " * self._last_width + "\r")
                    self._stream.flush()
            except _WRITE_ERRORS:
                self.enabled = False
        self._snap = None
        self._current_stage = None
        self._last_width = 0

    def summary(self, message):
        """完了行を1行残す(有効時のみ)。close で行を消した後に呼び、結果を1行で示すのに使う。"""
        if not self.enabled:
            return
        try:
            with STDERR_WRITE_LOCK:
                self._stream.write(message + "\n")
                self._stream.flush()
        except _WRITE_ERRORS:
            self.enabled = False

    def _heartbeat(self):
        while not self._stop.wait(self._interval):
            self._draw()

    def _stop_heartbeat(self):
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
        label, done, total, start, note = snap
        line = _format_line(label, done, total, self._now() - start, note)
        width = _display_width(line)
        pad = " " * max(0, self._last_width - width)
        try:
            with STDERR_WRITE_LOCK:
                self._stream.write("\r" + line + pad)
                self._stream.flush()
        except _WRITE_ERRORS:
            self.enabled = False
            return
        self._last_width = width
