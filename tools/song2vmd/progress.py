"""song2vmd 進捗表示。

重い処理(分離・認識・長尺分割の各チャンク・モーフ生成)の間、処理の段階を表示する。非機械モードでは
標準エラーが端末(TTY)かつ --quiet 未指定のときだけ人間向け1行を出し、機械モードでは TTY 判定に
関係なく progress イベントを標準出力へ発行する(--quiet は機械モードの送出に影響しない)。
"""


class ProgressReporter:
    """進捗表示の出し分けを1箇所にまとめる。"""

    def __init__(self, *, machine, quiet, emitter, stream):
        self._machine = machine
        self._show_human = (not machine) and (not quiet) and stream.isatty()
        self._emitter = emitter if machine else None
        self._stream = stream

    def stage(self, stage, *, done=0, total=None, note="", elapsed=0.0):
        """段の開始(または進行)を報告する。段id は機械モードの progress イベントで公開する固定の語彙。"""
        if self._machine:
            self._emitter.progress(stage=stage, done=done, total=total, note=note, elapsed=elapsed)
        elif self._show_human:
            self._stream.write(f"[{stage}] done={done} total={total} {note}\n")
