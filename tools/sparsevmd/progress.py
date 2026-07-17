"""進捗のライブ表示(実装待ち)。

重い処理(キーフレーム圧縮)の進行を stderr へ1本のライブ行で表示する予定のモジュール。
インターフェースは tools/sparsevmd/tests/test_progress.py が定義する。
"""


def _format_line(label, done, total, elapsed, note=""):
    raise NotImplementedError


def _display_width(text):
    raise NotImplementedError


class ProgressReporter:
    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15):
        raise NotImplementedError

    def stage(self, label):
        raise NotImplementedError

    def update(self, done, total, note=""):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def summary(self, message):
        raise NotImplementedError
