"""進捗のライブ表示(スタブ)。

本実装は未追加。libs/cli_progress/tests/test_progress.py が収集できるよう、公開 API の
シグネチャだけを NotImplementedError で仮置きする。
"""


def _format_line(label, done, total, elapsed, note=""):
    raise NotImplementedError


def _display_width(text):
    raise NotImplementedError


class ProgressReporter:
    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15, write_lock=None):
        raise NotImplementedError

    def stage(self, label):
        raise NotImplementedError

    def update(self, done, total, note=""):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def summary(self, message):
        raise NotImplementedError
