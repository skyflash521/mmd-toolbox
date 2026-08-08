"""進捗の報告先を、機械モードのイベント送出と人間向けライブ表示へ振り分ける。"""

from cli_progress import progress as _display_module


class ProgressEmitError(Exception):
    """機械モードの progress イベント送出の失敗。

    送出は外部処理の進捗コールバックからも呼ばれるため、素の例外のまま返すと呼び出し元が工程の
    失敗と区別できない。工程の失敗でないことを型で示し、想定外例外の経路へ落とす。
    """


class ProgressRouter:
    """1回の進捗報告を実行モードに応じて振り分ける。

    stage(stage_id, *, done=0, total=None, note="", elapsed=0.0) で工程の開始/進行を報告する。
    機械モードは受け取った5つの値をそのまま progress イベントへ載せる。非機械モードは stage_id が
    変わるたびに表示の段を切り替え、同じ stage_id での再呼び出しは進行を更新するだけで段は継続する。
    """

    def __init__(self, *, machine, quiet, emitter, stream, labels,
                 write_lock=None, now=None, interval=None):
        self._machine = machine
        self._emitter = emitter if machine else None
        self._labels = labels
        self._current_stage = None
        if machine:
            self._display = None
        else:
            # 再描画間隔は未指定なら渡さない(既定値をここへ書き写すと、表示側が既定を変えても
            # 追随しない別管理の値になる)。
            interval_option = {} if interval is None else {"interval": interval}
            self._display = _display_module.ProgressReporter(
                stream=stream, enabled=(not quiet) and stream.isatty(),
                now=now, write_lock=write_lock, **interval_option)

    def stage(self, stage_id, *, done=0, total=None, note="", elapsed=0.0):
        if self._machine:
            try:
                self._emitter.progress(
                    stage=stage_id, done=done, total=total, note=note, elapsed=elapsed)
            except Exception as e:
                # 失敗理由は専用例外の文字列だけが報告に出るので、元例外の型名と文言をそこへ残す。
                raise ProgressEmitError(f"{type(e).__name__}: {e}") from e
            return
        if stage_id != self._current_stage:
            self._current_stage = stage_id
            self._display.stage(self._labels.get(stage_id, stage_id))
        self._display.update(done, total, note)

    def close(self):
        """進捗を終える。次に同じ stage_id で stage() が呼ばれたとき新しい段として再開できるよう、
        machine の真偽に関わらず現在の段の記録を必ずリセットする。"""
        if not self._machine:
            self._display.close()
        self._current_stage = None

    def summary(self, message):
        """完了行を1行残す(非機械モードのみ)。"""
        if not self._machine:
            self._display.summary(message)
