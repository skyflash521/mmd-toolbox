"""song2vmd 進捗表示(ファサード)。

ProgressReporter は machine/quiet/stream から表示先を1箇所で出し分ける。機械モードでは TTY 判定に
関係なく progress イベントを emitter へ送出する(--quiet は機械モードの送出に影響しない)。非機械
モードでは、`stream.isatty()` と `--quiet` から求めた有効/無効を渡したうえで stage id を日本語の
工程名へ変換し、共有の `cli_progress.progress.ProgressReporter` へ委譲してライブ行を表示する
(ハートビート・書き込み失敗時の無効化等の実体は委譲先が持つ)。

fd 2 の排他: vocal_analysis.quiet.suppress_native_stderr は pyopenjtalk のネイティブ拡張呼び出し中、
標準エラー(fd 2)を os.dup2 で一時的に devnull へ差し替える(ネイティブ拡張が fd へ直接書く警告を
抑えるため)。この差し替え中に別スレッドが fd 2 へ書き込むと、Windows の実コンソールハンドルでは
OSError([WinError 1])が発生しうるため、共有表示器の構築時に同じ vocal_analysis.quiet.STDERR_WRITE_LOCK
を write_lock として注入し、suppress_native_stderr の fd 差し替えと重ならないようにする。
"""

from cli_progress import progress as _display_module
from vocal_analysis.quiet import STDERR_WRITE_LOCK

# load/separate/recognize/write は複数ツールで共有する工程、rms/events/generate は song2vmd 固有工程。
_STAGE_LABELS = {
    "load": "音声読み込み",
    "separate": "ボーカル分離",
    "recognize": "音素認識",
    "rms": "音量解析",
    "events": "口形イベント確定",
    "generate": "モーフ生成",
    "write": "書き出し",
}


def stage_label(stage_id):
    """stage id に対応する日本語の工程名。未知の id はそのまま返す。"""
    return _STAGE_LABELS.get(stage_id, stage_id)


class ProgressEmitError(Exception):
    """機械モードの progress イベント送出の失敗。

    送出は外部推論の進捗コールバックからも呼ばれるため、素の例外のまま返すと呼び出し元が工程の
    失敗と区別できない。工程の失敗でないことを型で示し、想定外例外の経路(内部エラー)へ落とす。
    """


class ProgressReporter:
    """song2vmd 進捗表示の出し分けを1箇所にまとめる。

    stage(stage_id, *, done=0, total=None, note="", elapsed=0.0) で工程の開始/進行を報告する。
    機械モードは emitter.progress へそのまま送出する。非機械モードは stage_id が変わるたびに
    共有表示器の段を切り替え、同じ stage_id での再呼び出しは進行(done/total/note)を更新するだけで
    段は継続する。

    close() でライブ行を消し、summary(message) は完了行を1行残す。
    """

    def __init__(self, *, machine, quiet, emitter, stream, now=None, interval=0.15):
        self._machine = machine
        self._emitter = emitter if machine else None
        self._current_stage = None
        if machine:
            self._display = None
        else:
            self._display = _display_module.ProgressReporter(
                stream=stream, enabled=(not quiet) and stream.isatty(),
                now=now, interval=interval, write_lock=STDERR_WRITE_LOCK)

    def stage(self, stage_id, *, done=0, total=None, note="", elapsed=0.0):
        """工程の開始/進行を報告する。machine モードは emitter.progress へそのまま送出し、非 machine
        モードはこのメソッド内で表示の出し分けを完結させる(呼び出し側は machine/non-machine を
        意識しない)。"""
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
            self._display.stage(stage_label(stage_id))
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
