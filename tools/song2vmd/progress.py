"""song2vmd の段ラベル表と、進捗の振り分けの構築設定。

報告先の振り分け(機械モードのイベント送出と人間向けライブ表示の切り替え)の実体は共有の
cli_progress_router が持つ。ここが持つのは song2vmd の段 id と日本語の工程名の対応表と、
構築時に渡す設定だけ。

fd 2 の排他: vocal_analysis.quiet.suppress_native_stderr は pyopenjtalk のネイティブ拡張呼び出し中、
標準エラー(fd 2)を os.dup2 で一時的に devnull へ差し替える(ネイティブ拡張が fd へ直接書く警告を
抑えるため)。この差し替え中に別スレッドが fd 2 へ書き込むと、Windows の実コンソールハンドルでは
OSError([WinError 1])が発生しうるため、同じ vocal_analysis.quiet.STDERR_WRITE_LOCK を write_lock
として注入し、suppress_native_stderr の fd 差し替えと重ならないようにする。
"""

from cli_progress_router import ProgressRouter
from vocal_analysis.quiet import STDERR_WRITE_LOCK

# load/separate/recognize/rms/write は複数ツールで共有する工程、events/generate は song2vmd 固有工程。
STAGE_LABELS = {
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
    return STAGE_LABELS.get(stage_id, stage_id)


def build_router(*, machine, quiet, emitter, stream):
    """song2vmd の段ラベル表と fd 2 の排他ロックを与えて進捗の振り分けを作る。"""
    return ProgressRouter(
        machine=machine, quiet=quiet, emitter=emitter, stream=stream,
        labels=STAGE_LABELS, write_lock=STDERR_WRITE_LOCK)
