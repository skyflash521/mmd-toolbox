"""song2vpr の段ラベル表と、進捗の振り分けの構築設定。

報告先の振り分け(機械モードのイベント送出と人間向け表示の切り替え)の実体は共有の
cli_progress_router が持つ。ここが持つのは song2vpr の段 id と日本語の工程名の対応表と、構築時に
渡す設定だけ。

標準エラーの排他: 音声前段は処理中に標準エラーの出力先を一時的に差し替えることがあり、その間に
人間向けのライブ表示が同じ標準エラーへ書くと書き込みに失敗しうる。前段が公開する排他の仕組みを
そのまま表示側へ渡して、差し替えと重ならないようにする。
"""

from cli_progress_router import ProgressRouter
from vocal_analysis.quiet import STDERR_WRITE_LOCK

# load/separate/recognize/rms/write は複数ツールで共有する工程、f0/notes は song2vpr 固有工程。
STAGE_LABELS = {
    "load": "音声読み込み",
    "separate": "ボーカル分離",
    "recognize": "音素認識",
    "rms": "音量解析",
    "f0": "ピッチ推定",
    "notes": "音符化",
    "write": "書き出し",
}


def stage_label(stage_id):
    """stage id に対応する日本語の工程名。未知の id はそのまま返す。"""
    return STAGE_LABELS.get(stage_id, stage_id)


def build_router(*, machine, quiet, emitter, stream):
    """song2vpr の段ラベル表と標準エラーの排他ロックを与えて進捗の振り分けを作る。"""
    return ProgressRouter(
        machine=machine, quiet=quiet, emitter=emitter, stream=stream,
        labels=STAGE_LABELS, write_lock=STDERR_WRITE_LOCK)
