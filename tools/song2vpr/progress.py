"""song2vpr の段ラベル表。

段 id と日本語の工程名の対応だけを持つ。報告先の振り分け(機械モードのイベント送出と人間向け表示の
切り替え)の実体は共有の cli_progress_router が持つ。
"""

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
