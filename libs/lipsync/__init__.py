"""lipsync — アニメ的口パクのキーフレーム生成共有コア(lipsync.md)。

複数の CLI が共有する。開き量を同梱した口形イベント列から、標準口モーフ
(あ・い・う・え・お)のモーフキーを生成する。VMD 形式の入出力は呼び出し側が
`vmd` 経由で行う。
"""

from .generate import generate_morph_keys
from .types import ConsonantClass, GenerationParams, MouthEvent, MouthShape

__all__ = [
    "ConsonantClass",
    "GenerationParams",
    "MouthEvent",
    "MouthShape",
    "generate_morph_keys",
]
