"""lipsync コアの公開データ型(implementation-plan.md §4.7/§4.8)。

時刻はすべてフレーム(30fps、float)。生成パラメータの意味は lipsync.md §2.3 が
正本で、既定値は implementation-plan.md §4.8 の初期目安。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MouthShape(Enum):
    """口形種別(implementation-plan.md §4.7)。

    BILABIAL と SILENCE は出力上どちらも閉口(全母音 0.0)だが、両唇閉鎖は
    閉口完成のタイミングを持つため生成上の扱いを区別する。
    """

    A = "a"
    I = "i"
    U = "u"
    E = "e"
    O = "o"
    BILABIAL = "bilabial"  # 両唇閉鎖(ま・ば・ぱ行)
    SILENCE = "silence"  # 無音・休符


@dataclass
class MouthEvent:
    """口形イベント。1イベント=1モーラ(implementation-plan.md §4.7)。

    イベント列は時間順・隙間なく連続(終端=次の始端)・非重複・全時間軸被覆を
    前提とする(呼び出し側の責務)。
    """

    shape: MouthShape
    start: float  # 開始フレーム
    end: float  # 終了フレーム
    open_amount: float = 0.0  # 開き量 0〜1。母音区間のみ有意


@dataclass
class GenerationParams:
    """キーフレーム生成パラメータ(implementation-plan.md §4.8)。

    既定値は出発点で、各CLIプリセットが上書きする。フレーム単位は 30fps。
    """

    open_cap: float = 0.8  # 開き量の上限
    vowel_scale: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    attack_frames: int = 2
    release_frames: int = 2
    min_hold_frames: int = 3
    coartic_overlap_max: int = 2  # 協調調音の重なり上限(=基準長)
    anticipation_frames: int = 1  # 母音口形の先行準備
    exaggeration: float = 1.0  # 母音合成プロファイルの誇張係数
    vibrato_threshold: int = 18  # 伸び表現を適用する保持プラトー長の下限
    vibrato_amp: float = 0.05
    vibrato_period: int = 15
