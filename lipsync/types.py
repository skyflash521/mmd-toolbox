"""lipsync コアの公開データ型。

時刻はすべてフレーム(30fps、float)。生成パラメータの意味は lipsync.md §2.3 が
正本で、既定値はその初期目安。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MouthShape(Enum):
    """口形種別。

    A〜O と N(撥音「ん」)は合成プロファイルを持つ母音的口形。N は閉口ではなく「ん」モーフを
    正に立てる有声的口形。BILABIAL と SILENCE は出力上どちらも閉口(全標準口モーフ 0.0)だが、
    両唇閉鎖は閉口完成のタイミングを持つため生成上の扱いを区別する。
    """

    A = "a"
    I = "i"
    U = "u"
    E = "e"
    O = "o"
    N = "n"  # 撥音「ん」(後続母音を持たない単独の鼻音)
    BILABIAL = "bilabial"  # 両唇閉鎖(ま・ば・ぱ行)
    SILENCE = "silence"  # 無音・休符
    LEGATO_GAP = "legato_gap"  # レガート間隙(非発音だが完全閉口でなく谷で繋ぐ)


@dataclass
class MouthEvent:
    """口形イベント。1イベント=1モーラ。

    イベント列は時間順・隙間なく連続(終端=次の始端)・非重複・全時間軸被覆を
    前提とする(呼び出し側の責務)。
    """

    shape: MouthShape
    start: float  # 開始フレーム
    end: float  # 終了フレーム
    open_amount: float = 0.0  # 開き量 0〜1。母音的口形(母音・撥音「ん」)区間のみ有意


@dataclass
class GenerationParams:
    """キーフレーム生成パラメータ。

    既定値は出発点で、各CLIプリセットが上書きする。フレーム単位は 30fps。
    """

    open_cap: float = 0.8  # 開き量の上限
    # 母音的口形別(a, i, u, e, o, n)の開き量倍率。
    vowel_scale: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    attack_frames: int = 2
    release_frames: int = 2
    min_hold_frames: int = 3
    # 三角形短区間の下限(フレーム)。L<triangle_min は吸収、triangle_min≤L<min_hold+2 は三角形ピーク。
    triangle_min_frames: float = 2.0
    coartic_overlap_max: int = 2  # 協調調音の重なり上限(=基準長)
    anticipation_frames: int = 1  # 母音口形の先行準備
    # レガート間隙(LEGATO_GAP)の谷係数 d を間隙長から決める線形関数のパラメータ。
    # d = clamp(shallow − slope×gap_len, deep, shallow)。間隙が長いほど d 小=谷が深い。
    legato_valley_shallow: float = 0.4  # 谷係数 d の上限(間隙長0、浅い谷)
    legato_valley_deep: float = 0.2  # 谷係数 d の下限(深い谷)
    legato_valley_slope: float = 0.025  # 間隙長1フレームあたりの d 減少
    exaggeration: float = 1.0  # 母音合成プロファイルの誇張係数
    vibrato_threshold: int = 18  # 伸び表現を適用する保持プラトー長の下限
    vibrato_amp: float = 0.05
    vibrato_period: int = 15
