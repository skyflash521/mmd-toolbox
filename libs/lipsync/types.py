"""lipsync コアの公開データ型。

時刻はすべてフレーム(30fps、float)。生成パラメータの既定値は初期目安。
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


class ConsonantClass(Enum):
    """母音的口形イベントの先頭子音の唇の丸め・横引き方向(母音合成を変調する)。

    標準口モーフは唇しか表せないため、子音は唇に影響するものだけが口形を変えられる。顎の開口量
    そのものは変えない(開口量は ApertureClass が独立に変調する)。NONE は子音なし、NEUTRAL は
    唇を動かさない子音(軟口蓋 か行・歯茎 さ/た/な/ら行・声門 は行 等。舌/喉が主体で唇効果なし)、
    ROUNDED は唇を丸める子音(ふ・わ)、SPREAD は視覚補助として い 方向へ寄せる子音(し・ち・じ・
    拗音)。両唇閉鎖(ま/ば/ぱ行)は `MouthShape.BILABIAL` で表しここには設けない(二重表現を
    避ける)。合成上 NONE と NEUTRAL は同値(純母音=主モーフ単独)。
    """

    NONE = "none"
    NEUTRAL = "neutral"
    ROUNDED = "rounded"
    SPREAD = "spread"


class ApertureClass(Enum):
    """母音的口形イベントの先頭子音がもたらす顎の開口減衰の強さ(母音合成の各モーフ最終重みを
    一律に減衰させる)。

    ConsonantClass(唇の丸め・横引き方向)とは独立な軸で、唇の方向を変えず顎の狭め具合だけを表す。
    各メンバー名・所属音は日本語の音韻論上の厳密な「調音位置」区分ではなく、開口減衰の強さで
    まとめた視覚チューニング用バケットである。両唇閉鎖(ま/ば/ぱ行)は `MouthShape.BILABIAL` で
    表しここには設けない。
    """

    NONE = "none"
    FIRM_CLOSURE = "firm_closure"
    NARROW_CHANNEL = "narrow_channel"
    SLIGHT_CLOSURE = "slight_closure"


@dataclass
class MouthEvent:
    """口形イベント。1イベントは1つのモーラに属するが、1つのモーラが複数の連続する
    口形イベントに対応することがある(呼び出し側が同一の母音的口形を持つ複数の連続イベントで
    表す場合)。分割されない通常の場合は1対1(1口形イベント=1モーラ)。

    イベント列は時間順・隙間なく連続(終端=次の始端)・非重複・全時間軸被覆を
    前提とする(呼び出し側の責務)。
    """

    shape: MouthShape
    start: float  # 開始フレーム
    end: float  # 終了フレーム
    open_amount: float = 0.0  # 開き量 0〜1。母音的口形(母音・撥音「ん」)区間のみ有意
    # 先頭子音の唇の丸め・横引き方向。母音合成を変調する。母音以外(両唇閉鎖・無音・レガート間隙)
    # では NONE。1音符に複数母音があるとき、および1つのモーラを複数の連続する口形イベントで
    # 表すときは、いずれも先頭のイベントにのみ付け、後続のイベントは NONE(呼び出し側の責務)。
    consonant_class: ConsonantClass = ConsonantClass.NONE
    # 先頭子音がもたらす顎の開口減衰。母音合成の各モーフ最終重みを一律に減衰させる。ConsonantClass
    # とは独立な軸。母音以外では NONE、複数母音の音符・1モーラを複数の口形イベントで表す場合の
    # いずれも先頭のイベントにのみ付ける(呼び出し側の責務)。
    aperture_class: ApertureClass = ApertureClass.NONE


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
    mora_valley_frames: float = 12.0  # モーラ境界の谷の片側半幅(非負)
    mora_valley_min_gap_frames: float = 4.0  # 連続する谷の間引きが要求する最低間隔(非負)
