"""song2vmd 歌い方スタイルプリセット。

--style が選ぶプリセットの具体値(開き量レンジ・アタック/リリース・協調調音の重なり上限・先行・
最小保持・合成誇張係数・開き量上限・母音別倍率・三角形下限・伸び表現・レガート谷)を確定する。
開き量上限・協調調音・先行・最小保持はCLIで明示指定(--open-max・--coarticulation・--anticipation・
--min-hold)があればプリセット値より優先する。母音別倍率は --vowel-gain が5母音値を要素ごとに
乗算して微調整する(撥音「ん」はプリセット値のまま)。それ以外の生成パラメータに
対応するCLIオプションは無く、プリセット値に固定される。
"""

from dataclasses import dataclass

# スタイル別プリセットの表。pop の母音別倍率・先行・協調調音・伸び表現・レガート谷は、同じ lipsync
# 生成コアを使うリップモーション生成系のMMD視覚チューニングで確定した標準値。他スタイルは未チューニングの
# 出発点値。モーラ境界の谷(半幅・最低間隔)はここに含めず、全プリセットが lipsync の生成既定を
# そのまま使う。
_PRESETS = {
    "pop": {
        "open_lo": 0.30, "open_hi": 0.75, "attack": 2, "release": 2,
        "coarticulation": 6, "anticipation": 11, "min_hold": 1, "exaggeration": 1.0, "open_max": 0.90,
        "vowel_scale": (1.30, 1.20, 1.70, 0.80, 1.70, 1.00),
        "triangle_min": 2.0,
        "vibrato_threshold": 10, "vibrato_amp": 0.05, "vibrato_period": 22,
        "legato_valley_shallow": 0.45, "legato_valley_deep": 0.30, "legato_valley_slope": 0.02,
    },
    "ballad": {
        "open_lo": 0.20, "open_hi": 0.55, "attack": 3, "release": 3,
        "coarticulation": 3, "anticipation": 1, "min_hold": 4, "exaggeration": 0.8, "open_max": 0.70,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
    "powerful": {
        "open_lo": 0.40, "open_hi": 0.95, "attack": 1, "release": 1,
        "coarticulation": 2, "anticipation": 2, "min_hold": 3, "exaggeration": 1.3, "open_max": 0.97,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
    "whisper": {
        "open_lo": 0.10, "open_hi": 0.35, "attack": 2, "release": 2,
        "coarticulation": 2, "anticipation": 1, "min_hold": 3, "exaggeration": 0.7, "open_max": 0.50,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
    "rap": {
        "open_lo": 0.30, "open_hi": 0.70, "attack": 1, "release": 1,
        "coarticulation": 1, "anticipation": 1, "min_hold": 2, "exaggeration": 1.0, "open_max": 0.85,
        "vowel_scale": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "triangle_min": 2.0,
        "vibrato_threshold": 18, "vibrato_amp": 0.05, "vibrato_period": 15,
        "legato_valley_shallow": 0.4, "legato_valley_deep": 0.2, "legato_valley_slope": 0.025,
    },
}

STYLE_NAMES = tuple(_PRESETS)


@dataclass(frozen=True)
class OpennessParams:
    """RMS→開き量写像に使うパラメータ。"""

    open_lo: float
    open_hi: float
    open_max: float


@dataclass(frozen=True)
class StyleGenParams:
    """lipsync へ渡す生成パラメータのうち song2vmd のスタイルプリセットが確定する部分
    (母音合成プロファイル等の残りは lipsync 側が持つ)。

    vowel_scale は --vowel-gain 乗算適用後の最終6要素(a, i, u, e, o, ん)。
    """

    attack_frames: int
    release_frames: int
    coartic_overlap_max: int
    anticipation_frames: int
    min_hold_frames: int
    exaggeration: float
    vowel_scale: tuple
    triangle_min_frames: float
    vibrato_threshold: int
    vibrato_amp: float
    vibrato_period: int
    legato_valley_shallow: float
    legato_valley_deep: float
    legato_valley_slope: float


def resolve(style, *, open_max=None, coarticulation=None, anticipation=None, min_hold=None,
            vowel_gain=(1.0, 1.0, 1.0, 1.0, 1.0)):
    """style のプリセット値に、CLI明示指定(Noneでない引数)を上書きして確定する。

    vowel_gain(--vowel-gain の5母音値)はプリセットの母音別倍率の a〜o へ要素ごとに乗算し、
    撥音「ん」はプリセット値のまま使う。戻り値は (OpennessParams, StyleGenParams)。
    開き量レンジ(open_lo/open_hi)・アタック・リリース・合成誇張係数・三角形下限・伸び表現・
    レガート谷はプリセット値に固定される(対応するCLIオプションが無いため)。
    """
    preset = _PRESETS[style]
    openness = OpennessParams(
        open_lo=preset["open_lo"],
        open_hi=preset["open_hi"],
        open_max=open_max if open_max is not None else preset["open_max"],
    )
    preset_scale = preset["vowel_scale"]
    vowel_scale = tuple(p * g for p, g in zip(preset_scale[:5], vowel_gain)) + (preset_scale[5],)
    gen = StyleGenParams(
        attack_frames=preset["attack"],
        release_frames=preset["release"],
        coartic_overlap_max=coarticulation if coarticulation is not None else preset["coarticulation"],
        anticipation_frames=anticipation if anticipation is not None else preset["anticipation"],
        min_hold_frames=min_hold if min_hold is not None else preset["min_hold"],
        exaggeration=preset["exaggeration"],
        vowel_scale=vowel_scale,
        triangle_min_frames=preset["triangle_min"],
        vibrato_threshold=preset["vibrato_threshold"],
        vibrato_amp=preset["vibrato_amp"],
        vibrato_period=preset["vibrato_period"],
        legato_valley_shallow=preset["legato_valley_shallow"],
        legato_valley_deep=preset["legato_valley_deep"],
        legato_valley_slope=preset["legato_valley_slope"],
    )
    return openness, gen


def describe_values():
    """--describe の presets 用に、公開引数名→プリセット値の対応を返す。

    載せるのは open_max・coarticulation・anticipation・min_hold のみ(CLIの対応オプションで
    プリセット値を上書きできる引数。開き量レンジ・アタック・リリース・合成誇張係数・母音別倍率・
    三角形下限・伸び表現・レガート谷はプリセット値に固定でCLIから上書きできない。母音別倍率は
    `--vowel-gain` で乗算による微調整はできるが上書きではないためここには載せない)。
    """
    return {
        name: {
            "open_max": preset["open_max"],
            "coarticulation": preset["coarticulation"],
            "anticipation": preset["anticipation"],
            "min_hold": preset["min_hold"],
        }
        for name, preset in _PRESETS.items()
    }
