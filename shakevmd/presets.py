"""shakevmd プリセット定義(shakevmd.md §2.7, §8)。

各プリセットは公開引数の束(ジャンル別の既定値)に加え、CLI 非公開の内蔵パラメーター
(現状は walking の歩調成分 gait_freq/gait_amp)を任意で持つ。`--preset NAME` 指定時、
公開引数は未指定引数の既定として適用し個別引数の明示指定が優先する(§2.7)。内蔵
パラメーターは CLI フラグを持たず、プリセット定義からのみ bake へ渡る(§8)。

オクターブ構成・静止/移動プロファイルのプリセット調整は後続サブステップへ繰延(値は
設計時の暫定値で、実利用での微調整は後続)。
"""

# 公開プリセット名(§2.7)。
PRESET_NAMES = ("handheld", "telephoto", "walking", "earthquake")

# 内蔵パラメーター名(CLI 非公開、プリセット/コアAPIのみ。§8)。bake() が受ける引数名と一致させ、
# cli が明示渡しする引数とは重複させない。cli はこの名前のキーがプリセットにあれば bake へ転送する。
# 歩調成分・静止/移動プロファイル(オクターブ重み構成)・settle収束時間・素朴な角度加算モード。
INTERNAL_PARAM_NAMES = (
    "gait_freq", "gait_amp", "still_profile", "moving_profile", "settle_time", "naive_rotation",
)

# 各プリセット = 公開引数(amp_rot/amp_pos/rot_weights/freq/motion_scale/settle/cut_threshold)の束。
# walking は加えて内蔵の歩調成分(gait_freq/gait_amp)を持つ(§95)。
_PRESETS = {
    # 手持ち撮影: 既定に近い自然な揺れ。
    "handheld": {
        "amp_rot": 0.9, "amp_pos": 0.06, "rot_weights": (1.0, 1.0, 0.35),
        "freq": 1.3, "motion_scale": 0.6, "settle": 0.4, "cut_threshold": (5.0, 20.0),
    },
    # 望遠: 画角が狭く角度揺れが拡大、低周波のゆったりしたドリフト主体。位置揺れは控えめ。
    "telephoto": {
        "amp_rot": 1.8, "amp_pos": 0.015, "rot_weights": (1.0, 1.0, 0.2),
        "freq": 0.7, "motion_scale": 0.8, "settle": 0.5, "cut_threshold": (5.0, 20.0),
    },
    # 歩き: 上下動の大きい周期的な揺れ。乱数ノイズに歩調成分を混合(左右=gait_freq、上下=2倍)。
    # gait_freq=1.0 → 上下のバウンドは 2.0Hz(歩行の歩調相当)。値は暫定(後続で微調整)。
    "walking": {
        "amp_rot": 1.0, "amp_pos": 0.18, "rot_weights": (1.3, 0.7, 0.3),
        "freq": 2.0, "motion_scale": 0.4, "settle": 0.3, "cut_threshold": (5.0, 20.0),
        "gait_freq": 1.0, "gait_amp": 0.1,
    },
    # 地震: 激しく高周波・大振幅。
    "earthquake": {
        "amp_rot": 3.5, "amp_pos": 0.5, "rot_weights": (1.0, 1.0, 0.8),
        "freq": 5.0, "motion_scale": 0.2, "settle": 0.6, "cut_threshold": (5.0, 20.0),
    },
}


def get_preset(name):
    """プリセット名から公開引数の束(dict)を返す。未知名は KeyError。"""
    return dict(_PRESETS[name])
