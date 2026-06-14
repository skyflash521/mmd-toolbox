"""shakevmd プリセット定義(shakevmd.md §2.7, §8)。

各プリセットは CLI 公開引数の束(ジャンル別の既定値)。`--preset NAME` 指定時に
未指定引数の既定として適用し、個別引数の明示指定が優先する(§2.7)。

内蔵パラメーター(オクターブ構成・静止/移動プロファイル)のプリセット調整と
walking の歩調成分ノイズ混合は、コア拡張を要するため後続サブステップへ繰延。
本段階では各プリセットを「全公開引数の束」として定義する(値は設計時の暫定値で、
実利用での微調整は後続)。
"""

# 公開プリセット名(§2.7)。
PRESET_NAMES = ("handheld", "telephoto", "walking", "earthquake")

# 各プリセット = 全公開引数(amp_rot/amp_pos/rot_weights/freq/motion_scale/settle/cut_threshold)の束。
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
    # 歩き: 上下動の大きい周期的な揺れ。歩調周波数寄り、ピッチ強め(歩調成分混合は後続)。
    "walking": {
        "amp_rot": 1.0, "amp_pos": 0.18, "rot_weights": (1.3, 0.7, 0.3),
        "freq": 2.0, "motion_scale": 0.4, "settle": 0.3, "cut_threshold": (5.0, 20.0),
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
