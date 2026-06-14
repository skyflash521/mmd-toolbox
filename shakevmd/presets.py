"""shakevmd プリセット定義(shakevmd.md §2.7, §8)。

各プリセットは CLI 公開引数の束(ジャンル別の既定値)。`--preset NAME` 指定時に
未指定引数の既定として適用し、個別引数の明示指定が優先する(§2.7)。

内蔵パラメーター(オクターブ構成・静止/移動プロファイル)のプリセット調整と
walking の歩調成分ノイズ混合は、コア拡張を要するため後続サブステップへ繰延。

実装は後続(現状スタブ)。
"""

# 公開プリセット名(§2.7)。値(公開引数の束)は実装で定義する。
PRESET_NAMES = ("handheld", "telephoto", "walking", "earthquake")


def get_preset(name):
    """プリセット名から公開引数の束(dict)を返す。未知名は KeyError。"""
    raise NotImplementedError
