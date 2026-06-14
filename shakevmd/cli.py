"""shakevmd CLI(shakevmd.md §2, §9)。

コアの薄いラッパー: 引数解析 → VMD読み(mmd_toolbox.vmd.io)→ bake() → VMD書き。
詳細パラメーター(オクターブ構成・persistence・プロファイル・手動カット等)は公開しない(§8)。

実装は後続(現状スタブ)。
"""


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(§9: 0/1/2/3)。"""
    raise NotImplementedError
