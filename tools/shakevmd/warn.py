"""構造化警告(shakevmd.md §12.3)。

警告を自由文字列でなく (code, message, section) の構造で持つ。機械モードでは warning
イベント {type, code, message, section} として送出し、非機械モードでは人間向けに1行表示する。
- code: 機械利用側の分岐に使う安定 id。shakevmd 由来は snake_case、ライブラリ(vmd.io)由来は
  ライブラリの形式(ハイフン区切り)をそのまま透過する。
- message: 人間向けの文言(元の自由文字列の文言を保持)。
- section: 対象セクション名の並び(該当が無ければ None)。frozen で重複畳み込み(set)に使えるよう
  tuple で持ち、機械モード送出時に配列へ変換する。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShakeWarning:
    code: str
    message: str
    section: tuple[str, ...] | None = None
