"""ボーン名から種別を推定する(mocapvmd.md §4.1)。

具体度の高い種別を優先し、最初に一致した種別を返す。標準ボーン名は包含関係を持ち、
例えば「右足ＩＫ」は legs(アスタリスク足アスタリスク)と foot_ik(アスタリスク足ＩＫ
アスタリスク)の両方に一致するため、foot_ik / toe_ik を legs より先に判定する。
分類できない名前は unknown とし、後段で除外せず保守的に処理する。

照合は名前とパターンをともに小文字化して行う。英語名の大文字小文字を無視するためで、
日本語・全角文字は小文字化の影響を受けない。半角IK・全角ＩＫ・英語名を対象に含む。
"""

from fnmatch import fnmatchcase

# (種別, glob パターン群)を具体度の高い順に並べる。最初に一致した種別を採る。
_RULES = (
    ("toe_ik", ("*つま先ik*", "*つま先ＩＫ*", "*toe ik*")),
    ("foot_ik", ("*足ik*", "*足ＩＫ*", "*foot ik*", "*leg ik*")),
    ("center", ("センター", "グルーブ", "center", "groove")),
    ("root", ("全ての親", "root")),
    ("torso", ("上半身*", "下半身", "首", "頭", "upper body*", "lower body", "neck", "head")),
    ("arms", ("*肩*", "*腕*", "*ひじ*", "*肘*", "*手首*",
              "*shoulder*", "*arm*", "*elbow*", "*wrist*")),
    ("fingers", ("*指*", "*finger*", "*thumb*", "*index*", "*middle*",
                 "*ring*", "*pinky*", "*little*")),
    ("legs", ("*足*", "*脚*", "*ひざ*", "*膝*", "*つま先*",
              "*leg*", "*knee*", "*ankle*", "*toe*")),
)

# 分類が返しうる種別の全集合(unknown を含む)。順序は種別表(§4.1)に対応する。
CATEGORIES = ("root", "center", "torso", "arms", "fingers", "legs", "foot_ik", "toe_ik", "unknown")


def classify(name):
    """ボーン名 name の種別を返す。どのパターンにも一致しなければ "unknown"。"""
    low = name.lower()
    for category, patterns in _RULES:
        if any(fnmatchcase(low, p.lower()) for p in patterns):
            return category
    return "unknown"
