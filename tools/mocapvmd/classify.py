from fnmatch import fnmatchcase

_RULES_MOST_SPECIFIC_FIRST = (
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

CATEGORIES = ("root", "center", "torso", "arms", "fingers", "legs", "foot_ik", "toe_ik", "unknown")


def classify(name: str) -> str:
    low = name.lower()
    for category, patterns in _RULES_MOST_SPECIFIC_FIRST:
        if any(fnmatchcase(low, p.lower()) for p in patterns):
            return category
    return "unknown"
