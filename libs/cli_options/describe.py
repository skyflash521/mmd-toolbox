"""自己記述で公開するオプション定義の配列を、parser の登録内容から組み立てる。"""

from .validators import CompoundValidator


def describe_options(parser, type_table):
    """自己記述の options を parser 定義から機械導出する。順序は add_argument 順。

    各要素は {name, type, constraint, default, help}(キー5つ)。type_table に無い引数
    (メタ操作・モード指定)は除外する。真偽フラグの否定形は肯定形の長形式で既に載るのでスキップする。
    """
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in type_table:
            continue
        type_, constraint = type_table[dest]
        if type_ is None:
            # 検証子を持つ引数は、型も範囲もその検証子が持つものをそのまま公開する
            # (手書きの複製を置かない)。
            validator = action.type
            type_ = "compound" if isinstance(validator, CompoundValidator) else validator.value_type
            constraint = validator.constraint
        if not action.option_strings:
            name = dest
        else:
            # 肯定形の長形式を採る。それを持たない action(否定形だけの登録・短形式だけの登録)は
            # 名前を採れないのでスキップする。否定形は肯定形が同じ格納先で既に載る。
            positive = [s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")]
            if not positive:
                continue
            name = positive[0]
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": action.default,
            "help": action.help,
        })
    return options
