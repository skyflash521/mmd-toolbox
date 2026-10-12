import argparse

from .validators import CompoundValidator

_REPEATING_ACTIONS = (argparse._AppendAction, argparse._ExtendAction)


def describe_options(parser, type_table, *, defaults=None):
    """type_table は格納先名から (型名, 制約) への対応で、載っている引数だけを返す。

    型名が None の引数は、型と制約を type に設定した検証子から取る。
    """
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in type_table:
            continue
        type_, constraint = type_table[dest]
        if type_ is None:
            validator = action.type
            type_ = "compound" if isinstance(validator, CompoundValidator) else validator.value_type
            constraint = validator.constraint
        if not action.option_strings:
            name = dest
        else:
            positive_long_forms = [
                s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")]
            if not positive_long_forms:
                continue
            name = positive_long_forms[0]
        repeat = isinstance(action, _REPEATING_ACTIONS)
        if defaults is not None and dest in defaults:
            default = defaults[dest]
        else:
            default = None if repeat else action.default
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": default,
            "help": action.help,
            "repeat": repeat,
        })
    return options
