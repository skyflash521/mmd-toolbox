"""引数値の受理範囲を1か所で決め、検証と自己記述の双方へ供給する検証子。"""

import argparse
import math


class RangeValidator:
    """範囲付きの数値引数の検証子。argparse の type と自己記述の constraint を1つの範囲から導く。

    受理判定はこの範囲そのもので行うので、公開する制約と実際に通る値が別管理にならない。
    解析できない値も自分で日本語の理由へ変換する(argparse は型が関数でないと理由の代わりに
    オブジェクトの repr を出すため、そのままでは実行ごとに変わる文字列が報告に載る)。
    """

    def __init__(self, *, value_type, minimum=None, maximum=None, exclusive_min=False):
        self.value_type = value_type  # "int" | "float"
        self.constraint = {"min": minimum, "max": maximum, "exclusive_min": exclusive_min}

    def __call__(self, text):
        try:
            value = int(text) if self.value_type == "int" else float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{self._unit()}が必要: {text!r}") from None
        if self.value_type == "float" and not math.isfinite(value):
            # 無限大・非数は大小比較をすり抜けるので、範囲判定の前に弾く。
            raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
        minimum, maximum = self.constraint["min"], self.constraint["max"]
        too_small = minimum is not None and (
            value <= minimum if self.constraint["exclusive_min"] else value < minimum)
        if too_small or (maximum is not None and value > maximum):
            raise argparse.ArgumentTypeError(f"{self._range_text()}が必要: {text!r}")
        return value

    def _unit(self):
        return "整数" if self.value_type == "int" else "数値"

    def _range_text(self):
        minimum, maximum = self.constraint["min"], self.constraint["max"]
        unit = self._unit()
        exclusive_min = self.constraint["exclusive_min"]
        if minimum is not None and maximum is not None:
            if exclusive_min:
                return f"{minimum} より大きく {maximum} 以下の{unit}"
            return f"{minimum}〜{maximum} の範囲の{unit}"
        if minimum is not None:
            if exclusive_min:
                return f"{minimum} より大きい{unit}"
            return f"{minimum} 以上の{unit}"
        return f"{maximum} 以下の{unit}"


class CompoundValidator:
    """複合トークンの引数の検証子。要素検証子の列と書式から自己記述の constraint を導く。

    要素間の関係の制約は constraint の形に載る場所が無いため、ここでは検証だけ行い公開しない
    (利用者へは help 文で示す)。文言に引数名は入れない(argparse が理由の前へ引数名を付けるので、
    入れると二重になる)。
    """

    def __init__(self, *, format, elements, relation=None, separator=":"):
        self.format = format
        self.elements = elements  # [(要素名, RangeValidator), ...]
        self.relation = relation  # (説明, 判定関数) または None
        self.separator = separator

    @property
    def constraint(self):
        return {
            "format": self.format,
            "fields": [{"name": name, "type": element.value_type, **element.constraint}
                       for name, element in self.elements],
        }

    def __call__(self, text):
        parts = text.split(self.separator)
        if len(parts) != len(self.elements):
            raise argparse.ArgumentTypeError(
                f"{self.format} の{len(self.elements)}要素が必要: {text!r}")
        values = []
        for part, (name, element) in zip(parts, self.elements, strict=True):
            try:
                values.append(element(part))
            except argparse.ArgumentTypeError as e:
                raise argparse.ArgumentTypeError(f"{name} は {e}") from None
        if self.relation is not None:
            description, holds = self.relation
            if not holds(values):
                raise argparse.ArgumentTypeError(f"{description}: {text!r}")
        return tuple(values)
