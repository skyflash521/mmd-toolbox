import argparse
import math
from collections.abc import Callable, Sequence
from typing import Literal


class RangeValidator:
    def __init__(self, *, value_type: Literal["int", "float"], minimum=None, maximum=None,
                 exclusive_min=False):
        self.value_type = value_type
        self.constraint = {"min": minimum, "max": maximum, "exclusive_min": exclusive_min}

    def __call__(self, text):
        try:
            value = int(text) if self.value_type == "int" else float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{self._unit()}が必要: {text!r}") from None
        # NaN は大小比較がすべて偽になり、範囲判定では弾けない。
        if self.value_type == "float" and not math.isfinite(value):
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
    def __init__(
        self,
        *,
        format: str,
        elements: Sequence[tuple[str, RangeValidator]],
        relation: tuple[str, Callable[[list[int | float]], bool]] | None = None,
        separator: str = ":",
    ):
        """format が separator で区切って示す要素の数と並びは、elements と一致させて渡す。

        relation は (違反時の理由, 判定関数)。判定関数は elements の順に並べた要素値を受け取り、成立なら真を返す。
        relation の条件は constraint に載らないので、この検証子を使う引数の help に書く。
        """
        self.format = format
        self.elements = elements
        self.relation = relation
        self.separator = separator

    @property
    def constraint(self):
        return {
            "format": self.format,
            "fields": [{"name": name, "type": element.value_type, **element.constraint}
                       for name, element in self.elements],
        }

    def __call__(self, text) -> tuple[int | float, ...]:
        """戻り値は elements の順に並べた要素値。"""
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
