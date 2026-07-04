"""S-1 認識ゲートの参照ラベル処理。

参照ラベルの音素記号を採点用カテゴリ(母音 a/i/u/e/o・子音 c・無音/息 sil)へ写像する。想定外記号は
黙って捨てず `UnknownReferenceSymbolError` で停止する。モノフォンラベル形式(秒単位・HTK 100ns単位)の
パーサも提供する。
"""

from dataclasses import dataclass
from pathlib import Path

# モノフォンラベルのHTK 100ns単位を秒へ変換する係数。
_HTK_100NS_UNITS_PER_SECOND = 1e7

_VOWEL_SYMBOLS = frozenset({"a", "i", "u", "e", "o"})
_SILENCE_SYMBOLS = frozenset({"pau", "br"})
# cl(促音の閉鎖)・N(撥音)・その他の全子音。sy/ty/zy に相当する音は sh/ch/j で表記されるため
# 別記号として含めない。
_CONSONANT_SYMBOLS = frozenset({
    "b", "by", "ch", "cl", "d", "f", "g", "gy", "h", "hy", "j", "k", "ky",
    "m", "my", "n", "N", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts",
    "v", "w", "y", "z",
})
_EXCLUDED_SYMBOLS = frozenset({"xx"})


class UnknownReferenceSymbolError(Exception):
    """参照ラベルの写像表に無い想定外の音素記号(写像表を補う必要がある)。"""


class MonophoneLabelFormatError(Exception):
    """モノフォンラベルファイルの行が「開始 終了 音素」の3列形式でない(原因特定用にファイルパスと
    行内容を含む)。"""


@dataclass
class CategorySegment:
    """参照ラベルの1区間(採点用カテゴリ)。

    `category` は `reference_symbol_to_category` の戻り値(母音 a/i/u/e/o・子音 c・無音/息 sil、
    または xx による除外印の None)。
    """

    category: str | None
    start_sec: float
    end_sec: float


def reference_symbol_to_category(symbol: str) -> str | None:
    """参照ラベルの音素記号を採点用カテゴリへ写像する。

    母音は a/i/u/e/o、pau・br は sil、子音は c を返す。xx(未定義区間)は採点から除外する
    印として None を返す。写像表に無い記号は `UnknownReferenceSymbolError` で停止する。
    """
    if symbol in _VOWEL_SYMBOLS:
        return symbol
    if symbol in _SILENCE_SYMBOLS:
        return "sil"
    if symbol in _CONSONANT_SYMBOLS:
        return "c"
    if symbol in _EXCLUDED_SYMBOLS:
        return None
    raise UnknownReferenceSymbolError(
        f"参照ラベルの写像表に無い音素記号です: {symbol!r}(写像表を補ってください)"
    )


def _parse_monophone_label_lines(
    path: Path, lines: list[str], time_scale: float
) -> list[CategorySegment]:
    """「開始 終了 音素」の3列形式(空行は無視)を、時刻を time_scale で除して秒へ変換しつつ
    カテゴリ区間列へ変換する。記号→カテゴリの写像は reference_symbol_to_category に
    委譲し、ここでは独自実装しない。"""
    segments = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise MonophoneLabelFormatError(
                f"{path}: 「開始 終了 音素」の3列形式ではない行です: {line!r}"
            )
        start_str, end_str, symbol = fields
        segments.append(
            CategorySegment(
                category=reference_symbol_to_category(symbol),
                start_sec=float(start_str) / time_scale,
                end_sec=float(end_str) / time_scale,
            )
        )
    return segments


def parse_seconds_monophone_label(path: str | Path) -> list[CategorySegment]:
    """モノフォンラベル(開始 終了 音素。時刻は秒)を読み、カテゴリ区間列へ変換する。"""
    path = Path(path)
    return _parse_monophone_label_lines(path, path.read_text(encoding="utf-8").splitlines(), time_scale=1.0)


def parse_htk100ns_monophone_label(path: str | Path) -> list[CategorySegment]:
    """モノフォンラベル(開始 終了 音素。時刻はHTK 100ns単位の整数)を読み、カテゴリ区間列へ
    変換する。"""
    path = Path(path)
    return _parse_monophone_label_lines(
        path, path.read_text(encoding="utf-8").splitlines(), time_scale=_HTK_100NS_UNITS_PER_SECOND
    )
