"""tools/<ツール>/CHANGELOG.md の共通パーサ。

PR 時の検査(check_changelog.py)とタグ push 時の Release 作成(create_release.py)が同じ
解析を使うことで、検査に通った CHANGELOG からは必ず Release 本文を抽出できることを保証する。
"""
import re
from pathlib import Path

_TITLE = "# 変更履歴"
_SECTION_RE = re.compile(r"^## (\d+\.\d+\.\d+) - \d{4}-\d{2}-\d{2}$")
_CATEGORY_RE = re.compile(r"^### (破壊的変更|追加|変更|修正)$")


class FormatError(ValueError):
    """CHANGELOG が規約の形式に適合しない。"""


def parse_sections(path: Path) -> dict[str, str]:
    """CHANGELOG を検証し、{バージョン: 節本文} を新しい順で返す。逸脱は FormatError。"""
    sections: dict[str, str] = {}
    current: str | None = None
    body: list[str] = []
    title_seen = False
    prev_key: tuple[int, ...] | None = None

    def close_section() -> None:
        if current is None:
            return
        if not any(text.strip() and not text.startswith("###") for text in body):
            raise FormatError(f"バージョン {current} の節の本文が空")
        category = None
        category_has_content = True
        for text in body:
            if text.startswith("###"):
                if not category_has_content:
                    raise FormatError(f"バージョン {current} の「{category}」に変更の行が無い(該当のない節は置かない)")
                category = text
                category_has_content = False
            elif text.strip():
                category_has_content = True
        if not category_has_content:
            raise FormatError(f"バージョン {current} の「{category}」に変更の行が無い(該当のない節は置かない)")
        sections[current] = "\n".join(body).strip()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## "):
            if not title_seen:
                raise FormatError(f"{lineno}行目: 「{_TITLE}」より前にバージョン見出しがある")
            matched = _SECTION_RE.match(line)
            if not matched:
                raise FormatError(
                    f"{lineno}行目: バージョン見出しが「## <MAJOR.MINOR.PATCH> - <YYYY-MM-DD>」の形でない: {line}"
                )
            close_section()
            current = matched.group(1)
            key = tuple(int(n) for n in current.split("."))
            if prev_key is not None and key >= prev_key:
                raise FormatError(f"{lineno}行目: バージョン {current} が降順(新しいバージョンが上)になっていない")
            prev_key = key
            body = []
        elif line.startswith("###"):
            if current is None:
                raise FormatError(f"{lineno}行目: カテゴリ見出しがバージョンの節の外にある: {line}")
            if not _CATEGORY_RE.match(line):
                raise FormatError(
                    f"{lineno}行目: カテゴリ見出しは 破壊的変更/追加/変更/修正 のどれか: {line}"
                )
            body.append(line)
        elif line.startswith("#"):
            if title_seen:
                raise FormatError(f"{lineno}行目: 想定外の見出し: {line}")
            if line != _TITLE:
                raise FormatError(f"{lineno}行目: 先頭見出しは「{_TITLE}」: {line}")
            title_seen = True
        elif current is not None:
            body.append(line)
        elif line.strip():
            raise FormatError(f"{lineno}行目: バージョン見出しの外に本文がある: {line}")

    if not title_seen:
        raise FormatError(f"先頭見出し「{_TITLE}」が無い")
    close_section()
    if not sections:
        raise FormatError("バージョンの節が1つも無い")
    return sections
