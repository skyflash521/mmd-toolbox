import pytest
from changelog_format import FormatError, parse_sections

VALID = """# 変更履歴

## 1.1.0 - 2026-02-01

### 追加

- 新機能

## 1.0.0 - 2026-01-01

最初の公開。
"""


def _parse(tmp_path, text):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(text, encoding="utf-8")
    return parse_sections(path)


def test_sections_are_returned_newest_first_with_stripped_bodies(tmp_path):
    sections = _parse(tmp_path, VALID)
    assert list(sections) == ["1.1.0", "1.0.0"]
    assert sections["1.1.0"] == "### 追加\n\n- 新機能"
    assert sections["1.0.0"] == "最初の公開。"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "## 1.0.0 - 2026-01-01\n\n本文\n",
        "# 別の題\n\n## 1.0.0 - 2026-01-01\n\n本文\n",
        "# 変更履歴\n",
        "# 変更履歴\n\n前置きの本文\n\n## 1.0.0 - 2026-01-01\n\n本文\n",
        "# 変更履歴\n\n## 1.0 - 2026-01-01\n\n本文\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-1-1\n\n本文\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-01-01\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-01-01\n\n### 追加\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-01-01\n\n### 雑多\n\n- 項目\n",
        "# 変更履歴\n\n### 追加\n\n- 項目\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-01-01\n\n# 想定外\n\n本文\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-01-01\n\n### 追加\n\n### 修正\n\n- 項目\n",
        "# 変更履歴\n\n## 1.0.0 - 2026-01-01\n\n### 追加\n\n- 項目\n\n### 修正\n",
    ],
    ids=[
        "empty-file",
        "version-before-title",
        "wrong-title",
        "no-version",
        "text-before-first-version",
        "version-not-three-parts",
        "date-not-zero-padded",
        "empty-body",
        "category-only-body",
        "unknown-category",
        "category-outside-version",
        "unexpected-heading",
        "empty-category-in-middle",
        "empty-category-at-end",
    ],
)
def test_format_violations_raise_format_error(tmp_path, text):
    with pytest.raises(FormatError):
        _parse(tmp_path, text)


@pytest.mark.parametrize("second", ["1.1.0", "1.2.0"], ids=["same-version", "ascending"])
def test_versions_must_be_strictly_descending(tmp_path, second):
    text = f"# 変更履歴\n\n## 1.1.0 - 2026-02-01\n\n本文\n\n## {second} - 2026-01-01\n\n本文\n"
    with pytest.raises(FormatError):
        _parse(tmp_path, text)


def test_versions_are_compared_numerically_not_lexically(tmp_path):
    text = "# 変更履歴\n\n## 1.10.0 - 2026-02-01\n\n本文\n\n## 1.9.0 - 2026-01-01\n\n本文\n"
    assert list(_parse(tmp_path, text)) == ["1.10.0", "1.9.0"]
