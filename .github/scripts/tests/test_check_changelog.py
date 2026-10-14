import sys

import check_changelog
import pytest

CHANGELOG = "# 変更履歴\n\n## {version} - 2026-01-01\n\n本文\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('__version__ = "1.2.3"\n', "1.2.3"),
        ('"""doc"""\n__version__="0.1.0"\n', "0.1.0"),
        ("x = 1\n", None),
        (None, None),
    ],
)
def test_version_of(text, expected):
    assert check_changelog.version_of(text) == expected


def _make_tool(root, name, version, changelog=None, published=True):
    tool = root / "tools" / name
    tool.mkdir(parents=True)
    (tool / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    if published:
        (tool / "README.md").write_text("# readme\n", encoding="utf-8")
    if changelog is not None:
        (tool / "CHANGELOG.md").write_text(changelog, encoding="utf-8")


def _exit_code(monkeypatch, tmp_path, base, main_versions=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_changelog.py", "--base", base])
    monkeypatch.setattr(check_changelog, "version_on_main", lambda tool: (main_versions or {}).get(tool))
    try:
        check_changelog.main()
    except SystemExit as exc:
        return exc.code
    return 0


def test_release_target_with_matching_top_section_passes(monkeypatch, tmp_path, capsys):
    _make_tool(tmp_path, "alpha", "1.1.0", CHANGELOG.format(version="1.1.0"))
    assert _exit_code(monkeypatch, tmp_path, "main", {"alpha": "1.0.0"}) == 0
    assert "リリース対象 alpha v1.1.0: CHANGELOG 検査 OK" in capsys.readouterr().out


def test_release_target_without_changelog_fails(monkeypatch, tmp_path, capsys):
    _make_tool(tmp_path, "alpha", "1.1.0")
    assert _exit_code(monkeypatch, tmp_path, "main", {"alpha": "1.0.0"}) == 1
    assert "リリース対象 alpha v1.1.0" in capsys.readouterr().out


def test_release_target_missing_the_version_section_fails(monkeypatch, tmp_path, capsys):
    _make_tool(tmp_path, "alpha", "1.1.0", CHANGELOG.format(version="1.0.0"))
    assert _exit_code(monkeypatch, tmp_path, "main", {"alpha": "1.0.0"}) == 1
    assert "にバージョン 1.1.0 の節が無い" in capsys.readouterr().out


def test_release_target_whose_top_section_is_newer_fails(monkeypatch, tmp_path, capsys):
    text = "# 変更履歴\n\n## 1.2.0 - 2026-02-01\n\n本文\n\n## 1.1.0 - 2026-01-01\n\n本文\n"
    _make_tool(tmp_path, "alpha", "1.1.0", text)
    assert _exit_code(monkeypatch, tmp_path, "main", {"alpha": "1.0.0"}) == 1
    assert "先頭の節が 1.1.0 でない" in capsys.readouterr().out


def test_unchanged_version_is_not_a_release_target(monkeypatch, tmp_path):
    _make_tool(tmp_path, "alpha", "1.0.0")
    assert _exit_code(monkeypatch, tmp_path, "main", {"alpha": "1.0.0"}) == 0


def test_version_sections_are_not_checked_for_other_base_branches(monkeypatch, tmp_path):
    _make_tool(tmp_path, "alpha", "1.1.0")
    assert _exit_code(monkeypatch, tmp_path, "develop", {"alpha": "1.0.0"}) == 0


def test_malformed_changelog_fails_regardless_of_base(monkeypatch, tmp_path, capsys):
    _make_tool(tmp_path, "alpha", "1.0.0", "# 別の題\n")
    assert _exit_code(monkeypatch, tmp_path, "develop") == 1
    assert "alpha" in capsys.readouterr().out


def test_unreadable_version_fails_for_published_tool(monkeypatch, tmp_path, capsys):
    _make_tool(tmp_path, "alpha", "1.0.0")
    (tmp_path / "tools" / "alpha" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    assert _exit_code(monkeypatch, tmp_path, "main") == 1
    assert "__version__ を読めない" in capsys.readouterr().out


def test_tool_without_readme_is_not_a_release_target(monkeypatch, tmp_path):
    _make_tool(tmp_path, "internal", "2.0.0", published=False)
    assert _exit_code(monkeypatch, tmp_path, "main", {"internal": "1.0.0"}) == 0
