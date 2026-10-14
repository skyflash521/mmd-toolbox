import subprocess
import sys

import create_release
import pytest

CHANGELOG = "# 変更履歴\n\n## 1.1.0 - 2026-02-01\n\n### 追加\n\n- 新機能\n\n## 1.0.0 - 2026-01-01\n\n最初の公開。\n"


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: recorded.append((cmd, kwargs)))
    return recorded


def _make_tool(root, name="alpha", changelog=CHANGELOG, published=True):
    tool = root / "tools" / name
    tool.mkdir(parents=True)
    if published:
        (tool / "README.md").write_text("# readme\n", encoding="utf-8")
    if changelog is not None:
        (tool / "CHANGELOG.md").write_text(changelog, encoding="utf-8")


def _run(monkeypatch, tmp_path, tag):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["create_release.py", tag])
    create_release.main()


def test_release_is_created_with_the_matching_section_as_notes(monkeypatch, tmp_path, calls):
    _make_tool(tmp_path)
    _run(monkeypatch, tmp_path, "alpha/v1.1.0")
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == ["gh", "release", "create", "alpha/v1.1.0", "--title", "alpha v1.1.0",
                   "--notes-file", "release_notes.md"]
    assert kwargs["check"] is True
    assert (tmp_path / "release_notes.md").read_text(encoding="utf-8") == "### 追加\n\n- 新機能\n"


def test_older_version_uses_its_own_section(monkeypatch, tmp_path, calls):
    _make_tool(tmp_path)
    _run(monkeypatch, tmp_path, "alpha/v1.0.0")
    assert (tmp_path / "release_notes.md").read_text(encoding="utf-8") == "最初の公開。\n"


@pytest.mark.parametrize("tag", ["alpha", "/v1.0.0", "alpha/v"], ids=["no-version", "no-tool", "empty-version"])
def test_malformed_tag_name_fails_without_release(monkeypatch, tmp_path, calls, tag):
    _make_tool(tmp_path)
    with pytest.raises(SystemExit) as raised:
        _run(monkeypatch, tmp_path, tag)
    assert raised.value.code != 0
    assert calls == []


def test_tag_of_a_tool_without_readme_creates_no_release(monkeypatch, tmp_path, calls, capsys):
    _make_tool(tmp_path, "internal", published=False)
    _run(monkeypatch, tmp_path, "internal/v1.0.0")
    assert calls == []
    assert not (tmp_path / "release_notes.md").exists()
    assert "Release を作らない" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("changelog", "tag"),
    [
        (None, "alpha/v1.1.0"),
        ("# 別の題\n", "alpha/v1.1.0"),
        (CHANGELOG, "alpha/v2.0.0"),
    ],
    ids=["no-changelog", "malformed-changelog", "missing-version-section"],
)
def test_changelog_problems_fail_without_release(monkeypatch, tmp_path, calls, changelog, tag):
    _make_tool(tmp_path, changelog=changelog)
    with pytest.raises(SystemExit) as raised:
        _run(monkeypatch, tmp_path, tag)
    assert raised.value.code != 0
    assert calls == []
    assert not (tmp_path / "release_notes.md").exists()
