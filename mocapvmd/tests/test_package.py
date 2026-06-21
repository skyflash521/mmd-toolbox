"""mocapvmd パッケージ土台のスモークテスト(mocapvmd.md §1)。

パッケージがインポート可能で、console script のエントリ(cli.main)が
公開・登録されていることを確認する。
"""

import tomllib
from pathlib import Path

import mocapvmd
from mocapvmd import cli


def test_package_imports():
    # namespace package ではなく実パッケージであること(__init__.py を持つ)。
    assert mocapvmd.__file__ is not None


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[2]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    # pyproject の [project.scripts] に mocapvmd = "mocapvmd.cli:main" が登録されていること。
    assert _pyproject()["project"]["scripts"]["mocapvmd"] == "mocapvmd.cli:main"


def test_package_included_in_setuptools_find():
    # 配布物に mocapvmd を含めるため packages.find の include に mocapvmd* があること。
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "mocapvmd*" in include
