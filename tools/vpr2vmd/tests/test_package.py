"""vpr2vmd パッケージ土台のスモークテスト。

独立 CLI `vpr2vmd` として提供する契約を静的に確認する: パッケージがインポート可能で、
console script のエントリ(cli.main)が公開・登録され、配布物に含まれること。
"""

import tomllib
from pathlib import Path

import vpr2vmd
from vpr2vmd import cli


def test_package_imports():
    # namespace package ではなく実パッケージであること(__init__.py を持つ)。
    assert vpr2vmd.__file__ is not None


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[3]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    # pyproject の [project.scripts] に vpr2vmd = "vpr2vmd.cli:main" が登録されていること。
    assert _pyproject()["project"]["scripts"]["vpr2vmd"] == "vpr2vmd.cli:main"


def test_package_included_in_setuptools_find():
    # 配布物に vpr2vmd を含めるため packages.find の include に vpr2vmd* があること。
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "vpr2vmd*" in include
