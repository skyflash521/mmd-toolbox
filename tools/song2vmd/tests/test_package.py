"""song2vmd パッケージ土台のスモークテスト。

独立 CLI `song2vmd` として提供する契約を静的に確認する: パッケージがインポート可能で、
console script のエントリ(cli.main)が公開・登録され、配布物に含まれること。
"""

import re
import tomllib
from pathlib import Path

import song2vmd
from song2vmd import cli


def test_package_imports():
    # namespace package(__init__.py 不在)ではなく実パッケージであること。
    assert song2vmd.__file__ is not None
    # バージョンは上がっていくので、番号は固定せず形式だけを見る。
    assert re.fullmatch(r"\d+\.\d+\.\d+", song2vmd.__version__)


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[3]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    # pyproject の [project.scripts] に song2vmd = "song2vmd.cli:main" が登録されていること。
    assert _pyproject()["project"]["scripts"]["song2vmd"] == "song2vmd.cli:main"


def test_package_included_in_setuptools_find():
    # 配布物に song2vmd を含めるため packages.find の include に song2vmd* があること。
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "song2vmd*" in include
