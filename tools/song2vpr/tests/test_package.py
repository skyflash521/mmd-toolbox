"""song2vpr パッケージ土台のスモークテスト。

独立 CLI `song2vpr` として提供する契約を静的に確認する: パッケージがインポート可能で、
console script のエントリ(cli.main)が公開・登録され、配布物に含まれ、テスト対象パスに入ること。
"""

import tomllib
from pathlib import Path

import pytest

song2vpr = pytest.importorskip("song2vpr", reason="パッケージがまだ無い")
cli = pytest.importorskip("song2vpr.cli", reason="CLI モジュールがまだ無い")


def test_package_imports():
    # namespace package(__init__.py 不在)ではなく実パッケージであること。
    assert song2vpr.__file__ is not None
    assert song2vpr.__version__ == "0.0.1"


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[3]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    # pyproject の [project.scripts] に song2vpr = "song2vpr.cli:main" が登録されていること。
    assert _pyproject()["project"]["scripts"]["song2vpr"] == "song2vpr.cli:main"


def test_package_included_in_setuptools_find():
    # 配布物に song2vpr を含めるため packages.find の include に song2vpr* があること。
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "song2vpr*" in include


def test_tests_included_in_pytest_testpaths():
    # 既定の pytest 実行で song2vpr のテストが収集されること。
    testpaths = _pyproject()["tool"]["pytest"]["ini_options"]["testpaths"]
    assert "tools/song2vpr" in testpaths
