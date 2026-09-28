import re
import tomllib
from pathlib import Path

import song2vpr
from song2vpr import cli


def test_package_is_a_regular_package_with_a_semver_version():
    assert song2vpr.__file__ is not None
    assert re.fullmatch(r"\d+\.\d+\.\d+", song2vpr.__version__)


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[3]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    assert _pyproject()["project"]["scripts"]["song2vpr"] == "song2vpr.cli:main"


def test_package_included_in_setuptools_find():
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "song2vpr*" in include


def test_tests_included_in_pytest_testpaths():
    testpaths = _pyproject()["tool"]["pytest"]["ini_options"]["testpaths"]
    assert "tools/song2vpr" in testpaths
