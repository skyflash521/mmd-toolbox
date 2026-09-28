import re
import tomllib
from pathlib import Path

import song2vmd
from song2vmd import cli


def test_package_is_a_regular_package_with_semver_version():
    assert song2vmd.__file__ is not None
    assert re.fullmatch(r"\d+\.\d+\.\d+", song2vmd.__version__)


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[3]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    assert _pyproject()["project"]["scripts"]["song2vmd"] == "song2vmd.cli:main"


def test_package_included_in_setuptools_find():
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "song2vmd*" in include
