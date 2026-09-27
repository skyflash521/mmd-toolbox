import tomllib
from pathlib import Path

import mocapvmd
from mocapvmd import cli


def test_package_is_regular_package_with_init_file():
    assert mocapvmd.__file__ is not None


def test_console_script_entry_point_callable():
    assert callable(cli.main)


def _pyproject():
    root = Path(__file__).resolve().parents[3]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_console_script_registered_in_pyproject():
    assert _pyproject()["project"]["scripts"]["mocapvmd"] == "mocapvmd.cli:main"


def test_package_included_in_setuptools_find():
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "mocapvmd*" in include
