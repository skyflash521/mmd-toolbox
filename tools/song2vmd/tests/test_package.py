"""song2vmd パッケージ土台のスモークテスト(song2vmd.md)。

パッケージがインポート可能で、console script のエントリ(cli.main)が公開されていることを確認する。
"""

import pytest

pytest.importorskip("song2vmd.cli", reason="impl pending: song2vmd cli")


def test_package_imports():
    import song2vmd

    # namespace package(__init__.py 不在)ではなく実パッケージであること。
    assert song2vmd.__file__ is not None
    assert song2vmd.__version__ == "0.0.1"


def test_console_script_entry_point_callable():
    # song2vmd.cli:main(pyproject の [project.scripts] へ登録する対象)が import 解決可能。
    from song2vmd.cli import main

    assert callable(main)
