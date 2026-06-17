"""sparsevmd パッケージ土台のスモークテスト(sparsevmd.md §8)。

パッケージがインポート可能で、コア(presets)と console script のエントリ(cli.main)が
公開されていることを確認する。
"""


def test_package_imports():
    import sparsevmd

    # namespace package(__init__.py 不在)ではなく実パッケージであること(§8)。
    assert sparsevmd.__file__ is not None


def test_presets_module_available():
    from sparsevmd import presets

    assert hasattr(presets, "resolve_tolerances")
    assert hasattr(presets, "PRESET_NAMES")


def test_console_script_entry_point_callable():
    # pyproject の [project.scripts] sparsevmd = "sparsevmd.cli:main" の対象が解決可能。
    from sparsevmd.cli import main

    assert callable(main)
