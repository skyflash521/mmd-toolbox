"""sparsevmd パッケージ土台のスモークテスト(sparsevmd.md §8)。

パッケージがインポート可能で、コア(presets)が公開されていることを確認する。
CLI(cli.main)は後続ステップで追加するため、ここでは検査しない。
"""


def test_package_imports():
    import sparsevmd

    # namespace package(__init__.py 不在)ではなく実パッケージであること(§8)。
    assert sparsevmd.__file__ is not None


def test_presets_module_available():
    from sparsevmd import presets

    assert hasattr(presets, "resolve_tolerances")
    assert hasattr(presets, "PRESET_NAMES")
