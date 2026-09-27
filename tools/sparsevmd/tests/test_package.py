import importlib

import pytest


def test_package_is_regular_package_not_namespace():
    import sparsevmd

    assert sparsevmd.__file__ is not None


def test_presets_module_available():
    from sparsevmd import presets

    assert hasattr(presets, "resolve_tolerances")
    assert hasattr(presets, "PRESET_NAMES")


def test_console_script_entry_point_callable():
    from sparsevmd.cli import main

    assert callable(main)


@pytest.mark.parametrize(
    ("legacy_module", "vmd_module", "symbol"),
    [
        pytest.param("sparsevmd.fit", "vmd.fit", "fit_bezier_curve", id="fit"),
        pytest.param("sparsevmd.reduce", "vmd.reduce", "reduce_bone_track", id="reduce"),
        pytest.param("sparsevmd.cuts", "vmd.cuts", "detect_cuts_camera", id="cuts"),
        pytest.param("sparsevmd.sample", "vmd.sample", "perspective_series", id="sample"),
        pytest.param("sparsevmd.presets", "vmd.reduce", "Tolerances", id="presets"),
    ],
)
def test_legacy_import_path_reexports_vmd_symbol(legacy_module, vmd_module, symbol):
    legacy = importlib.import_module(legacy_module)
    origin = importlib.import_module(vmd_module)
    assert getattr(legacy, symbol) is getattr(origin, symbol)
