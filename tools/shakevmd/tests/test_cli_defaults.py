import inspect

from shakevmd import cli
from shakevmd.bake import bake
from vmd import io
from vmd.types import CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6

_BAKE_DEFAULTS = {
    name: p.default for name, p in inspect.signature(bake).parameters.items()
    if p.default is not inspect.Parameter.empty
}


def _write_input(path):
    keys = [
        CameraKey(0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), LINEAR, 30, 0),
        CameraKey(30, -25.0, (10.0, 5.0, 2.0), (0.2, 0.1, 0.0), LINEAR, 30, 0),
        CameraKey(60, -20.0, (20.0, 0.0, -3.0), (-0.1, 0.3, 0.05), LINEAR, 30, 0),
    ]
    io.write_file(VmdDocument(camera=keys), str(path))
    return str(path)


def test_cut_threshold_passed_to_bake_without_options_equals_bake_signature_default(tmp_path, monkeypatch):
    received = {}

    def spy(camera_keys, *args, **kwargs):
        received.update(kwargs)
        return bake(camera_keys, *args, **kwargs)

    monkeypatch.setattr(cli, "bake", spy)
    assert cli.main([_write_input(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--no-smooth"]) == 0
    for name in ("cut_pos_threshold", "cut_rot_threshold"):
        assert received[name] == _BAKE_DEFAULTS[name], name
