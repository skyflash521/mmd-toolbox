import inspect
import json

import pytest

from shakevmd import cli
from shakevmd.bake import bake
from vmd import io
from vmd.types import CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6

_BAKE_DEFAULTS = {
    name: p.default for name, p in inspect.signature(bake).parameters.items()
    if p.default is not inspect.Parameter.empty
}

_BAKE_PARAMS_BY_CLI_DEST = {
    "amp_rot": ("amp_rot",),
    "amp_pos": ("amp_pos",),
    "rot_weights": ("rot_weights",),
    "freq": ("freq",),
    "motion_damp": ("motion_damp",),
    "settle": ("settle",),
    "fade": ("fade_sec",),
    "cut_threshold": ("cut_pos_threshold", "cut_rot_threshold"),
    "seed": ("seed",),
    "ranges": ("ranges",),
    "impulses": ("impulses",),
}

_DESCRIBE_NAME_BY_CLI_DEST = {
    "amp_rot": "--amp-rot",
    "amp_pos": "--amp-pos",
    "rot_weights": "--rot-weights",
    "freq": "--freq",
    "motion_damp": "--motion-damp",
    "settle": "--settle",
    "fade": "--fade",
    "cut_threshold": "--cut-threshold",
    "seed": "--seed",
}


def _bake_default(dest):
    values = tuple(_BAKE_DEFAULTS[name] for name in _BAKE_PARAMS_BY_CLI_DEST[dest])
    return values[0] if len(values) == 1 else values


def _as_json_value(value):
    if isinstance(value, (tuple, list)):
        return [_as_json_value(v) for v in value]
    return value


def _write_input(path):
    keys = [
        CameraKey(0, -30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), LINEAR, 30, 0),
        CameraKey(30, -25.0, (10.0, 5.0, 2.0), (0.2, 0.1, 0.0), LINEAR, 30, 0),
        CameraKey(60, -20.0, (20.0, 0.0, -3.0), (-0.1, 0.3, 0.05), LINEAR, 30, 0),
    ]
    io.write_file(VmdDocument(camera=keys), str(path))
    return str(path)


def _describe_defaults(capsysbinary):
    assert cli.main(["--describe"]) == 0
    event = json.loads(capsysbinary.readouterr().out.decode("utf-8").splitlines()[0])
    return {o["name"]: o["default"] for o in event["options"]}


@pytest.mark.parametrize("dest", list(_DESCRIBE_NAME_BY_CLI_DEST))
def test_describe_default_equals_bake_signature_default(capsysbinary, dest):
    defaults = _describe_defaults(capsysbinary)
    assert defaults[_DESCRIBE_NAME_BY_CLI_DEST[dest]] == _as_json_value(_bake_default(dest))


def test_argparse_seed_default_equals_bake_signature_default():
    assert cli._build_parser().parse_args([]).seed == _BAKE_DEFAULTS["seed"]


@pytest.mark.parametrize("dest", list(_BAKE_PARAMS_BY_CLI_DEST))
def test_value_passed_to_bake_without_options_equals_bake_signature_default(tmp_path, monkeypatch, dest):
    received = {}

    def spy(camera_keys, *args, **kwargs):
        received.update(kwargs)
        return bake(camera_keys, *args, **kwargs)

    monkeypatch.setattr(cli, "bake", spy)
    assert cli.main([_write_input(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--no-smooth"]) == 0
    for name in _BAKE_PARAMS_BY_CLI_DEST[dest]:
        assert _as_json_value(received[name]) == _as_json_value(_BAKE_DEFAULTS[name]), name
