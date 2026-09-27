import json

from sparsevmd import cli, presets

_GROUPS = ["core", "arms", "legs", "fingers", "ik", "mocap"]
_NONNEG = {"min": 0, "max": None, "exclusive_min": False}
_FOV = {"min": 0.5, "max": None, "exclusive_min": False}
_AT_LEAST_1 = {"min": 1, "max": None, "exclusive_min": False}
_FRAME = {"min": 0, "max": None, "exclusive_min": False}


def _num_field(name, type_, mn):
    return {"name": name, "type": type_, "min": mn, "max": None, "exclusive_min": False}


EXPECTED = {
    "input": ("str", None, None, False),
    "--output": ("str", None, None, False),
    "--overwrite": ("flag", None, False, False),
    "--target": ("enum", {"choices": ["camera", "bone", "all"]}, "all", False),
    "--bone": ("str", None, None, True),
    "--bone-glob": ("str", None, None, True),
    "--bone-group": ("enum", {"choices": _GROUPS}, None, True),
    "--bone-file": ("str", None, None, False),
    "--exclude-bone": ("str", None, None, True),
    "--exclude-bone-glob": ("str", None, None, True),
    "--exclude-bone-group": ("enum", {"choices": _GROUPS}, None, True),
    "--list-bones": ("flag", None, False, False),
    "--range": ("compound",
                {"format": "START:END",
                 "fields": [_num_field("START", "int", 0), _num_field("END", "int", 0)]},
                None, True),
    "--preset": ("enum", {"choices": list(presets.PRESET_NAMES)}, "balanced", False),
    "--bone-pos-tol": ("float", _NONNEG, None, False),
    "--bone-rot-tol": ("float", _NONNEG, None, False),
    "--camera-pos-tol": ("float", _NONNEG, None, False),
    "--camera-rot-tol": ("float", _NONNEG, None, False),
    "--camera-distance-tol": ("float", _NONNEG, None, False),
    "--camera-fov-tol": ("float", _FOV, None, False),
    "--max-segment-frames": ("int", _AT_LEAST_1, None, False),
    "--min-segment-frames": ("int", _AT_LEAST_1, 1, False),
    "--curve-mode": ("enum", {"choices": ["bezier", "linear"]}, "bezier", False),
    "--strict": ("flag", None, False, False),
    "--cut-threshold-camera": ("compound",
                               {"format": "POS,ROT,DIST",
                                "fields": [_num_field("POS", "float", 0), _num_field("ROT", "float", 0),
                                           _num_field("DIST", "float", 0)]},
                               [5.0, 20.0, 5.0], False),
    "--cut-threshold-bone": ("compound",
                             {"format": "POS,ROT",
                              "fields": [_num_field("POS", "float", 0), _num_field("ROT", "float", 0)]},
                             [1.0, 30.0], False),
    "--cut-detect": ("flag", None, True, False),
    "--keep-frame": ("int", _FRAME, None, True),
    "--dry-run": ("flag", None, False, False),
    "--verbose": ("flag", None, False, False),
    "--quiet": ("flag", None, False, False),
}


def describe_result(capsysbinary):
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_emits_only_options_and_presets_without_input(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list)
    assert set(r) == {"type", "mode", "options", "presets"}


def test_describe_works_without_machine_flag(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_ignores_input_and_does_not_read_vmd(capsysbinary):
    rc = cli.main(["--describe", "does_not_exist.vmd"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_match_expected_table_without_meta_or_negated_flags(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    assert "--no-cut-detect" not in by_name
    assert len(r["options"]) == 31
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help", "repeat"}
        assert isinstance(o["help"], str) and o["help"]
        assert isinstance(o["repeat"], bool)
    assert set(by_name) == set(EXPECTED)
    for name, (type_, constraint, default, repeat) in EXPECTED.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name
        assert o["repeat"] == repeat, name


def test_describe_presets_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"bone_pos_tol", "bone_rot_tol", "camera_pos_tol",
                                    "camera_rot_tol", "camera_distance_tol", "camera_fov_tol"}
    expected = {
        "precise": {"bone_pos_tol": 0.005, "bone_rot_tol": 0.05, "camera_pos_tol": 0.01,
                    "camera_rot_tol": 0.02, "camera_distance_tol": 0.01, "camera_fov_tol": 0.50},
        "balanced": {"bone_pos_tol": 0.01, "bone_rot_tol": 0.10, "camera_pos_tol": 0.02,
                     "camera_rot_tol": 0.05, "camera_distance_tol": 0.02, "camera_fov_tol": 0.50},
        "aggressive": {"bone_pos_tol": 0.05, "bone_rot_tol": 0.50, "camera_pos_tol": 0.10,
                       "camera_rot_tol": 0.25, "camera_distance_tol": 0.10, "camera_fov_tol": 1.00},
    }
    assert {p["name"]: p["values"] for p in r["presets"]} == expected


def test_describe_type_table_covers_non_meta_args():
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_without_machine_reports_arg_error_as_error_event(capsysbinary):
    rc = cli.main(["--describe", "--max-segment-frames", "abc"])
    assert rc == 2
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-segment-frames" and events[-1]["exit_code"] == 2
