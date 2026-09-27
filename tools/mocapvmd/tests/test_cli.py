import pytest

from mocapvmd import cli
from mocapvmd import report as mocap_report
from vmd import io

from .helpers import (
    BONE_LINEAR_INTERP,
    BONE_NONLINEAR,
    CAM_NONLINEAR,
    bone,
    cam,
    ik_property,
    light,
    morph,
    self_shadow,
    write_vmd,
)

_FORMAT_DRY_RUN = mocap_report.format_dry_run
_DEFAULT_MODEL_NAME_RAW = b"TestModel".ljust(20, b"\x00")

_EXIT_OK = 0
_EXIT_INPUT_ERROR = 1
_EXIT_ARG_ERROR = 2

_WINERROR_PRIVILEGE_NOT_HELD = 1314

_FOOT_IK_BASE_POS_STRENGTH = 0.65

_DENSE_FRAME_COUNT = 11


def _write_all_sections_doc(path, *, model_name_raw=_DEFAULT_MODEL_NAME_RAW):
    write_vmd(
        path,
        model_name_raw=model_name_raw,
        bone=[
            bone("センター", 0, interp=BONE_NONLINEAR),
            bone("センター", 1, pos=(1.0, 0.0, 0.0), interp=BONE_NONLINEAR),
            bone("右足ＩＫ", 0),
            bone("右足ＩＫ", 1, pos=(0.0, 0.0, 0.5)),
        ],
        morph=[morph("まばたき", 0, 0.0), morph("まばたき", 5, 1.0)],
        camera=[cam(0, interp=CAM_NONLINEAR), cam(10, center=(1.0, 1.0, 1.0), interp=CAM_NONLINEAR)],
        light=[light(0)],
        self_shadow=[self_shadow(0)],
        ik_property=[ik_property(0, [("右足ＩＫ", 1), ("左足ＩＫ", 0)])],
    )


def test_missing_input_is_input_error(tmp_path):
    code = cli.main([str(tmp_path / "nope.vmd")])
    assert code == _EXIT_INPUT_ERROR


def test_input_directory_is_input_error(tmp_path):
    d = tmp_path / "indir"
    d.mkdir()
    code = cli.main([str(d)])
    assert code == _EXIT_INPUT_ERROR


def test_non_vmd_input_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    src.write_bytes(b"not a vmd file at all")
    code = cli.main([str(src)])
    assert code == _EXIT_INPUT_ERROR


def test_non_finite_bone_value_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(
        src,
        bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0)), bone("センター", 1)],
    )
    assert cli.main([str(src), "-o", str(out)]) == _EXIT_INPUT_ERROR


def test_single_key_bone_kept_verbatim(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)])
    assert cli.main([str(src), "-o", str(out)]) == _EXIT_OK
    out_doc, _ = io.read(str(out))
    assert len(out_doc.bone) == 1
    assert out_doc.bone[0].position == pytest.approx((1.0, 2.0, 3.0))
    assert out_doc.bone[0].interpolation == BONE_NONLINEAR


def test_single_key_non_finite_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0))])
    assert cli.main([str(src), "-o", str(out)]) == _EXIT_INPUT_ERROR


def test_single_key_zero_norm_quaternion_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0, rot=(0.0, 0.0, 0.0, 0.0))])
    assert cli.main([str(src), "-o", str(out)]) == _EXIT_INPUT_ERROR


def test_no_denoise_non_finite_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(4)]
    keys.append(bone("センター", 4, pos=(float("inf"), 0.0, 0.0)))
    write_vmd(src, bone=keys)
    assert cli.main([str(src), "-o", str(out), "--no-denoise"]) == _EXIT_INPUT_ERROR


def test_no_denoise_zero_norm_quaternion_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [bone("センター", f) for f in range(4)]
    keys.append(bone("センター", 4, rot=(0.0, 0.0, 0.0, 0.0)))
    write_vmd(src, bone=keys)
    assert cli.main([str(src), "-o", str(out), "--no-denoise"]) == _EXIT_INPUT_ERROR


def test_overwrite_guard_rejects_output_equal_to_input_and_leaves_input_untouched(tmp_path):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    before = src.read_bytes()
    code = cli.main([str(src), "-o", str(src)])
    assert code == _EXIT_ARG_ERROR
    assert src.read_bytes() == before


def test_overwrite_guard_rejects_symlink_to_input(tmp_path):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    before = src.read_bytes()
    link = tmp_path / "link.vmd"
    try:
        link.symlink_to(src)
    except OSError as e:
        if getattr(e, "winerror", None) != _WINERROR_PRIVILEGE_NOT_HELD:
            raise
        pytest.skip("シンボリックリンク作成権限なし(開発者モード/管理者権限が必要)")
    assert cli.main([str(src), "-o", str(link)]) == _EXIT_ARG_ERROR
    assert src.read_bytes() == before


def test_overwrite_flag_allows_same_path_and_keeps_nonbone_sections(tmp_path):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    in_doc, _ = io.read(str(src))
    code = cli.main([str(src), "-o", str(src), "--overwrite"])
    assert code == _EXIT_OK
    out_doc, _ = io.read(str(src))
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.ik_property == in_doc.ik_property
    assert len(out_doc.bone) >= 1


def test_existing_distinct_output_blocked_without_overwrite(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    out.write_bytes(b"old content")
    code = cli.main([str(src), "-o", str(out)])
    assert code == _EXIT_ARG_ERROR
    assert out.read_bytes() == b"old content"


def test_existing_distinct_output_allowed_with_overwrite(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    out.write_bytes(b"old content")
    code = cli.main([str(src), "-o", str(out), "--overwrite"])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.morph == in_doc.morph


def test_default_output_name_appends_mocap_suffix(tmp_path):
    src = tmp_path / "dance.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src)])
    assert code == _EXIT_OK
    assert (tmp_path / "dance_mocap.vmd").exists()


def test_explicit_output_written(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "-o", str(out)])
    assert code == _EXIT_OK
    assert out.exists()


def test_nonbone_sections_and_model_name_pass_through_with_denoise(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "-o", str(out)])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.light == in_doc.light
    assert out_doc.self_shadow == in_doc.self_shadow
    assert out_doc.ik_property == in_doc.ik_property
    assert out_doc.camera[0].interpolation == CAM_NONLINEAR
    assert out_doc.model_name_raw == in_doc.model_name_raw


def _assert_frames_and_values_kept_with_linear_interp(in_doc, out_doc, names):
    for name in names:
        in_keys = sorted((k for k in in_doc.bone if k.name == name), key=lambda k: k.frame)
        out_keys = sorted((k for k in out_doc.bone if k.name == name), key=lambda k: k.frame)
        assert [k.frame for k in out_keys] == [k.frame for k in in_keys]
        for ki, ko in zip(in_keys, out_keys, strict=True):
            assert ko.position == ki.position
            assert ko.rotation == pytest.approx(ki.rotation, abs=1e-6)
            assert ko.interpolation == BONE_LINEAR_INTERP


def test_no_denoise_no_stabilize_keeps_bone_values_and_linearizes_interp(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    _assert_frames_and_values_kept_with_linear_interp(in_doc, out_doc, ("センター", "右足ＩＫ"))


def test_dry_run_does_not_write_explicit_output(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "-o", str(out), "--dry-run"])
    assert code == _EXIT_OK
    assert not out.exists()


def test_dry_run_does_not_write_default_output(tmp_path):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == _EXIT_OK
    assert not (tmp_path / "in_mocap.vmd").exists()


def _dry_run_report(src, monkeypatch, *args):
    captured = {}

    def capture(rep):
        captured["report"] = rep
        return _FORMAT_DRY_RUN(rep)

    monkeypatch.setattr(mocap_report, "format_dry_run", capture)
    assert cli.main([str(src), "--dry-run", *args]) == _EXIT_OK
    return captured["report"]


def test_dry_run_report_lists_bone_categories_and_foot_ik_candidates(tmp_path, monkeypatch):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    data = _dry_run_report(src, monkeypatch)
    names = [e["name"] for e in data["bones"]]
    assert "センター" in names
    center = next(e for e in data["bones"] if e["name"] == "センター")
    assert center["category"] == "center"
    assert "右足ＩＫ" in data["foot_ik_candidates"]


def test_report_reflects_denoise_flag(tmp_path, monkeypatch):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    assert _dry_run_report(src, monkeypatch)["denoise"] is True
    assert _dry_run_report(src, monkeypatch, "--no-denoise")["denoise"] is False


def test_dry_run_prints_bone_names(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == _EXIT_OK
    out = capsys.readouterr().out
    assert "センター" in out


def test_dry_run_does_not_write_output_when_report_is_built(tmp_path):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == _EXIT_OK
    assert not (tmp_path / "in_mocap.vmd").exists()


def test_clean_strength_option_resolves_cleaning_in_report(tmp_path, monkeypatch):
    from mocapvmd import presets

    clean_strength = 1.4
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    data = _dry_run_report(src, monkeypatch, "--clean-strength", str(clean_strength))
    foot = next(e for e in data["bones"] if e["name"] == "右足ＩＫ")
    assert foot["cleaning"]["pos_strength"] == pytest.approx(
        min(1.0, _FOOT_IK_BASE_POS_STRENGTH * clean_strength))
    assert foot["cleaning"] == presets.resolve_cleaning(clean_strength, "foot_ik")


@pytest.mark.parametrize("bad", [
    pytest.param("-0.1", id="negative"),
    pytest.param("nan", id="nan"),
    pytest.param("inf", id="inf"),
])
def test_negative_or_non_finite_clean_strength_is_arg_error(tmp_path, bad):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    assert cli.main([str(src), "--clean-strength", bad]) == _EXIT_ARG_ERROR


def test_unknown_preset_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    assert cli.main([str(src), "--preset", "turbo"]) == _EXIT_ARG_ERROR


def test_default_clean_strength_is_unit_in_dry_run_report(tmp_path, monkeypatch):
    from mocapvmd import presets

    src = tmp_path / "in.vmd"
    _write_all_sections_doc(src)
    data = _dry_run_report(src, monkeypatch)
    center = next(e for e in data["bones"] if e["name"] == "センター")
    assert center["cleaning"] == presets.resolve_cleaning(1.0, "center")


_JITTER_BONE_CATEGORIES = {
    "全ての親": "root",
    "センター": "center",
    "上半身": "torso",
    "右腕": "arms",
    "右人指1": "fingers",
    "右足": "legs",
    "右足ＩＫ": "foot_ik",
    "右つま先ＩＫ": "toe_ik",
    "謎ボーン": "unknown",
}
_JITTER_BONES = tuple(_JITTER_BONE_CATEGORIES)

_ALTERNATING_X_JITTER = [0.0, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.0]


def _write_nonlinear_x_jitter_for_every_category(path):
    keys = []
    for name in _JITTER_BONES:
        keys += [
            bone(name, f, pos=(x, 0.0, 0.0), interp=BONE_NONLINEAR)
            for f, x in enumerate(_ALTERNATING_X_JITTER)
        ]
    write_vmd(path, bone=keys)


def _x_variation(keys, name):
    ks = sorted((k for k in keys if k.name == name), key=lambda k: k.frame)
    return sum(abs(ks[i + 1].position[0] - ks[i].position[0]) for i in range(len(ks) - 1))


def test_default_denoise_reduces_x_variation_for_every_category(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_nonlinear_x_jitter_for_every_category(src)
    code = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        assert _x_variation(out_doc.bone, name) < _x_variation(in_doc.bone, name)


def test_explicit_denoise_flag_reduces_x_variation_for_every_category(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_nonlinear_x_jitter_for_every_category(src)
    code = cli.main([str(src), "-o", str(out), "--denoise", "--no-reduce"])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        assert _x_variation(out_doc.bone, name) < _x_variation(in_doc.bone, name)


def test_no_denoise_no_stabilize_keeps_bone_values_for_every_category(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_nonlinear_x_jitter_for_every_category(src)
    code = cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    _assert_frames_and_values_kept_with_linear_interp(in_doc, out_doc, _JITTER_BONES)


def test_no_denoise_preserves_nonbone_sections(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-denoise"])
    assert code == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.light == in_doc.light
    assert out_doc.self_shadow == in_doc.self_shadow
    assert out_doc.ik_property == in_doc.ik_property


_FOOT_IK_BONES = ("右足ＩＫ", "右つま先ＩＫ")
_GROUNDED_X_WOBBLE = [0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0]
_GROUNDED_SLOW_X_RAMP = [round(0.05 * i, 6) for i in range(_DENSE_FRAME_COUNT)]


def _write_grounded_wobble_for_foot_ik_and_center(path):
    keys = []
    for name in (*_FOOT_IK_BONES, "センター"):
        keys += [
            bone(name, f, pos=(x, 0.0, 0.0), interp=BONE_NONLINEAR)
            for f, x in enumerate(_GROUNDED_X_WOBBLE)
        ]
    write_vmd(path, bone=keys)


def test_default_foot_ik_stabilize_reduces_foot_drift_under_no_denoise_and_keeps_center(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_grounded_wobble_for_foot_ik_and_center(src)
    assert cli.main([str(src), "-o", str(out), "--no-denoise", "--no-reduce"]) == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in _FOOT_IK_BONES:
        assert _x_variation(out_doc.bone, name) < _x_variation(in_doc.bone, name)
    _assert_frames_and_values_kept_with_linear_interp(in_doc, out_doc, ("センター",))


def test_default_foot_ik_output_equals_denoise_then_stabilize(tmp_path):
    from mocapvmd import denoise, footik, presets

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(_GROUNDED_SLOW_X_RAMP)])
    assert cli.main([str(src), "-o", str(out), "--no-reduce"]) == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    foot = sorted((k for k in in_doc.bone if k.name == "右足ＩＫ"), key=lambda k: k.frame)
    params = presets.resolve_cleaning(1.0, "foot_ik")
    cpos, _ = denoise.apply_denoise(
        [k.position for k in foot], [k.rotation for k in foot],
        pos_window=params["pos_window"], rot_window=params["rot_window"],
        pos_strength=params["pos_strength"], rot_strength=params["rot_strength"],
    )
    expected = footik.stabilize_foot_ik(
        {"右足ＩＫ": ("foot_ik", [k.frame for k in foot], cpos)}, 1.0
    )["右足ＩＫ"].locked_positions
    out_foot = sorted((k for k in out_doc.bone if k.name == "右足ＩＫ"), key=lambda k: k.frame)
    assert [k.frame for k in out_foot] == [k.frame for k in foot]
    for got, exp in zip(out_foot, expected, strict=True):
        assert got.position == pytest.approx(exp)


def test_explicit_foot_ik_stabilize_matches_default(tmp_path):
    src = tmp_path / "in.vmd"
    out_default = tmp_path / "default.vmd"
    out_explicit = tmp_path / "explicit.vmd"
    _write_grounded_wobble_for_foot_ik_and_center(src)
    assert cli.main([str(src), "-o", str(out_default), "--no-denoise"]) == _EXIT_OK
    assert cli.main([str(src), "-o", str(out_explicit), "--no-denoise", "--foot-ik-stabilize"]) == _EXIT_OK
    assert io.read(str(out_explicit))[0].bone == io.read(str(out_default))[0].bone


def test_no_foot_ik_stabilize_keeps_foot_and_toe_values(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_grounded_wobble_for_foot_ik_and_center(src)
    assert cli.main([
        str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce",
    ]) == _EXIT_OK
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    _assert_frames_and_values_kept_with_linear_interp(in_doc, out_doc, _FOOT_IK_BONES)


def test_foot_slide_suppression_outside_unit_interval_or_nan_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(0.05 * f, 0.0, 0.0)) for f in range(6)])
    for bad in ("1.5", "-0.1", "nan"):
        assert cli.main([str(src), "-o", str(out), "--foot-slide-suppression", bad]) == _EXIT_ARG_ERROR


def _foot_ik_x_series(doc):
    ks = sorted((k for k in doc.bone if k.name == "右足ＩＫ"), key=lambda k: k.frame)
    return [k.position[0] for k in ks]


def test_foot_slide_suppression_zero_keeps_slide_and_one_narrows_x_range(tmp_path):
    src = tmp_path / "in.vmd"
    out0 = tmp_path / "out0.vmd"
    out1 = tmp_path / "out1.vmd"
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(_GROUNDED_SLOW_X_RAMP)])

    assert cli.main([str(src), "-o", str(out0), "--no-denoise", "--no-reduce",
                     "--foot-slide-suppression", "0"]) == _EXIT_OK
    assert cli.main([str(src), "-o", str(out1), "--no-denoise", "--no-reduce",
                     "--foot-slide-suppression", "1"]) == _EXIT_OK

    in_x = _foot_ik_x_series(io.read(str(src))[0])
    x0 = _foot_ik_x_series(io.read(str(out0))[0])
    x1 = _foot_ik_x_series(io.read(str(out1))[0])
    assert x0 == pytest.approx(in_x)
    assert (max(x1) - min(x1)) < (max(in_x) - min(in_x))


def _write_center_linear_ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(_DENSE_FRAME_COUNT)])


def _center_frames(bones):
    return sorted(k.frame for k in bones if k.name == "センター")


def _write_center_quadratic_curve(path):
    write_vmd(path, bone=[
        bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(_DENSE_FRAME_COUNT)
    ])


def _center_keys(path):
    return sorted((k for k in io.read(str(path))[0].bone if k.name == "センター"), key=lambda k: k.frame)


def test_default_output_is_reduced_with_bezier_curve_mode(tmp_path):
    src = tmp_path / "in.vmd"
    out_default = tmp_path / "default.vmd"
    out_bezier = tmp_path / "bezier.vmd"
    out_linear = tmp_path / "linear.vmd"
    _write_center_quadratic_curve(src)
    assert cli.main([str(src), "-o", str(out_default)]) == _EXIT_OK
    assert cli.main([str(src), "-o", str(out_bezier), "--curve-mode", "bezier"]) == _EXIT_OK
    assert cli.main([str(src), "-o", str(out_linear), "--curve-mode", "linear"]) == _EXIT_OK
    default_keys = _center_keys(out_default)
    assert len(default_keys) < _DENSE_FRAME_COUNT
    assert default_keys == _center_keys(out_bezier)
    assert default_keys != _center_keys(out_linear)


def test_no_reduce_keeps_dense_linear(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    assert cli.main([str(src), "-o", str(out), "--no-reduce"]) == _EXIT_OK
    out_doc, _ = io.read(str(out))
    assert _center_frames(out_doc.bone) == list(range(_DENSE_FRAME_COUNT))
    for k in out_doc.bone:
        if k.name == "センター":
            assert k.interpolation == BONE_LINEAR_INTERP


def test_preset_slower_accepted_and_unknown_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    assert cli.main([str(src), "-o", str(out), "--preset", "slower"]) == _EXIT_OK
    assert cli.main([str(src), "--preset", "turbo"]) == _EXIT_ARG_ERROR


def test_curve_mode_linear_accepted_and_unknown_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    assert cli.main([str(src), "-o", str(out), "--curve-mode", "linear"]) == _EXIT_OK
    assert cli.main([str(src), "--curve-mode", "spline"]) == _EXIT_ARG_ERROR


def test_reduce_error_overrides_accept_valid_and_reject_negative_or_non_finite(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_linear_ramp(src)
    assert cli.main([str(src), "-o", str(out), "--reduce-error-bone-pos", "0.05"]) == _EXIT_OK
    assert cli.main([str(src), "-o", str(out), "--overwrite", "--reduce-error-bone-rot", "0.5"]) == _EXIT_OK
    assert cli.main([str(src), "--reduce-error-bone-pos", "-1"]) == _EXIT_ARG_ERROR
    assert cli.main([str(src), "--reduce-error-bone-rot", "nan"]) == _EXIT_ARG_ERROR


def test_no_reduce_output_is_dense_linear_for_every_category(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_nonlinear_x_jitter_for_every_category(src)
    code = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert code == _EXIT_OK
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        frames = sorted(k.frame for k in out_doc.bone if k.name == name)
        assert frames == list(range(_DENSE_FRAME_COUNT))
    for k in out_doc.bone:
        assert k.interpolation == BONE_LINEAR_INTERP


def test_report_has_reduction_section_for_multi_and_single_key_bones(tmp_path, monkeypatch):
    src = tmp_path / "in.vmd"
    keys = [bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(_DENSE_FRAME_COUNT)]
    single_key_bone = bone("右腕", 0, pos=(1.0, 0.0, 0.0))
    keys.append(single_key_bone)
    write_vmd(src, bone=keys)
    data = _dry_run_report(src, monkeypatch)
    for name in ("センター", "右腕"):
        r = next(e for e in data["bones"] if e["name"] == name)["reduction"]
        assert set(r) == {"output_keys", "reduction_rate", "tol_pos", "tol_rot", "cuts", "errors"}
        assert set(r["errors"]) == {"pos_x", "pos_y", "pos_z", "rot_deg"}


def test_report_reduce_flag_follows_reduce_option(tmp_path, monkeypatch):
    src = tmp_path / "in.vmd"
    _write_center_quadratic_curve(src)
    on = _dry_run_report(src, monkeypatch)
    off = _dry_run_report(src, monkeypatch, "--no-reduce")
    assert on["reduce"] is True
    assert off["reduce"] is False
    assert "reduction" not in next(e for e in off["bones"] if e["name"] == "センター")


def test_report_reduction_equals_reduce_bones_diagnostics_when_only_reducing(tmp_path, monkeypatch):
    from mocapvmd import reduce as mreduce

    src = tmp_path / "in.vmd"
    _write_center_quadratic_curve(src)
    data = _dry_run_report(src, monkeypatch, "--no-denoise", "--no-foot-ik-stabilize")
    in_doc, _ = io.read(str(src))
    diag = {}
    mreduce.reduce_bones(in_doc.bone, "medium", diagnostics_out=diag)
    d = diag["センター"]
    r = next(e for e in data["bones"] if e["name"] == "センター")["reduction"]
    assert r["output_keys"] == d["output_keys"]
    assert r["cuts"] == d["cuts"]
    assert r["tol_pos"] == d["tol_pos"]
    assert r["tol_rot"] == d["tol_rot"]
    assert r["errors"] == d["errors"]
    assert r["reduction_rate"] == pytest.approx(1.0 - d["output_keys"] / d["input_keys"])


def test_dry_run_reduction_equals_clean_stabilize_reduce_diagnostics(tmp_path, monkeypatch):
    from mocapvmd import reduce as mreduce
    from mocapvmd.cli import _clean_bones, _stabilize_bones

    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(_GROUNDED_X_WOBBLE)])
    data = _dry_run_report(src, monkeypatch)
    in_doc, _ = io.read(str(src))
    cleaned = _clean_bones(in_doc.bone, 1.0)
    full = {}
    mreduce.reduce_bones(_stabilize_bones(cleaned, 1.0), "medium", diagnostics_out=full)
    d = full["右足ＩＫ"]
    stage_skipped_inputs = {
        "no_clean_no_stabilize": in_doc.bone,
        "no_clean": _stabilize_bones(in_doc.bone, 1.0),
        "no_stabilize": cleaned,
    }
    for skipped_name, skipped in stage_skipped_inputs.items():
        diag = {}
        mreduce.reduce_bones(skipped, "medium", diagnostics_out=diag)
        assert diag["右足ＩＫ"]["errors"] != d["errors"], skipped_name
    r = next(e for e in data["bones"] if e["name"] == "右足ＩＫ")["reduction"]
    assert r["output_keys"] == d["output_keys"]
    assert r["cuts"] == d["cuts"]
    assert r["errors"] == d["errors"]


@pytest.mark.parametrize(
    "bad_key",
    [
        pytest.param(bone("センター", 4, pos=(float("inf"), 0.0, 0.0)), id="non_finite_pos"),
        pytest.param(bone("センター", 4, rot=(0.0, 0.0, 0.0, 0.0)), id="zero_norm_quat"),
    ],
)
def test_dry_run_rejects_invalid_bone_value_as_input_error(tmp_path, bad_key):
    src = tmp_path / "in.vmd"
    keys = [bone("センター", f) for f in range(4)]
    keys.append(bad_key)
    write_vmd(src, bone=keys)
    assert cli.main([str(src), "--dry-run"]) == _EXIT_INPUT_ERROR


def _list_lines(capsys):
    return capsys.readouterr().out.splitlines()


def _line_with(lines, name):
    return next(ln for ln in lines if name in ln)


def test_list_bones_line_shows_only_its_own_category(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("右足ＩＫ", 0), bone("謎ボーン", 0)])
    assert cli.main([str(src), "--list-bones"]) == _EXIT_OK
    lines = _list_lines(capsys)
    cats = {"センター": "center", "右足ＩＫ": "foot_ik", "謎ボーン": "unknown"}
    all_cats = set(cats.values())
    for name, cat in cats.items():
        ln = _line_with(lines, name)
        assert cat in ln
        assert all(other not in ln for other in all_cats - {cat})


def test_list_bones_in_first_appearance_order_once_per_name(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("右腕", 0), bone("センター", 0), bone("右腕", 5)])
    assert cli.main([str(src), "--list-bones"]) == _EXIT_OK
    lines = _list_lines(capsys)

    def first_line_index(name):
        return next(i for i, ln in enumerate(lines) if name in ln)

    assert first_line_index("右腕") < first_line_index("センター")
    assert sum(1 for ln in lines if "右腕" in ln) == 1
    assert sum(1 for ln in lines if "センター" in ln) == 1


def test_list_bones_does_not_write_explicit_or_default_output(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "explicit.vmd"
    write_vmd(src, bone=[bone("センター", 0)])
    assert cli.main([str(src), "-o", str(out), "--list-bones"]) == _EXIT_OK
    assert not out.exists()
    assert not (tmp_path / "in_mocap.vmd").exists()


def test_list_bones_bypasses_overwrite_guard_reduce_override_and_value_validation(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    non_finite_key = bone("センター", 1, pos=(float("inf"), 0.0, 0.0))
    write_vmd(src, bone=[bone("センター", 0), non_finite_key])
    before = src.read_bytes()
    assert cli.main([str(src), "-o", str(src), "--reduce-error-bone-pos", "nan", "--list-bones"]) == _EXIT_OK
    assert "center" in _line_with(_list_lines(capsys), "センター")
    assert src.read_bytes() == before


def test_list_bones_takes_precedence_over_dry_run(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 1)])
    assert cli.main([str(src), "--list-bones", "--dry-run"]) == _EXIT_OK
    assert _list_lines(capsys) == ["センター [center]"]


def test_verbose_prints_report_to_stdout_and_still_writes_output(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_all_sections_doc(src)
    assert cli.main([str(src), "-o", str(out), "-v"]) == _EXIT_OK
    assert out.exists()
    stdout = capsys.readouterr().out
    assert "preset: medium" in stdout
    assert "センター [center]" in stdout


def test_overwrite_guard_is_checked_before_reading_input(tmp_path):
    src = tmp_path / "in.vmd"
    src.write_bytes(b"not a vmd")
    assert cli.main([str(src), "-o", str(src)]) == _EXIT_ARG_ERROR


def test_reduce_override_is_validated_before_reading_input(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    src.write_bytes(b"not a vmd")
    assert cli.main([str(src), "-o", str(out), "--reduce-error-bone-pos", "nan"]) == _EXIT_ARG_ERROR


def test_decode_warning_is_single_common_format_line_on_stderr_only(tmp_path, capsys):
    from vmd.reduce import BONE_LINEAR_INTERP
    from vmd.types import BoneKey, VmdDocument
    cp932_undecodable_name = b"\x81\x20name".ljust(15, b"\x00")
    keys = [BoneKey(cp932_undecodable_name, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
            for f in range(4)]
    io.write_file(VmdDocument(bone=keys), str(tmp_path / "in.vmd"))
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--no-reduce"])
    assert rc == _EXIT_OK
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: decode-error: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert not body.startswith(" ")
    assert "警告:" not in err


def test_version_flag_prints_name_and_version_and_exits_zero(capsys):
    from mocapvmd import __version__

    assert cli.main(["--version"]) == _EXIT_OK
    assert f"mocapvmd {__version__}" in capsys.readouterr().out
