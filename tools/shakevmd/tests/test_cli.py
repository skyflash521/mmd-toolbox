import math
import sys

import pytest

from shakevmd import cli, presets
from vmd import interp, io
from vmd.reduce import Tolerances, reduce_camera_track
from vmd.sample import perspective_series
from vmd.types import BoneKey, CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6
_WINDOWS_ERROR_PRIVILEGE_NOT_HELD = 1314


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0), fov=30, persp=1),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05), fov=30, persp=1),
]

PAN_STOP_AT_30_KEYS = [
    cam(0, rot=(0.0, 0.0, 0.0)),
    cam(30, rot=(0.0, 0.5, 0.0)),
    cam(60, rot=(0.0, 0.5, 0.0)),
]

POSITION_CUT_AT_30_KEYS = [
    cam(0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, center=(0.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0)),
    cam(30, center=(40.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0)),
    cam(60, center=(40.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0)),
]

ANGLE_CUT_AT_30_KEYS = [
    cam(0, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.6, 0.0, 0.0)),
    cam(60, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.6, 0.0, 0.0)),
]

DISTANCE_CUT_AT_30_KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-5.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(60, dist=-5.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
]

SMALL_CUT_AT_30_KEYS = [
    cam(0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, center=(4.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(60, center=(4.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
]

CUTS_AT_21_AND_41_KEYS = [
    cam(0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(20, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(21, center=(40.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(40, center=(40.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(41, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(60, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
]


def write_input(path, keys=KEYS):
    io.write_file(VmdDocument(camera=list(keys)), str(path))
    return str(path)


def read_camera(path):
    doc, _ = io.read(str(path))
    return doc.camera


class TestCli:
    def test_console_script_entry_point_declared(self):
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["scripts"]["shakevmd"] == "shakevmd.cli:main"

    def test_main_falls_back_to_sys_argv(self, tmp_path, monkeypatch):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        monkeypatch.setattr(sys, "argv", ["shakevmd", inp, "-o", str(out), "--no-smooth"])
        assert cli.main() == 0
        assert out.exists()

    def test_writes_dense_bake_to_input_name_with_shake_suffix(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        rc = cli.main([inp, "--no-smooth"])
        assert rc == 0
        out = tmp_path / "in_shake.vmd"
        assert out.exists()
        baked = read_camera(out)
        assert sorted(k.frame for k in baked) == list(range(0, 61))

    def test_default_output_always_vmd_extension(self, tmp_path):
        inp = write_input(tmp_path / "take.dat")
        rc = cli.main([inp, "--no-smooth"])
        assert rc == 0
        assert (tmp_path / "take_shake.vmd").exists()
        assert not (tmp_path / "take_shake.dat").exists()

    def test_explicit_output_path_is_written(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "sub" / "out.vmd"
        out.parent.mkdir()
        rc = cli.main([inp, "-o", str(out), "--no-smooth"])
        assert rc == 0 and out.exists()

    def test_invalid_vmd_exit1(self, tmp_path):
        bad = tmp_path / "bad.vmd"
        bad.write_bytes(b"not a vmd file at all")
        assert cli.main([str(bad)]) == 1

    def test_missing_input_file_exit1(self, tmp_path):
        assert cli.main([str(tmp_path / "nope.vmd")]) == 1

    def test_no_camera_keys_exit1(self, tmp_path):
        inp = write_input(tmp_path / "empty.vmd", keys=[])
        assert cli.main([inp]) == 1

    def test_output_same_as_input_without_overwrite_exit2_and_keeps_input(self, tmp_path):
        p = tmp_path / "in.vmd"
        inp = write_input(p)
        before = p.read_bytes()
        assert cli.main([inp, "-o", inp]) == 2
        assert p.read_bytes() == before

    def test_output_same_as_input_with_overwrite_replaces_it(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        rc = cli.main([inp, "-o", inp, "--overwrite", "--no-smooth"])
        assert rc == 0
        assert sorted(k.frame for k in read_camera(inp)) == list(range(0, 61))

    def test_existing_distinct_output_blocked_without_overwrite(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        out.write_bytes(b"old content")
        assert cli.main([inp, "-o", str(out), "--no-smooth"]) == 2
        assert out.read_bytes() == b"old content"

    def test_existing_distinct_output_allowed_with_overwrite(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        out.write_bytes(b"old content")
        assert cli.main([inp, "-o", str(out), "--overwrite", "--no-smooth"]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_overwrite_guard_via_symlink_exit2(self, tmp_path):
        p = tmp_path / "in.vmd"
        inp = write_input(p)
        before = p.read_bytes()
        link = tmp_path / "link.vmd"
        try:
            link.symlink_to(p)
        except OSError as e:
            if getattr(e, "winerror", None) != _WINDOWS_ERROR_PRIVILEGE_NOT_HELD:
                raise
            pytest.skip("シンボリックリンク作成権限なし(開発者モード/管理者権限が必要)")
        assert cli.main([inp, "-o", str(link)]) == 2
        assert p.read_bytes() == before

    def test_overlapping_ranges_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "0:60", "--range", "30:60"]) == 2

    def test_output_long_form_is_accepted(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "--output", str(out), "--no-smooth"]) == 0 and out.exists()

    def test_bad_range_format_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for bad in ("abc", "1.5:30", "x:30", "30:y", "10:20:30", "30"):
            assert cli.main([inp, "--range", bad]) == 2

    def test_reversed_range_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "60:0"]) == 2

    def test_start_beyond_resolved_open_end_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "999:"]) == 2

    def test_colon_only_range_is_full(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--range", ":", "--no-smooth"]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_ranges_overlapping_after_snap_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "25:55", "--range", "28:58"]) == 2

    def test_output_under_file_path_exit3(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        clash = tmp_path / "afile"
        clash.write_bytes(b"x")
        rc = cli.main([inp, "-o", str(clash / "out.vmd"), "--no-smooth"])
        assert rc == 3

    def test_range_bakes_inside_and_keeps_outside_key_intact(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", "30:60", "--no-smooth"])
        assert rc == 0
        out_keys = {k.frame: k for k in read_camera(out)}
        in_keys = {k.frame: k for k in read_camera(inp)}
        assert sorted(out_keys) == [0] + list(range(30, 61))
        assert out_keys[0] == in_keys[0]

    def test_open_ended_range_start_omitted(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", ":30", "--no-smooth"])
        assert rc == 0
        frames = sorted(k.frame for k in read_camera(out))
        assert frames == list(range(0, 31)) + [60]

    def test_open_ended_range_end_omitted(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", "30:", "--no-smooth"])
        assert rc == 0
        frames = sorted(k.frame for k in read_camera(out))
        assert frames == [0] + list(range(30, 61))

    def test_multiple_nonoverlapping_ranges(self, tmp_path):
        keys = [cam(0), cam(15), cam(30, persp=1), cam(45), cam(60)]
        inp = write_input(tmp_path / "in.vmd", keys)
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", "0:15", "--range", "45:60", "--no-smooth"])
        assert rc == 0
        frames = sorted(k.frame for k in read_camera(out))
        assert frames == list(range(0, 16)) + [30] + list(range(45, 61))

    def test_seed_reproducible_and_seed_dependent(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        o1, o2, o3 = tmp_path / "o1.vmd", tmp_path / "o2.vmd", tmp_path / "o3.vmd"
        assert cli.main([inp, "-o", str(o1), "--seed", "7", "--amp-rot", "5", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(o2), "--seed", "7", "--amp-rot", "5", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(o3), "--seed", "8", "--amp-rot", "5", "--no-smooth"]) == 0
        assert o1.read_bytes() == o2.read_bytes()
        assert o1.read_bytes() != o3.read_bytes()

    def test_duplicate_frame_warning_is_displayed(self, tmp_path, capsys):
        dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
        inp = write_input(tmp_path / "dup.vmd", dup)
        rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--no-smooth"])
        assert rc == 0
        text = capsys.readouterr()
        assert "warning:" in (text.out + text.err).lower()

    def test_octave_clamp_warns_only_when_freq_exceeds_band(self, tmp_path, capsys):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "d.vmd"), "--no-smooth"]) == 0
        base = capsys.readouterr()
        assert "warning:" not in (base.out + base.err).lower()
        assert cli.main([inp, "-o", str(tmp_path / "f.vmd"), "--freq", "3.0", "--no-smooth"]) == 0
        clamped = capsys.readouterr()
        assert "warning:" in (clamped.out + clamped.err).lower()

    def test_amp_rot_zero_no_rotation_shake(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--amp-rot", "0", "--amp-pos", "0", "--settle", "0", "--no-smooth"]) == 0
        baked = {k.frame: k for k in read_camera(out)}
        for f in range(0, 61):
            s = interp.sample_camera(KEYS, f)
            assert baked[f].rotation == pytest.approx(s["rotation"], abs=1e-5)

    def test_public_options_affect_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")

        def out_bytes(args, name):
            p = tmp_path / name
            assert cli.main([inp, "-o", str(p), *args, "--no-smooth"]) == 0
            return p.read_bytes()

        base = out_bytes([], "base.vmd")
        assert out_bytes(["--freq", "3.0"], "freq.vmd") != base
        assert out_bytes(["--rot-weights", "1,1,0.9"], "rw.vmd") != base
        assert out_bytes(["--fade", "0.2"], "fade.vmd") != base
        assert out_bytes(["--motion-damp", "2.0"], "ms.vmd") != base
        assert cli.main([inp, "-o", str(tmp_path / "ct.vmd"), "--cut-threshold", "4,15", "--no-smooth"]) == 0

    def test_multiple_impulses_are_all_applied_and_summed(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        base, one, two = tmp_path / "b.vmd", tmp_path / "1.vmd", tmp_path / "2.vmd"
        last = tmp_path / "last.vmd"
        common = ["--amp-rot", "0", "--amp-pos", "0", "--settle", "0"]
        assert cli.main([inp, "-o", str(base), *common, "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(one), *common, "--impulse", "20:10:0.5", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(two), *common,
                         "--impulse", "20:10:0.5", "--impulse", "45:10:0.5", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(last), *common, "--impulse", "45:10:0.5", "--no-smooth"]) == 0
        assert base.read_bytes() != one.read_bytes()
        assert one.read_bytes() != two.read_bytes()
        assert two.read_bytes() != last.read_bytes()

    def test_amp_rot_affects_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        assert cli.main([inp, "-o", str(a), "--amp-rot", "1", "--amp-pos", "0", "--settle", "0", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(b), "--amp-rot", "9", "--amp-pos", "0", "--settle", "0", "--no-smooth"]) == 0
        assert a.read_bytes() != b.read_bytes()

    def test_amp_pos_affects_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        assert cli.main([inp, "-o", str(a), "--amp-pos", "0", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(b), "--amp-pos", "0.5", "--no-smooth"]) == 0
        assert a.read_bytes() != b.read_bytes()

    def test_omitted_options_equal_explicit_defaults(self, tmp_path):
        explicit_defaults = [
            "--amp-rot", "0.8", "--amp-pos", "0.05", "--rot-weights", "1,1,0.3",
            "--freq", "1.2", "--seed", "1", "--fade", "0.7",
            "--motion-damp", "1.0", "--settle", "0", "--cut-threshold", "5,20",
        ]
        for name, keys in (("plain", KEYS), ("stop", PAN_STOP_AT_30_KEYS),
                           ("cut", POSITION_CUT_AT_30_KEYS), ("anglecut", ANGLE_CUT_AT_30_KEYS)):
            inp = write_input(tmp_path / f"{name}.vmd", keys)
            d, e = tmp_path / f"{name}_d.vmd", tmp_path / f"{name}_e.vmd"
            assert cli.main([inp, "-o", str(d), "--no-smooth"]) == 0
            assert cli.main([inp, "-o", str(e), *explicit_defaults, "--no-smooth"]) == 0
            assert d.read_bytes() == e.read_bytes(), f"default != explicit for {name}"

    def test_default_seed_is_one(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        d, s1 = tmp_path / "d.vmd", tmp_path / "s1.vmd"
        assert cli.main([inp, "-o", str(d), "--amp-rot", "5", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(s1), "--amp-rot", "5", "--seed", "1", "--no-smooth"]) == 0
        assert d.read_bytes() == s1.read_bytes()

    def test_settle_affects_output_and_zero_disables_it(self, tmp_path):
        inp = write_input(tmp_path / "ps.vmd", PAN_STOP_AT_30_KEYS)
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        common = ["--amp-rot", "0", "--amp-pos", "0", "--no-smooth"]
        assert cli.main([inp, "-o", str(a), *common, "--settle", "0"]) == 0
        assert cli.main([inp, "-o", str(b), *common, "--settle", "5"]) == 0
        assert a.read_bytes() != b.read_bytes()
        baked0 = {k.frame: k for k in read_camera(a)}
        for f in range(0, 61):
            s = interp.sample_camera(PAN_STOP_AT_30_KEYS, f)
            assert baked0[f].rotation == pytest.approx(s["rotation"], abs=1e-5)

    def test_cut_threshold_position_side_affects_output(self, tmp_path):
        inp = write_input(tmp_path / "cut.vmd", POSITION_CUT_AT_30_KEYS)
        lo, hi = tmp_path / "lo.vmd", tmp_path / "hi.vmd"
        assert cli.main([inp, "-o", str(lo), "--cut-threshold", "5,20", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(hi), "--cut-threshold", "100,200", "--no-smooth"]) == 0
        assert lo.read_bytes() != hi.read_bytes()

    def test_cut_threshold_angle_side_affects_output(self, tmp_path):
        inp = write_input(tmp_path / "acut.vmd", ANGLE_CUT_AT_30_KEYS)
        lo, hi = tmp_path / "lo.vmd", tmp_path / "hi.vmd"
        assert cli.main([inp, "-o", str(lo), "--cut-threshold", "5,20", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(hi), "--cut-threshold", "5,200", "--no-smooth"]) == 0
        assert lo.read_bytes() != hi.read_bytes()

    def test_cut_threshold_position_side_applies_to_distance_jump(self, tmp_path):
        inp = write_input(tmp_path / "zcut.vmd", DISTANCE_CUT_AT_30_KEYS)
        lo, hi = tmp_path / "lo.vmd", tmp_path / "hi.vmd"
        assert cli.main([inp, "-o", str(lo), "--cut-threshold", "5,20", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(hi), "--cut-threshold", "100,200", "--no-smooth"]) == 0
        assert lo.read_bytes() != hi.read_bytes()

    def test_cli_flags_wire_to_correct_bake_params(self, tmp_path):
        from shakevmd.bake import bake
        cases = [
            (["--freq", "3.0"], dict(freq=3.0), KEYS),
            (["--rot-weights", "0.5,0.7,0.9"], dict(rot_weights=(0.5, 0.7, 0.9)), KEYS),
            (["--amp-rot", "2.0"], dict(amp_rot=2.0), KEYS),
            (["--amp-pos", "0.3"], dict(amp_pos=0.3), KEYS),
            (["--fade", "0.2"], dict(fade_sec=0.2), KEYS),
            (["--motion-damp", "2.0"], dict(motion_damp=2.0), KEYS),
            (["--motion-damp", "0"], dict(motion_damp=0.0), KEYS),
            (["--settle", "1.5"], dict(settle=1.5), PAN_STOP_AT_30_KEYS),
            (["--settle", "0"], dict(settle=0.0), PAN_STOP_AT_30_KEYS),
            (["--seed", "9"], dict(seed=9), KEYS),
            (["--cut-threshold", "3,10"], dict(cut_pos_threshold=3.0, cut_rot_threshold=10.0), SMALL_CUT_AT_30_KEYS),
            (["--impulse", "20:10:0.5"], dict(impulses=[(20, 10.0, 0.5)]), KEYS),
        ]
        for i, (flag_args, kw, keys) in enumerate(cases):
            inp = write_input(tmp_path / f"in_{i}.vmd", keys)
            src = read_camera(inp)
            out = tmp_path / f"cli_{i}.vmd"
            exp = tmp_path / f"exp_{i}.vmd"
            assert cli.main([inp, "-o", str(out), *flag_args, "--no-smooth"]) == 0
            io.write_file(VmdDocument(camera=bake(list(src), **kw).camera_keys), str(exp))
            assert read_camera(out) == read_camera(exp), f"flag misw-wired: {flag_args}"

    def test_invalid_compound_option_formats_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for args in (
            ["--rot-weights", "1,1"],
            ["--rot-weights", "1,1,1,1"],
            ["--rot-weights", "a,b,c"],
            ["--cut-threshold", "5"],
            ["--cut-threshold", "5,20,30"],
            ["--cut-threshold", "a,b"],
            ["--cut-threshold", "5,deg"],
            ["--impulse", "30:10"],
            ["--impulse", "30:10:0.5:x"],
            ["--impulse", "x:10:0.5"],
            ["--impulse", "30:s:0.5"],
            ["--impulse", "30:10:d"],
        ):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), *args]) == 2

    def test_range_ends_snap_to_nearest_key_not_floor_or_ceil(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        assert cli.main([inp, "-o", str(a), "--range", "26:60", "--no-smooth"]) == 0
        assert sorted(k.frame for k in read_camera(a)) == [0] + list(range(30, 61))
        assert cli.main([inp, "-o", str(b), "--range", "0:34", "--no-smooth"]) == 0
        assert sorted(k.frame for k in read_camera(b)) == list(range(0, 31)) + [60]

    def test_range_end_beyond_last_key_snaps_to_it(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--range", "0:999", "--no-smooth"]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_range_start_before_first_key_snaps_to_it(self, tmp_path):
        keys = [cam(30), cam(45), cam(60)]
        inp = write_input(tmp_path / "off.vmd", keys)
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--range", "10:60", "--no-smooth"]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(30, 61))

    def test_missing_input_arg_exit2(self):
        assert cli.main([]) == 2

    def test_surplus_positional_arg_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        extra = write_input(tmp_path / "extra.vmd")
        assert cli.main([inp, extra]) == 2

    def test_non_integer_seed_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--seed", "abc"]) == 2

    def test_invalid_numeric_scalars_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for opt in ("--amp-rot", "--amp-pos", "--freq", "--fade", "--motion-damp", "--settle"):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), opt, "xyz"]) == 2

    def test_non_finite_numeric_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for opt, val in (("--fade", "inf"), ("--amp-rot", "nan"), ("--freq", "inf"),
                         ("--motion-damp", "-inf"), ("--rot-weights", "1,inf,1"),
                         ("--cut-threshold", "inf,20"), ("--impulse", "20:inf:0.5")):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), opt, val]) == 2

    def test_huge_finite_fade_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--fade", "1e308"]) == 2

    def test_out_of_domain_numeric_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        o = str(tmp_path / "o.vmd")
        cases = [
            ["--fade", "-0.7"], ["--amp-rot", "-1"], ["--amp-pos", "-0.1"],
            ["--motion-damp", "-1"], ["--settle", "-1"],
            ["--freq", "0"], ["--freq", "-1"],
            ["--cut-threshold", "-5,20"], ["--cut-threshold", "5,-20"],
            ["--impulse", "20:-1:0.5"],
            ["--impulse", "20:10:0"],
            ["--impulse", "20:10:-0.5"],
        ]
        for args in cases:
            assert cli.main([inp, "-o", o, *args]) == 2, args

    def test_zero_disable_values_allowed(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        o = str(tmp_path / "o.vmd")
        for args in (["--amp-rot", "0"], ["--amp-pos", "0"], ["--motion-damp", "0"],
                     ["--settle", "0"], ["--fade", "0"], ["--cut-threshold", "0,0"],
                     ["--impulse", "20:0:0.5"]):
            assert cli.main([inp, "-o", o, "--overwrite", *args, "--no-smooth"]) == 0, args

    def test_negative_impulse_frame_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--impulse=-5:10:0.5"]) == 2

    def test_read_warnings_propagated(self, tmp_path, capsys, monkeypatch):
        from vmd.types import VmdWarning
        inp = write_input(tmp_path / "in.vmd")
        real_read = io.read

        def fake_read(path):
            doc, warns = real_read(path)
            warns.append(VmdWarning(code="decode-error", message="注入した読込警告"))
            return doc, warns

        monkeypatch.setattr(cli.io, "read", fake_read)
        assert cli.main([str(inp), "-o", str(tmp_path / "out.vmd"), "--no-smooth"]) == 0
        cap = capsys.readouterr()
        text = (cap.out + cap.err).lower()
        assert "warning:" in text and "decode-error" in text

    def test_negative_range_endpoint_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range=-10:0"]) == 2
        assert cli.main([inp, "--range=0:-5"]) == 2

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_non_finite_baked_output_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        rc = cli.main([inp, "-o", str(tmp_path / "o.vmd"),
                       "--amp-rot", "1e308", "--rot-weights", "1e308,1,1"])
        assert rc == 2

    def test_serialization_overflow_is_arg_error_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--amp-rot", "1e308", "--no-smooth"]) == 2

    def test_abbreviated_flag_rejected_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--over"]) == 2

    def test_missing_option_operand_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for opt in ("--output", "--range", "--seed", "--amp-rot", "--amp-pos",
                    "--rot-weights", "--freq", "--fade", "--motion-damp",
                    "--settle", "--cut-threshold", "--impulse",
                    "--preset"):
            assert cli.main([inp, opt]) == 2

    def test_rejects_internal_params_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for opt in (["--octaves", "5"], ["--persistence", "0.7"], ["--settle-time", "2"],
                    ["--gait-freq", "2.0"], ["--gait-amp", "0.2"]):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), *opt]) == 2

    def test_non_camera_sections_passthrough_with_warning(self, tmp_path, capsys):
        bone = BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                       position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                       interpolation=bytes(64))
        model_name_raw = b"TestModel".ljust(20, b"\x00")
        doc = VmdDocument(bone=[bone], camera=list(KEYS), model_name_raw=model_name_raw)
        inp = tmp_path / "mixed.vmd"
        io.write_file(doc, str(inp))
        out = tmp_path / "out.vmd"
        rc = cli.main([str(inp), "-o", str(out), "--no-smooth"])
        assert rc == 0
        outdoc, _ = io.read(str(out))
        assert len(outdoc.bone) == 1 and outdoc.bone[0].name == "bone"
        assert sorted(k.frame for k in outdoc.camera) == list(range(0, 61))
        assert outdoc.model_name_raw == model_name_raw
        text = capsys.readouterr()
        assert "warning:" in (text.out + text.err).lower()


class TestCliOps:
    PUBLIC_PARAMS = {"amp_rot", "amp_pos", "rot_weights", "freq",
                     "motion_damp", "settle", "cut_threshold"}
    INTERNAL_PARAMS = set(presets.INTERNAL_PARAM_NAMES)

    def test_presets_defined_for_all_names(self):
        assert set(presets.PRESET_NAMES) == {"handheld", "telephoto", "walking", "earthquake"}
        for name in presets.PRESET_NAMES:
            params = presets.get_preset(name)
            assert isinstance(params, dict)
            assert self.PUBLIC_PARAMS <= set(params), name
            assert set(params) - self.PUBLIC_PARAMS <= self.INTERNAL_PARAMS, name
        with pytest.raises(KeyError):
            presets.get_preset("nonexistent-preset")

    def test_only_walking_has_gait_component(self):
        w = presets.get_preset("walking")
        assert w["gait_freq"] > 0.0 and w["gait_amp"] > 0.0
        for name in ("handheld", "telephoto", "earthquake"):
            q = presets.get_preset(name)
            assert q.get("gait_freq", 0.0) == 0.0, name
            assert q.get("gait_amp", 0.0) == 0.0, name

    def test_internal_params_forwardable_and_are_bake_kwargs(self):
        import inspect

        from shakevmd.bake import bake
        params = inspect.signature(bake).parameters
        for name in presets.INTERNAL_PARAM_NAMES:
            assert name in params, name
        assert {"gait_freq", "gait_amp", "still_profile", "moving_profile",
                "settle_time", "naive_rotation"} <= set(presets.INTERNAL_PARAM_NAMES)

    def test_preset_internal_params_forward_without_collision(self, tmp_path, monkeypatch):
        p = dict(presets.get_preset("handheld"))
        p.update(still_profile=(1.0, 0.5, 0.25), moving_profile=(1.0, 0.7, 0.4),
                 settle_time=2.0, naive_rotation=True, gait_freq=1.5, gait_amp=0.1,
                 speed_ref_world=2.0, speed_ref_angle=0.05)
        assert set(presets.INTERNAL_PARAM_NAMES) <= set(p)
        monkeypatch.setitem(presets._PRESETS, "handheld", p)
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--preset", "handheld", "--no-smooth"]) == 0

    def test_all_internal_params_forwarded_and_effective(self, tmp_path, monkeypatch):
        from shakevmd.bake import bake
        internal = dict(still_profile=(1.0, 0.5, 0.25), moving_profile=(1.0, 0.7, 0.4),
                        settle_time=2.5, naive_rotation=True, gait_freq=1.5, gait_amp=0.1,
                        speed_ref_world=2.0, speed_ref_angle=0.05)
        assert set(internal) >= set(presets.INTERNAL_PARAM_NAMES)
        p = dict(presets.get_preset("handheld"))
        p.update(internal)
        monkeypatch.setitem(presets._PRESETS, "handheld", p)
        inp = write_input(tmp_path / "in.vmd", PAN_STOP_AT_30_KEYS)
        src = read_camera(inp)
        out, exp = tmp_path / "cli.vmd", tmp_path / "exp.vmd"
        assert cli.main([inp, "-o", str(out), "--preset", "handheld", "--no-smooth"]) == 0
        hp = presets.get_preset("handheld")
        baked = bake(
            list(src), seed=1,
            amp_rot=hp["amp_rot"], amp_pos=hp["amp_pos"], rot_weights=hp["rot_weights"],
            freq=hp["freq"], motion_damp=hp["motion_damp"], settle=hp["settle"],
            cut_pos_threshold=hp["cut_threshold"][0], cut_rot_threshold=hp["cut_threshold"][1],
            fade_sec=0.7, impulses=(), **internal,
        )
        io.write_file(VmdDocument(camera=baked.camera_keys), str(exp))
        assert read_camera(out) == read_camera(exp)

    def test_walking_preset_forwards_gait_to_bake(self, tmp_path):
        from shakevmd.bake import bake
        inp = write_input(tmp_path / "in.vmd", KEYS)
        src = read_camera(inp)
        out, exp = tmp_path / "cli.vmd", tmp_path / "exp.vmd"
        assert cli.main([inp, "-o", str(out), "--preset", "walking", "--no-smooth"]) == 0
        p = presets.get_preset("walking")
        baked = bake(
            list(src), seed=1,
            amp_rot=p["amp_rot"], amp_pos=p["amp_pos"], rot_weights=p["rot_weights"],
            freq=p["freq"], motion_damp=p["motion_damp"], settle=p["settle"],
            cut_pos_threshold=p["cut_threshold"][0], cut_rot_threshold=p["cut_threshold"][1],
            fade_sec=0.7, impulses=(), gait_freq=p["gait_freq"], gait_amp=p["gait_amp"],
        )
        io.write_file(VmdDocument(camera=baked.camera_keys), str(exp))
        assert read_camera(out) == read_camera(exp)

    def test_all_presets_run_via_cli(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for name in ("handheld", "telephoto", "walking", "earthquake"):
            assert cli.main([inp, "-o", str(tmp_path / f"{name}.vmd"), "--preset", name, "--no-smooth"]) == 0, name

    def test_unknown_preset_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--preset", "bogus"]) == 2

    def test_preset_changes_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        base, eq = tmp_path / "base.vmd", tmp_path / "eq.vmd"
        assert cli.main([inp, "-o", str(base), "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(eq), "--preset", "earthquake", "--no-smooth"]) == 0
        assert base.read_bytes() != eq.read_bytes()

    def test_individual_arg_overrides_preset(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        base, full, part = tmp_path / "base.vmd", tmp_path / "full.vmd", tmp_path / "part.vmd"
        assert cli.main([inp, "-o", str(base), "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(full), "--preset", "earthquake", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(part), "--preset", "earthquake",
                         "--amp-rot", "0.8", "--no-smooth"]) == 0
        assert part.read_bytes() != full.read_bytes()
        assert part.read_bytes() != base.read_bytes()

    def test_full_explicit_args_supersede_preset(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        d, e = tmp_path / "d.vmd", tmp_path / "e.vmd"
        explicit_defaults = [
            "--amp-rot", "0.8", "--amp-pos", "0.05", "--rot-weights", "1,1,0.3",
            "--freq", "1.2", "--motion-damp", "1.0", "--settle", "0",
            "--cut-threshold", "5,20",
        ]
        assert cli.main([inp, "-o", str(d), "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(e), "--preset", "earthquake", *explicit_defaults, "--no-smooth"]) == 0
        assert d.read_bytes() == e.read_bytes()

    def test_preset_equals_its_explicit_public_values_unless_gait_is_active(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        for name in presets.PRESET_NAMES:
            p = presets.get_preset(name)
            explicit = [
                "--amp-rot", str(p["amp_rot"]), "--amp-pos", str(p["amp_pos"]),
                "--rot-weights", "{},{},{}".format(*p["rot_weights"]),
                "--freq", str(p["freq"]), "--motion-damp", str(p["motion_damp"]),
                "--settle", str(p["settle"]),
                "--cut-threshold", "{},{}".format(*p["cut_threshold"]),
            ]
            a, b = tmp_path / f"{name}_a.vmd", tmp_path / f"{name}_b.vmd"
            assert cli.main([inp, "-o", str(a), "--preset", name, "--no-smooth"]) == 0, name
            assert cli.main([inp, "-o", str(b), *explicit, "--no-smooth"]) == 0, name
            active_gait = p.get("gait_freq", 0.0) > 0.0 and p.get("gait_amp", 0.0) != 0.0
            if active_gait:
                assert a.read_bytes() != b.read_bytes(), name
            else:
                assert a.read_bytes() == b.read_bytes(), name

    def test_individual_override_is_order_independent(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        before, after = tmp_path / "before.vmd", tmp_path / "after.vmd"
        assert cli.main([inp, "-o", str(before), "--amp-rot", "0.8", "--preset", "earthquake", "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(after), "--preset", "earthquake", "--amp-rot", "0.8", "--no-smooth"]) == 0
        assert before.read_bytes() == after.read_bytes()

    def test_dry_run_writes_no_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--dry-run"]) == 0
        assert not out.exists()

    def test_dry_run_still_rejects_existing_output_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", inp, "--dry-run"]) == 2

    def test_dry_run_writes_no_default_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--dry-run"]) == 0
        assert not (tmp_path / "in_shake.vmd").exists()

    def test_dry_run_reports_range_key_count_amplitude_and_cuts(self, tmp_path, capsys):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = (cap.out + cap.err).lower()
        assert "61" in text
        for label in ("range", "key", "amplitude", "cut"):
            assert label in text, label

    def test_dry_run_reports_detected_cut(self, tmp_path, capsys):
        inp = write_input(tmp_path / "cut.vmd", POSITION_CUT_AT_30_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = (cap.out + cap.err).lower()
        assert "cut" in text and "30" in text

    def test_dry_run_reports_all_cuts(self, tmp_path, capsys):
        inp = write_input(tmp_path / "mcut.vmd", CUTS_AT_21_AND_41_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "21" in text and "41" in text

    def test_dry_run_reports_all_snapped_ranges(self, tmp_path, capsys):
        keys = [cam(0), cam(15), cam(30, persp=1), cam(45), cam(60)]
        inp = write_input(tmp_path / "mr.vmd", keys)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "0:14", "--range", "44:60", "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "15" in text and "45" in text

    def test_dry_run_reports_warning(self, tmp_path, capsys):
        dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
        inp = write_input(tmp_path / "dup.vmd", dup)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        assert "warning:" in (cap.out + cap.err).lower()

    def test_dry_run_reports_snapped_range(self, tmp_path, capsys):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "26:60", "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "30" in text and "60" in text

    def test_verbose_reports_range_and_cuts_but_plain_run_does_not(self, tmp_path, capsys):
        inp = write_input(tmp_path / "cut.vmd", POSITION_CUT_AT_30_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--no-smooth"]) == 0
        q = capsys.readouterr()
        quiet_text = (q.out + q.err).lower()
        assert "range" not in quiet_text and "cut" not in quiet_text
        assert cli.main([inp, "-o", str(tmp_path / "b.vmd"), "--verbose", "--no-smooth"]) == 0
        cap = capsys.readouterr()
        verbose_text = (cap.out + cap.err).lower()
        assert len(cap.out + cap.err) > len(q.out + q.err)
        assert "range" in verbose_text and "cut" in verbose_text and "30" in verbose_text

    def test_verbose_reports_snapped_range(self, tmp_path, capsys):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "26:60", "--verbose", "--no-smooth"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "30" in text and "60" in text

    def test_verbose_reports_all_cuts(self, tmp_path, capsys):
        inp = write_input(tmp_path / "mcut.vmd", CUTS_AT_21_AND_41_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--verbose", "--no-smooth"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "21" in text and "41" in text

    def test_verbose_reports_all_snapped_ranges(self, tmp_path, capsys):
        keys = [cam(0), cam(15), cam(30, persp=1), cam(45), cam(60)]
        inp = write_input(tmp_path / "mr.vmd", keys)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "0:14", "--range", "44:60", "--verbose", "--no-smooth"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "15" in text and "45" in text

    def test_verbose_still_writes_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--verbose", "--no-smooth"]) == 0
        assert out.exists()
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_verbose_does_not_change_vmd(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        q, v = tmp_path / "q.vmd", tmp_path / "v.vmd"
        assert cli.main([inp, "-o", str(q), "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(v), "--verbose", "--no-smooth"]) == 0
        assert v.read_bytes() == q.read_bytes()

    def test_verbose_short_alias_equals_long(self, tmp_path, capsys):
        inp = write_input(tmp_path / "in.vmd")
        out = str(tmp_path / "out.vmd")
        assert cli.main([inp, "-o", out, "-v", "--no-smooth"]) == 0
        short = capsys.readouterr()
        short_text = short.out + short.err
        assert cli.main([inp, "-o", out, "--verbose", "--no-smooth", "--overwrite"]) == 0
        long = capsys.readouterr()
        assert short_text == (long.out + long.err)


SMOOTH_POS_TOL = 0.10
SMOOTH_ROT_TOL_DEG = 0.25
SMOOTH_DIST_TOL = 0.10
SMOOTH_FOV_TOL = 1.00

FOV_JUMP_AT_30_KEYS = [
    cam(0, fov=30), cam(29, fov=30), cam(30, fov=45), cam(60, fov=45),
]


_POSITION_CHANNEL_OFFSETS = (0, 4, 8)
_ROTATION_CHANNEL_OFFSET = 12
_ALL_CHANNEL_OFFSETS = (0, 4, 8, 12, 16, 20)


def _is_curved(ax_bx_ay_by):
    ax, bx, ay, by = ax_bx_ay_by
    return ax != ay or bx != by


def _channel(interpolation, offset):
    return interpolation[offset:offset + 4]


_SHAKE_ARGS = ["--amp-rot", "8.0", "--amp-pos", "1.0", "--seed", "1", "--fade", "0.1"]


class TestSmooth:
    def _bake(self, out_path, *extra):
        inp = write_input(out_path.parent / "in.vmd")
        rc = cli.main([inp, "-o", str(out_path), *_SHAKE_ARGS, *extra])
        return rc

    def test_smooth_flag_accepted(self, tmp_path):
        out = tmp_path / "smooth.vmd"
        assert self._bake(out, "--smooth") == 0
        assert out.exists()

    def test_smooth_reduces_key_count(self, tmp_path):
        dense = tmp_path / "dense.vmd"
        smooth = tmp_path / "smooth.vmd"
        assert self._bake(dense, "--no-smooth") == 0
        assert self._bake(smooth, "--smooth") == 0
        n_dense = len(read_camera(dense))
        n_smooth = len(read_camera(smooth))
        assert n_dense == 61
        assert n_smooth < n_dense * 0.6

    def test_smooth_position_and_rotation_channels_are_bezier(self, tmp_path):
        smooth = tmp_path / "smooth.vmd"
        assert self._bake(smooth, "--smooth") == 0
        ks = [bytes(k.interpolation) for k in read_camera(smooth)[1:]]
        assert any(any(_is_curved(_channel(b, j)) for j in _POSITION_CHANNEL_OFFSETS) for b in ks)
        assert any(_is_curved(_channel(b, _ROTATION_CHANNEL_OFFSET)) for b in ks)

    def test_smooth_preserves_shake_within_tolerance(self, tmp_path):
        dense = tmp_path / "dense.vmd"
        smooth = tmp_path / "smooth.vmd"
        assert self._bake(dense, "--no-smooth") == 0
        assert self._bake(smooth, "--smooth") == 0
        dk, sk = read_camera(dense), read_camera(smooth)
        for i in range(0, 241):
            f = i * 0.25
            d = interp.sample_camera(dk, f)
            s = interp.sample_camera(sk, f)
            assert math.dist(s["position"], d["position"]) <= SMOOTH_POS_TOL + 1e-6
            for ax in range(3):
                diff = s["rotation"][ax] - d["rotation"][ax]
                diff_deg = abs(math.degrees(math.atan2(math.sin(diff), math.cos(diff))))
                assert diff_deg <= SMOOTH_ROT_TOL_DEG + 1e-6
            assert abs(s["distance"] - d["distance"]) <= SMOOTH_DIST_TOL + 1e-6
        assert perspective_series(sk, 0, 60) == perspective_series(dk, 0, 60)

    def test_smooth_preserves_fov(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd", keys=FOV_JUMP_AT_30_KEYS)
        dense = tmp_path / "dense.vmd"
        smooth = tmp_path / "smooth.vmd"
        assert cli.main([inp, "-o", str(dense), *_SHAKE_ARGS, "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(smooth), *_SHAKE_ARGS, "--smooth"]) == 0
        dk, sk = read_camera(dense), read_camera(smooth)
        for i in range(0, 241):
            f = i * 0.25
            sf = interp.sample_camera(sk, f)["fov"]
            df = interp.sample_camera(dk, f)["fov"]
            assert abs(sf - df) <= SMOOTH_FOV_TOL + 1e-6

    def test_smooth_preserves_cut_boundary(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd", keys=POSITION_CUT_AT_30_KEYS)
        dense = tmp_path / "dense.vmd"
        smooth = tmp_path / "smooth.vmd"
        assert cli.main([inp, "-o", str(dense), *_SHAKE_ARGS, "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(smooth), *_SHAKE_ARGS, "--smooth"]) == 0
        dk, sk = read_camera(dense), read_camera(smooth)
        for i in range(0, 241):
            f = i * 0.25
            d = interp.sample_camera(dk, f)
            s = interp.sample_camera(sk, f)
            assert math.dist(s["position"], d["position"]) <= SMOOTH_POS_TOL + 1e-6

    def test_smooth_passes_bake_cuts_as_keep_frames(self, tmp_path, monkeypatch):
        inp = write_input(tmp_path / "in.vmd", keys=POSITION_CUT_AT_30_KEYS)
        out = tmp_path / "smooth.vmd"
        captured = {}
        real = cli.reduce_camera_track
        def capturing(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)
        monkeypatch.setattr(cli, "reduce_camera_track", capturing)
        assert cli.main([inp, "-o", str(out), *_SHAKE_ARGS, "--smooth"]) == 0
        assert captured["no_cut_detect"] is True
        assert {29, 30} <= set(captured["keep_frames"])

    def test_smooth_writes_single_vmd_no_intermediate(self, tmp_path, monkeypatch):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "smooth.vmd"
        calls = []
        real_write = cli.io.write_file
        def counting(doc, path):
            calls.append(str(path))
            return real_write(doc, path)
        monkeypatch.setattr(cli.io, "write_file", counting)
        assert cli.main([inp, "-o", str(out), *_SHAKE_ARGS, "--smooth"]) == 0
        assert calls == [str(out)]
        assert out.exists()

    def test_smooth_matches_shared_engine_pipeline(self, tmp_path):
        dense = tmp_path / "dense.vmd"
        smooth = tmp_path / "smooth.vmd"
        assert self._bake(dense, "--no-smooth") == 0
        assert self._bake(smooth, "--smooth") == 0
        dk, sk = read_camera(dense), read_camera(smooth)
        tols = Tolerances(
            bone_pos=1.0, bone_rot=30.0,
            camera_pos=SMOOTH_POS_TOL, camera_rot=SMOOTH_ROT_TOL_DEG,
            camera_distance=SMOOTH_DIST_TOL, camera_fov=SMOOTH_FOV_TOL,
        )
        first, last = dk[0].frame, dk[-1].frame
        keep = tuple(range(first, last + 1, 5))
        pk = reduce_camera_track(
            dk, [(first, last)], tols,
            cut_thresholds=(5.0, 20.0, 5.0),
            keep_frames=keep,
            no_cut_detect=True,
            min_seg=1, max_seg=5, strict=False, curve_mode="bezier",
            force_bezier=True,
        )
        for i in range(0, 241):
            f = i * 0.25
            s = interp.sample_camera(sk, f)
            p = interp.sample_camera(pk, f)
            assert math.dist(s["position"], p["position"]) <= SMOOTH_POS_TOL + 1e-6
            for ax in range(3):
                diff = s["rotation"][ax] - p["rotation"][ax]
                diff_deg = abs(math.degrees(math.atan2(math.sin(diff), math.cos(diff))))
                assert diff_deg <= SMOOTH_ROT_TOL_DEG + 1e-6
            assert abs(s["distance"] - p["distance"]) <= SMOOTH_DIST_TOL + 1e-6
            assert abs(s["fov"] - p["fov"]) <= SMOOTH_FOV_TOL + 1e-6

    def test_smooth_on_by_default(self, tmp_path):
        default_out = tmp_path / "default.vmd"
        smooth_out = tmp_path / "smooth.vmd"
        assert self._bake(default_out) == 0
        assert self._bake(smooth_out, "--smooth") == 0
        assert default_out.read_bytes() == smooth_out.read_bytes()
        assert len(read_camera(default_out)) < 61 * 0.6

    def test_no_smooth_produces_dense_linear_keys(self, tmp_path):
        out = tmp_path / "raw.vmd"
        assert self._bake(out, "--no-smooth") == 0
        ks = read_camera(out)
        assert len(ks) == 61
        for k in ks[1:]:
            b = bytes(k.interpolation)
            assert all(not _is_curved(_channel(b, j)) for j in _ALL_CHANNEL_OFFSETS)

    def test_smooth_preserves_high_frequency_shake(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd", keys=[cam(0), cam(60)])
        dense = tmp_path / "dense.vmd"
        smooth = tmp_path / "smooth.vmd"
        hf = [*_SHAKE_ARGS, "--freq", "2.0", "--impulse", "30:60:0.08"]
        assert cli.main([inp, "-o", str(dense), *hf, "--no-smooth"]) == 0
        assert cli.main([inp, "-o", str(smooth), *hf]) == 0
        dk, sk = read_camera(dense), read_camera(smooth)
        for f in range(0, 61):
            d = interp.sample_camera(dk, f)
            s = interp.sample_camera(sk, f)
            assert math.dist(s["position"], d["position"]) <= SMOOTH_POS_TOL + 1e-6
            for ax in range(3):
                diff = s["rotation"][ax] - d["rotation"][ax]
                diff_deg = abs(math.degrees(math.atan2(math.sin(diff), math.cos(diff))))
                assert diff_deg <= SMOOTH_ROT_TOL_DEG + 1e-6
        drx = [interp.sample_camera(dk, f)["rotation"][0] for f in range(61)]
        srx = [interp.sample_camera(sk, f)["rotation"][0] for f in range(61)]
        max_step = max(abs(math.degrees(drx[f + 1] - drx[f])) for f in range(60))
        assert max_step > 2.0
        dpp = math.degrees(max(drx) - min(drx))
        spp = math.degrees(max(srx) - min(srx))
        assert dpp > 5.0
        assert spp >= dpp - 2 * SMOOTH_ROT_TOL_DEG

    def test_smooth_passes_force_bezier_true(self, tmp_path, monkeypatch):
        captured = {}
        real = cli.reduce_camera_track
        def capturing(*args, **kwargs):
            captured.update(kwargs)
            return real(*args, **kwargs)
        monkeypatch.setattr(cli, "reduce_camera_track", capturing)
        assert self._bake(tmp_path / "smooth.vmd", "--smooth") == 0
        assert captured
        assert captured.get("force_bezier") is True

    def test_smooth_makes_at_least_85_percent_of_keys_curved(self, tmp_path):
        out = tmp_path / "smooth.vmd"
        assert self._bake(out, "--smooth") == 0
        ks = [bytes(k.interpolation) for k in read_camera(out)[1:]]
        position_and_rotation = (*_POSITION_CHANNEL_OFFSETS, _ROTATION_CHANNEL_OFFSET)
        curved_keys = sum(1 for b in ks if any(_is_curved(_channel(b, j)) for j in position_and_rotation))
        assert curved_keys / len(ks) >= 0.85
