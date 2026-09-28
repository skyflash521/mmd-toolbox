import math

import pytest

from mocapvmd import presets, report

from .helpers import bone


def _quat_y(deg):
    """Y軸まわり deg 度回転の quaternion を (x, y, z, w) の順で返す。"""
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _entry(rep, name):
    return next(e for e in rep["bones"] if e["name"] == name)


_FOOT_AND_TOE_GROUNDING_FIELDS = (
    "grounding_candidates", "grounding_segments", "max_change",
    "mean_change", "lock_applied_ratio", "clamp_warnings",
)


def _slow_grounded_ramp_frames_0_to_10(name="右足ＩＫ"):
    xs = [round(0.05 * i, 6) for i in range(11)]
    return [bone(name, f, pos=(x, 0.0, 0.0)) for f, x in enumerate(xs)]


def test_report_adds_grounding_diagnostics_for_foot():
    rep = report.build_report(_slow_grounded_ramp_frames_0_to_10(), denoise=False)
    e = _entry(rep, "右足ＩＫ")
    assert e["grounding_candidates"] == 11
    assert e["grounding_segments"] == [[0, 10]]
    assert e["lock_applied_ratio"] == pytest.approx(1.0)
    assert e["max_change"] > 0.0
    assert e["mean_change"] == pytest.approx(e["mean_change"])
    assert e["clamp_warnings"] == 0


def test_report_grounding_segments_are_absolute_frames():
    keys = [bone("右足ＩＫ", 100 + f, pos=(round(0.05 * f, 6), 0.0, 0.0)) for f in range(11)]
    rep = report.build_report(keys, denoise=False)
    assert _entry(rep, "右足ＩＫ")["grounding_segments"] == [[100, 110]]


def test_report_adds_grounding_diagnostics_for_toe():
    rep = report.build_report(_slow_grounded_ramp_frames_0_to_10("右つま先ＩＫ"), denoise=False)
    e = _entry(rep, "右つま先ＩＫ")
    assert e["grounding_candidates"] == 11
    assert e["grounding_segments"] == [[0, 10]]
    assert e["lock_applied_ratio"] == pytest.approx(1.0)


def test_report_grounding_reflects_denoise_then_stabilize():
    from mocapvmd import denoise as dn
    from mocapvmd import footik

    keys = _slow_grounded_ramp_frames_0_to_10()
    rep = report.build_report(keys, denoise=True)
    e = _entry(rep, "右足ＩＫ")
    foot = sorted(keys, key=lambda k: k.frame)
    params = presets.resolve_cleaning(1.0, "foot_ik")
    cpos, _ = dn.apply_denoise(
        [k.position for k in foot], [k.rotation for k in foot],
        pos_window=params["pos_window"], rot_window=params["rot_window"],
        pos_strength=params["pos_strength"], rot_strength=params["rot_strength"],
    )
    ts = footik.stabilize_foot_ik(
        {"右足ＩＫ": ("foot_ik", [k.frame for k in foot], cpos)}, 1.0
    )["右足ＩＫ"]
    assert e["grounding_segments"] == [[s.start, s.end] for s in ts.grounding.segments]
    assert e["max_change"] == pytest.approx(ts.max_change)
    assert e["mean_change"] == pytest.approx(ts.mean_change)
    assert e["lock_applied_ratio"] == pytest.approx(ts.lock_applied_ratio)


def test_report_non_foot_bone_has_no_grounding_fields():
    rep = report.build_report([bone("センター", 0), bone("センター", 10)])
    entry = _entry(rep, "センター")
    assert all(f not in entry for f in _FOOT_AND_TOE_GROUNDING_FIELDS)


def test_report_foot_ik_stabilize_flag_and_off_skips_grounding():
    rep_on = report.build_report(_slow_grounded_ramp_frames_0_to_10())
    rep_off = report.build_report(_slow_grounded_ramp_frames_0_to_10(), foot_ik_stabilize=False)
    assert rep_on["foot_ik_stabilize"] is True
    assert rep_off["foot_ik_stabilize"] is False
    assert "grounding_segments" in _entry(rep_on, "右足ＩＫ")
    off_entry = _entry(rep_off, "右足ＩＫ")
    assert all(f not in off_entry for f in _FOOT_AND_TOE_GROUNDING_FIELDS)


def test_report_counts_clamp_warning_when_ramp_exceeds_max_lock_displacement():
    step_per_frame = 0.3
    n_frames = 24
    xs = [round(step_per_frame * i, 6) for i in range(n_frames)]
    keys = [bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(xs)]
    rep = report.build_report(keys, denoise=False)
    assert _entry(rep, "右足ＩＫ")["clamp_warnings"] >= 1


def test_format_dry_run_shows_grounding_segment_and_clamp_warning_counts_for_foot():
    text = report.format_dry_run(report.build_report(_slow_grounded_ramp_frames_0_to_10(), denoise=False))
    foot_line = next(line for line in text.splitlines() if "右足ＩＫ" in line and "foot_ik" in line)
    tokens = foot_line.split()
    assert "ground_seg=1" in tokens
    assert "warn=0" in tokens


def test_report_includes_resolved_cleaning_params():
    keys = [bone("センター", 0), bone("右足ＩＫ", 0)]
    rep = report.build_report(keys, clean_strength=1.4)
    center = _entry(rep, "センター")
    assert center["cleaning"] == presets.resolve_cleaning(1.4, "center")
    foot = _entry(rep, "右足ＩＫ")
    assert foot["cleaning"] == presets.resolve_cleaning(1.4, "foot_ik")


def test_report_default_clean_strength_is_unit():
    keys = [bone("センター", 0)]
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["cleaning"] == presets.resolve_cleaning(1.0, "center")
    assert rep["clean_strength"] == pytest.approx(1.0)


def test_format_dry_run_shows_center_pos_blend_scaled_by_clean_strength():
    keys = [bone("センター", 0), bone("センター", 10)]
    rep = report.build_report(keys, clean_strength=1.4)
    text = report.format_dry_run(rep)
    center_line = next(line for line in text.splitlines() if "センター" in line and "center" in line)
    center_pos_blend_0_45_times_1_4 = "0.63"
    assert center_pos_blend_0_45_times_1_4 in center_line


def test_report_counts_spike_candidate_frames():
    keys = [bone("センター", f) for f in range(11)]
    keys[5] = bone("センター", 5, pos=(0.5, 0.0, 0.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 1


def test_report_spike_candidates_are_frame_based_not_axis_based():
    keys = [bone("センター", f) for f in range(11)]
    keys[5] = bone("センター", 5, pos=(0.5, 0.5, 0.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 1


def test_report_counts_rotation_spike_candidates():
    keys = [bone("頭", f) for f in range(11)]
    keys[5] = bone("頭", 5, rot=_quat_y(10.0))
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["spike_candidates"] == 1


def test_report_counts_rotation_accent_protected():
    keys = [bone("頭", f, rot=_quat_y(8.0 * f)) for f in range(11)]
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["protected_frames"] == 11


def test_spike_candidates_count_pos_and_rot_on_same_frame_once():
    keys = [bone("センター", f) for f in range(11)]
    keys[3] = bone("センター", 3, pos=(0.5, 0.0, 0.0), rot=_quat_y(10.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 1


def test_spike_candidates_count_pos_and_rot_on_distinct_frames_separately():
    keys = [bone("センター", f) for f in range(11)]
    keys[3] = bone("センター", 3, pos=(0.5, 0.0, 0.0))
    keys[7] = bone("センター", 7, rot=_quat_y(10.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 2


def test_report_protected_frames_cover_every_frame_of_steady_accent():
    keys = [bone("センター", f, pos=(0.5 * f, 0.0, 0.0)) for f in range(11)]
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["protected_frames"] == 11


def test_protected_frames_union_combines_boundaries_pos_rot():
    xs = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ds = [0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 16.0, 16.0, 16.0, 16.0]
    keys = [bone("センター", f, pos=(xs[f], 0.0, 0.0), rot=_quat_y(ds[f])) for f in range(11)]
    rep = report.build_report(keys)
    range_ends = {0, 10}
    pos_accent = {2, 3, 4}
    rot_accent = {4, 5, 6}
    assert _entry(rep, "センター")["protected_frames"] == len(range_ends | pos_accent | rot_accent)


def test_report_still_track_has_no_spikes_and_protects_only_range_ends():
    keys = [bone("センター", f) for f in range(11)]
    rep = report.build_report(keys)
    e = _entry(rep, "センター")
    assert e["spike_candidates"] == 0
    assert e["protected_frames"] == 2


def test_report_single_key_track_zero_spikes_and_protected():
    keys = [bone("センター", 0)]
    rep = report.build_report(keys)
    e = _entry(rep, "センター")
    assert e["spike_candidates"] == 0
    assert e["protected_frames"] == 0


def test_format_dry_run_shows_spike_and_protected_counts():
    keys = [bone("センター", f) for f in range(11)]
    keys[5] = bone("センター", 5, pos=(0.5, 0.0, 0.0))
    rep = report.build_report(keys)
    text = report.format_dry_run(rep)
    center_line = next(line for line in text.splitlines() if "センター" in line and "center" in line)
    tokens = center_line.split()
    assert "spikes=1" in tokens
    assert "protected=2" in tokens


def test_report_classifies_and_counts():
    keys = [
        bone("センター", 0),
        bone("センター", 10),
        bone("右足ＩＫ", 0),
        bone("右足ＩＫ", 5),
        bone("右足ＩＫ", 10),
    ]
    rep = report.build_report(keys)
    center = _entry(rep, "センター")
    assert center["category"] == "center"
    assert center["input_keys"] == 2
    assert center["frame_first"] == 0
    assert center["frame_last"] == 10
    foot = _entry(rep, "右足ＩＫ")
    assert foot["category"] == "foot_ik"
    assert foot["input_keys"] == 3


def test_foot_and_toe_candidates_listed():
    keys = [
        bone("右足ＩＫ", 0),
        bone("右つま先ＩＫ", 0),
        bone("センター", 0),
    ]
    rep = report.build_report(keys)
    assert rep["foot_ik_candidates"] == ["右足ＩＫ"]
    assert rep["toe_ik_candidates"] == ["右つま先ＩＫ"]


def test_max_speed_is_largest_per_frame_position_step():
    keys = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 1, pos=(1.0, 0.0, 0.0)),
        bone("センター", 2, pos=(3.0, 0.0, 0.0)),
    ]
    rep = report.build_report(keys)
    center = _entry(rep, "センター")
    assert center["max_speed"] == pytest.approx(2.0)


def test_max_ang_speed_is_rotation_degrees_per_frame():
    keys = [
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 1, rot=_quat_y(90.0)),
    ]
    rep = report.build_report(keys)
    head = _entry(rep, "頭")
    assert head["max_ang_speed_deg"] == pytest.approx(90.0, abs=1e-3)


def test_angular_velocity_normalized_by_frame_gap():
    keys = [
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 4, rot=_quat_y(90.0)),
    ]
    rep = report.build_report(keys)
    ninety_degrees_over_four_frames = 22.5
    assert _entry(rep, "頭")["max_ang_speed_deg"] == pytest.approx(ninety_degrees_over_four_frames, abs=1e-3)


def test_sign_flipped_quaternion_is_zero_angular_speed():
    keys = [
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 1, rot=(0.0, 0.0, 0.0, -1.0)),
    ]
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["max_ang_speed_deg"] == pytest.approx(0.0, abs=1e-6)


def test_velocity_uses_time_order_not_input_order():
    keys = [
        bone("センター", 4, pos=(10.0, 0.0, 0.0)),
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 2, pos=(2.0, 0.0, 0.0)),
    ]
    rep = report.build_report(keys)
    frame2_to_frame4_speed = (10.0 - 2.0) / 2
    assert _entry(rep, "センター")["max_speed"] == pytest.approx(frame2_to_frame4_speed)


def test_angular_velocity_uses_time_order_not_input_order():
    keys = [
        bone("頭", 4, rot=_quat_y(100.0)),
        bone("頭", 0, rot=_quat_y(0.0)),
        bone("頭", 2, rot=_quat_y(20.0)),
    ]
    rep = report.build_report(keys)
    frame2_to_frame4_deg_per_frame = (100.0 - 20.0) / 2
    assert _entry(rep, "頭")["max_ang_speed_deg"] == pytest.approx(frame2_to_frame4_deg_per_frame, abs=1e-3)


def test_velocity_normalized_by_frame_gap():
    keys = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 4, pos=(4.0, 0.0, 0.0)),
    ]
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["max_speed"] == pytest.approx(1.0)


def test_single_key_track_has_zero_speed():
    keys = [bone("センター", 0)]
    rep = report.build_report(keys)
    center = _entry(rep, "センター")
    assert center["input_keys"] == 1
    assert center["max_speed"] == 0.0
    assert center["max_ang_speed_deg"] == 0.0


def test_unknown_bone_kept_not_excluded():
    keys = [bone("謎ボーン", 0), bone("謎ボーン", 5)]
    rep = report.build_report(keys)
    entry = _entry(rep, "謎ボーン")
    assert entry["category"] == "unknown"


def test_overall_range():
    keys = [bone("センター", 3), bone("右腕", 9)]
    rep = report.build_report(keys)
    assert rep["range"] == [3, 9]


def test_empty_input_gives_empty_sections_and_formattable_dry_run():
    rep = report.build_report([])
    assert rep["bones"] == []
    assert rep["foot_ik_candidates"] == []
    assert rep["toe_ik_candidates"] == []
    assert rep["range"] == []
    assert isinstance(report.format_dry_run(rep), str)


def test_bones_in_input_appearance_order():
    keys = [bone("右腕", 0), bone("センター", 0), bone("右腕", 1)]
    rep = report.build_report(keys)
    assert [e["name"] for e in rep["bones"]] == ["右腕", "センター"]


def _reduction_diagnostics(name, input_keys, output_keys, *, tol_pos=0.014, tol_rot=0.14, cuts=0, errors=None):
    return {
        name: {
            "input_keys": input_keys,
            "output_keys": output_keys,
            "tol_pos": tol_pos,
            "tol_rot": tol_rot,
            "cuts": cuts,
            "errors": errors or {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0},
        }
    }


def test_report_adds_reduction_section_when_provided():
    keys = [bone("センター", f) for f in range(11)]
    errors = {"pos_x": 0.012, "pos_y": 0.003, "pos_z": 0.0, "rot_deg": 0.8}
    red = _reduction_diagnostics("センター", 11, 3, tol_pos=0.014, tol_rot=0.14, cuts=2, errors=errors)
    rep = report.build_report(keys, reduction=red)
    r = _entry(rep, "センター")["reduction"]
    assert r["output_keys"] == 3
    assert r["reduction_rate"] == pytest.approx(1.0 - 3 / 11)
    assert r["tol_pos"] == 0.014
    assert r["tol_rot"] == 0.14
    assert r["cuts"] == 2
    assert r["errors"] == errors


def test_report_reduce_flag_follows_whether_reduction_is_passed_even_if_empty():
    keys = [bone("センター", f) for f in range(11)]
    rep_off = report.build_report(keys)
    assert rep_off["reduce"] is False
    assert "reduction" not in _entry(rep_off, "センター")
    rep_on = report.build_report(keys, reduction=_reduction_diagnostics("センター", 11, 3))
    assert rep_on["reduce"] is True
    assert "reduction" in _entry(rep_on, "センター")
    rep_empty = report.build_report([], reduction={})
    assert rep_empty["reduce"] is True


def test_format_dry_run_shows_reduction():
    keys = [bone("センター", f) for f in range(11)]
    errors = {"pos_x": 0.012, "pos_y": 0.003, "pos_z": 0.0, "rot_deg": 0.8}
    red = _reduction_diagnostics("センター", 11, 3, cuts=2, errors=errors)
    text = report.format_dry_run(report.build_report(keys, reduction=red))
    assert "reduce: on" in text
    center_line = next(line for line in text.splitlines() if "センター" in line and "center" in line)
    tokens = center_line.split()
    assert "out_keys=3" in tokens
    assert "cuts=2" in tokens
    assert "red=72.7%" in tokens
    assert "tol=[0.014,0.14]" in tokens
    assert "err_pos=0.012" in tokens
    assert "err_rot=0.8deg" in tokens


def _pose_denoise_diagnostics():
    return {
        "enabled": True,
        "pmx": None,
        "frames": 11,
        "markers": {"available": 16, "required_bones_ok": True},
        "marker_displacement": {
            "max": 0.23,
            "mean": 0.05,
            "by_marker": {"head": {"max": 0.23, "mean": 0.06}},
        },
        "fit": {
            "frames": 11,
            "fallback_frames": 2,
            "mean_error_before": 0.08,
            "mean_error_after": 0.03,
            "max_bone_delta_deg": 2.4,
            "max_center_delta": 0.12,
        },
    }


def test_build_report_includes_pose_denoise_block_unchanged():
    keys = [bone("センター", 0), bone("センター", 10)]
    diag = _pose_denoise_diagnostics()
    rep = report.build_report(keys, pose_denoise=diag)
    assert rep["pose_denoise"] == diag


def test_build_report_omits_pose_denoise_when_absent():
    rep = report.build_report([bone("センター", 0), bone("センター", 10)])
    assert "pose_denoise" not in rep


def test_format_dry_run_shows_pose_summary():
    rep = report.build_report(
        [bone("センター", 0), bone("センター", 10)], pose_denoise=_pose_denoise_diagnostics()
    )
    text = report.format_dry_run(rep)
    assert "pose" in text.lower()
    assert "既定モデルプロファイル" in text
    assert "available=16" in text
    assert "required_bones_ok=True" in text
    assert "max=0.23" in text
    assert "0.08" in text and "0.03" in text
    assert "fallback=2" in text


def test_report_reduction_rate_derives_from_diagnostics_keys_and_is_zero_for_zero_input():
    keys = [bone("センター", f) for f in range(11)]
    diagnostics_input_keys = 10
    rep = report.build_report(keys, reduction=_reduction_diagnostics("センター", diagnostics_input_keys, 3))
    assert _entry(rep, "センター")["reduction"]["reduction_rate"] == pytest.approx(1.0 - 3 / diagnostics_input_keys)
    rep0 = report.build_report([bone("センター", 0)], reduction=_reduction_diagnostics("センター", 0, 0))
    assert _entry(rep0, "センター")["reduction"]["reduction_rate"] == 0.0


def test_report_reduction_only_on_named_bones():
    keys = [bone("センター", 0), bone("センター", 10), bone("右腕", 0), bone("右腕", 10)]
    rep = report.build_report(keys, reduction=_reduction_diagnostics("センター", 2, 2))
    assert "reduction" in _entry(rep, "センター")
    assert "reduction" not in _entry(rep, "右腕")


def test_format_dry_run_reduce_off_when_no_reduction():
    text = report.format_dry_run(report.build_report([bone("センター", 0)]))
    assert "reduce: off" in text


def test_format_dry_run_err_pos_is_max_of_position_axes():
    keys = [bone("センター", f) for f in range(11)]
    errors = {"pos_x": 0.002, "pos_y": 0.004, "pos_z": 0.013, "rot_deg": 0.5}
    text = report.format_dry_run(
        report.build_report(keys, reduction=_reduction_diagnostics("センター", 11, 3, errors=errors))
    )
    center_line = next(line for line in text.splitlines() if "センター" in line and "center" in line)
    assert "err_pos=0.013" in center_line.split()


def test_format_dry_run_shows_values_and_candidates():
    keys = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 10, pos=(5.0, 0.0, 0.0)),
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 1, rot=_quat_y(90.0)),
        bone("右足ＩＫ", 0),
        bone("右足ＩＫ", 40),
        bone("右つま先ＩＫ", 0),
    ]
    rep = report.build_report(keys)
    text = report.format_dry_run(rep)
    lines = text.splitlines()
    center_line = next(line for line in lines if "センター" in line and "center" in line)
    center_key_count, center_last_frame, center_max_speed = "2", "10", "0.5"
    assert center_key_count in center_line
    assert center_last_frame in center_line
    assert center_max_speed in center_line
    head_line = next(line for line in lines if "頭" in line and "torso" in line)
    head_max_ang_speed_deg = "90"
    assert head_max_ang_speed_deg in head_line
    foot_line = next(line for line in lines if "右足ＩＫ" in line and "foot_ik" in line)
    foot_last_frame = "40"
    assert foot_last_frame in foot_line
    bone_line_plus_candidate_list = 2
    assert text.count("右足ＩＫ") >= bone_line_plus_candidate_list
    assert text.count("右つま先ＩＫ") >= bone_line_plus_candidate_list


def _two_arm_bones_and_cleaned():
    keys = [bone(n, f) for n in ("右手首", "左手首") for f in range(3)]
    cleaned = [
        bone("右手首", 0), bone("右手首", 1, pos=(0.3, 0.0, 0.0)), bone("右手首", 2),
        bone("左手首", 0), bone("左手首", 1), bone("左手首", 2, rot=_quat_y(6.0)),
    ]
    return keys, cleaned


def test_report_records_change_from_input_to_cleaned_keys_per_bone():
    keys, cleaned = _two_arm_bones_and_cleaned()
    rep = report.build_report(keys, denoise=False, foot_ik_stabilize=False, cleaned_bone_keys=cleaned)
    right = _entry(rep, "右手首")["clean_change"]
    assert right["max_pos"] == pytest.approx(0.3)
    assert right["mean_pos"] == pytest.approx(0.1)
    assert right["max_rot_deg"] == pytest.approx(0.0)
    left = _entry(rep, "左手首")["clean_change"]
    assert left["max_rot_deg"] == pytest.approx(6.0)
    assert left["mean_rot_deg"] == pytest.approx(2.0)


def test_report_summarizes_change_over_all_frames_of_each_category():
    keys, cleaned = _two_arm_bones_and_cleaned()
    rep = report.build_report(keys, denoise=False, foot_ik_stabilize=False, cleaned_bone_keys=cleaned)
    arms = next(c for c in rep["categories"] if c["category"] == "arms")
    assert arms["bones"] == 2
    assert arms["max_pos_change"] == pytest.approx(0.3)
    assert arms["mean_pos_change"] == pytest.approx(0.3 / 6)
    assert arms["max_rot_change_deg"] == pytest.approx(6.0)
    assert arms["mean_rot_change_deg"] == pytest.approx(6.0 / 6)
    assert "lock_applied_ratio" not in arms


def test_report_category_summary_sums_protected_frames_and_averages_lock_ratio():
    keys = _slow_grounded_ramp_frames_0_to_10("右足ＩＫ") + [bone("左足ＩＫ", f) for f in range(11)]
    rep = report.build_report(keys, denoise=False)
    foot = next(c for c in rep["categories"] if c["category"] == "foot_ik")
    entries = [_entry(rep, "右足ＩＫ"), _entry(rep, "左足ＩＫ")]
    assert foot["protected_frames"] == sum(e["protected_frames"] for e in entries)
    assert foot["lock_applied_ratio"] == pytest.approx(sum(e["lock_applied_ratio"] for e in entries) / 2)


def test_report_without_cleaned_keys_has_no_change_statistics():
    keys, _cleaned = _two_arm_bones_and_cleaned()
    rep = report.build_report(keys, denoise=False, foot_ik_stabilize=False)
    assert "clean_change" not in _entry(rep, "右手首")
    arms = next(c for c in rep["categories"] if c["category"] == "arms")
    assert "max_pos_change" not in arms


def test_format_dry_run_shows_category_summary_line():
    keys, cleaned = _two_arm_bones_and_cleaned()
    rep = report.build_report(keys, denoise=False, foot_ik_stabilize=False, cleaned_bone_keys=cleaned)
    line = next(x for x in report.format_dry_run(rep).splitlines() if x.startswith("種別 arms"))
    assert "bones=2" in line
    assert "max_pos=0.3" in line
    assert "max_rot=6deg" in line


def test_rotation_change_is_zero_for_unchanged_non_unit_quaternion():
    keys = [bone("右手首", 0, rot=(0.0, 0.0, 0.0, 0.5))]
    rep = report.build_report(keys, denoise=False, foot_ik_stabilize=False, cleaned_bone_keys=keys)
    assert _entry(rep, "右手首")["clean_change"]["max_rot_deg"] == pytest.approx(0.0, abs=1e-9)


def test_grounding_diagnostics_come_from_denoised_keys_when_given():
    fast = [bone("右足ＩＫ", f, pos=(3.0 * f, 0.0, 0.0)) for f in range(11)]
    rep = report.build_report(
        fast, denoise=True, denoised_bone_keys=_slow_grounded_ramp_frames_0_to_10(),
    )
    e = _entry(rep, "右足ＩＫ")
    assert e["grounding_candidates"] == 11
    assert e["lock_applied_ratio"] == pytest.approx(1.0)


def test_grounding_segments_use_frames_of_denoised_keys_when_ranges_differ():
    short_input = [bone("右足ＩＫ", f) for f in range(100, 111)]
    expanded = [bone("右足ＩＫ", f) for f in range(0, 111)]
    rep = report.build_report(short_input, denoise=True, denoised_bone_keys=expanded)
    assert _entry(rep, "右足ＩＫ")["grounding_segments"] == [[0, 110]]
