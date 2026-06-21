"""診断レポートのテスト(mocapvmd.md §4.4)。

ボーン一覧・分類結果・キー数・フレーム範囲・足IK/つま先IK候補・フレーム間速度・
回転角速度をまとめる。速度は連続フレーム差をキー間フレーム差で1フレームあたりへ正規化する。
"""

import json
import math

import pytest

from mocapvmd import presets, report

from .helpers import bone


def _quat_y(deg):
    """Y軸まわり deg 度回転の quaternion (x,y,z,w)。"""
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _entry(rep, name):
    return next(e for e in rep["bones"] if e["name"] == name)


def test_report_includes_resolved_cleaning_params():
    # 各ボーンに、選択プリセットで解決したクリーニングパラメータが付く(チューニング確認用、§4.5)。
    keys = [bone("センター", 0), bone("右足ＩＫ", 0)]
    rep = report.build_report(keys, preset="stable-foot")
    center = _entry(rep, "センター")
    assert center["cleaning"] == presets.resolve_cleaning("stable-foot", "center")
    foot = _entry(rep, "右足ＩＫ")
    assert foot["cleaning"] == presets.resolve_cleaning("stable-foot", "foot_ik")


def test_report_default_preset_is_balanced():
    keys = [bone("センター", 0)]
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["cleaning"] == presets.resolve_cleaning("balanced", "center")


def test_format_dry_run_shows_cleaning_params():
    # dry-run も適用プリセットで解決したクリーニング強度を表示する(§4.4 は dry-run と report-json の双方に要求)。
    keys = [bone("センター", 0), bone("センター", 10)]
    rep = report.build_report(keys, preset="strong")
    text = report.format_dry_run(rep)
    center_line = next(line for line in text.splitlines() if "センター" in line and "center" in line)
    assert "0.63" in center_line  # center 位置強度 0.45×1.4


def test_report_counts_spike_candidate_frames():
    # 単発スパイクを持つボーンは spike_candidates(候補フレーム数)が厳密に 1(frame5 の X のみ)。
    keys = [bone("センター", f) for f in range(11)]
    keys[5] = bone("センター", 5, pos=(0.5, 0.0, 0.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 1


def test_report_spike_candidates_are_frame_based_not_axis_based():
    # 同一フレームで複数軸が候補でも、spike_candidates はフレーム単位で数える(軸イベント数でない)。
    keys = [bone("センター", f) for f in range(11)]
    keys[5] = bone("センター", 5, pos=(0.5, 0.5, 0.0))  # X と Y が同時に候補
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 1


def test_report_counts_rotation_spike_candidates():
    # 回転の単発スパイク(frame5 で10度跳ねて戻る)も spike_candidates に数える(位置だけの集計を排除)。
    keys = [bone("頭", f) for f in range(11)]
    keys[5] = bone("頭", 5, rot=_quat_y(10.0))
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["spike_candidates"] == 1


def test_report_counts_rotation_accent_protected():
    # 回転の同方向継続(+8度/frame)アクセントも protected_frames に数える(位置だけの集計を排除)。
    keys = [bone("頭", f, rot=_quat_y(8.0 * f)) for f in range(11)]
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["protected_frames"] == 11


def test_spike_candidates_union_same_frame():
    # 位置と回転が同一フレームで候補なら、和集合のフレーム数は 1(加算でなく和集合)。
    keys = [bone("センター", f) for f in range(11)]
    keys[3] = bone("センター", 3, pos=(0.5, 0.0, 0.0), rot=_quat_y(10.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 1


def test_spike_candidates_union_distinct_frames():
    # 位置と回転が別フレームで候補なら、和集合のフレーム数は 2(max でなく和集合)。
    keys = [bone("センター", f) for f in range(11)]
    keys[3] = bone("センター", 3, pos=(0.5, 0.0, 0.0))
    keys[7] = bone("センター", 7, rot=_quat_y(10.0))
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["spike_candidates"] == 2


def test_report_counts_protected_frames():
    # 同方向継続(+0.5/frame)は全11フレームがアクセント保護対象。範囲端も含め protected_frames == 11。
    keys = [bone("センター", f, pos=(0.5 * f, 0.0, 0.0)) for f in range(11)]
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["protected_frames"] == 11


def test_protected_frames_union_combines_boundaries_pos_rot():
    # 保護フレームは boundaries・位置アクセント・回転アクセントの和集合。重複と別フレームを混ぜ、
    # 和集合数だけが正解になるよう構成する。位置アクセント frames{2,3,4}、回転アクセント frames{4,5,6}
    # (frame4 で重複)、範囲端{0,10} → 和集合 {0,2,3,4,5,6,10} = 7(sum=8・max=3 を排除)。
    xs = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ds = [0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 16.0, 16.0, 16.0, 16.0]
    keys = [bone("センター", f, pos=(xs[f], 0.0, 0.0), rot=_quat_y(ds[f])) for f in range(11)]
    rep = report.build_report(keys)
    assert _entry(rep, "センター")["protected_frames"] == 7


def test_report_still_track_candidates_and_protected():
    # 完全静止: スパイク候補0、保護は範囲端の2フレームのみ。
    keys = [bone("センター", f) for f in range(11)]
    rep = report.build_report(keys)
    e = _entry(rep, "センター")
    assert e["spike_candidates"] == 0
    assert e["protected_frames"] == 2


def test_report_single_key_track_zero_spikes_and_protected():
    # キー1個のトラックは検出できないので spike_candidates・protected_frames とも 0。
    keys = [bone("センター", 0)]
    rep = report.build_report(keys)
    e = _entry(rep, "センター")
    assert e["spike_candidates"] == 0
    assert e["protected_frames"] == 0


def test_format_dry_run_shows_spike_and_protected_counts():
    # dry-run 表示にもスパイク候補数・保護フレーム数が出る(§4.4 は dry-run と report-json の双方に要求)。
    keys = [bone("センター", f) for f in range(11)]
    keys[5] = bone("センター", 5, pos=(0.5, 0.0, 0.0))
    rep = report.build_report(keys)
    text = report.format_dry_run(rep)
    center_line = next(line for line in text.splitlines() if "センター" in line and "center" in line)
    # スパイク候補1・保護フレーム2(範囲端のみ。アクセント・カットなし)。空白トークン完全一致で
    # 桁違い(spikes=10 等)の誤実装を排除する。
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


def test_velocity_diagnostics():
    # 位置は frame0->1 で 1.0、frame1->2 で 2.0 動く。フレーム差で正規化した最大速度=2.0。
    keys = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 1, pos=(1.0, 0.0, 0.0)),
        bone("センター", 2, pos=(3.0, 0.0, 0.0)),
    ]
    rep = report.build_report(keys)
    center = _entry(rep, "センター")
    assert center["max_speed"] == pytest.approx(2.0)


def test_angular_velocity_diagnostics():
    # 1フレームで Y軸 90度回転 → 最大角速度 ≈ 90度/frame。
    keys = [
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 1, rot=_quat_y(90.0)),
    ]
    rep = report.build_report(keys)
    head = _entry(rep, "頭")
    assert head["max_ang_speed_deg"] == pytest.approx(90.0, abs=1e-3)


def test_angular_velocity_normalized_by_frame_gap():
    # キーが疎(フレーム差4)でも1フレームあたりへ正規化される。90度/4frame = 22.5度/frame。
    keys = [
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 4, rot=_quat_y(90.0)),
    ]
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["max_ang_speed_deg"] == pytest.approx(22.5, abs=1e-3)


def test_sign_flipped_quaternion_is_zero_angular_speed():
    # q と -q は同じ姿勢。符号反転を見かけの大角速度にしない(§4.2)。
    keys = [
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 1, rot=(0.0, 0.0, 0.0, -1.0)),
    ]
    rep = report.build_report(keys)
    assert _entry(rep, "頭")["max_ang_speed_deg"] == pytest.approx(0.0, abs=1e-6)


def test_velocity_uses_time_order_not_input_order():
    # io.read はキーをソートしないため順不同VMDが入力されうる。診断はトラック内を時系列順に
    # 並べてから差分を取る。入力順 frame 4,0,2 で時系列順の最大速度を検証する。
    keys = [
        bone("センター", 4, pos=(10.0, 0.0, 0.0)),
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 2, pos=(2.0, 0.0, 0.0)),
    ]
    rep = report.build_report(keys)
    # 時系列順: f0(0)->f2(2)=1.0/frame, f2(2)->f4(10)=8/2=4.0/frame。最大=4.0。
    # 入力順のまま差分を取ると f4(10)->f0(0)=10/4=2.5 が最大になり、誤実装を検出できる。
    assert _entry(rep, "センター")["max_speed"] == pytest.approx(4.0)


def test_angular_velocity_uses_time_order_not_input_order():
    # 回転角速度も時系列順で計算する。入力順 frame 4=100度, 0=0度, 2=20度(Y軸回転)。
    keys = [
        bone("頭", 4, rot=_quat_y(100.0)),
        bone("頭", 0, rot=_quat_y(0.0)),
        bone("頭", 2, rot=_quat_y(20.0)),
    ]
    rep = report.build_report(keys)
    # 時系列順: f0(0)->f2(20)=10度/frame, f2(20)->f4(100)=80/2=40度/frame。最大=40。
    # 入力順だと f4(100)->f0(0)=100/4=25 が最大になり、誤実装を検出できる。
    assert _entry(rep, "頭")["max_ang_speed_deg"] == pytest.approx(40.0, abs=1e-3)


def test_velocity_normalized_by_frame_gap():
    # キーが疎(フレーム差>1)でも、1フレームあたりへ正規化される。
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


def test_empty_bone_section():
    rep = report.build_report([])
    assert rep["bones"] == []
    assert rep["foot_ik_candidates"] == []
    assert rep["toe_ik_candidates"] == []
    # 空入力ではフレームが無いので range は空。
    assert rep["range"] == []
    # 空レポートでも dry-run 整形は例外を出さない。
    assert isinstance(report.format_dry_run(rep), str)


def test_bones_in_input_appearance_order():
    keys = [bone("右腕", 0), bone("センター", 0), bone("右腕", 1)]
    rep = report.build_report(keys)
    assert [e["name"] for e in rep["bones"]] == ["右腕", "センター"]


def test_write_json_roundtrip(tmp_path):
    keys = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 10, pos=(1.0, 0.0, 0.0)),
        bone("右足ＩＫ", 0),
        bone("右つま先ＩＫ", 0),
    ]
    rep = report.build_report(keys)
    path = tmp_path / "report.json"
    report.write_json(rep, str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    # 速度・範囲・候補一覧まで含めた完全な往復を保証する(JSON ネイティブ型のみで構成)。
    assert loaded == rep


def test_format_dry_run_shows_values_and_candidates():
    # dry-run は分類だけでなくキー数・フレーム範囲・速度を値として表示する(§4.4)。ラベル文言には
    # 依存せず、入力から確定する値とボーン名・分類・候補名で検証する。
    keys = [
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 10, pos=(5.0, 0.0, 0.0)),  # max_speed = 5/10 = 0.5
        bone("頭", 0, rot=(0.0, 0.0, 0.0, 1.0)),
        bone("頭", 1, rot=_quat_y(90.0)),           # max_ang_speed_deg = 90/frame
        bone("右足ＩＫ", 0),
        bone("右足ＩＫ", 40),
        bone("右つま先ＩＫ", 0),
    ]
    rep = report.build_report(keys)
    text = report.format_dry_run(rep)
    lines = text.splitlines()
    center_line = next(line for line in lines if "センター" in line and "center" in line)
    assert "2" in center_line       # キー数
    assert "10" in center_line      # フレーム範囲終端
    assert "0.5" in center_line     # 最大速度
    head_line = next(line for line in lines if "頭" in line and "torso" in line)
    assert "90" in head_line        # 最大回転角速度の表示
    foot_line = next(line for line in lines if "右足ＩＫ" in line and "foot_ik" in line)
    assert "40" in foot_line        # フレーム範囲終端
    # 候補一覧はボーン行とは独立して表示される(各候補名がボーン行と候補一覧で2回以上現れる)。
    assert text.count("右足ＩＫ") >= 2
    assert text.count("右つま先ＩＫ") >= 2
