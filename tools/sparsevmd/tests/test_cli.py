"""CLI のテスト。

CLI はコアの薄いラッパー: 引数解析 → VMD読み(vmd.io)→ トラック削減 →
VMD書き。終了コード: 0 正常 / 1 入力不正(VMDでない・対象セクションにキーなし)/
2 引数エラー / 3 出力書き込み失敗 / 4 strict で許容誤差を満たせない。
"""

import numpy as np
import pytest

from vmd import interp, io
from vmd.types import BoneKey, CameraKey, VmdDocument
from sparsevmd import cli

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, center=(0.0, 0.0, 0.0)):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))


def linear_camera_doc():
    # 連続フレームの線形移動(密)。
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]


def long_linear_camera_doc(n):
    # n+1 フレームの定速線形移動(密)。単一線形区間で誤差ゼロにフィットできる。
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(n + 1)]


# --- 基本動作 ---------------------------------------------------------------


def test_camera_reduce_writes_output(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    assert out.exists()
    doc, _ = io.read(str(out))
    # 連続フレームの線形移動 → 端点の2キーへ削減。
    assert [k.frame for k in doc.camera] == [0, 30]


def test_max_segment_frames_default_is_unlimited(tmp_path):
    # 既定(--max-segment-frames 未指定)は上限なし=無制限。長い定速線形区間を
    # maxspan-cap で機械的に分割せず端点2キーへ削減する。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=long_linear_camera_doc(360))
    out = tmp_path / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 360]


def test_max_segment_frames_explicit_caps_span(tmp_path):
    # 明示した有限上限は維持する。同じ定速線形でも上限を超える非定数区間は maxspan-cap で分割する。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=long_linear_camera_doc(360))
    out = tmp_path / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                     "--max-segment-frames", "180"])
    assert code == 0
    doc, _ = io.read(str(out))
    frames = [k.frame for k in doc.camera]
    assert frames[0] == 0 and frames[-1] == 360
    assert len(frames) > 2


def test_max_segment_frames_zero_is_arg_error(tmp_path):
    # 明示指定は1以上の整数。0 は引数エラー(終了コード2)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--max-segment-frames", "0"])
    assert code == 2


EASE = (96, 0, 96, 30)


def eased_camera_doc():
    # 位置Xが1本の強いイージング曲線で動く密なカメラ(曲線フィット検証用)。
    return [
        cam(f, center=(30.0 * interp._solve_factor(*EASE, f / 30.0), 0.0, 0.0))
        for f in range(31)
    ]


def test_curve_mode_bezier_reduces_curved_motion(tmp_path):
    # 既定(bezier)は曲線をベジェ1本で表し両端2キーへ削減、linear は多数に分割する。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=eased_camera_doc())

    bez_out = tmp_path / "bez.vmd"
    assert cli.main([str(src), "-o", str(bez_out), "--target", "camera", "--no-cut-detect"]) == 0
    bez, _ = io.read(str(bez_out))

    lin_out = tmp_path / "lin.vmd"
    assert cli.main(
        [str(src), "-o", str(lin_out), "--target", "camera", "--curve-mode", "linear",
         "--no-cut-detect"]
    ) == 0
    lin, _ = io.read(str(lin_out))

    assert [k.frame for k in bez.camera] == [0, 30]
    assert len(lin.camera) > 2
    # bezier の到達キーには非線形の位置X制御点が入る(線形ブロックと異なる)。
    assert bez.camera[-1].interpolation[0:4] != bytes([20, 107, 20, 107])


def test_default_output_path(tmp_path):
    src = tmp_path / "motion.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    assert (tmp_path / "motion_sparse.vmd").exists()


def test_non_target_section_passthrough(tmp_path):
    # bone を対象にし、camera セクションは透過(バイト保持)される。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    cam_keys = [cam(0), cam(30)]
    bone_keys = [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    write_vmd(src, camera=cam_keys, bone=bone_keys)
    code = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert code == 0
    doc, _ = io.read(str(out))
    # camera は対象外なのでバイト等価で透過(値・補間ブロックがそのまま)。
    assert doc.camera == cam_keys
    # bone は削減される。
    assert len(doc.bone) < 31


# --- 入力正規化 ------------------------------------------------------


def test_duplicate_frame_camera_last_wins(tmp_path):
    # 同一フレーム重複は後勝ち。frame0 を 2 回入れ、後の値が採用される。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [cam(0, center=(0.0, 0.0, 0.0)), cam(0, center=(5.0, 0.0, 0.0)),
            cam(10, center=(5.0, 0.0, 0.0))]
    write_vmd(src, camera=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 0
    doc, _ = io.read(str(out))
    frames = [k.frame for k in doc.camera]
    assert len(frames) == len(set(frames))  # 重複フレームなし
    assert interp.sample(doc.camera, "pos_x", 0) == pytest.approx(5.0)  # 後勝ち


def test_single_key_camera_preserved_verbatim(tmp_path):
    # 1 キー以下のトラックは削減不能として逐語保持(値・補間ブロック不変)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    ease_blk = bytes([40, 90, 10, 118]) + bytes([20, 107, 20, 107]) * 5
    key = CameraKey(7, -30.0, (3.0, 0.0, 0.0), (0.0, 0.0, 0.0), ease_blk, 30, 0)
    write_vmd(src, camera=[key])
    code = cli.main([str(src), "-o", str(out), "--target", "camera"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert doc.camera == [key]  # 補間ブロック含め完全一致


def test_single_key_bone_preserved_verbatim(tmp_path):
    # 選択ボーンでもキー1件なら逐語保持(補間ブロック不変)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    nonlinear = bytearray(_bone_linear())
    nonlinear[0] = 40  # X 軸 x1 を非線形値に
    key = BoneKey("センター".encode("cp932").ljust(15, b"\x00"), 5, (1.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 1.0), bytes(nonlinear))
    write_vmd(src, bone=[key])
    code = cli.main([str(src), "-o", str(out), "--target", "bone"])
    assert code == 0
    doc, _ = io.read(str(out))
    assert doc.bone == [key]


# --- 削減対象なし記録 / verbose ---------------


def test_dry_run_records_no_reduction_target(tmp_path, capsys):
    # 全トラック1キー以下 → dry-run に「削減対象なし」を記録。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=[cam(7, center=(3.0, 0.0, 0.0))])
    code = cli.main([str(src), "--target", "camera", "--dry-run"])
    assert code == 0
    assert "削減対象なし" in capsys.readouterr().out


def test_verbose_logs_diagnostics(tmp_path, capsys):
    # -v 指定で不連続検出位置などの診断を stderr に出す。レポート系フラグ無しでも。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0))
            for f in range(31)]
    write_vmd(src, camera=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "-v"])
    assert code == 0
    assert "15" in capsys.readouterr().err  # 不連続検出位置 frame15 が verbose ログに出る


# --- CLI 堅牢化 -----------------------------------------


def test_bone_file_decode_error_is_arg_error(tmp_path):
    # --bone-file が UTF-8 でデコードできない場合は引数エラー(終了コード2)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(11)])
    bf = tmp_path / "bones.txt"
    bf.write_bytes(b"\xff\xfe\x00 invalid utf8")
    code = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "bone",
                     "--bone-file", str(bf)])
    assert code == 2


def test_keep_frame_out_of_range_warns(tmp_path, capsys):
    # 全削減範囲外の keep-frame は警告して無視する。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                     "--range", "0:10", "--keep-frame", "50"])
    assert code == 0
    err = capsys.readouterr().err
    assert "50" in err and ("警告" in err or "keep" in err.lower())


def test_list_bones_zero_bones_warns_unmatched(tmp_path, capsys):
    # ボーン0件の VMD で --bone 指定 + --list-bones → 未一致選択子を警告。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=[cam(0), cam(30)])
    code = cli.main([str(src), "--list-bones", "--bone", "存在しない"])
    assert code == 0
    err = capsys.readouterr().err
    assert "警告" in err


# --- list-bones -------------------------------------------------------------


def test_list_bones_shows_name_count_and_selection(tmp_path, capsys):
    # ボーン名・キー数・選択状態を表示する。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30), bone("頭", 0)])
    code = cli.main([str(src), "--bone", "センター", "--list-bones"])
    assert code == 0
    out = capsys.readouterr().out
    assert "センター" in out and "頭" in out
    assert "2" in out  # センターのキー数
    # 選択状態(選択/非選択)が表示される。
    low = out.lower()
    assert "select" in low or "選択" in out


def test_list_bones_does_not_write_output_and_exits_zero(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0)])
    code = cli.main([str(src), "-o", str(out), "--list-bones"])
    assert code == 0
    assert not out.exists()


# --- 終了コード -------------------------------------------------------------


def test_missing_input_is_arg_error(tmp_path):
    # 入力パスが存在しない → 引数エラー(コード2、読み込み前のパス検証)。
    code = cli.main([str(tmp_path / "nope.vmd"), "--target", "camera"])
    assert code == 2


def test_bad_range_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--range", "20:10"])
    assert code == 2


def test_target_camera_with_bone_selection_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", 0)])
    code = cli.main([str(src), "--target", "camera", "--bone", "センター"])
    assert code == 2


def test_fov_tol_below_half_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--camera-fov-tol", "0.4"])
    assert code == 2


def test_same_path_without_overwrite_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(src), "--target", "camera"])
    assert code == 2


def test_same_path_with_overwrite_ok(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(src), "--overwrite", "--target", "camera", "--curve-mode", "linear"])
    assert code == 0


def test_target_camera_no_camera_keys_is_input_error(tmp_path):
    # camera 対象だがカメラキー無し → 入力不正(コード1)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    code = cli.main([str(src), "--target", "camera"])
    assert code == 1


def test_bone_file_missing_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    code = cli.main([str(src), "--target", "bone", "--bone-file", str(tmp_path / "nope.txt")])
    assert code == 2


def test_keep_frame_negative_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--keep-frame", "-3"])
    assert code == 2


def test_target_all_explicit_bone_absent_section_is_arg_error(tmp_path):
    # target all・カメラあり・ボーンキー無し・明示 --bone NAME → 引数エラー。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "all", "--bone", "存在しない"])
    assert code == 2


def test_target_bone_explicit_missing_name_is_arg_error_not_input(tmp_path):
    # target bone・ボーンキー無し・明示 --bone NAME → 引数エラー2(空セクションの入力不正1より優先)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())  # bone セクション無し
    code = cli.main([str(src), "--target", "bone", "--bone", "存在しない"])
    assert code == 2


def test_unmatched_glob_warns_in_reduce_path(tmp_path, capsys):
    # 削減パスでも不一致 glob の警告を出す(list-bones だけでなく)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    bones = [bone("頭", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    write_vmd(src, bone=bones)
    code = cli.main(
        [str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear",
         "--bone", "頭", "--bone-glob", "幻*"]
    )
    assert code == 0
    assert "幻*" in capsys.readouterr().err


def test_sole_unmatched_glob_warns_before_exit2(tmp_path, capsys):
    # 唯一の include が不一致 glob → 最終0件で SelectionError(コード2)。
    # 終了前に不一致警告を stderr に出す。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    code = cli.main([str(src), "--target", "bone", "--bone-glob", "幻*"])
    assert code == 2
    assert "幻*" in capsys.readouterr().err


def test_range_intersect_preserves_outside(tmp_path):
    # --range 0:10 のみ削減。範囲内[0,10]は端点へ、範囲外(15..30)は元キー保持。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    cam_keys = [cam(f, center=(float(f), 0.0, 0.0)) for f in (0, 5, 10, 15, 20, 25, 30)]
    write_vmd(src, camera=cam_keys)
    code = cli.main(
        [str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--range", "0:10"]
    )
    assert code == 0
    doc, _ = io.read(str(out))
    fr = [k.frame for k in doc.camera]
    assert 0 in fr and 10 in fr  # 範囲内は端点に削減
    assert 5 not in fr  # 範囲内の中間は削減
    assert 15 in fr and 20 in fr and 25 in fr and 30 in fr  # 範囲外は保持


def test_range_expand_start_after_end_is_arg_error(tmp_path):
    # 999: は対象トラック末尾(30)に展開され 999>30 → 引数エラー。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--range", "999:"])
    assert code == 2


def test_dry_run_prints_stats_and_no_output(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--dry-run"])
    assert code == 0
    assert not out.exists()  # dry-run は出力VMDを書かない
    text = capsys.readouterr().out
    assert "camera" in text and "31" in text  # 統計が表示される


def test_report_json_option_is_removed(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--report-json", str(tmp_path / "report.json")])
    assert code == 2


def test_preview_csv_option_is_removed(tmp_path):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "--target", "camera", "--preview-csv", str(tmp_path / "preview.csv")])
    assert code == 2


def test_output_parent_missing_is_write_error(tmp_path):
    # 出力先の親ディレクトリが存在しない → 出力書き込み失敗(コード3)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "nodir" / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 3


def test_strict_unsatisfiable_is_exit_4(tmp_path):
    # ジグザグで線形表現不能。min-segment-frames を大きくし strict にすると、分割下限まで
    # 分割しても許容を満たせず終了コード4。
    src = tmp_path / "in.vmd"
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    out = tmp_path / "out.vmd"
    code = cli.main(
        [
            str(src),
            "-o",
            str(out),
            "--target",
            "camera",
            "--curve-mode",
            "linear",
            "--strict",
            "--min-segment-frames",
            "8",
            "--max-segment-frames",
            "180",
        ]
    )
    assert code == 4


# --- デコード不能ボーン名 -------------------------------------------

def undec_bone(frame, pos=(0.0, 0.0, 0.0)):
    # CP932 デコード不能なボーン名フィールド(孤立したリードバイト 0x81)。
    return BoneKey(b"\x81".ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def test_undecodable_bone_name_warns_and_reduces(tmp_path, capsys):
    # デコード不能名は警告しつつ、トラックの削減自体は継続する。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [undec_bone(0), undec_bone(10, (1.0, 0.0, 0.0)), undec_bone(20, (2.0, 0.0, 0.0))]
    write_vmd(src, bone=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert code == 0
    err = capsys.readouterr().err
    assert "デコード" in err or "Shift-JIS" in err
    assert out.exists()
    # 線形に並んだ中間キーが落ち、端点2キー(frame 0/20)へ削減されたことを確認する。
    out_doc, _ = io.read(str(out))
    assert [k.frame for k in out_doc.bone] == [0, 20]


def test_undecodable_name_not_targetable_by_bone(tmp_path):
    # 置換文字名を --bone で明示しても一致せず、入力に存在しない扱いでコード2。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [undec_bone(0), undec_bone(10), undec_bone(20)]
    write_vmd(src, bone=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "bone", "--bone", "�"])
    assert code == 2


# --- 警告行の共通書式 -------------------------------------------------


def _single_stderr_line(err):
    lines = err.splitlines()
    assert len(lines) == 1
    return lines[0]


@pytest.mark.xfail(reason="impl pending: sparsevmd-warning-line", strict=True)
def test_read_warning_line_uses_common_format(tmp_path, capsys):
    # デコード不能なボーン名の読み込み警告(vmd.io の decode-error)が共通書式で1行にまとまる。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [undec_bone(0), undec_bone(10, (1.0, 0.0, 0.0)), undec_bone(20, (2.0, 0.0, 0.0))]
    write_vmd(src, bone=keys)
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert out_text == ""
    line = _single_stderr_line(err)
    prefix = "warning: decode-error: "
    assert line.startswith(prefix)
    body = line[len(prefix):]
    assert body.strip()
    assert body.lstrip() == body
    assert "警告:" not in err


@pytest.mark.xfail(reason="impl pending: sparsevmd-warning-line", strict=True)
def test_selector_unmatched_warning_line_uses_common_format(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("頭", f) for f in range(31)])
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear",
                   "--bone", "頭", "--bone-glob", "幻*"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert out_text == ""
    line = _single_stderr_line(err)
    prefix = "warning: selector_unmatched: "
    assert line.startswith(prefix)
    body = line[len(prefix):]
    assert body.strip()
    assert body.lstrip() == body
    assert "警告:" not in err


@pytest.mark.xfail(reason="impl pending: sparsevmd-warning-line", strict=True)
def test_keep_frame_ignored_warning_line_uses_common_format(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                   "--range", "0:10", "--keep-frame", "50"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    assert out_text == ""
    line = _single_stderr_line(err)
    prefix = "warning: keep_frame_ignored: "
    assert line.startswith(prefix)
    body = line[len(prefix):]
    assert body.strip()
    assert body.lstrip() == body
    assert "警告:" not in err


@pytest.mark.xfail(reason="impl pending: sparsevmd-warning-line", strict=True)
def test_selection_unresolved_warning_line_uses_common_format(tmp_path, capsys):
    # selector_unmatched(蓄積分)と selection_unresolved(理由)の2行が、それぞれ共通書式になる。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    rc = cli.main([str(src), "--list-bones", "--bone-glob", "幻*"])
    assert rc == 0
    out_text, err = capsys.readouterr()
    # --list-bones の一覧テキストは仕様どおり標準出力へ出るが(name\tkeys=n\tstate)、
    # 警告(旧ラベル含む)が標準出力へ紛れ込んでいないことを確認する。
    assert "warning" not in out_text
    assert "警告:" not in out_text
    lines = err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("warning: selector_unmatched: ")
    assert lines[1].startswith("warning: selection_unresolved: ")
    for line in lines:
        label, code, body = line.split(": ", 2)
        assert label == "warning" and code
        assert body.strip()
        assert body.lstrip() == body
    assert "警告:" not in err
