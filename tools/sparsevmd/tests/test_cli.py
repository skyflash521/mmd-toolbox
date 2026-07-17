"""CLI のテスト。

CLI はコアの薄いラッパー: 引数解析 → VMD読み(vmd.io)→ トラック削減 →
VMD書き。終了コード: 0 正常 / 1 入力不正(VMDでない・対象セクションにキーなし)/
2 引数エラー / 3 出力書き込み失敗 / 4 strict で許容誤差を満たせない。
"""

import sys

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
    # -v 指定で不連続検出位置などの診断を標準出力に出す(利用者が要求した詳細診断は標準出力へ、の
    # 規約に従う)。レポート系フラグ無しでも。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0))
            for f in range(31)]
    write_vmd(src, camera=keys)
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "-v"])
    assert code == 0
    cap = capsys.readouterr()
    assert "15" in cap.out  # 不連続検出位置 frame15 が verbose ログに出る
    assert "15" not in cap.err


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
    assert "50" in err and "warning: keep_frame_ignored: " in err


def test_list_bones_zero_bones_warns_unmatched(tmp_path, capsys):
    # ボーン0件の VMD で --bone 指定 + --list-bones → 未一致選択子を警告。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=[cam(0), cam(30)])
    code = cli.main([str(src), "--list-bones", "--bone", "存在しない"])
    assert code == 0
    err = capsys.readouterr().err
    assert "warning: selection_unresolved: " in err


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


# --- 進捗ライブ表示の終端処理(全終了経路で close、正常終了時のみ完了行) -------------------


class _SpyProgressReporter:
    """ProgressReporter の差し替え。close/summary の呼び出しを、共有の calls リストへ記録する
    (下記 spy_progress フィクスチャが標準エラーへの print 呼び出しも同じリストへ記録するため、
    close とエラー行表示の相対順序を1本のタイムラインで検証できる)。"""

    calls = None  # クラス変数: monkeypatch 先のコンストラクタから書けるよう、テストごとにリセットする

    def __init__(self, *args, **kwargs):
        pass

    def stage(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        _SpyProgressReporter.calls.append("close")

    def summary(self, message):
        _SpyProgressReporter.calls.append(("summary", message))


@pytest.fixture
def spy_progress(monkeypatch):
    calls = []
    _SpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _SpyProgressReporter)

    real_print = print

    def spy_print(*args, **kwargs):
        if kwargs.get("file") is sys.stderr and args:
            calls.append(("stderr_print", args[0]))
        real_print(*args, **kwargs)

    monkeypatch.setattr(cli, "print", spy_print, raising=False)
    return calls


def test_normal_run_closes_progress_then_shows_completion(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    # close は冪等なので、正常終了の明示的な close/summary の後に、全終了経路を保証する
    # 保険としての再呼び出しが続いてもよい(先頭2件の順序だけを固定する)。
    assert spy_progress[:2] == ["close", ("summary", f"完了 {out}")]
    assert all(call == "close" for call in spy_progress[2:])


def test_dry_run_closes_progress_without_completion_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "--target", "camera", "--curve-mode", "linear", "--dry-run"])
    assert rc == 0
    assert spy_progress[0] == "close"
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_strict_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    out = tmp_path / "out.vmd"
    rc = cli.main([
        str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
        "--strict", "--min-segment-frames", "8", "--max-segment-frames", "8",
    ])
    assert rc == 4
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_write_failure_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "nodir" / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 3
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_keyboard_interrupt_closes_progress_before_error_line(tmp_path, monkeypatch, spy_progress):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())

    def raise_interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "reduce_camera_track", raise_interrupt)
    rc = cli.main([str(src), "--target", "camera", "--curve-mode", "linear", "--dry-run"])
    assert rc == 130
    assert spy_progress[0] == "close"
    assert any(entry[0] == "stderr_print" for entry in spy_progress[1:] if isinstance(entry, tuple))
    assert not any(isinstance(c, tuple) and c[0] == "summary" for c in spy_progress)


def test_verbose_closes_progress_before_diagnostics(tmp_path, monkeypatch, spy_progress):
    # --verbose の詳細診断は標準出力へ出るが、同じ端末画面上でライブ行と重ならないよう、
    # 診断を出す前に close していることをタイミング非依存に検証する
    # (_log_diagnostics 呼び出し自体を記録し、close との相対順序を見る)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    monkeypatch.setattr(cli, "_log_diagnostics", lambda *a, **k: spy_progress.append("log_diagnostics"))
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "-v"])
    assert rc == 0
    assert "close" in spy_progress and "log_diagnostics" in spy_progress
    assert spy_progress.index("close") < spy_progress.index("log_diagnostics")


# --- 進捗ライブ表示の工程名・対象補足 ----------------------------------------
#
# 実際の描画結果(stderr 上の文字列)はハートビートスレッドの再描画タイミング(既定 0.15 秒間隔)に
# 依存し、小さい入力では描画前に処理が終わって非決定的になりうる。そのため stage/update の
# 呼び出し引数そのものをスパイで記録し、描画待ちに依存せず工程名・対象補足を検証する。


class _LabelSpyProgressReporter:
    """ProgressReporter の差し替え。stage/update の呼び出し引数を記録する(描画タイミングに
    依存しない検証のため)。close/summary は no-op。"""

    calls = None

    def __init__(self, *args, **kwargs):
        pass

    def stage(self, label):
        _LabelSpyProgressReporter.calls.append(("stage", label))

    def update(self, done, total, note=""):
        _LabelSpyProgressReporter.calls.append(("update", done, total, note))

    def close(self):
        pass

    def summary(self, message):
        pass


def test_progress_label_is_keyframe_reduction_with_target_note(tmp_path, monkeypatch):
    # 進捗ライブ表示の工程名は camera・bone とも同じ共有名称に統一し、対象(カメラ/ボーン名)は
    # 行末の補足(note)として出す(工程を対象ごとに個別登録しない不変条件)。
    calls = []
    _LabelSpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _LabelSpyProgressReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", f, pos=(0.0, float(f), 0.0))
                                                      for f in range(31)])
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "all", "--curve-mode", "linear"])
    assert rc == 0
    stages = [c for c in calls if c[0] == "stage"]
    updates = [c for c in calls if c[0] == "update"]
    assert stages and all(label == "キーフレーム圧縮" for _, label in stages)
    assert any(note == "カメラ" for _, _done, _total, note in updates)
    assert any(note == "センター" for _, _done, _total, note in updates)


def test_progress_camera_note_preserves_reducer_note(tmp_path, monkeypatch):
    # reduce_camera_track が渡す非空 note(出力後検証区間の補足等)を対象名の後ろへ残すことを、
    # 実際のリデューサ挙動に依存せずコールバック引数だけで検証する。
    calls = []
    _LabelSpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _LabelSpyProgressReporter)

    def fake_reduce_camera_track(cam, cam_ranges, tols, cut_thresholds=None, diagnostics=None,
                                  progress=None, **kw):
        if progress is not None:
            progress(1, 2, "出力後検証")
        return cam

    monkeypatch.setattr(cli, "reduce_camera_track", fake_reduce_camera_track)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    updates = [c for c in calls if c[0] == "update"]
    assert any(note == "カメラ 出力後検証" for _, _done, _total, note in updates)


def test_progress_bone_note_shows_target_before_processing(tmp_path, monkeypatch):
    # bone 段の note は「今から処理するボーン名」を示す(§6.1 の「現在対象」)。処理完了後の
    # 名前ではないことを、reduce_bone_track 呼び出し時点で直前の update が既に自分の名前で
    # あることを確認して検証する(呼び出し順序に直接依存し描画タイミングに依存しない)。
    calls = []
    _LabelSpyProgressReporter.calls = calls
    monkeypatch.setattr(cli.progress, "ProgressReporter", _LabelSpyProgressReporter)

    seen_notes_at_call = []

    def fake_reduce_bone_track(keys, track_ranges, tols, cut_thresholds=None, diagnostics=None, **kw):
        updates = [c for c in calls if c[0] == "update"]
        seen_notes_at_call.append(updates[-1][3] if updates else None)
        return keys

    monkeypatch.setattr(cli, "reduce_bone_track", fake_reduce_bone_track)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=(
        [bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
        + [bone("上半身", f, pos=(0.0, float(f), 0.0)) for f in range(31)]
    ))
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear"])
    assert rc == 0
    assert seen_notes_at_call == ["センター", "上半身"]


# --- 進捗ライブ表示の有効化配線(TTY・quiet・機械モード) --------------------------


class _EnabledCapturingReporter:
    """ProgressReporter の差し替え。コンストラクタへ渡された enabled 値だけを記録する。"""

    captured_enabled = None

    def __init__(self, *args, **kwargs):
        _EnabledCapturingReporter.captured_enabled = kwargs.get("enabled")

    def stage(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def summary(self, message):
        pass


class _TTYWrapper:
    """既存の stream をラップし、isatty() だけ True を返す(capsys 捕捉ストリームへの委譲)。"""

    def __init__(self, stream):
        self._stream = stream

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self._stream, name)


def test_progress_disabled_when_not_tty(tmp_path, monkeypatch):
    # pytest の capsys 捕捉ストリームは非TTYなので、既定(--quiet 無し・非機械)でも無効。
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is False


def test_progress_enabled_when_tty_and_not_quiet_and_not_machine(tmp_path, monkeypatch):
    # TTY かつ --quiet 無し・非機械なら有効(肯定ケース)。
    monkeypatch.setattr(sys, "stderr", _TTYWrapper(sys.stderr))
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is True


def test_progress_disabled_with_quiet_flag_even_when_tty(tmp_path, monkeypatch):
    # TTY であっても --quiet 指定時は無効(isatty だけを見る誤実装を弾くため TTY 化して検証する)。
    monkeypatch.setattr(sys, "stderr", _TTYWrapper(sys.stderr))
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--quiet"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is False


def test_progress_disabled_in_machine_mode_even_when_tty(tmp_path, monkeypatch):
    # 機械モードは進捗を progress イベントで出すため、人間向けライブ表示は無効。TTY であっても
    # 機械モードなら無効になることを、isatty だけを見る誤実装を弾く形で検証する。
    monkeypatch.setattr(sys, "stderr", _TTYWrapper(sys.stderr))
    monkeypatch.setattr(cli.progress, "ProgressReporter", _EnabledCapturingReporter)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--machine"])
    assert rc == 0
    assert _EnabledCapturingReporter.captured_enabled is False
