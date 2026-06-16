"""CLI のテスト(sparsevmd.md §2, §9)。

CLI はコアの薄いラッパー: 引数解析 → VMD読み(mmd_toolbox.vmd.io)→ トラック削減 →
VMD書き。終了コード(§9): 0 正常 / 1 入力不正(VMDでない・対象セクションにキーなし)/
2 引数エラー / 3 出力書き込み失敗 / 4 strict で許容誤差を満たせない。
"""

import numpy as np
import pytest

from mmd_toolbox.vmd import io
from mmd_toolbox.vmd.types import BoneKey, CameraKey, VmdDocument
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


def test_curve_mode_bezier_falls_back_to_linear(tmp_path):
    # bezier 未実装のため、bezier 指定(および既定)でも linear で削減して正常終了する。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    code = cli.main([str(src), "-o", str(out), "--target", "camera"])  # curve-mode 既定
    assert code == 0
    doc, _ = io.read(str(out))
    assert [k.frame for k in doc.camera] == [0, 30]


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


# --- list-bones -------------------------------------------------------------


def test_list_bones_shows_name_count_and_selection(tmp_path, capsys):
    # ボーン名・キー数・選択状態を表示する(§2.7)。
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
    # 入力パスが存在しない → 引数エラー(コード2、読み込み前のパス検証。§2.2)。
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


def test_output_parent_missing_is_write_error(tmp_path):
    # 出力先の親ディレクトリが存在しない → 出力書き込み失敗(コード3。§9)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    out = tmp_path / "nodir" / "out.vmd"
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear"])
    assert code == 3


def test_strict_unsatisfiable_is_exit_4(tmp_path):
    # ジグザグで線形表現不能。min-segment-frames を大きくし strict にすると、分割下限まで
    # 分割しても許容を満たせず終了コード4(§9)。
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
