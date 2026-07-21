"""mocapvmd CLI の表現空間ノイズ除去(pose モード)統合テスト。

--denoise-mode bone|pose と --pmx を扱う。pose モードは既定モデルプロファイル
または指定PMXで動き、必須標準ボーン不足・PMX形式不正・パス不在/非通常ファイルは
いずれも入力不正(終了コード1)になる。
"""

import pytest

from vmd import io
from mocapvmd import cli
from mocapvmd.model_profile import STANDARD_BONE_NAMES

from .helpers import bone, build_standard_pmx, write_vmd


def _write_input(path):
    write_vmd(
        path,
        bone=[
            bone("センター", 0),
            bone("センター", 10, pos=(0.3, 0.0, 0.0)),
            bone("頭", 0),
            bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
        ],
    )


def test_bone_mode_is_default(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_input(src)
    code = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert code == 0
    assert out.is_file()


def test_pose_mode_default_profile(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_input(src)
    code = cli.main([str(src), "-o", str(out), "--denoise-mode", "pose", "--no-reduce"])
    assert code == 0
    doc, _ = io.read(str(out))
    names = {k.name for k in doc.bone}
    assert "センター" in names and "頭" in names
    # pose 経路は密キー(全フレーム)を出す(bone 経路は入力フレームのまま=0,10)。
    center_frames = sorted(k.frame for k in doc.bone if k.name == "センター")
    assert center_frames == list(range(0, 11))


def test_pose_mode_with_pmx(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    pmx = tmp_path / "model.pmx"
    _write_input(src)
    pmx.write_bytes(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    code = cli.main(
        [str(src), "-o", str(out), "--denoise-mode", "pose", "--pmx", str(pmx), "--no-reduce"]
    )
    assert code == 0
    doc, _ = io.read(str(out))
    # 指定PMXの pose 経路でも密キー化される。
    center_frames = sorted(k.frame for k in doc.bone if k.name == "センター")
    assert center_frames == list(range(0, 11))


def test_pose_pmx_missing_path_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main(
        [
            str(src),
            "-o",
            str(tmp_path / "out.vmd"),
            "--denoise-mode",
            "pose",
            "--pmx",
            str(tmp_path / "nope.pmx"),
        ]
    )
    assert code == 1


def test_pose_pmx_directory_is_input_error(tmp_path):
    # --pmx が非通常ファイル(ディレクトリ)でも入力不正。
    src = tmp_path / "in.vmd"
    _write_input(src)
    d = tmp_path / "pmxdir"
    d.mkdir()
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(d)]
    )
    assert code == 1


def test_pose_pmx_unsupported_encoding_is_input_error(tmp_path):
    # 未対応文字コードのPMXは PmxFormatError → 入力不正(終了コード1)。
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "model.pmx"
    _write_input(src)
    data = bytearray(build_standard_pmx(list(STANDARD_BONE_NAMES.values())))
    data[9] = 5  # globals[0] エンコード方式(0/1以外)
    pmx.write_bytes(bytes(data))
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(pmx)]
    )
    assert code == 1


def test_pose_pmx_missing_required_role_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "model.pmx"
    _write_input(src)
    # 必須標準ボーンを1つ欠いたPMX。
    names = [n for n in STANDARD_BONE_NAMES.values() if n != STANDARD_BONE_NAMES["wrist_r"]]
    pmx.write_bytes(build_standard_pmx(names))
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(pmx)]
    )
    assert code == 1


def test_pose_pmx_format_error_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    pmx = tmp_path / "bad.pmx"
    _write_input(src)
    pmx.write_bytes(b"NOTPMX")
    code = cli.main(
        [str(src), "-o", str(tmp_path / "out.vmd"), "--denoise-mode", "pose", "--pmx", str(pmx)]
    )
    assert code == 1


def test_no_denoise_pose_mode_needs_no_pmx(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_input(src)
    # --no-denoise は pose モード指定でもPMX不要で動く。
    code = cli.main(
        [str(src), "-o", str(out), "--denoise-mode", "pose", "--no-denoise", "--no-reduce"]
    )
    assert code == 0
    assert out.is_file()


# --- pose モードの診断表示(dry-run) ----------------------------------


def test_pose_dry_run_has_pose_denoise_summary(tmp_path, capsys):
    # pose モードの dry-run に pose_denoise 要約が出る。
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main([str(src), "--dry-run", "--denoise-mode", "pose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "pose_denoise" in out
    assert "既定モデルプロファイル" in out  # --pmx 未指定=既定モデルプロファイル
    assert "available=" in out
    assert "required_bones_ok=True" in out
    assert "marker_disp:" in out
    assert "fit:" in out


def test_pose_dry_run_shows_pose_summary(tmp_path, capsys):
    # pose モードの dry-run 表示に pose_denoise 要約が出る。
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main([str(src), "--dry-run", "--denoise-mode", "pose"])
    assert code == 0
    out = capsys.readouterr().out
    assert "pose_denoise" in out
    assert "既定モデルプロファイル" in out  # --pmx 未指定の明示
    assert "available=" in out
    assert "fallback=" in out


def test_bone_mode_dry_run_omits_pose_denoise(tmp_path, capsys):
    # 既定 bone モードの dry-run に pose_denoise 要約は出ない。
    src = tmp_path / "in.vmd"
    _write_input(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == 0
    assert "pose_denoise" not in capsys.readouterr().out
