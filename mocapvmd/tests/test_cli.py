"""mocapvmd CLI のテスト(mocapvmd.md §3)。

CLI は引数解析 → VMD読み → クリーニング → 疎化 → VMD書き。終了コード:
0 正常 / 1 入力不正 / 2 引数エラー / 3 出力書き込み失敗。

本テストは基盤(入出力・パス検証・上書きガード・dry-run・対象外セクション透過)を扱う。
"""

import json

import pytest

from mmd_toolbox.vmd import io
from mocapvmd import cli

from .helpers import (
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


def _full_doc(path):
    """全セクションにキーを持つVMDを書き出す。透過確認用。

    ボーン・カメラには非線形の補間バイトを入れ、再構築・線形化されれば検出できるようにする。
    """
    write_vmd(
        path,
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


# --- パス検証・ガード -------------------------------------------------------


def test_missing_input_is_arg_error(tmp_path):
    code = cli.main([str(tmp_path / "nope.vmd")])
    assert code == 2


def test_non_vmd_input_is_input_error(tmp_path):
    src = tmp_path / "in.vmd"
    src.write_bytes(b"not a vmd file at all")
    code = cli.main([str(src)])
    assert code == 1


def test_overwrite_guard_blocks_same_path(tmp_path):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    before = src.read_bytes()
    # 既定出力でなく入力と同一パスを明示 → --overwrite 無しは引数エラー。
    code = cli.main([str(src), "-o", str(src)])
    assert code == 2
    # exit2 を返すだけでなく、ガード時は入力を書き換えない(「上書きしてから2を返す」実装を排除)。
    assert src.read_bytes() == before


def test_overwrite_guard_via_symlink(tmp_path):
    # 入力へのシンボリックリンク経由の出力も「同一ファイル」としてガードが効く。
    # 文字列比較でなく samefile/realpath で判定することの検証。
    src = tmp_path / "in.vmd"
    _full_doc(src)
    before = src.read_bytes()
    link = tmp_path / "link.vmd"
    try:
        link.symlink_to(src)
    except OSError as e:
        # 権限不足(Windows ERROR_PRIVILEGE_NOT_HELD)のときだけ skip。それ以外は表面化させる。
        if getattr(e, "winerror", None) != 1314:
            raise
        pytest.skip("シンボリックリンク作成権限なし(開発者モード/管理者権限が必要)")
    assert cli.main([str(src), "-o", str(link)]) == 2
    assert src.read_bytes() == before


def test_overwrite_allows_same_path(tmp_path):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    in_doc, _ = io.read(str(src))
    code = cli.main([str(src), "-o", str(src), "--overwrite"])
    assert code == 0
    out_doc, _ = io.read(str(src))
    # 同一パス上書きでも基盤段階は全セクション逐語透過(処理せず0を返す/入力を破損する実装を排除)。
    assert out_doc.bone == in_doc.bone
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.ik_property == in_doc.ik_property


def test_existing_distinct_output_allowed(tmp_path):
    # 上書きガードは「入力と同一パス」限定。入力と別の既存ファイルへの出力は --overwrite なしでも許可
    # (全既存出力を拒否する実装を排除)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    out.write_bytes(b"old content")
    code = cli.main([str(src), "-o", str(out)])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.bone == in_doc.bone


# --- 既定出力・書き出し -----------------------------------------------------


def test_default_output_name(tmp_path):
    src = tmp_path / "dance.vmd"
    _full_doc(src)
    code = cli.main([str(src)])
    assert code == 0
    assert (tmp_path / "dance_mocap.vmd").exists()


def test_explicit_output_written(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    code = cli.main([str(src), "-o", str(out)])
    assert code == 0
    assert out.exists()


# --- 対象外セクションの透過 -------------------------------------------------


def test_passthrough_preserves_all_sections(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    code = cli.main([str(src), "-o", str(out)])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    # 対象外セクション(モーフ・カメラ・照明・セルフ影・IKプロパティ)は無加工で透過する。
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.light == in_doc.light
    assert out_doc.self_shadow == in_doc.self_shadow
    assert out_doc.ik_property == in_doc.ik_property
    # 現基盤ではボーンも逐語透過する(クリーニング・疎化は後続実装)。非線形補間バイトも保持される。
    assert out_doc.bone == in_doc.bone
    assert out_doc.bone[0].interpolation == BONE_NONLINEAR
    assert out_doc.camera[0].interpolation == CAM_NONLINEAR


# --- dry-run ----------------------------------------------------------------


def test_dry_run_does_not_write_explicit_output(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    code = cli.main([str(src), "-o", str(out), "--dry-run"])
    assert code == 0
    assert not out.exists()


def test_dry_run_does_not_write_default_output(tmp_path):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == 0
    assert not (tmp_path / "in_mocap.vmd").exists()


# --- 診断レポート(report-json / dry-run 表示) ----------------------------


def test_report_json_written(tmp_path):
    src = tmp_path / "in.vmd"
    rep = tmp_path / "report.json"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run", "--report-json", str(rep)])
    assert code == 0
    data = json.loads(rep.read_text(encoding="utf-8"))
    names = [e["name"] for e in data["bones"]]
    assert "センター" in names
    center = next(e for e in data["bones"] if e["name"] == "センター")
    assert center["category"] == "center"
    # 足IK候補が分類結果として出る。
    assert "右足ＩＫ" in data["foot_ik_candidates"]


def test_dry_run_prints_report(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "センター" in out


def test_report_json_with_dry_run_does_not_write_output(tmp_path):
    src = tmp_path / "in.vmd"
    rep = tmp_path / "report.json"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run", "--report-json", str(rep)])
    assert code == 0
    assert rep.exists()
    assert not (tmp_path / "in_mocap.vmd").exists()


# --- --preset(クリーニング強度) -------------------------------------------


@pytest.mark.xfail(reason="impl pending: Step 2b --preset", strict=True)
def test_preset_option_resolves_cleaning_in_report(tmp_path):
    from mocapvmd import presets

    src = tmp_path / "in.vmd"
    rep = tmp_path / "report.json"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run", "--preset", "stable-foot", "--report-json", str(rep)])
    assert code == 0
    data = json.loads(rep.read_text(encoding="utf-8"))
    foot = next(e for e in data["bones"] if e["name"] == "右足ＩＫ")
    assert foot["cleaning"]["pos_strength"] == pytest.approx(0.65 * 1.5)
    assert foot["cleaning"] == presets.resolve_cleaning("stable-foot", "foot_ik")


def test_invalid_preset_value_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    # 未知のプリセット値は引数エラー(終了コード2)。
    assert cli.main([str(src), "--preset", "turbo"]) == 2


@pytest.mark.xfail(reason="impl pending: Step 2b --preset", strict=True)
def test_default_preset_is_balanced_in_report_json(tmp_path):
    # --preset 省略時は balanced が適用されることを CLI レベルで検証する(誤って別プリセットを
    # 明示渡しする実装を排除)。
    from mocapvmd import presets

    src = tmp_path / "in.vmd"
    rep = tmp_path / "report.json"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run", "--report-json", str(rep)])
    assert code == 0
    data = json.loads(rep.read_text(encoding="utf-8"))
    center = next(e for e in data["bones"] if e["name"] == "センター")
    assert center["cleaning"] == presets.resolve_cleaning("balanced", "center")
