"""mocapvmd CLI のテスト。

CLI は引数解析 → VMD読み → 全ボーンの一般ノイズ軽減(クリーニング)→ VMD書き。終了コード:
0 正常 / 1 入力不正 / 2 引数エラー / 3 出力書き込み失敗。

本テストは入出力・パス検証・上書きガード・dry-run・対象外セクション透過・診断表示・
クリーニング(--denoise/--no-denoise)を扱う。
"""

import pytest

from vmd import io
from mocapvmd import cli, report as mocap_report

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

_FORMAT_DRY_RUN = mocap_report.format_dry_run


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


def test_non_finite_bone_value_is_input_error(tmp_path):
    # クリーニング対象の値が非有限(NaN)ならクリーニングが ValueError を送出し、入力不正=終了コード1。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(
        src,
        bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0)), bone("センター", 1)],
    )
    assert cli.main([str(src), "-o", str(out)]) == 1


def test_single_key_bone_kept_verbatim(tmp_path):
    # キー1個のトラックは平滑化できないため逐語保持(値・補間そのまま)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)])
    assert cli.main([str(src), "-o", str(out)]) == 0
    out_doc, _ = io.read(str(out))
    assert len(out_doc.bone) == 1
    assert out_doc.bone[0].position == pytest.approx((1.0, 2.0, 3.0))
    assert out_doc.bone[0].interpolation == BONE_NONLINEAR


def test_single_key_non_finite_is_input_error(tmp_path):
    # キー1個でも非有限値は検証を迂回せず入力不正=終了コード1。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0))])
    assert cli.main([str(src), "-o", str(out)]) == 1


def test_single_key_zero_norm_quaternion_is_input_error(tmp_path):
    # キー1個でもゼロノルム quaternion は入力不正=終了コード1。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0, rot=(0.0, 0.0, 0.0, 0.0))])
    assert cli.main([str(src), "-o", str(out)]) == 1


def test_no_denoise_non_finite_is_input_error(tmp_path):
    # クリーニングを切っても入力値検証は迂回されず、非有限値は疎化に渡る前に入力不正=終了コード1。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(4)]
    keys.append(bone("センター", 4, pos=(float("inf"), 0.0, 0.0)))
    write_vmd(src, bone=keys)
    assert cli.main([str(src), "-o", str(out), "--no-denoise"]) == 1


def test_no_denoise_zero_norm_quaternion_is_input_error(tmp_path):
    # クリーニングを切った密トラックのゼロノルム quaternion も、疎化(既定 on)に渡る前に入力不正=終了コード1。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    keys = [bone("センター", f) for f in range(4)]
    keys.append(bone("センター", 4, rot=(0.0, 0.0, 0.0, 0.0)))
    write_vmd(src, bone=keys)
    assert cli.main([str(src), "-o", str(out), "--no-denoise"]) == 1


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
    # 同一パス上書きが許可され、出力が有効なVMDとして読め、対象外セクションが保持される
    # (処理せず0を返す/入力を破損する実装を排除)。ボーン処理の有無に依存しない検証。
    src = tmp_path / "in.vmd"
    _full_doc(src)
    in_doc, _ = io.read(str(src))
    code = cli.main([str(src), "-o", str(src), "--overwrite"])
    assert code == 0
    out_doc, _ = io.read(str(src))
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.ik_property == in_doc.ik_property
    assert len(out_doc.bone) >= 1


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
    assert out_doc.morph == in_doc.morph


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


def test_nonbone_sections_passthrough_with_denoise(tmp_path):
    # 既定(denoise on)でも対象外セクション(モーフ・カメラ・照明・セルフ影・IKプロパティ)は無加工透過。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    code = cli.main([str(src), "-o", str(out)])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.light == in_doc.light
    assert out_doc.self_shadow == in_doc.self_shadow
    assert out_doc.ik_property == in_doc.ik_property
    assert out_doc.camera[0].interpolation == CAM_NONLINEAR


def test_no_denoise_keeps_bones_verbatim(tmp_path):
    # 一般ノイズ軽減・足IK安定化をともに無効化するとボーンは逐語透過する(非線形補間バイトも保持)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.bone == in_doc.bone
    assert out_doc.bone[0].interpolation == BONE_NONLINEAR


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


# --- 診断表示(dry-run) ---------------------------------------------------


def _dry_run_report(src, monkeypatch, *args):
    captured = {}

    def capture(rep):
        captured["report"] = rep
        return _FORMAT_DRY_RUN(rep)

    monkeypatch.setattr(mocap_report, "format_dry_run", capture)
    assert cli.main([str(src), "--dry-run", *args]) == 0
    return captured["report"]


def test_dry_run_builds_report(tmp_path, monkeypatch):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    data = _dry_run_report(src, monkeypatch)
    names = [e["name"] for e in data["bones"]]
    assert "センター" in names
    center = next(e for e in data["bones"] if e["name"] == "センター")
    assert center["category"] == "center"
    # 足IK候補が分類結果として出る。
    assert "右足ＩＫ" in data["foot_ik_candidates"]


def test_report_reflects_denoise_flag(tmp_path, monkeypatch):
    # dry-run の処理計画にクリーニング有効/無効が反映される。
    src = tmp_path / "in.vmd"
    _full_doc(src)
    assert _dry_run_report(src, monkeypatch)["denoise"] is True
    assert _dry_run_report(src, monkeypatch, "--no-denoise")["denoise"] is False


def test_dry_run_prints_report(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "センター" in out


def test_dry_run_does_not_write_output_when_report_is_built(tmp_path):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    code = cli.main([str(src), "--dry-run"])
    assert code == 0
    assert not (tmp_path / "in_mocap.vmd").exists()


# --- --clean-strength(クリーニング強度) / --preset(疎化プリセット) ---------


def test_clean_strength_option_resolves_cleaning_in_report(tmp_path, monkeypatch):
    from mocapvmd import presets

    src = tmp_path / "in.vmd"
    _full_doc(src)
    data = _dry_run_report(src, monkeypatch, "--clean-strength", "1.4")
    foot = next(e for e in data["bones"] if e["name"] == "右足ＩＫ")
    assert foot["cleaning"]["pos_strength"] == pytest.approx(min(1.0, 0.65 * 1.4))
    assert foot["cleaning"] == presets.resolve_cleaning(1.4, "foot_ik")


@pytest.mark.parametrize("bad", ["-0.1", "nan", "inf"])
def test_invalid_clean_strength_is_arg_error(tmp_path, bad):
    # 非有限・負のクリーニング強度倍率は引数エラー(終了コード2)。
    src = tmp_path / "in.vmd"
    _full_doc(src)
    assert cli.main([str(src), "--clean-strength", bad]) == 2


def test_invalid_preset_value_is_arg_error(tmp_path):
    src = tmp_path / "in.vmd"
    _full_doc(src)
    # 未知の疎化プリセット値は引数エラー(終了コード2)。
    assert cli.main([str(src), "--preset", "turbo"]) == 2


def test_default_clean_strength_is_unit_in_dry_run_report(tmp_path, monkeypatch):
    # --clean-strength 省略時は倍率1.0(基準値)が適用されることを CLI レベルで検証する。
    from mocapvmd import presets

    src = tmp_path / "in.vmd"
    _full_doc(src)
    data = _dry_run_report(src, monkeypatch)
    center = next(e for e in data["bones"] if e["name"] == "センター")
    assert center["cleaning"] == presets.resolve_cleaning(1.0, "center")


# --- denoise クリーニング統合 -----------------------------------------------


# 全9カテゴリ(root/center/torso/arms/fingers/legs/foot_ik/toe_ik/unknown)の代表ボーン。
# 一般ノイズ軽減は全ボーンへ適用されるため、カテゴリ分岐で一部をスキップする実装を排除する。
_JITTER_BONES = (
    "全ての親",      # root
    "センター",      # center
    "上半身",        # torso
    "右腕",          # arms
    "右人指1",       # fingers
    "右足",          # legs
    "右足ＩＫ",      # foot_ik
    "右つま先ＩＫ",  # toe_ik
    "謎ボーン",      # unknown
)


def _jitter_doc(path):
    """X に微小ジッタを載せた密トラックを、全カテゴリの代表ボーンに対して書き出す(連続フレーム0-10)。

    補間は非線形にしておき、クリーニングで線形へ組み直されたかを検出できるようにする。
    """
    xs = [0.0, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.0]
    keys = []
    for name in _JITTER_BONES:
        keys += [bone(name, f, pos=(x, 0.0, 0.0), interp=BONE_NONLINEAR) for f, x in enumerate(xs)]
    write_vmd(path, bone=keys)


def _x_variation(keys, name):
    ks = sorted((k for k in keys if k.name == name), key=lambda k: k.frame)
    return sum(abs(ks[i + 1].position[0] - ks[i].position[0]) for i in range(len(ks) - 1))


def test_denoise_default_smooths_jitter(tmp_path):
    # 既定(denoise on)で全ボーン(center/torso/arms/legs)の微小ジッタが平滑化され、X方向の総変動が減る。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _jitter_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        assert _x_variation(out_doc.bone, name) < _x_variation(in_doc.bone, name)


def test_explicit_denoise_smooths_jitter(tmp_path):
    # 明示 --denoise でも(既定と同じく)微小ジッタが平滑化される(--denoise を受理しない実装を排除)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _jitter_doc(src)
    code = cli.main([str(src), "-o", str(out), "--denoise", "--no-reduce"])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        assert _x_variation(out_doc.bone, name) < _x_variation(in_doc.bone, name)


def test_no_denoise_keeps_bones_verbatim_all_categories(tmp_path):
    # 一般ノイズ軽減・足IK安定化をともに無効化すると、全カテゴリのボーンがキー列そのまま(値・フレーム・
    # 補間)逐語保持される(総変動量だけ一致させて中身を変える実装を排除)。foot_ik/toe_ik も含む。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _jitter_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        in_keys = sorted((k for k in in_doc.bone if k.name == name), key=lambda k: k.frame)
        out_keys = sorted((k for k in out_doc.bone if k.name == name), key=lambda k: k.frame)
        assert out_keys == in_keys


def test_no_denoise_preserves_nonbone_sections(tmp_path):
    # --no-denoise 経路でも対象外セクション(モーフ・カメラ・照明・セルフ影・IKプロパティ)を無加工透過する。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _full_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-denoise"])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    assert out_doc.morph == in_doc.morph
    assert out_doc.camera == in_doc.camera
    assert out_doc.light == in_doc.light
    assert out_doc.self_shadow == in_doc.self_shadow
    assert out_doc.ik_property == in_doc.ik_property


# --- 足IK安定化統合(--foot-ik-stabilize) -----------------------------------


def _foot_jitter_doc(path):
    """右足ＩＫ・右つま先ＩＫ(接地中の遅い水平ぐらつき)とセンター(同じ揺れ)を密トラックで書き出す。

    足IK・つま先IK は X が ±0.05 で揺れる(各ステップ <= 0.08 で接地・Y=0)。足IK安定化でアンカー
    (中央値0)へ寄り、水平変動が減るべき対象。センターは足IK安定化の対象外で、--no-denoise なら逐語の
    まま。補間は非線形にし、安定化で線形へ組み直されたかを検出できるようにする。
    """
    xs = [0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0]
    keys = []
    for name in ("右足ＩＫ", "右つま先ＩＫ", "センター"):
        keys += [bone(name, f, pos=(x, 0.0, 0.0), interp=BONE_NONLINEAR) for f, x in enumerate(xs)]
    write_vmd(path, bone=keys)


def test_foot_ik_stabilize_default_reduces_grounded_foot_drift(tmp_path):
    # 既定 on の足IK安定化は、一般平滑化を切った(--no-denoise)状態でも接地中の足IK水平ぐらつきを抑える。
    # 足IK安定化の対象外であるセンターは --no-denoise なので逐語(変動不変)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _foot_jitter_doc(src)
    assert cli.main([str(src), "-o", str(out), "--no-denoise", "--no-reduce"]) == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    # 足IK・つま先IK とも接地中の水平変動が減る。
    for name in ("右足ＩＫ", "右つま先ＩＫ"):
        assert _x_variation(out_doc.bone, name) < _x_variation(in_doc.bone, name)
    # センターは足IK安定化の対象外で、--no-denoise なのでキー列そのまま(値・フレーム・補間)逐語保持。
    in_center = sorted((k for k in in_doc.bone if k.name == "センター"), key=lambda k: k.frame)
    out_center = sorted((k for k in out_doc.bone if k.name == "センター"), key=lambda k: k.frame)
    assert out_center == in_center


def test_foot_ik_stabilize_runs_after_denoise(tmp_path):
    # 既定(denoise on + stabilize on)で、足IK出力が denoise→stabilize の順に処理された結果と一致する。
    # 逆順(stabilize→denoise)では結果が変わるため、処理順を固定する。
    from mocapvmd import denoise, footik, presets

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    xs = [round(0.05 * i, 6) for i in range(11)]  # 接地中の遅いランプ(中央値0.25からアンカー寄せが効く)
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(xs)])
    assert cli.main([str(src), "-o", str(out), "--no-reduce"]) == 0
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
    assert [k.frame for k in out_foot] == [k.frame for k in foot]  # 件数・フレーム列の一致
    for got, exp in zip(out_foot, expected):
        assert got.position == pytest.approx(exp)


def test_explicit_foot_ik_stabilize_matches_default(tmp_path):
    # 明示 --foot-ik-stabilize は既定(省略)と同一結果(別プリセット/強度を使う誤実装を排除)。
    src = tmp_path / "in.vmd"
    out_default = tmp_path / "default.vmd"
    out_explicit = tmp_path / "explicit.vmd"
    _foot_jitter_doc(src)
    assert cli.main([str(src), "-o", str(out_default), "--no-denoise"]) == 0
    assert cli.main([str(src), "-o", str(out_explicit), "--no-denoise", "--foot-ik-stabilize"]) == 0
    assert io.read(str(out_explicit))[0].bone == io.read(str(out_default))[0].bone


def test_no_foot_ik_stabilize_keeps_foot_and_toe_verbatim(tmp_path):
    # --no-denoise --no-foot-ik-stabilize では足IK・つま先IKも逐語保持(値・フレーム・非線形補間)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _foot_jitter_doc(src)
    assert cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"]) == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    for name in ("右足ＩＫ", "右つま先ＩＫ"):
        in_keys = sorted((k for k in in_doc.bone if k.name == name), key=lambda k: k.frame)
        out_keys = sorted((k for k in out_doc.bone if k.name == name), key=lambda k: k.frame)
        assert out_keys == in_keys


def test_foot_slide_suppression_out_of_range_is_arg_error(tmp_path):
    # 横滑り抑制は 0〜1 のみ。範囲外・非有限は引数エラー(終了コード2)。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(0.05 * f, 0.0, 0.0)) for f in range(6)])
    for bad in ("1.5", "-0.1", "nan"):
        assert cli.main([str(src), "-o", str(out), "--foot-slide-suppression", bad]) == 2


def test_foot_slide_suppression_zero_preserves_slide_one_removes(tmp_path):
    # S=0 は接地中の横滑りを保持(出力=入力)、S=1(既定)はアンカーへ寄せて横方向の振れ幅を縮める。
    src = tmp_path / "in.vmd"
    out0 = tmp_path / "out0.vmd"
    out1 = tmp_path / "out1.vmd"
    xs = [round(0.05 * i, 6) for i in range(11)]  # 接地中の遅いドリフト
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(xs)])

    assert cli.main([str(src), "-o", str(out0), "--no-denoise", "--no-reduce",
                     "--foot-slide-suppression", "0"]) == 0
    assert cli.main([str(src), "-o", str(out1), "--no-denoise", "--no-reduce",
                     "--foot-slide-suppression", "1"]) == 0

    def foot_x(doc):
        ks = sorted((k for k in doc.bone if k.name == "右足ＩＫ"), key=lambda k: k.frame)
        return [k.position[0] for k in ks]

    in_x = foot_x(io.read(str(src))[0])
    x0 = foot_x(io.read(str(out0))[0])
    x1 = foot_x(io.read(str(out1))[0])
    assert x0 == pytest.approx(in_x)                                   # S=0: 横滑りそのまま保持
    assert (max(x1) - min(x1)) < (max(in_x) - min(in_x))              # S=1: 振れ幅が縮む


# --- インプロセス疎化統合(--preset / --curve-mode / --no-reduce) ---------


def _ramp_doc(path):
    # センターの直線ランプ(密11フレーム)。疎化で端2キーへ削減される。
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


def _center_frames(bones):
    return sorted(k.frame for k in bones if k.name == "センター")


def _curve_doc(path):
    # センターの曲線(2次)。疎化でキーが減り、bezier と linear で結果が相違する。
    write_vmd(path, bone=[bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(11)])


def test_default_output_is_reduced(tmp_path):
    # 既定でクリーニング後に疎化し(キー数減)、既定の curve-mode は bezier(明示 bezier と一致・linear と相違)。
    src = tmp_path / "in.vmd"
    out_default = tmp_path / "default.vmd"
    out_bezier = tmp_path / "bezier.vmd"
    out_linear = tmp_path / "linear.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out_default)]) == 0
    assert cli.main([str(src), "-o", str(out_bezier), "--curve-mode", "bezier"]) == 0
    assert cli.main([str(src), "-o", str(out_linear), "--curve-mode", "linear"]) == 0
    d = sorted((k for k in io.read(str(out_default))[0].bone if k.name == "センター"), key=lambda k: k.frame)
    b = sorted((k for k in io.read(str(out_bezier))[0].bone if k.name == "センター"), key=lambda k: k.frame)
    ll = sorted((k for k in io.read(str(out_linear))[0].bone if k.name == "センター"), key=lambda k: k.frame)
    assert len(d) < 11  # 疎化されている
    assert d == b       # 既定の curve-mode は bezier
    assert d != ll      # 曲線入力で bezier と linear は相違(既定が linear へ退行していない)


def test_no_reduce_keeps_dense_linear(tmp_path):
    # --no-reduce ではクリーニング後の密キー(全フレーム・線形補間)を出力する。
    from vmd.reduce import BONE_LINEAR_INTERP

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    assert cli.main([str(src), "-o", str(out), "--no-reduce"]) == 0
    out_doc, _ = io.read(str(out))
    assert _center_frames(out_doc.bone) == list(range(11))
    for k in out_doc.bone:
        if k.name == "センター":
            assert k.interpolation == BONE_LINEAR_INTERP


def test_reduce_preset_validation(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    assert cli.main([str(src), "-o", str(out), "--preset", "slower"]) == 0
    assert cli.main([str(src), "--preset", "turbo"]) == 2  # 未知プリセットは引数エラー


def test_curve_mode_validation(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    assert cli.main([str(src), "-o", str(out), "--curve-mode", "linear"]) == 0
    assert cli.main([str(src), "--curve-mode", "spline"]) == 2  # 未知 curve-mode は引数エラー


def test_reduce_error_override_validation(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    assert cli.main([str(src), "-o", str(out), "--reduce-error-bone-pos", "0.05"]) == 0
    assert cli.main([str(src), "-o", str(out), "--reduce-error-bone-rot", "0.5"]) == 0  # 有効な回転許容値は受理
    assert cli.main([str(src), "--reduce-error-bone-pos", "-1"]) == 2  # 負の許容値は引数エラー
    assert cli.main([str(src), "--reduce-error-bone-rot", "nan"]) == 2  # 非有限は引数エラー


def test_denoise_output_is_dense_linear(tmp_path):
    # クリーニング後は連続フレームの密キーで、補間ブロックは線形(クリーニング後の密キー形式)。
    from vmd.reduce import BONE_LINEAR_INTERP

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _jitter_doc(src)
    code = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert code == 0
    out_doc, _ = io.read(str(out))
    for name in _JITTER_BONES:
        frames = sorted(k.frame for k in out_doc.bone if k.name == name)
        assert frames == list(range(11))
    for k in out_doc.bone:
        assert k.interpolation == BONE_LINEAR_INTERP


# --- 疎化レポートの CLI 配線(疎化を実行して reduction 診断をレポートへ載せる) ------


def test_report_includes_reduction_section(tmp_path, monkeypatch):
    # 既定(疎化 on)の dry-run は、全ボーン(多キー・単一キー)に疎化レポートを載せる。
    # 特定ボーンだけ診断を渡す不完全な配線を排除する。
    src = tmp_path / "in.vmd"
    keys = [bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(11)]
    keys.append(bone("右腕", 0, pos=(1.0, 0.0, 0.0)))  # 単一キー(逐語・削減なし)も診断に載る
    write_vmd(src, bone=keys)
    data = _dry_run_report(src, monkeypatch)
    for name in ("センター", "右腕"):
        r = next(e for e in data["bones"] if e["name"] == name)["reduction"]
        assert set(r) == {"output_keys", "reduction_rate", "tol_pos", "tol_rot", "cuts", "errors"}
        assert set(r["errors"]) == {"pos_x", "pos_y", "pos_z", "rot_deg"}


def test_report_reduce_flag_follows_reduce_option(tmp_path, monkeypatch):
    # 既定はレポート reduce: true、--no-reduce は false かつ reduction セクション無し。
    src = tmp_path / "in.vmd"
    _curve_doc(src)
    on = _dry_run_report(src, monkeypatch)
    off = _dry_run_report(src, monkeypatch, "--no-reduce")
    assert on["reduce"] is True
    assert off["reduce"] is False
    assert "reduction" not in next(e for e in off["bones"] if e["name"] == "センター")


def test_report_reduction_matches_reduce_bones(tmp_path, monkeypatch):
    # レポートの reduction 診断は、同じ入力を reduce_bones に diagnostics_out 付きで通した素データと一致する。
    # --no-denoise --no-foot-ik-stabilize でパイプラインを疎化だけに絞り、配線(CLI が診断を載せる)を固定する。
    from mocapvmd import reduce as mreduce

    src = tmp_path / "in.vmd"
    _curve_doc(src)
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


def test_dry_run_reduction_matches_full_pipeline(tmp_path, monkeypatch):
    # dry-run のレポート reduction は、クリーニング→足IK安定化→疎化の全段を通した診断と一致する
    # (output_keys だけでなく最大再生誤差 errors まで)。clean・stabilize が実際に値を変える入力(接地中の
    # ジッタ)を使い、各段を飛ばすと errors が全段と変わる(=どの段の省略も検出できる)ことを負例で保証する:
    # 全前処理省略(raw 直接)・clean だけ省略(stabilize(raw))・stabilize だけ省略(clean(raw))。
    from mocapvmd import reduce as mreduce
    from mocapvmd.cli import _clean_bones, _stabilize_bones

    src = tmp_path / "in.vmd"
    xs = [0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05, 0.0]  # 接地中のジッタ(各ステップ<=0.08)
    write_vmd(src, bone=[bone("右足ＩＫ", f, pos=(x, 0.0, 0.0)) for f, x in enumerate(xs)])
    data = _dry_run_report(src, monkeypatch)
    in_doc, _ = io.read(str(src))
    cleaned = _clean_bones(in_doc.bone, 1.0)
    full = {}
    mreduce.reduce_bones(_stabilize_bones(cleaned, 1.0), "medium", diagnostics_out=full)
    d = full["右足ＩＫ"]
    # 負例: 全前処理省略・clean 省略・stabilize 省略は、いずれも errors が全段と異なる(各段が結果に効く入力)。
    for skipped in (in_doc.bone, _stabilize_bones(in_doc.bone, 1.0), cleaned):
        diag = {}
        mreduce.reduce_bones(skipped, "medium", diagnostics_out=diag)
        assert diag["右足ＩＫ"]["errors"] != d["errors"]
    r = next(e for e in data["bones"] if e["name"] == "右足ＩＫ")["reduction"]
    assert r["output_keys"] == d["output_keys"]
    assert r["cuts"] == d["cuts"]
    assert r["errors"] == d["errors"]  # 各段を省略した負例(errors != d)では一致しない


@pytest.mark.parametrize(
    "bad_key",
    [
        bone("センター", 4, pos=(float("inf"), 0.0, 0.0)),  # 非有限位置
        bone("センター", 4, rot=(0.0, 0.0, 0.0, 0.0)),      # ゼロノルム quaternion
    ],
    ids=["non_finite_pos", "zero_norm_quat"],
)
def test_dry_run_invalid_input_is_error(tmp_path, bad_key):
    # dry-run でも疎化レポートのため入力を検証し、非有限値・ゼロノルム quaternion とも入力不正=終了コード1
    # (検証が dry-run 分岐より前段にあること、両ケースを弾くことを固定する)。
    src = tmp_path / "in.vmd"
    keys = [bone("センター", f) for f in range(4)]
    keys.append(bad_key)
    write_vmd(src, bone=keys)
    assert cli.main([str(src), "--dry-run"]) == 1


# --- --list-bones(ボーン一覧と分類を表示して終了) --------------------------


def _list_lines(capsys):
    return capsys.readouterr().out.splitlines()


def _line_with(lines, name):
    # 指定ボーン名を含む行を返す(表示形式を仮定せず、名前と分類が同一行に並ぶことだけを要求)。
    return next(ln for ln in lines if name in ln)


def test_list_bones_pairs_name_and_category_per_line(tmp_path, capsys):
    # 各ボーンの行に「自分の分類だけ」が並ぶ。他分類を含まないことも検証し、全分類を各行へ出す誤実装
    # (例「センター center foot_ik unknown」)も排除する。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("右足ＩＫ", 0), bone("謎ボーン", 0)])
    assert cli.main([str(src), "--list-bones"]) == 0
    lines = _list_lines(capsys)
    cats = {"センター": "center", "右足ＩＫ": "foot_ik", "謎ボーン": "unknown"}
    all_cats = set(cats.values())
    for name, cat in cats.items():
        ln = _line_with(lines, name)
        assert cat in ln
        assert all(other not in ln for other in all_cats - {cat})  # 他ボーンの分類は混在しない


def test_list_bones_appearance_order_and_dedup(tmp_path, capsys):
    # 一覧は初出順・名前ごとに1回(重複キーで同名を複数行に出さない)。表示形式は仮定せず、各名前を含む
    # 行の初出位置の順序と出現回数で検証する。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("右腕", 0), bone("センター", 0), bone("右腕", 5)])
    assert cli.main([str(src), "--list-bones"]) == 0
    lines = _list_lines(capsys)
    first = lambda name: next(i for i, ln in enumerate(lines) if name in ln)
    assert first("右腕") < first("センター")                       # 初出順
    assert sum(1 for ln in lines if "右腕" in ln) == 1            # 重複キーでも1回
    assert sum(1 for ln in lines if "センター" in ln) == 1


def test_list_bones_does_not_write_output(tmp_path):
    # --list-bones は表示して終了し、明示出力先(-o)も既定出力先も書かない。
    src = tmp_path / "in.vmd"
    out = tmp_path / "explicit.vmd"
    write_vmd(src, bone=[bone("センター", 0)])
    assert cli.main([str(src), "-o", str(out), "--list-bones"]) == 0
    assert not out.exists()
    assert not (tmp_path / "in_mocap.vmd").exists()


def test_list_bones_unaffected_by_write_and_reduce_validation(tmp_path, capsys):
    # --list-bones は書き込み・疎化をしないので、処理固有の検証(出力先=入力の上書きガード、不正な疎化
    # 許容値、非有限ボーン値)に阻まれず一覧を表示して終了コード0(短絡が両ガード・値検証より前にある)。
    src = tmp_path / "in.vmd"
    keys = [bone("センター", 0), bone("センター", 1, pos=(float("inf"), 0.0, 0.0))]  # 非有限値を含む
    write_vmd(src, bone=keys)
    before = src.read_bytes()
    # 出力先が入力自身(上書きガード対象)・不正 override(終了コード2対象)でも一覧は成功する。
    assert cli.main([str(src), "-o", str(src), "--reduce-error-bone-pos", "nan", "--list-bones"]) == 0
    assert "center" in _line_with(_list_lines(capsys), "センター")
    assert src.read_bytes() == before  # 入力は書き換えられない


def test_overwrite_guard_priority_over_unreadable_input(tmp_path):
    # 通常経路では上書きガード(引数エラー2)を読み込み(入力不正1)より先に判定する。読めない入力でも
    # 出力先=入力なら終了コード2(--list-bones 配線で読み込みを早期化しても通常経路の優先順位を保つ)。
    src = tmp_path / "in.vmd"
    src.write_bytes(b"not a vmd")
    assert cli.main([str(src), "-o", str(src)]) == 2


def test_reduce_override_validation_priority_over_unreadable_input(tmp_path):
    # 通常経路では override 検証(引数エラー2)を読み込み(入力不正1)より先に判定する。読めない入力でも
    # 不正 override なら終了コード2。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    src.write_bytes(b"not a vmd")
    assert cli.main([str(src), "-o", str(out), "--reduce-error-bone-pos", "nan"]) == 2


@pytest.mark.xfail(reason="impl pending: mocapvmd-warning-line", strict=True)
def test_human_warning_line_uses_common_format(tmp_path, capsys):
    # 警告行は共通コードのラベルで1行にまとめて標準エラーへ出す(安定コードは機械モードの
    # warning イベントと同じ値)。旧来の日本語ラベルは出さず、標準出力には何も漏らさない。
    from vmd.types import BoneKey, VmdDocument
    from vmd.reduce import BONE_LINEAR_INTERP
    bad_name = b"\x81\x20name".ljust(15, b"\x00")  # cp932 で復号できないバイト列
    keys = [BoneKey(bad_name, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
            for f in range(4)]
    io.write_file(VmdDocument(bone=keys), str(tmp_path / "in.vmd"))
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--no-reduce"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()  # 空行混入も検出するため除外しない
    assert len(lines) == 1
    prefix = "warning: decode-error: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()  # 本文が空白のみでない
    assert not body.startswith(" ")  # 接頭辞直後の空白が1つだけ(既に prefix に含む)
    assert "警告:" not in err


def test_version_flag_prints_name_and_version_and_exits_zero(capsys):
    # --version は版を表示して終了コード0。argparse の version アクションは SystemExit を投げるが、
    # main はそれを捕捉して終了コードへ変換する(--help と同じ)ため戻り値で確認する。
    # 版番号は __version__ を正本とし、表示文字列にツール名と版を含む。
    from mocapvmd import __version__

    assert cli.main(["--version"]) == 0
    assert f"mocapvmd {__version__}" in capsys.readouterr().out
