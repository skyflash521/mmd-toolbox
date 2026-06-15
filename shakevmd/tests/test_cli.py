"""shakevmd CLI のテスト(shakevmd.md §2, §9)。

CLI はコアの薄いラッパー: 引数解析 → VMD読み(mmd_toolbox.vmd.io)→ bake() → VMD書き。
終了コード(§9): 0 正常 / 1 入力不正(VMDでない・カメラキーなし)/ 2 引数エラー
(範囲不正・重複・上書き未許可)/ 3 出力書き込み失敗。

`TestCli` はコア CLI(**§2.1-2.6 + §9**: I/O・範囲・主要揺れパラメーター・終了コード)を、
`TestCliOps` は **§2.7 運用/プリセット系**(`--preset`・`--dry-run`・`--preview-csv`・`-v/--verbose`)
を検証する。§2.7 のうち内蔵パラメーター(オクターブ/プロファイル)のプリセット調整と walking の
歩調成分ノイズは、コア拡張を要するため後続サブステップへ繰延(本段階のプリセットは公開引数の束)。
"""

import sys

import numpy as np
import pytest

from shakevmd import cli, presets
from mmd_toolbox.vmd import io
from mmd_toolbox.vmd.types import BoneKey, CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


# 動きのあるカメラ列(順不同でも可)。
KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0), fov=45, persp=1),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05), fov=30, persp=1),
]

# パン→停止(settle 検証用、等速=線形補間)。
PAN_STOP_KEYS = [
    cam(0, rot=(0.0, 0.0, 0.0)),
    cam(30, rot=(0.0, 0.5, 0.0)),
    cam(60, rot=(0.0, 0.5, 0.0)),
]

# frame30 で中心が大きく跳ぶ=位置カット(cut-threshold 位置側の検証用)。
CUT_KEYS = [
    cam(0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, center=(0.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0)),
    cam(30, center=(40.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0)),
    cam(60, center=(40.0, 0.0, 0.0), rot=(0.0, 0.5, 0.0)),
]

# frame30 で角度が大きく跳ぶ=角度カット(cut-threshold 角度側の検証用)。
# frame29→30 で pitch 0→0.6rad(≈34°)。dist=0 にしてカメラワールド位置を中心(不動)に
# 固定し、回転による world 位置移動(dist≠0 だと ~17.7 単位)を排除=位置側を確実に0にする。
ANGLE_CUT_KEYS = [
    cam(0, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.6, 0.0, 0.0)),
    cam(60, dist=0.0, center=(0.0, 0.0, 0.0), rot=(0.6, 0.0, 0.0)),
]

# frame30 で distance が大きく跳ぶ=ズーム(中心も角度も不動だがカメラワールド位置が移動)。
# §2.6「カメラ中心またはカメラワールド位置」の world 側検出を確認する。
ZOOM_CUT_KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-5.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(60, dist=-5.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
]

# frame29→30 で中心が 4 単位だけ跳ぶ「小さなカット」。cut-threshold 値の配線検証用:
# 位置閾値3なら 4>3 で検出、既定5なら 4<5 で非検出。閾値値が出力を変えるので、
# CLI が --cut-threshold の値を無視して既定を使う誤配線を等価比較で判別できる。
SMALL_CUT_KEYS = [
    cam(0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(29, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, center=(4.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(60, center=(4.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
]

# frame21 と frame41 の2か所で中心が大きく跳ぶ=複数カット(報告が全位置を含むかの検証用)。
MULTI_CUT_KEYS = [
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
    # --- コマンド公開(§2.1 `shakevmd INPUT.vmd [options]`) ----------------
    def test_console_script_entry_point_declared(self):
        # §2.1 の `shakevmd` コマンドが cli:main として公開されている(pyproject 宣言)。
        # 直接 cli.main() を呼ぶだけではコマンド露出を保証できないため契約を静的に確認する。
        import tomllib
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["scripts"]["shakevmd"] == "shakevmd.cli:main"

    def test_main_falls_back_to_sys_argv(self, tmp_path, monkeypatch):
        # console script は main() を引数なし(argv=None)で呼ぶ。main は sys.argv[1:] へ
        # フォールバックする(§2.1)。明示 argv だけ動いて sys.argv 経路が壊れる実装を排除。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        monkeypatch.setattr(sys, "argv", ["shakevmd", inp, "-o", str(out)])
        assert cli.main() == 0
        assert out.exists()

    # --- 正常系(§9 コード0) ---------------------------------------------
    def test_bakes_and_writes_default_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        rc = cli.main([inp])
        assert rc == 0
        out = tmp_path / "in_shake.vmd"          # 既定出力 = <入力名>_shake.vmd
        assert out.exists()
        baked = read_camera(out)
        # 1フレーム間隔の高密度キー(0..60 全61)
        assert sorted(k.frame for k in baked) == list(range(0, 61))

    def test_default_output_always_vmd_extension(self, tmp_path):
        # §2.2: 既定出力は拡張子に依らず `<入力名>_shake.vmd`。非 .vmd 入力でも .vmd で出す。
        inp = write_input(tmp_path / "take.dat")     # 中身は有効な VMD、拡張子のみ .dat
        rc = cli.main([inp])
        assert rc == 0
        assert (tmp_path / "take_shake.vmd").exists()
        assert not (tmp_path / "take_shake.dat").exists()

    def test_explicit_output(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "sub" / "out.vmd"
        out.parent.mkdir()
        rc = cli.main([inp, "-o", str(out)])
        assert rc == 0 and out.exists()

    # --- 入力不正(§9 コード1) -------------------------------------------
    def test_invalid_vmd_exit1(self, tmp_path):
        bad = tmp_path / "bad.vmd"
        bad.write_bytes(b"not a vmd file at all")
        assert cli.main([str(bad)]) == 1

    def test_missing_input_exit1(self, tmp_path):
        # §9 は欠落ファイルの専用コードを規定しない。「入力から有効なカメラデータを得られない」
        # (欠落・非VMD・カメラなし)は一律 exit 1(入力不正)に括る設計とする。
        assert cli.main([str(tmp_path / "nope.vmd")]) == 1

    def test_no_camera_keys_exit1(self, tmp_path):
        inp = write_input(tmp_path / "empty.vmd", keys=[])   # カメラキーなし
        assert cli.main([inp]) == 1

    # --- 引数エラー(§9 コード2) -----------------------------------------
    def test_overwrite_guard_exit2(self, tmp_path):
        p = tmp_path / "in.vmd"
        inp = write_input(p)
        before = p.read_bytes()
        # 入力と同一パスへ出力 & --overwrite なし → エラー
        assert cli.main([inp, "-o", inp]) == 2
        # exit2 を返すだけでなく、ガード時は入力を書き換えない(§2.2)。
        # 「上書きしてから2を返す」実装を排除するため原本一致も検証する。
        assert p.read_bytes() == before

    def test_overwrite_allowed(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        rc = cli.main([inp, "-o", inp, "--overwrite"])
        assert rc == 0
        assert sorted(k.frame for k in read_camera(inp)) == list(range(0, 61))

    def test_existing_distinct_output_allowed(self, tmp_path):
        # 上書きガードは「入力と同一パス」限定(§2.2)。入力と別の既存ファイルへの出力は
        # --overwrite なしでも許可され、既存内容を上書きする。全既存出力を拒否する実装を排除。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        out.write_bytes(b"old content")          # 既存だが入力とは別パス
        assert cli.main([inp, "-o", str(out)]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_overwrite_guard_via_symlink_exit2(self, tmp_path):
        # 入力へのシンボリックリンク経由の出力も「同一ファイル」として上書きガードが効く
        # (§2.2)。abspath 文字列比較だと取りこぼすので samefile/realpath で判定する。
        p = tmp_path / "in.vmd"
        inp = write_input(p)
        before = p.read_bytes()
        link = tmp_path / "link.vmd"
        try:
            link.symlink_to(p)                    # link は入力と同一実体
        except OSError as e:
            # 権限不足(Windows ERROR_PRIVILEGE_NOT_HELD)のときだけ skip。
            # 開発者モード/管理者権限が無いとシンボリックリンクを作成できない。
            # それ以外の OSError は本物の失敗としてそのまま表面化させる。
            if getattr(e, "winerror", None) != 1314:
                raise
            pytest.skip("シンボリックリンク作成権限なし(開発者モード/管理者権限が必要)")
        assert cli.main([inp, "-o", str(link)]) == 2
        assert p.read_bytes() == before           # ガード時は原本を書き換えない

    def test_overlapping_ranges_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "0:60", "--range", "30:60"]) == 2

    def test_output_long_alias(self, tmp_path):
        # -o の長形式 --output(§2.2)
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "--output", str(out)]) == 0 and out.exists()

    def test_bad_range_format_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        # 非整数・余分なコロン・コロン無し(START:END 書式でない単一フレーム省略形)等
        for bad in ("abc", "1.5:30", "x:30", "30:y", "10:20:30", "30"):
            assert cli.main([inp, "--range", bad]) == 2

    def test_reversed_range_exit2(self, tmp_path):
        # START>END は不正(§2 引数エラー)
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "60:0"]) == 2

    def test_open_ended_start_beyond_end_exit2(self, tmp_path):
        # START が(省略された)END より後になる範囲は引数エラー(§2.2 START>END)。
        # 末尾キー60の入力で `999:` は END=60 に解決され 999>60 → exit 2。
        # 省略端の解決「後」にも逆順検査することを保証する(parse 時は END=None で素通り)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "999:"]) == 2

    def test_colon_only_range_is_full(self, tmp_path):
        # `:`(両端省略)は全範囲 = 範囲指定なしと同じ(有効)
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--range", ":"]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_post_snap_overlap_exit2(self, tmp_path):
        # スナップ「後」に重複したらエラー(§5.2)。25:55 と 28:58 は共に 30:60 へスナップ→重複
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range", "25:55", "--range", "28:58"]) == 2

    # --- 出力書き込み失敗(§9 コード3) -----------------------------------
    def test_output_write_failure_exit3(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        # 出力先の親がファイル(ディレクトリでない)→ 書き込み不可
        clash = tmp_path / "afile"
        clash.write_bytes(b"x")
        rc = cli.main([inp, "-o", str(clash / "out.vmd")])
        assert rc == 3

    # --- 範囲・パラメーター ------------------------------------------------
    def test_range_limits_baking(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", "30:60"])
        assert rc == 0
        out_keys = {k.frame: k for k in read_camera(out)}
        in_keys = {k.frame: k for k in read_camera(inp)}
        # frame0 は範囲外で原本のまま、30..60 が高密度
        assert sorted(out_keys) == [0] + list(range(30, 61))
        # 範囲外の frame0 は揺らさず原本を保持する(§2.2: --range は揺れ適用範囲)。
        # フレーム番号だけでなく内容(原本一致)も検証し、範囲外を書き換える実装を排除する。
        assert out_keys[0] == in_keys[0]

    def test_open_ended_range_start_omitted(self, tmp_path):
        # START 省略(:30)→ 先頭から30まで
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", ":30"])
        assert rc == 0
        frames = sorted(k.frame for k in read_camera(out))
        assert frames == list(range(0, 31)) + [60]   # 0..30 ベイク、frame60 原本

    def test_open_ended_range_end_omitted(self, tmp_path):
        # END 省略(30:)→ 30 から末尾まで
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", "30:"])
        assert rc == 0
        frames = sorted(k.frame for k in read_camera(out))
        assert frames == [0] + list(range(30, 61))   # frame0 原本、30..60 ベイク

    def test_multiple_nonoverlapping_ranges(self, tmp_path):
        # 複数 --range(非重複)は両方ベイク。中間キーを持つ入力で [0:0]…ではなく両端を分離。
        keys = [cam(0), cam(15), cam(30, persp=1), cam(45), cam(60)]
        inp = write_input(tmp_path / "in.vmd", keys)
        out = tmp_path / "out.vmd"
        rc = cli.main([inp, "-o", str(out), "--range", "0:15", "--range", "45:60"])
        assert rc == 0
        frames = sorted(k.frame for k in read_camera(out))
        assert frames == list(range(0, 16)) + [30] + list(range(45, 61))

    def test_seed_reproducible_and_seed_dependent(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        o1, o2, o3 = tmp_path / "o1.vmd", tmp_path / "o2.vmd", tmp_path / "o3.vmd"
        assert cli.main([inp, "-o", str(o1), "--seed", "7", "--amp-rot", "5"]) == 0
        assert cli.main([inp, "-o", str(o2), "--seed", "7", "--amp-rot", "5"]) == 0
        assert cli.main([inp, "-o", str(o3), "--seed", "8", "--amp-rot", "5"]) == 0
        assert o1.read_bytes() == o2.read_bytes()   # 同一シード→バイナリ一致(§7.5)
        assert o1.read_bytes() != o3.read_bytes()   # 異なるシード→変わる(seed が効いている)

    def test_normalization_warning_displayed(self, tmp_path, capsys):
        # 正規化警告(同一フレーム重複の後勝ち破棄など)はユーザーに表示する(§3.1)。
        dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]   # frame30 重複
        inp = write_input(tmp_path / "dup.vmd", dup)
        rc = cli.main([inp, "-o", str(tmp_path / "out.vmd")])
        assert rc == 0
        text = capsys.readouterr()
        # 警告は安定マーカー "warning:" 付きで表示される(文言非依存)
        assert "warning:" in (text.out + text.err).lower()

    def test_freq_bandlimit_clamp_warning(self, tmp_path, capsys):
        # §6.1: 実効周波数>8Hz のオクターブは自動クランプし「警告」を出す。
        # 内蔵 octaves=3 では freq×4>8(=freq>2)でクランプ発生。
        # 既定 freq=1.2(1.2/2.4/4.8≤8)はクランプなし=警告なし、--freq 3.0(→12Hz)は警告あり、
        # と判別することで「クランプ警告がユーザーに伝播される」ことを検証する(KEYS は重複なし)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "d.vmd")]) == 0           # 既定 freq
        base = capsys.readouterr()
        assert "warning:" not in (base.out + base.err).lower()
        assert cli.main([inp, "-o", str(tmp_path / "f.vmd"), "--freq", "3.0"]) == 0
        clamped = capsys.readouterr()
        assert "warning:" in (clamped.out + clamped.err).lower()

    def test_amp_rot_zero_no_rotation_shake(self, tmp_path):
        # --amp-rot 0 かつ settle/impulse なし → 回転は原本サンプリングと一致(揺れなし)
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--amp-rot", "0", "--amp-pos", "0", "--settle", "0"]) == 0
        from mmd_toolbox.vmd import interp
        baked = {k.frame: k for k in read_camera(out)}
        for f in range(0, 61):
            s = interp.sample_camera(KEYS, f)
            assert baked[f].rotation == pytest.approx(s["rotation"], abs=1e-5)

    # --- 主要オプションの透過(§2.3-2.6) -----------------------------------
    def test_public_options_affect_output(self, tmp_path):
        # 主要オプションが bake に効く(パースして無視する実装を排除=出力が既定と変わる)。
        inp = write_input(tmp_path / "in.vmd")

        def out_bytes(args, name):
            p = tmp_path / name
            assert cli.main([inp, "-o", str(p), *args]) == 0
            return p.read_bytes()

        base = out_bytes([], "base.vmd")
        assert out_bytes(["--freq", "3.0"], "freq.vmd") != base
        assert out_bytes(["--rot-weights", "1,1,0.9"], "rw.vmd") != base
        assert out_bytes(["--fade", "0.2"], "fade.vmd") != base
        assert out_bytes(["--motion-scale", "2.0"], "ms.vmd") != base
        # --cut-threshold は KEYS にカット(隣接フレーム)がないため出力は変わらないが、
        # パースされ exit0 になることは確認(効果は test_cuts.py / bake のカットテストで担保)
        assert cli.main([inp, "-o", str(tmp_path / "ct.vmd"), "--cut-threshold", "4,15"]) == 0

    def test_impulse_option_applies_and_multiple(self, tmp_path):
        # --impulse F:S:D が効く。複数指定可(§2.4)。base ノイズは切って衝撃だけ見る。
        inp = write_input(tmp_path / "in.vmd")
        base, one, two = tmp_path / "b.vmd", tmp_path / "1.vmd", tmp_path / "2.vmd"
        last = tmp_path / "last.vmd"
        common = ["--amp-rot", "0", "--amp-pos", "0", "--settle", "0"]
        assert cli.main([inp, "-o", str(base), *common]) == 0
        assert cli.main([inp, "-o", str(one), *common, "--impulse", "20:10:0.5"]) == 0
        assert cli.main([inp, "-o", str(two), *common, "--impulse", "20:10:0.5", "--impulse", "45:10:0.5"]) == 0
        assert cli.main([inp, "-o", str(last), *common, "--impulse", "45:10:0.5"]) == 0
        assert base.read_bytes() != one.read_bytes()    # 1つ目が効く
        assert one.read_bytes() != two.read_bytes()      # 2つ目(複数指定)も効く
        # 加算合成(§6.3): 2指定は「45単独」とも異なる。これにより「最後の1つだけ保持」
        # する実装(その場合 two==last になる)を排除する。
        assert two.read_bytes() != last.read_bytes()

    def test_amp_rot_affects_output(self, tmp_path):
        # --amp-rot 非ゼロ値が効く(回転シェイク振幅が出力に影響)
        inp = write_input(tmp_path / "in.vmd")
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        assert cli.main([inp, "-o", str(a), "--amp-rot", "1", "--amp-pos", "0", "--settle", "0"]) == 0
        assert cli.main([inp, "-o", str(b), "--amp-rot", "9", "--amp-pos", "0", "--settle", "0"]) == 0
        assert a.read_bytes() != b.read_bytes()

    def test_amp_pos_affects_output(self, tmp_path):
        # --amp-pos が効く(位置シェイクが出力に影響)
        inp = write_input(tmp_path / "in.vmd")
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        assert cli.main([inp, "-o", str(a), "--amp-pos", "0"]) == 0
        assert cli.main([inp, "-o", str(b), "--amp-pos", "0.5"]) == 0
        assert a.read_bytes() != b.read_bytes()

    def test_defaults_match_spec(self, tmp_path):
        # 既定値が spec(§2.3-2.6)どおり: 既定実行と spec 既定値の明示実行がバイナリ一致。
        # settle/cut-threshold の既定を実際に発火させるため、停止を含む入力(PAN_STOP_KEYS:
        # frame30 で停止→settle 発火)、位置カットを含む入力(CUT_KEYS: 29→30 で位置ジャンプ
        # →cut-threshold 位置側5.0が活性)、角度カットを含む入力(ANGLE_CUT_KEYS: 34°ジャンプ
        # →cut-threshold 角度側20.0が活性)でも検証する。KEYS だけだと全て不活性で誤既定を見逃す。
        spec_defaults = [
            "--amp-rot", "0.8", "--amp-pos", "0.05", "--rot-weights", "1,1,0.3",
            "--freq", "1.2", "--seed", "1", "--fade", "0.7",
            "--motion-scale", "0.5", "--settle", "0.3", "--cut-threshold", "5,20",
        ]
        for name, keys in (("plain", KEYS), ("stop", PAN_STOP_KEYS),
                           ("cut", CUT_KEYS), ("anglecut", ANGLE_CUT_KEYS)):
            inp = write_input(tmp_path / f"{name}.vmd", keys)
            d, e = tmp_path / f"{name}_d.vmd", tmp_path / f"{name}_e.vmd"
            assert cli.main([inp, "-o", str(d)]) == 0                    # 既定
            assert cli.main([inp, "-o", str(e), *spec_defaults]) == 0    # spec 既定値を明示
            assert d.read_bytes() == e.read_bytes(), f"default != spec-explicit for {name}"

    def test_default_seed_is_one(self, tmp_path):
        # 既定 seed は 1(§2.3)。既定実行と --seed 1 が一致する
        inp = write_input(tmp_path / "in.vmd")
        d, s1 = tmp_path / "d.vmd", tmp_path / "s1.vmd"
        assert cli.main([inp, "-o", str(d), "--amp-rot", "5"]) == 0
        assert cli.main([inp, "-o", str(s1), "--amp-rot", "5", "--seed", "1"]) == 0
        assert d.read_bytes() == s1.read_bytes()

    def test_settle_affects_output(self, tmp_path):
        # --settle が効く(パン→停止後の減衰振動)。base ノイズは切る。
        inp = write_input(tmp_path / "ps.vmd", PAN_STOP_KEYS)
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        common = ["--amp-rot", "0", "--amp-pos", "0"]
        assert cli.main([inp, "-o", str(a), *common, "--settle", "0"]) == 0
        assert cli.main([inp, "-o", str(b), *common, "--settle", "5"]) == 0
        assert a.read_bytes() != b.read_bytes()
        # settle=0 は「無効化」(§2.5)。停止入力でも振動を加えず、原本サンプリングと一致する。
        # これにより `args.settle or 0.3` のように 0 を既定へ落とす実装(振動が出る)を排除する。
        from mmd_toolbox.vmd import interp
        baked0 = {k.frame: k for k in read_camera(a)}
        for f in range(0, 61):
            s = interp.sample_camera(PAN_STOP_KEYS, f)
            assert baked0[f].rotation == pytest.approx(s["rotation"], abs=1e-5)

    def test_cut_threshold_affects_output(self, tmp_path):
        # --cut-threshold がカット検出へ転送される(閾値で分割が変わり出力が変わる)
        inp = write_input(tmp_path / "cut.vmd", CUT_KEYS)
        lo, hi = tmp_path / "lo.vmd", tmp_path / "hi.vmd"
        assert cli.main([inp, "-o", str(lo), "--cut-threshold", "5,20"]) == 0    # frame30 をカット検出
        assert cli.main([inp, "-o", str(hi), "--cut-threshold", "100,200"]) == 0  # カット検出なし
        assert lo.read_bytes() != hi.read_bytes()

    def test_cut_threshold_angle_jump(self, tmp_path):
        # 角度ジャンプでもカット検出(§2.6 位置,角度のいずれか)。中心不動・角度のみ跳ぶ入力。
        inp = write_input(tmp_path / "acut.vmd", ANGLE_CUT_KEYS)
        lo, hi = tmp_path / "lo.vmd", tmp_path / "hi.vmd"
        assert cli.main([inp, "-o", str(lo), "--cut-threshold", "5,20"]) == 0   # 角度34°>20 → カット
        assert cli.main([inp, "-o", str(hi), "--cut-threshold", "5,200"]) == 0  # 角度閾値200° → カットなし
        assert lo.read_bytes() != hi.read_bytes()

    def test_cut_threshold_world_position_zoom(self, tmp_path):
        # ズーム(distance 変化)はカメラ中心不動でもカメラワールド位置が跳ぶ→位置カット
        # (§2.6 「カメラ中心またはカメラワールド位置」)。位置側閾値で分割が変わる。
        inp = write_input(tmp_path / "zcut.vmd", ZOOM_CUT_KEYS)
        lo, hi = tmp_path / "lo.vmd", tmp_path / "hi.vmd"
        assert cli.main([inp, "-o", str(lo), "--cut-threshold", "5,20"]) == 0   # world_jump≈25>5 → カット
        assert cli.main([inp, "-o", str(hi), "--cut-threshold", "100,200"]) == 0  # 位置閾値100 → カットなし
        assert lo.read_bytes() != hi.read_bytes()

    def test_cli_flags_wire_to_correct_bake_params(self, tmp_path):
        # CLI は bake() の薄いラッパー。各フラグが bake の「対応」パラメーターへ正しく
        # 配線されることを等価比較で検証する(値が別パラメーターへ渡る誤配線を排除)。
        # CLI が実際に読む入力キーで bake し、同じ writer で round-trip して比較するため
        # float32 精度差は相殺される。motion_scale=0 / settle=0 のケースは「0で無効化」
        # (§2.5)も兼ね、`args.x or default` のように 0 を既定へ落とす実装を排除する。
        # 各ケースは「フラグ値が実際に出力へ効く入力」で検証する(不活性な入力だと値を
        # 無視する誤配線でも等価が成立してしまう)。settle は停止入力(PAN_STOP_KEYS)、
        # cut-threshold は閾値3で検出・既定5で非検出となる小カット入力(SMALL_CUT_KEYS)を使う。
        from shakevmd.bake import bake
        cases = [
            (["--freq", "3.0"], dict(freq=3.0), KEYS),
            (["--rot-weights", "0.5,0.7,0.9"], dict(rot_weights=(0.5, 0.7, 0.9)), KEYS),
            (["--amp-rot", "2.0"], dict(amp_rot=2.0), KEYS),
            (["--amp-pos", "0.3"], dict(amp_pos=0.3), KEYS),
            (["--fade", "0.2"], dict(fade_sec=0.2), KEYS),
            (["--motion-scale", "2.0"], dict(motion_scale=2.0), KEYS),
            (["--motion-scale", "0"], dict(motion_scale=0.0), KEYS),     # 0で無効化(§2.5)
            (["--settle", "1.5"], dict(settle=1.5), PAN_STOP_KEYS),      # 停止入力で settle 発火
            (["--settle", "0"], dict(settle=0.0), PAN_STOP_KEYS),        # 0で無効化(§2.5)
            (["--seed", "9"], dict(seed=9), KEYS),
            # 閾値3で検出・既定5で非検出の小カット入力。閾値値が効くので誤配線を判別できる。
            (["--cut-threshold", "3,10"], dict(cut_pos_threshold=3.0, cut_rot_threshold=10.0), SMALL_CUT_KEYS),
            # --impulse F:S:D の各要素が正しい順序で配線される(F=フレーム=int, S=度, D=秒)。
            (["--impulse", "20:10:0.5"], dict(impulses=[(20, 10.0, 0.5)]), KEYS),
        ]
        for i, (flag_args, kw, keys) in enumerate(cases):
            inp = write_input(tmp_path / f"in_{i}.vmd", keys)
            src = read_camera(inp)                  # CLI が実際にベイクする入力キー
            out = tmp_path / f"cli_{i}.vmd"
            exp = tmp_path / f"exp_{i}.vmd"
            assert cli.main([inp, "-o", str(out), *flag_args]) == 0
            # bake() は BakeResult(camera_keys, warnings) を返す。camera_keys を書き出して比較。
            io.write_file(VmdDocument(camera=bake(list(src), **kw).camera_keys), str(exp))
            assert read_camera(out) == read_camera(exp), f"flag misw-wired: {flag_args}"

    def test_invalid_compound_option_formats_exit2(self, tmp_path):
        # 複合フォーマットの要素数不足・型エラーは引数エラー(exit2)
        inp = write_input(tmp_path / "in.vmd")
        for args in (
            ["--rot-weights", "1,1"],        # 3要素でない(不足)
            ["--rot-weights", "1,1,1,1"],    # 3要素でない(余剰)
            ["--rot-weights", "a,b,c"],      # 非数値
            ["--cut-threshold", "5"],        # 位置,角度の2要素でない(不足)
            ["--cut-threshold", "5,20,30"],  # 2要素でない(余剰)
            ["--cut-threshold", "a,b"],      # 非数値(位置,角度とも)
            ["--cut-threshold", "5,deg"],    # 角度が非数値
            ["--impulse", "30:10"],          # F:S:D の3要素でない(不足)
            ["--impulse", "30:10:0.5:x"],    # 3要素でない(余剰)
            ["--impulse", "x:10:0.5"],       # F が非数値
            ["--impulse", "30:s:0.5"],       # S が非数値
            ["--impulse", "30:10:d"],        # D が非数値
        ):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), *args]) == 2

    def test_range_snaps_at_cli(self, tmp_path):
        # 非キー端点は「最近接」の既存キー(0/30/60)へスナップ(§5.2)。
        # 最近接を floor(常に直前キー) / ceil(常に切り上げ)の両方から区別するため、
        # 上側キーが最近接の端点と下側キーが最近接の端点を別々に使う。
        inp = write_input(tmp_path / "in.vmd")
        a, b = tmp_path / "a.vmd", tmp_path / "b.vmd"
        # 26 は上側キー30が最近接(|26-30|=4 < |26-0|=26)。floor なら 0 になるので区別できる。
        assert cli.main([inp, "-o", str(a), "--range", "26:60"]) == 0
        assert sorted(k.frame for k in read_camera(a)) == [0] + list(range(30, 61))
        # 34 は下側キー30が最近接(|34-30|=4 < |34-60|=26)。ceil なら 60 になるので区別できる。
        assert cli.main([inp, "-o", str(b), "--range", "0:34"]) == 0
        assert sorted(k.frame for k in read_camera(b)) == list(range(0, 31)) + [60]

    def test_range_out_of_span_endpoint_snaps(self, tmp_path):
        # キー範囲外の端点も「最近接の既存キー」へスナップする(§5.2)。拒否ではない。
        # END=999 は最終キー60が最近接 → [0,60] として全域ベイク。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--range", "0:999"]) == 0
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_range_out_of_span_start_snaps(self, tmp_path):
        # START 側のスナップ対称性(§5.2)。フレームは負にできないので、先頭キーが frame30 の
        # 入力で START=10(先頭キーより前)を使い、最近接の先頭キー30へスナップすることを確認。
        keys = [cam(30), cam(45), cam(60)]
        inp = write_input(tmp_path / "off.vmd", keys)
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--range", "10:60"]) == 0
        # START 10→先頭キー30へスナップ → [30,60] 全域ベイク(30より前のフレームは無い)。
        assert sorted(k.frame for k in read_camera(out)) == list(range(30, 61))

    # --- 引数エラー(§9 コード2) -----------------------------------------
    def test_missing_input_arg_exit2(self):
        assert cli.main([]) == 2                          # INPUT 必須

    def test_surplus_positional_arg_exit2(self, tmp_path):
        # §2.1: INPUT はちょうど1個。余剰ポジショナル引数は引数エラー(exit 2)。
        # 受け入れ・無視する実装(2個目を黙って捨てる)を排除する。
        inp = write_input(tmp_path / "in.vmd")
        extra = write_input(tmp_path / "extra.vmd")
        assert cli.main([inp, extra]) == 2

    def test_invalid_option_value_exit2(self, tmp_path):
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--seed", "abc"]) == 2      # int でない

    def test_invalid_numeric_scalars_exit2(self, tmp_path):
        # §9: 数値スカラーの不正入力は引数エラー(exit 2)。float 系も --seed 同様に弾く。
        inp = write_input(tmp_path / "in.vmd")
        for opt in ("--amp-rot", "--amp-pos", "--freq", "--fade", "--motion-scale", "--settle"):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), opt, "xyz"]) == 2

    def test_non_finite_numeric_exit2(self, tmp_path):
        # inf/nan は有限数でない → 引数エラー(exit 2)。bake へ渡すと OverflowError 等で
        # クラッシュしうるため、CLI 境界で弾く(スカラー・複合フォーマット双方)。
        inp = write_input(tmp_path / "in.vmd")
        for opt, val in (("--fade", "inf"), ("--amp-rot", "nan"), ("--freq", "inf"),
                         ("--motion-scale", "-inf"), ("--rot-weights", "1,inf,1"),
                         ("--cut-threshold", "inf,20"), ("--impulse", "20:inf:0.5")):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), opt, val]) == 2

    def test_overflowing_numeric_exit2(self, tmp_path):
        # 有限でも過大な値は下流(int(round(fade*FPS)))で OverflowError になりうる。
        # CLI はこれも引数エラー(exit 2)として扱い、Python 例外を漏らさない(§9)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--fade", "1e308"]) == 2

    def test_out_of_domain_numeric_exit2(self, tmp_path):
        # 物理量の定義域違反は引数エラー(exit 2)。振幅/秒数/係数は非負、周波数は正、
        # cut-threshold(感度)は非負、--impulse の S は非負・D は正(§2.3-2.6)。
        # 0 が有効な無効化値である項目(amp/motion-scale/settle/cut-threshold)は別途 0 許容。
        inp = write_input(tmp_path / "in.vmd")
        o = str(tmp_path / "o.vmd")
        cases = [
            ["--fade", "-0.7"], ["--amp-rot", "-1"], ["--amp-pos", "-0.1"],
            ["--motion-scale", "-1"], ["--settle", "-1"],
            ["--freq", "0"], ["--freq", "-1"],
            ["--cut-threshold", "-5,20"], ["--cut-threshold", "5,-20"],
            ["--impulse", "20:-1:0.5"],   # S(強さ)が負
            ["--impulse", "20:10:0"],     # D(減衰秒)が 0
            ["--impulse", "20:10:-0.5"],  # D が負
        ]
        for args in cases:
            assert cli.main([inp, "-o", o, *args]) == 2, args

    def test_zero_disable_values_allowed(self, tmp_path):
        # 0 が有効な無効化/中立値である項目は exit 0(過剰拒否しない)。
        inp = write_input(tmp_path / "in.vmd")
        o = str(tmp_path / "o.vmd")
        for args in (["--amp-rot", "0"], ["--amp-pos", "0"], ["--motion-scale", "0"],
                     ["--settle", "0"], ["--fade", "0"], ["--cut-threshold", "0,0"],
                     ["--impulse", "20:0:0.5"]):
            assert cli.main([inp, "-o", o, *args]) == 0, args

    def test_negative_impulse_frame_exit2(self, tmp_path):
        # --impulse の F はフレーム=非負(§2.4)。負フレームは引数エラー(exit 2)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--impulse=-5:10:0.5"]) == 2

    def test_read_warnings_propagated(self, tmp_path, capsys, monkeypatch):
        # io.read() の継続可能警告(名前デコード不可・旧版セクション欠落など)もユーザーへ
        # 伝播する(VMD I/O は mmd_toolbox へ委譲する設計、§3.1)。捨てる実装を排除する。
        # 実トリガは write/read の round-trip 依存で脆いため、io.read に警告を注入して
        # 「CLI が io.read の警告を surface する」配線そのものを検証する。
        from mmd_toolbox.vmd.types import VmdWarning
        inp = write_input(tmp_path / "in.vmd")
        real_read = io.read

        def fake_read(path):
            doc, warns = real_read(path)
            warns.append(VmdWarning(code="decode-error", message="注入した読込警告"))
            return doc, warns

        monkeypatch.setattr(cli.io, "read", fake_read)
        assert cli.main([str(inp), "-o", str(tmp_path / "out.vmd")]) == 0
        cap = capsys.readouterr()
        text = (cap.out + cap.err).lower()
        assert "warning:" in text and "decode-error" in text

    def test_negative_range_endpoint_exit2(self, tmp_path):
        # フレーム番号は非負(VMD は uint)。負の範囲端は引数エラー(exit 2)。
        # `=`形式で渡し argparse がオプションと誤認しないようにする。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--range=-10:0"]) == 2
        assert cli.main([inp, "--range=0:-5"]) == 2

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")  # 意図的に bake 内で overflow させる
    def test_non_finite_baked_output_exit2(self, tmp_path):
        # 引数は有限でも bake 内の乗算で出力が inf 化しうる(amp-pos×motion-scale)。
        # float32 は inf を例外なく pack するため、焼き後の有限性検査で exit 2 に倒す(§9)。
        inp = write_input(tmp_path / "in.vmd")
        rc = cli.main([inp, "-o", str(tmp_path / "o.vmd"),
                       "--amp-pos", "1e308", "--motion-scale", "1e308"])
        assert rc == 2

    def test_serialization_overflow_is_arg_error_exit2(self, tmp_path):
        # 有限でも過大な振幅は bake で巨大な回転値となり、VMD の float32 書き出しで
        # OverflowError になる。これは引数起因なので「出力書き込み失敗(3)」ではなく
        # 引数エラー(2)に分類する(§9)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--amp-rot", "1e308"]) == 2

    def test_abbreviated_flag_rejected_exit2(self, tmp_path):
        # 仕様外の前置き省略形(--over 等)は受理しない(allow_abbrev=False)→ exit 2。
        # 省略形がフラグとして通ると非仕様の挙動(ガード回避等)を招くため。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--over"]) == 2

    def test_missing_option_operand_exit2(self, tmp_path):
        # §9: 値を要するオプションに値が無い(オペランド欠落)も引数エラー(exit 2)。
        inp = write_input(tmp_path / "in.vmd")
        for opt in ("--output", "--range", "--seed", "--amp-rot", "--amp-pos",
                    "--rot-weights", "--freq", "--fade", "--motion-scale",
                    "--settle", "--cut-threshold", "--impulse",
                    "--preset", "--preview-csv"):
            assert cli.main([inp, opt]) == 2

    def test_rejects_internal_params_exit2(self, tmp_path):
        # §8: 詳細内部パラメーター(オクターブ構成・persistence・settle収束時間等)は
        # CLI 非公開 → 未知オプションとして引数エラー(exit 2)。
        # 内部フラグ名は spec で規定されないため代表確認。網羅の本質は「未知オプション
        # は一律 exit 2」で、これは argparse がすべての未知フラグに対し保証する。
        inp = write_input(tmp_path / "in.vmd")
        for opt in (["--octaves", "5"], ["--persistence", "0.7"], ["--settle-time", "2"],
                    ["--gait-freq", "2.0"], ["--gait-amp", "0.2"]):
            assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), *opt]) == 2

    # --- 非カメラセクション透過(§3.1) -------------------------------------
    def test_non_camera_sections_passthrough_with_warning(self, tmp_path, capsys):
        # ボーン等のセクションは無加工で透過し、警告を表示する(§3.1)。
        bone = BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                       position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                       interpolation=bytes(64))
        doc = VmdDocument(bone=[bone], camera=list(KEYS))
        inp = tmp_path / "mixed.vmd"
        io.write_file(doc, str(inp))
        out = tmp_path / "out.vmd"
        rc = cli.main([str(inp), "-o", str(out)])
        assert rc == 0
        outdoc, _ = io.read(str(out))
        # ボーンキーが無傷で残る + カメラはベイクされている
        assert len(outdoc.bone) == 1 and outdoc.bone[0].name == "bone"
        assert sorted(k.frame for k in outdoc.camera) == list(range(0, 61))
        text = capsys.readouterr()
        assert "warning:" in (text.out + text.err).lower()   # 非カメラ透過の警告(安定マーカー)


import csv as _csv


class TestCliOps:
    """§2.7 運用/プリセット系: --preset / --dry-run / --preview-csv / -v,--verbose。

    本段階のプリセットは公開引数の束(個別引数が優先)。内蔵パラメーター調整と
    walking 歩調成分はコア拡張後の後続サブステップへ繰延(test_cli.py docstring 参照)。
    """

    # --- プリセット定義(presets.py) -------------------------------------
    PUBLIC_PARAMS = {"amp_rot", "amp_pos", "rot_weights", "freq",
                     "motion_scale", "settle", "cut_threshold"}
    # 内蔵パラメーター(CLI 非公開、プリセット/コアAPIのみ。§8)。許可集合は presets の
    # 単一の真実源から導出する(ハードコードしない。新規内蔵パラメータ追加時に自動で同期)。
    INTERNAL_PARAMS = set(presets.INTERNAL_PARAM_NAMES)

    def test_presets_defined_for_all_names(self):
        # §2.7 の4プリセットが定義され、各々が公開引数の完全な束を含む(一括設定)。
        # 余剰キーは内蔵パラメーター(歩調等)に限る。未知名は KeyError。
        assert set(presets.PRESET_NAMES) == {"handheld", "telephoto", "walking", "earthquake"}
        for name in presets.PRESET_NAMES:
            params = presets.get_preset(name)
            assert isinstance(params, dict)
            assert self.PUBLIC_PARAMS <= set(params), name              # 全公開引数を含む
            assert set(params) - self.PUBLIC_PARAMS <= self.INTERNAL_PARAMS, name  # 余剰は内蔵のみ
        with pytest.raises(KeyError):
            presets.get_preset("nonexistent-preset")

    def test_only_walking_has_gait_component(self):
        # §95: 歩調成分は walking のみ。walking は gait_freq>0・gait_amp>0、他は無効(0/未設定)。
        w = presets.get_preset("walking")
        assert w["gait_freq"] > 0.0 and w["gait_amp"] > 0.0
        for name in ("handheld", "telephoto", "earthquake"):
            q = presets.get_preset(name)
            assert q.get("gait_freq", 0.0) == 0.0, name
            assert q.get("gait_amp", 0.0) == 0.0, name

    def test_internal_params_forwardable_and_are_bake_kwargs(self):
        # §8: 内蔵パラメータはプリセット定義から bake へ転送可能(coreAPI かつ preset 調整可能)。
        # INTERNAL_PARAM_NAMES は全て bake() のキーワード引数(転送先が実在)で、歩調に加え
        # 静止/移動プロファイル・settle収束時間・素朴な角度加算モードを含む(転送ループの実効は
        # walking 歩調の等価テストで担保済み=同一機構)。
        import inspect
        from shakevmd.bake import bake
        params = inspect.signature(bake).parameters
        for name in presets.INTERNAL_PARAM_NAMES:
            assert name in params, name
        assert {"gait_freq", "gait_amp", "still_profile", "moving_profile",
                "settle_time", "naive_rotation"} <= set(presets.INTERNAL_PARAM_NAMES)

    def test_preset_internal_params_forward_without_collision(self, tmp_path, monkeypatch):
        # 実際の転送経路で重複キーワード衝突がないことを検証する(cli が明示渡しする引数名を
        # ハードコードして列挙すると漏れる=drift)。全内蔵パラメータをプリセットへ入れ、
        # cli を通して例外なく exit 0 になることを確認する。衝突があれば **internal で TypeError。
        p = dict(presets.get_preset("handheld"))
        p.update(still_profile=(1.0, 0.5, 0.25), moving_profile=(1.0, 0.7, 0.4),
                 settle_time=2.0, naive_rotation=True, gait_freq=1.5, gait_amp=0.1)
        # 全内蔵パラメータを実際に行使する(将来の追加で取りこぼさない)。
        assert set(presets.INTERNAL_PARAM_NAMES) <= set(p)
        monkeypatch.setitem(presets._PRESETS, "handheld", p)
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--preset", "handheld"]) == 0

    def test_all_internal_params_forwarded_and_effective(self, tmp_path, monkeypatch):
        # 内蔵パラメータが「すべて実際に転送され出力に効く」ことを behavioral に検証する(no-collision の
        # exit0 や settle_time 1つだけでは、他を黙って落とす実装を排除できない)。
        # 全内蔵を既定と異なる値でプリセットへ入れ、cli 出力が同値の直接 bake と一致することを確認する。
        # PAN_STOP_KEYS は移動区間・停止・静止区間を含むので、profiles/settle_time/naive/gait すべてが
        # 出力に効く → どれか1つでも転送漏れがあれば直接 bake と不一致で落ちる。
        from shakevmd.bake import bake
        internal = dict(still_profile=(1.0, 0.5, 0.25), moving_profile=(1.0, 0.7, 0.4),
                        settle_time=2.5, naive_rotation=True, gait_freq=1.5, gait_amp=0.1)
        assert set(internal) >= set(presets.INTERNAL_PARAM_NAMES)   # 全内蔵を網羅(将来追加も強制)
        p = dict(presets.get_preset("handheld"))
        p.update(internal)
        monkeypatch.setitem(presets._PRESETS, "handheld", p)
        inp = write_input(tmp_path / "in.vmd", PAN_STOP_KEYS)
        src = read_camera(inp)
        out, exp = tmp_path / "cli.vmd", tmp_path / "exp.vmd"
        assert cli.main([inp, "-o", str(out), "--preset", "handheld"]) == 0
        hp = presets.get_preset("handheld")
        baked = bake(
            list(src), seed=1,
            amp_rot=hp["amp_rot"], amp_pos=hp["amp_pos"], rot_weights=hp["rot_weights"],
            freq=hp["freq"], motion_scale=hp["motion_scale"], settle=hp["settle"],
            cut_pos_threshold=hp["cut_threshold"][0], cut_rot_threshold=hp["cut_threshold"][1],
            fade_sec=0.7, impulses=(), **internal,
        )
        io.write_file(VmdDocument(camera=baked.camera_keys), str(exp))
        assert read_camera(out) == read_camera(exp)

    def test_walking_preset_forwards_gait_to_bake(self, tmp_path):
        # §95: --preset walking が gait_freq/gait_amp を bake へ配線する。CLI 出力が、同じ公開引数+
        # 歩調引数で直接 bake した結果と一致することで検証する(歩調を渡さない実装は不一致で落ちる)。
        from shakevmd.bake import bake
        inp = write_input(tmp_path / "in.vmd", KEYS)
        src = read_camera(inp)
        out, exp = tmp_path / "cli.vmd", tmp_path / "exp.vmd"
        assert cli.main([inp, "-o", str(out), "--preset", "walking"]) == 0
        p = presets.get_preset("walking")
        baked = bake(
            list(src), seed=1,
            amp_rot=p["amp_rot"], amp_pos=p["amp_pos"], rot_weights=p["rot_weights"],
            freq=p["freq"], motion_scale=p["motion_scale"], settle=p["settle"],
            cut_pos_threshold=p["cut_threshold"][0], cut_rot_threshold=p["cut_threshold"][1],
            fade_sec=0.7, impulses=(), gait_freq=p["gait_freq"], gait_amp=p["gait_amp"],
        )
        io.write_file(VmdDocument(camera=baked.camera_keys), str(exp))
        assert read_camera(out) == read_camera(exp)

    def test_all_presets_run_via_cli(self, tmp_path):
        # 4プリセット名すべてが CLI で受理され正常終了する(earthquake だけでなく全名)。
        inp = write_input(tmp_path / "in.vmd")
        for name in ("handheld", "telephoto", "walking", "earthquake"):
            assert cli.main([inp, "-o", str(tmp_path / f"{name}.vmd"), "--preset", name]) == 0, name

    def test_unknown_preset_exit2(self, tmp_path):
        # 未知のプリセット名は引数エラー(exit 2)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "o.vmd"), "--preset", "bogus"]) == 2

    def test_preset_changes_output(self, tmp_path):
        # --preset はベイクに効く(既定実行と異なる出力になる)。earthquake は既定と明確に異なる。
        inp = write_input(tmp_path / "in.vmd")
        base, eq = tmp_path / "base.vmd", tmp_path / "eq.vmd"
        assert cli.main([inp, "-o", str(base)]) == 0
        assert cli.main([inp, "-o", str(eq), "--preset", "earthquake"]) == 0
        assert base.read_bytes() != eq.read_bytes()

    def test_individual_arg_overrides_preset(self, tmp_path):
        # 個別引数はプリセットより優先。かつ「1つ明示してもプリセット全体は無効化されない」
        # ことを判別する(§2.7)。1引数だけ上書きし、全適用とも既定とも異なることを確認。
        inp = write_input(tmp_path / "in.vmd")
        base, full, part = tmp_path / "base.vmd", tmp_path / "full.vmd", tmp_path / "part.vmd"
        assert cli.main([inp, "-o", str(base)]) == 0                              # 既定
        assert cli.main([inp, "-o", str(full), "--preset", "earthquake"]) == 0     # earthquake 全適用
        assert cli.main([inp, "-o", str(part), "--preset", "earthquake",
                         "--amp-rot", "0.8"]) == 0                                  # amp-rot だけ上書き
        assert part.read_bytes() != full.read_bytes()   # amp-rot 上書きが効く(全適用と異なる)
        assert part.read_bytes() != base.read_bytes()   # 残りの earthquake 引数は有効(既定とも異なる)

    def test_full_explicit_args_supersede_preset(self, tmp_path):
        # 全公開引数を既定値で明示すれば、--preset を付けても既定実行とバイナリ一致(完全上書き)。
        inp = write_input(tmp_path / "in.vmd")
        d, e = tmp_path / "d.vmd", tmp_path / "e.vmd"
        spec_defaults = [
            "--amp-rot", "0.8", "--amp-pos", "0.05", "--rot-weights", "1,1,0.3",
            "--freq", "1.2", "--motion-scale", "0.5", "--settle", "0.3",
            "--cut-threshold", "5,20",
        ]
        assert cli.main([inp, "-o", str(d)]) == 0
        assert cli.main([inp, "-o", str(e), "--preset", "earthquake", *spec_defaults]) == 0
        assert d.read_bytes() == e.read_bytes()

    def test_preset_matches_explicit_params(self, tmp_path):
        # CLI の --preset は presets.get_preset() の値をそのまま適用する(§8: presets.py が
        # プリセット定義の境界)。公開引数のみのプリセットは「--preset NAME == その公開引数の
        # 明示指定」と一致する。内蔵パラメーター(歩調等)を持つプリセットは、公開引数だけの
        # 明示指定では内蔵分が欠けるため一致しない(=内蔵パラメーターも実際に効いている証拠)。
        inp = write_input(tmp_path / "in.vmd")
        for name in presets.PRESET_NAMES:
            p = presets.get_preset(name)
            explicit = [
                "--amp-rot", str(p["amp_rot"]), "--amp-pos", str(p["amp_pos"]),
                "--rot-weights", "{},{},{}".format(*p["rot_weights"]),
                "--freq", str(p["freq"]), "--motion-scale", str(p["motion_scale"]),
                "--settle", str(p["settle"]),
                "--cut-threshold", "{},{}".format(*p["cut_threshold"]),
            ]
            a, b = tmp_path / f"{name}_a.vmd", tmp_path / f"{name}_b.vmd"
            assert cli.main([inp, "-o", str(a), "--preset", name]) == 0, name
            assert cli.main([inp, "-o", str(b), *explicit]) == 0, name
            # 「効く」内蔵パラメーターを持つか(キーの有無でなく実効値で判定。非 walking が
            # gait_freq=0 を明示しても無効=公開引数のみと一致、を誤判定しない)。
            active_gait = p.get("gait_freq", 0.0) > 0.0 and p.get("gait_amp", 0.0) != 0.0
            if active_gait:
                assert a.read_bytes() != b.read_bytes(), name   # 内蔵分が効くので不一致
            else:
                assert a.read_bytes() == b.read_bytes(), name   # 実効する内蔵なし → 一致

    def test_individual_override_is_order_independent(self, tmp_path):
        # 個別引数の優先は指定順に依らない(§2.7)。個別引数を --preset の前に置いても上書きが効く
        # (後続 preset が先行の個別引数を潰す実装を排除)。
        inp = write_input(tmp_path / "in.vmd")
        before, after = tmp_path / "before.vmd", tmp_path / "after.vmd"
        assert cli.main([inp, "-o", str(before), "--amp-rot", "0.8", "--preset", "earthquake"]) == 0
        assert cli.main([inp, "-o", str(after), "--preset", "earthquake", "--amp-rot", "0.8"]) == 0
        assert before.read_bytes() == after.read_bytes()

    # --- --dry-run ---------------------------------------------------------
    def test_dry_run_writes_no_output(self, tmp_path):
        # --dry-run は出力ファイルを書かない(§2.7)。exit 0。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--dry-run"]) == 0
        assert not out.exists()

    def test_dry_run_writes_no_default_output(self, tmp_path):
        # -o 省略時も --dry-run は既定出力(<入力>_shake.vmd)を書かない(§2.7「出力せず」)。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "--dry-run"]) == 0
        assert not (tmp_path / "in_shake.vmd").exists()

    def test_dry_run_reports_stats(self, tmp_path, capsys):
        # --dry-run は統計を表示する(§2.7: 適用範囲・出力キー数・最大振幅・カット位置・警告)。
        # ラベル(安定マーカー)と出力キー数(KEYS 全域=61)の双方を確認する。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = (cap.out + cap.err).lower()
        assert "61" in text                                   # 出力キー数(0..60)
        for label in ("range", "key", "amplitude", "cut"):    # 各統計項目のラベル
            assert label in text, label

    def test_dry_run_reports_detected_cut(self, tmp_path, capsys):
        # --dry-run はカット検出位置を報告する(§5.3, §10)。frame30 でカットする入力で
        # "cut" ラベル付きで検出フレーム "30" が現れる(偶発的な "30" を排除)。
        inp = write_input(tmp_path / "cut.vmd", CUT_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = (cap.out + cap.err).lower()
        assert "cut" in text and "30" in text

    def test_dry_run_reports_all_cuts(self, tmp_path, capsys):
        # 複数カットは全位置を報告する(§5.3「検出されたカット位置は…必ず報告」)。
        # frame21・frame41 の2カット入力で両方が現れる(先頭1個だけ報告する実装を排除)。
        inp = write_input(tmp_path / "mcut.vmd", MULTI_CUT_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "21" in text and "41" in text

    def test_dry_run_reports_all_snapped_ranges(self, tmp_path, capsys):
        # 複数 --range の解決後範囲を全て報告する(§5.2)。キー 0/15/30/45/60 の入力で
        # 0:14→0:15, 44:60→45:60。両範囲端(15 と 45)が現れる。
        keys = [cam(0), cam(15), cam(30, persp=1), cam(45), cam(60)]
        inp = write_input(tmp_path / "mr.vmd", keys)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "0:14", "--range", "44:60", "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "15" in text and "45" in text     # 2範囲のスナップ後端

    def test_dry_run_reports_warning(self, tmp_path, capsys):
        # --dry-run は警告も表示する(§2.7)。同一フレーム重複の正規化警告が出る入力で確認。
        dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
        inp = write_input(tmp_path / "dup.vmd", dup)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
        cap = capsys.readouterr()
        assert "warning:" in (cap.out + cap.err).lower()

    def test_dry_run_reports_snapped_range(self, tmp_path, capsys):
        # --dry-run はスナップ「後」の実適用範囲を報告する(§5.2)。26:60 は 30:60 へスナップ
        # (KEYS のキーは 0/30/60)。生の入力ではなく解決後の 30・60 が現れる。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "26:60", "--dry-run"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "30" in text and "60" in text     # スナップ後の範囲端

    # --- --preview-csv -----------------------------------------------------
    def _read_csv(self, path):
        rows = list(_csv.reader(path.read_text(encoding="utf-8").splitlines()))
        return rows[0], rows[1:]

    def test_preview_csv_written(self, tmp_path):
        # --preview-csv はフレームごとの揺れ量(各チャンネル)を CSV 出力する(§2.7)。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        csv_path = tmp_path / "preview.csv"
        assert cli.main([inp, "-o", str(out), "--preview-csv", str(csv_path)]) == 0
        assert csv_path.exists()
        header, data = self._read_csv(csv_path)
        # frame 列 + 回転3 + 位置3 の計7列以上、データ行はベイクフレーム数(61)。
        assert len(header) >= 7
        assert "frame" in header[0].lower()
        assert len(data) == 61
        assert all(len(r) == len(header) for r in data)
        # frame 列は実フレーム(0..60)。全行同一フレーム等を排除(§2.7「フレームごと」)。
        assert [int(r[0]) for r in data] == list(range(0, 61))

    def test_preview_csv_holds_shake_amounts_not_absolute(self, tmp_path):
        # CSV の値は「揺れ量」(=ベイク値−元サンプリング)であって絶対カメラ値ではない(§2.7)。
        # 揺れ無し(amp/settle=0)なら全チャンネルほぼ0、既定(揺れ有り)なら非ゼロが現れる、で判別。
        # 絶対カメラ値なら揺れ無しでも非ゼロ(KEYS の位置は非0)になるはず=これを排除する。
        inp = write_input(tmp_path / "in.vmd")
        zero_csv, shake_csv = tmp_path / "zero.csv", tmp_path / "shake.csv"
        assert cli.main([inp, "-o", str(tmp_path / "z.vmd"), "--preview-csv", str(zero_csv),
                         "--amp-rot", "0", "--amp-pos", "0", "--settle", "0"]) == 0
        assert cli.main([inp, "-o", str(tmp_path / "s.vmd"), "--preview-csv", str(shake_csv)]) == 0

        def channel_vals(path):
            _, data = self._read_csv(path)
            return [abs(float(c)) for r in data for c in r[1:]]  # frame 列以外

        assert max(channel_vals(zero_csv)) < 1e-4    # 揺れ無し → ほぼ0(揺れ量である証拠)
        assert max(channel_vals(shake_csv)) > 1e-4   # 揺れ有り → 非ゼロ

    def test_preview_csv_deltas_reconstruct_baked(self, tmp_path):
        # CSV の各チャンネル値は「ベイク値 − 元サンプリング」の揺れ量で、列ごとに正しい(§2.7)。
        # 全フレームで サンプリング + CSV差分 == ベイク値 を確認(無関係/重複列の実装を排除)。
        from mmd_toolbox.vmd import interp
        inp = write_input(tmp_path / "in.vmd")
        out, csv_path = tmp_path / "out.vmd", tmp_path / "p.csv"
        assert cli.main([inp, "-o", str(out), "--preview-csv", str(csv_path), "--seed", "3"]) == 0
        header, data = self._read_csv(csv_path)
        col = {name: header.index(name)
               for name in ("rot_x", "rot_y", "rot_z", "pos_x", "pos_y", "pos_z")}
        baked = {k.frame: k for k in read_camera(out)}
        for row in data:
            f = int(row[0])
            s = interp.sample_camera(KEYS, f)
            for j, axis in enumerate(("rot_x", "rot_y", "rot_z")):
                assert baked[f].rotation[j] == pytest.approx(
                    s["rotation"][j] + float(row[col[axis]]), abs=1e-4)
            for j, axis in enumerate(("pos_x", "pos_y", "pos_z")):
                assert baked[f].position[j] == pytest.approx(
                    s["position"][j] + float(row[col[axis]]), abs=1e-4)

    def test_preview_csv_still_writes_vmd(self, tmp_path):
        # --preview-csv は通常出力(VMD)も書く(--dry-run とは異なり出力を抑止しない)。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--preview-csv", str(tmp_path / "p.csv")]) == 0
        assert out.exists()

    def test_preview_csv_does_not_change_vmd(self, tmp_path):
        # --preview-csv は付加出力で、ベイクされる VMD を変えない(§2.7)。
        # 非 preview 実行とバイナリ一致する。
        inp = write_input(tmp_path / "in.vmd")
        without, with_csv = tmp_path / "wo.vmd", tmp_path / "w.vmd"
        assert cli.main([inp, "-o", str(without)]) == 0
        assert cli.main([inp, "-o", str(with_csv), "--preview-csv", str(tmp_path / "p.csv")]) == 0
        assert with_csv.read_bytes() == without.read_bytes()

    def test_preview_csv_path_collision_exit2(self, tmp_path):
        # --preview-csv が出力 VMD と同一パスなら CSV が VMD を上書きしてしまう→引数エラー(exit 2)。
        # 入力と同一でも同様。付加出力の契約(§2.7)を守れない組合せを排除する。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--preview-csv", str(out)]) == 2
        assert cli.main([inp, "-o", str(out), "--preview-csv", inp]) == 2

    def test_preview_csv_write_failure_exit3(self, tmp_path):
        # CSV 出力の書き込み失敗も出力書き込み失敗(§9 コード3)として扱う。
        # 親がファイル(ディレクトリでない)の CSV パス → 書き込み不可。
        inp = write_input(tmp_path / "in.vmd")
        clash = tmp_path / "afile"
        clash.write_bytes(b"x")
        rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--preview-csv", str(clash / "p.csv")])
        assert rc == 3

    def test_dry_run_suppresses_preview_csv(self, tmp_path):
        # --dry-run は「出力せず」(§2.7)。--preview-csv 併用でも CSV を書かない。
        inp = write_input(tmp_path / "in.vmd")
        csv_path = tmp_path / "p.csv"
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--preview-csv", str(csv_path), "--dry-run"]) == 0
        assert not csv_path.exists()

    # --- -v / --verbose ----------------------------------------------------
    def test_verbose_reports_range_and_cuts(self, tmp_path, capsys):
        # --verbose は詳細ログ(適用範囲・カット位置等)を出す(§2.7, §5.2, §5.3)。
        # かつ非 verbose 実行ではこれら詳細を出さない(範囲/カット報告は dry-run/verbose 限定)。
        inp = write_input(tmp_path / "cut.vmd", CUT_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "a.vmd")]) == 0
        q = capsys.readouterr()
        quiet_text = (q.out + q.err).lower()
        assert "range" not in quiet_text and "cut" not in quiet_text   # 非verboseは詳細を出さない
        assert cli.main([inp, "-o", str(tmp_path / "b.vmd"), "--verbose"]) == 0
        cap = capsys.readouterr()
        verbose_text = (cap.out + cap.err).lower()
        assert len(cap.out + cap.err) > len(q.out + q.err)            # 出力が増える
        assert "range" in verbose_text and "cut" in verbose_text and "30" in verbose_text

    def test_verbose_reports_snapped_range(self, tmp_path, capsys):
        # --verbose もスナップ後の実適用範囲を報告する(§5.2)。KEYS で 26:60 → 30:60。
        inp = write_input(tmp_path / "in.vmd")
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "26:60", "--verbose"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "30" in text and "60" in text

    def test_verbose_reports_all_cuts(self, tmp_path, capsys):
        # verbose も検出カット全件を報告する(§5.3)。frame21・frame41 の2カット入力。
        inp = write_input(tmp_path / "mcut.vmd", MULTI_CUT_KEYS)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--verbose"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "21" in text and "41" in text

    def test_verbose_reports_all_snapped_ranges(self, tmp_path, capsys):
        # verbose も複数 --range の解決後範囲を全て報告する(§5.2)。0:14→0:15, 44:60→45:60。
        keys = [cam(0), cam(15), cam(30, persp=1), cam(45), cam(60)]
        inp = write_input(tmp_path / "mr.vmd", keys)
        assert cli.main([inp, "-o", str(tmp_path / "out.vmd"),
                         "--range", "0:14", "--range", "44:60", "--verbose"]) == 0
        cap = capsys.readouterr()
        text = cap.out + cap.err
        assert "15" in text and "45" in text

    def test_verbose_still_writes_output(self, tmp_path):
        # --verbose は通常出力(VMD)を書く。出力を抑止するのは --dry-run のみ(§2.7)。
        inp = write_input(tmp_path / "in.vmd")
        out = tmp_path / "out.vmd"
        assert cli.main([inp, "-o", str(out), "--verbose"]) == 0
        assert out.exists()
        assert sorted(k.frame for k in read_camera(out)) == list(range(0, 61))

    def test_verbose_does_not_change_vmd(self, tmp_path):
        # --verbose はログのみで、ベイクされる VMD を変えない(§2.7)。非 verbose と一致する。
        inp = write_input(tmp_path / "in.vmd")
        q, v = tmp_path / "q.vmd", tmp_path / "v.vmd"
        assert cli.main([inp, "-o", str(q)]) == 0
        assert cli.main([inp, "-o", str(v), "--verbose"]) == 0
        assert v.read_bytes() == q.read_bytes()

    def test_verbose_short_alias_equals_long(self, tmp_path, capsys):
        # -v は --verbose と同義(§2.7)。出力パスを同一にして(ログがパスを含んでも差が出ない)
        # 両者の詳細ログが一致することを確認する。
        inp = write_input(tmp_path / "in.vmd")
        out = str(tmp_path / "out.vmd")            # 同一パス(入力とは別なので上書きガード対象外)
        assert cli.main([inp, "-o", out, "-v"]) == 0
        short = capsys.readouterr()
        short_text = short.out + short.err
        assert cli.main([inp, "-o", out, "--verbose"]) == 0
        long = capsys.readouterr()
        assert short_text == (long.out + long.err)
