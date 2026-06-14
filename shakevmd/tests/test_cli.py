"""shakevmd CLI のテスト(shakevmd.md §2, §9)。

CLI はコアの薄いラッパー: 引数解析 → VMD読み(mmd_toolbox.vmd.io)→ bake() → VMD書き。
終了コード(§9): 0 正常 / 1 入力不正(VMDでない・カメラキーなし)/ 2 引数エラー
(範囲不正・重複・上書き未許可)/ 3 出力書き込み失敗。

本サブステップの範囲は **§2.1-2.6 + §9 のコア CLI**(I/O・範囲・主要揺れパラメーター・終了コード)。
§2.7 の運用/プリセット flags(`--preset`・`--dry-run`・`--preview-csv`・`-v/--verbose`)は
**後続「CLI 運用機能」サブステップへ繰り延べ**る(--preset は presets.py を要し、--dry-run/
--preview-csv は bake 結果の追加情報=最大振幅・チャンネル別揺れ量の公開を要するため)。
未実装の間、これらは未知オプションとして exit 2(test_rejects_internal_params_exit2 と同じ扱い)。
"""

import sys

import numpy as np
import pytest

from shakevmd import cli
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

    def test_deferred_section27_options_exit2(self, tmp_path):
        # §2.7 運用/プリセット flags は本サブステップ未実装 → 未知オプションとして exit2
        # (後続「CLI 運用機能」サブステップで実装。方針=docstring と契約=本テストを一致させる)
        inp = write_input(tmp_path / "in.vmd")
        for args in (["--preset", "handheld"], ["--dry-run"], ["--preview-csv", "x.csv"], ["--verbose"], ["-v"]):
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

    def test_missing_option_operand_exit2(self, tmp_path):
        # §9: 値を要するオプションに値が無い(オペランド欠落)も引数エラー(exit 2)。
        inp = write_input(tmp_path / "in.vmd")
        for opt in ("--output", "--range", "--seed", "--amp-rot", "--amp-pos",
                    "--rot-weights", "--freq", "--fade", "--motion-scale",
                    "--settle", "--cut-threshold", "--impulse"):
            assert cli.main([inp, opt]) == 2

    def test_rejects_internal_params_exit2(self, tmp_path):
        # §8: 詳細内部パラメーター(オクターブ構成・persistence・settle収束時間等)は
        # CLI 非公開 → 未知オプションとして引数エラー(exit 2)。
        # 内部フラグ名は spec で規定されないため代表確認。網羅の本質は「未知オプション
        # は一律 exit 2」で、これは argparse がすべての未知フラグに対し保証する。
        inp = write_input(tmp_path / "in.vmd")
        for opt in (["--octaves", "5"], ["--persistence", "0.7"], ["--settle-time", "2"]):
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
