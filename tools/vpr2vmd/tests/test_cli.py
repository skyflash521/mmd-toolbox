"""vpr2vmd CLI 骨組みのテスト。

範囲は CLI の起動・引数解析・検証と `--dry-run` の空実行(出力を書かない)に限る。
vpr 読み込み・口形イベント確定・VMD 生成の統合は test_convert.py が検証する。

終了コードは既存ツール(shakevmd 等)の規約に倣う:
0 正常 / 1 入力不正(欠落・非vpr 等) / 2 引数エラー(未知オプション・範囲不正・上書きガード) /
3 出力書き込み失敗。
"""

import pytest

from vpr import Note, Part, TempoEvent, Track, VprFormatError, VprProject, VprWarning
from vpr2vmd import cli


def _assert_error_line(err):
    """非機械の失敗理由が『error: <理由>』の行(理由は非空)で出て、トレースバックを含まないこと。"""
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert any(ln.startswith("error: ") and ln[len("error: "):].strip() for ln in lines), err
    assert "Traceback" not in err, err


def _touch(path):
    path.write_bytes(b"")
    return str(path)


@pytest.fixture(autouse=True)
def _stub_read(monkeypatch):
    """--dry-run も vpr を読み処理するため、read を最小の合成プロジェクトへ差し替える。

    引数解析・検証の経路を実 vpr 無しで決定論的に通す(引数エラーは読み込み前に確定するので、
    その分岐は本スタブを使わない)。トラック名は `--track` 名指定テストが一致できるよう `Vocal`。
    """
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(
            name="p", start_tick=0,
            notes=[Note(
                start_tick=0, duration_tick=480, pitch=60,
                lyric="x", velocity=64, phonemes=["a"],
            )],
        )])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))


def test_parses_full_option_set_in_dry_run(tmp_path):
    """主要オプションを一通り受理し、--dry-run が 0 で空実行する。"""
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([
        src,
        "-o", str(out),
        "--track", "0",
        "--model-name", "Model",
        "--style", "ballad",
        "--open-max", "0.8",
        "--default-open", "0.5",
        "--dry-run",
    ])
    assert rc == 0


def test_dry_run_plan_shows_resolved_values(tmp_path, capsys, monkeypatch):
    # 処理計画は解決した値を示す(機械モードの入力検査と同じ解決結果)。未指定を代替表示で
    # 済ませると、利用者はどの値で動くのかを読み取れない。
    # 代表テンポは基準テンポの既定(120)と紛れないよう、入力側を別の値にして見分ける。
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 150.0)],
        tracks=[Track(name="Vocal", parts=[Part(
            name="p", start_tick=0,
            notes=[Note(start_tick=0, duration_tick=480, pitch=60,
                        lyric="x", velocity=64, phonemes=["a"])],
        )])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--dry-run"]) == 0
    # 診断要約は別の表示なので、処理計画の行だけを対象にする。
    plan_text = capsys.readouterr().out.split("--- 診断 ---")[0]
    plan = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip()
            for ln in plan_text.splitlines() if ":" in ln}

    # 解決した対象トラックの番号と名前。
    assert "0" in plan["track"] and "Vocal" in plan["track"]
    # 調整パラメータは代替表示でなく数値(複合表示は区切りで分けて全要素を見る)。
    numeric = ["open-max", "default-open", "legato-max", "coartic-overlap", "anticipation"]
    compound = ["valley(shallow/deep/slope)", "tempo(ref-bpm/scale-min)"]
    for key in numeric:
        float(plan[key])
    for key in compound:
        for value in plan[key].split("/"):
            float(value)
    # 解決の前提になる代表テンポも示す(スタイル解決のテンポ補正がこの値で決まる)。
    assert float(plan["representative-bpm"]) == pytest.approx(150.0)


@pytest.mark.parametrize("velocities, expected", [
    ((64, 64), "default"), ((40, 100), "velocity"), ((), "なし"),
])
def test_dry_run_diagnostics_show_open_source(tmp_path, capsys, monkeypatch,
                                              velocities, expected):
    # 開き量が一定に見えるとき、それが声量曲線由来か既定値かを人間向けの診断でも判別できるようにする。
    # 表示が経路に連動することを見るため、一様と非一様の両方を通す。
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(
            name="p", start_tick=0,
            notes=[Note(start_tick=480 * i, duration_tick=480, pitch=60, lyric="x",
                        velocity=v, phonemes=["a"])
                   for i, v in enumerate(velocities)],
        )])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--dry-run"]) == 0
    diagnostics = capsys.readouterr().out.split("--- 診断 ---")[1]
    shown = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip()
             for ln in diagnostics.splitlines() if ":" in ln}
    assert shown["開き量の決定経路"] == expected


def test_dry_run_writes_no_output(tmp_path):
    """--dry-run は出力 VMD を書かない。"""
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


def test_report_json_is_now_unknown_option(tmp_path):
    """--report-json は廃止。未知オプションとして引数エラー(コード2)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--report-json", str(tmp_path / "rep.json"), "--dry-run"]) == 2


def test_track_accepts_non_integer_name(tmp_path):
    """--track は半角数字だけの INDEX 指定に限らず Track 名も受理する。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--track", "Vocal", "--dry-run"]) == 0


def test_unknown_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--bogus"]) == 2


def test_missing_positional_is_arg_error():
    assert cli.main([]) == 2


def test_unknown_style_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--style", "nope", "--dry-run"]) == 2


def test_open_max_non_float_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "abc", "--dry-run"]) == 2


def test_default_open_non_float_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--default-open", "abc", "--dry-run"]) == 2


def test_open_max_negative_is_arg_error(tmp_path):
    """開き量は 0〜1。負の --open-max は引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "-0.1", "--dry-run"]) == 2


def test_open_max_over_one_is_arg_error(tmp_path):
    """開き量は 0〜1。1 を超える --open-max は引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "1.5", "--dry-run"]) == 2


def test_default_open_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--default-open", "-0.5", "--dry-run"]) == 2


def test_open_amount_bounds_are_accepted(tmp_path):
    """境界値 0.0 と 1.0 は受理する。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "1.0", "--default-open", "0.0", "--dry-run"]) == 0


def test_missing_input_file_is_input_error(tmp_path):
    """存在しない入力 vpr は入力不正(終了コード 1)。"""
    missing = str(tmp_path / "nope.vpr")
    assert cli.main([missing]) == 1


def test_existing_other_path_output_blocked_without_overwrite(tmp_path):
    """別パスの既存出力も --overwrite 無しでは上書きガードで拒否する。"""
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")  # 既存出力(入力とは別パス)
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_existing_other_path_output_not_overwritten_without_overwrite(tmp_path):
    """別パスの既存出力を通常実行(--overwrite 無し)で書き換えないこと。"""
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    out.write_bytes(b"stale")  # 既存の別パス出力(入力とは別パス)
    rc = cli.main([src, "-o", str(out)])
    assert rc == 2
    assert out.read_bytes() == b"stale"  # ガードに阻まれ上書きされていない


def test_overwrite_allows_existing_output(tmp_path):
    """--overwrite 指定時は既存出力があってもガードで弾かれない。"""
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


def test_overwrite_guard_blocks_input_overwrite(tmp_path):
    """出力先が入力と同一パスのとき、--overwrite 無しは引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "-o", src, "--dry-run"]) == 2


def test_missing_same_path_output_is_input_error_not_overwrite_guard(tmp_path):
    """入力・出力が同一の未存在パスなら、出力先に既存ファイルが無いため上書きガードは発火せず、
    後続の入力存在確認で入力不正(コード1)になる。
    """
    missing = str(tmp_path / "missing.vpr")
    assert cli.main([missing, "-o", missing, "--dry-run"]) == 1


def test_model_name_over_20_bytes_is_arg_error(tmp_path):
    """--model-name が cp932 で 20 バイト超なら引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "x" * 21, "--dry-run"]) == 2


def test_no_n_morph_accepted_in_dry_run(tmp_path):
    """--no-n-morph を受理し --dry-run で 0。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--no-n-morph", "--dry-run"]) == 0


def test_dry_run_plan_reflects_n_morph_off_by_default(tmp_path, capsys):
    """既定では撥音を無音へ倒す。--dry-run の計画表示に n-morph off を出す。"""
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--dry-run"])
    assert "n-morph: off" in capsys.readouterr().out


def test_dry_run_plan_reflects_no_n_morph(tmp_path, capsys):
    """--no-n-morph 明示指定時も既定と同じく撥音を無音へ倒す旨(n-morph off)を計画表示に出す。"""
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--no-n-morph", "--dry-run"])
    assert "n-morph: off" in capsys.readouterr().out


def test_dry_run_plan_reflects_n_morph_on(tmp_path, capsys):
    """--n-morph 指定時は「ん」モーフを使う旨(n-morph on)を計画表示に出す。"""
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--n-morph", "--dry-run"])
    assert "n-morph: on" in capsys.readouterr().out


def test_model_name_multibyte_over_20_bytes_is_arg_error(tmp_path):
    """制約はバイト長。cp932 で 2 バイトの文字 11 個=22 バイトは引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "あ" * 11, "--dry-run"]) == 2


def test_model_name_multibyte_at_20_byte_limit_is_accepted(tmp_path):
    """cp932 で 2 バイトの文字 10 個=20 バイトちょうどは受理する。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "あ" * 10, "--dry-run"]) == 0


def test_model_name_non_cp932_is_arg_error(tmp_path):
    """--model-name が Shift-JIS(cp932)で表現できなければ引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "\U0001F600", "--dry-run"]) == 2


def test_model_name_at_20_byte_limit_is_accepted(tmp_path):
    """上限ちょうど(cp932 で 20 バイト)の --model-name は受理する。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "x" * 20, "--dry-run"]) == 0


def test_existing_default_output_blocked_without_overwrite(tmp_path):
    """既定出力 <入力名>.vmd が既存なら、-o 無し・--overwrite 無しは上書きガードで拒否する。"""
    src = _touch(tmp_path / "song.vpr")
    _touch(tmp_path / "song.vmd")  # 既定出力が既に存在
    assert cli.main([src, "--dry-run"]) == 2


def test_existing_default_output_allowed_with_overwrite(tmp_path):
    src = _touch(tmp_path / "song.vpr")
    _touch(tmp_path / "song.vmd")
    assert cli.main([src, "--overwrite", "--dry-run"]) == 0


def test_default_output_is_vmd_alongside_input(tmp_path):
    """既定出力は <入力名>.vmd。--dry-run では書かないがパス解決でエラーにならない。"""
    src = _touch(tmp_path / "song.vpr")
    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "song.vmd").exists()


# --- 調整パラメータの CLI オプション ---


def test_parses_tuning_options_in_dry_run(tmp_path):
    """調整パラメータ一式を受理し --dry-run が 0 で空実行する。"""
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([
        src,
        "--legato-max", "10",
        "--valley-shallow", "0.5",
        "--valley-deep", "0.25",
        "--valley-slope", "0.03",
        "--coartic-overlap", "5",
        "--anticipation", "8",
        "--ref-bpm", "160",
        "--tempo-scale-min", "0.4",
        "--dry-run",
    ])
    assert rc == 0


def test_legato_max_non_positive_is_arg_error(tmp_path):
    """--legato-max は正値。0 以下は引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--legato-max", "0", "--dry-run"]) == 2


def test_legato_max_non_float_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--legato-max", "abc", "--dry-run"]) == 2


def test_valley_shallow_out_of_range_is_arg_error(tmp_path):
    """谷係数は 0.0〜1.0。範囲外は引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--valley-shallow", "1.5", "--dry-run"]) == 2
    assert cli.main([src, "--valley-deep", "-0.1", "--dry-run"]) == 2


def test_valley_slope_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--valley-slope", "-0.1", "--dry-run"]) == 2


def test_coartic_overlap_below_one_is_arg_error(tmp_path):
    """協調調音の重なりは 1 以上。0 は引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--coartic-overlap", "0", "--dry-run"]) == 2


def test_anticipation_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--anticipation", "-1", "--dry-run"]) == 2


def test_anticipation_zero_is_accepted(tmp_path):
    """先行準備 0(無効化)は受理する。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--anticipation", "0", "--dry-run"]) == 0


def test_ref_bpm_non_positive_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--ref-bpm", "0", "--dry-run"]) == 2


def test_tempo_scale_min_out_of_range_is_arg_error(tmp_path):
    """下げ止まり係数は 0 超〜1.0。0 と 1 超は引数エラー、1.0 は受理。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--tempo-scale-min", "0", "--dry-run"]) == 2
    assert cli.main([src, "--tempo-scale-min", "1.5", "--dry-run"]) == 2
    assert cli.main([src, "--tempo-scale-min", "1.0", "--dry-run"]) == 0


def test_valley_deep_above_shallow_is_arg_error_even_in_dry_run(tmp_path):
    """谷係数の逆転(下限>上限)は vpr 内容に依らない引数エラー。dry-run でも弾く。"""
    src = _touch(tmp_path / "in.vpr")
    # pop 既定 shallow=0.45 に対し下限を上回らせる(片側指定×プリセット値の組み合わせ)。
    assert cli.main([src, "--valley-deep", "0.6", "--dry-run"]) == 2
    # 明示両指定の逆転も弾く。
    assert cli.main([src, "--valley-shallow", "0.1", "--valley-deep", "0.2", "--dry-run"]) == 2
    # 正当な範囲(下限≤上限)は dry-run で受理する。
    assert cli.main([src, "--valley-shallow", "0.5", "--valley-deep", "0.2", "--dry-run"]) == 0


def test_dry_run_plan_shows_tuning_overrides(tmp_path, capsys):
    """--dry-run の計画表示に調整パラメータの上書き値を出す。"""
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--legato-max", "12", "--anticipation", "9", "--dry-run"])
    out = capsys.readouterr().out
    assert "legato-max: 12" in out
    assert "anticipation: 9" in out


# --- バージョン・--version/--n-morph/--verbose・上書きガード統一・非機械の失敗理由 ---


def _project_with_notes(notes):
    """指定音符列を持つ単一トラック `Vocal` の合成プロジェクト。"""
    return VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(name="p", start_tick=0, notes=notes)])],
    )


def test_version_attr_exposed():
    """パッケージがバージョン属性 __version__(非空の文字列)を公開すること(--version が表示するバージョンの源)。"""
    import vpr2vmd

    assert isinstance(vpr2vmd.__version__, str) and vpr2vmd.__version__


def test_version_flag_prints_and_exits_zero(capsys):
    """--version はバージョン(__version__)を表示して終了コード 0 で終わる。"""
    import vpr2vmd

    rc = cli.main(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vpr2vmd" in out
    assert vpr2vmd.__version__ in out


def test_n_morph_positive_flag_accepted(tmp_path):
    """--n-morph(肯定形)を受理し --dry-run で 0(--no-n-morph の対)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--n-morph", "--dry-run"]) == 0


def test_verbose_prints_plan_on_normal_run(tmp_path, capsys):
    """--verbose は通常実行でも処理計画・診断を標準出力へ出し、出力 VMD を書く。"""
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--verbose"])
    captured = capsys.readouterr().out
    assert rc == 0
    assert out.exists()
    assert "採用音符数" in captured


def test_missing_input_reports_reason(tmp_path, capsys):
    """存在しない入力 vpr は理由 1 行 + 終了コード 1(input_not_found)。"""
    missing = str(tmp_path / "nope.vpr")
    rc = cli.main([missing])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_missing_positional_reports_reason(capsys):
    """input 欠落は理由 1 行 + 終了コード 2(bad_argument)。

    input は nargs=? なので argparse は通し、CLI 本体の検査が失敗報告経路へ流す。
    """
    rc = cli.main([])
    assert rc == 2
    assert capsys.readouterr().err.strip()


def test_not_vpr_reports_reason(tmp_path, capsys, monkeypatch):
    """非 vpr(VprFormatError)は理由 1 行 + 終了コード 1(not_vpr)。"""
    src = _touch(tmp_path / "in.vpr")

    def _raise(_src):
        raise VprFormatError("壊れた vpr")

    monkeypatch.setattr(cli, "read", _raise)
    rc = cli.main([src])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_no_tracks_reports_reason(tmp_path, capsys, monkeypatch):
    """トラックが 1 件も無い入力は理由 1 行 + 終了コード 1(no_tracks)。"""
    src = _touch(tmp_path / "in.vpr")
    empty = VprProject(resolution=480, tempos=[TempoEvent(0, 120.0)], tracks=[])
    monkeypatch.setattr(cli, "read", lambda _src: (empty, []))
    rc = cli.main([src])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_bad_track_reports_reason(tmp_path, capsys):
    """--track の INDEX 範囲外は理由 1 行 + 終了コード 2(bad_track)。"""
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--track", "5"])
    assert rc == 2
    _assert_error_line(capsys.readouterr().err)


def test_valley_inverted_reports_reason(tmp_path, capsys):
    """谷係数の逆転は理由 1 行 + 終了コード 2(valley_bounds_inverted)。"""
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--valley-deep", "0.6", "--dry-run"])
    assert rc == 2
    _assert_error_line(capsys.readouterr().err)


def test_output_exists_reports_reason(tmp_path, capsys):
    """出力先に既存ファイルがあれば理由 1 行 + 終了コード 2(output_exists)。"""
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "-o", src, "--dry-run"])
    assert rc == 2
    _assert_error_line(capsys.readouterr().err)


def test_write_failure_reports_reason(tmp_path, capsys, monkeypatch):
    """出力書き込み失敗(OSError)は理由 1 行 + 終了コード 3(write_failed)。"""
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"

    def _raise(_doc, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_file", _raise)
    rc = cli.main([src, "-o", str(out)])
    assert rc == 3
    _assert_error_line(capsys.readouterr().err)


def test_internal_error_reports_reason_without_traceback(tmp_path, capsys, monkeypatch):
    """想定外例外はトレースバックを漏らさず理由 1 行 + 終了コード 1(internal_error)。"""
    src = _touch(tmp_path / "in.vpr")

    def _boom(*_a, **_k):
        raise RuntimeError("想定外")

    monkeypatch.setattr(cli, "generate_morph_keys", _boom)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_vpr_read_warning_surfaced_to_stderr(tmp_path, capsys, monkeypatch):
    """vpr 読み込みが返す構造化警告(重なり音符)を標準エラーへ出す。"""
    src = _touch(tmp_path / "in.vpr")
    project = _project_with_notes(
        [Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])]
    )
    warning = VprWarning(
        code="overlapping_notes", message="重なり音符", track_index=0, part_index=0,
        note_index=1, related_note_index=0, tick=0,
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, [warning]))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    err = capsys.readouterr().err
    assert rc == 0
    assert "重なり音符" in err or "overlapping_notes" in err


def test_no_adopted_notes_warns_and_succeeds(tmp_path, capsys, monkeypatch):
    """採用音符が空でもエラーにせず正常終了し、標準エラーへ警告を出す。"""
    src = _touch(tmp_path / "in.vpr")
    empty_track = _project_with_notes([])  # 発音の無いトラック
    monkeypatch.setattr(cli, "read", lambda _src: (empty_track, []))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 0
    assert "発音" in capsys.readouterr().err


def test_read_warning_line_uses_common_format(tmp_path, capsys, monkeypatch):
    # vpr 読み込みが返す構造化警告も同じ共通書式(warning: コード: 本文)で1行にまとめる。
    src = _touch(tmp_path / "in.vpr")
    project = _project_with_notes(
        [Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])]
    )
    warning = VprWarning(
        code="overlapping_notes", message="重なり音符", track_index=0, part_index=0,
        note_index=1, related_note_index=0, tick=0,
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, [warning]))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: overlapping_notes: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert not body.startswith(" ")
    assert "警告:" not in err


def test_no_adopted_notes_warning_line_uses_common_format(tmp_path, capsys, monkeypatch):
    # 採用音符が空のときの警告も同じ共通書式(warning: コード: 本文)で1行にまとめる。
    src = _touch(tmp_path / "in.vpr")
    empty_track = _project_with_notes([])
    monkeypatch.setattr(cli, "read", lambda _src: (empty_track, []))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: no_adopted_notes: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert not body.startswith(" ")
    assert "警告:" not in err


def test_non_machine_usage_error_is_single_error_line(capsys):
    """非機械モードの使用法エラーも人間向けのエラー行1行だけを出し、argparse 素の用法は出さない。"""
    rc = cli.main(["in.vpr", "--bogus"])
    assert rc == 2
    # 標準エラー全体との完全一致で、物理的に1行であること・書式・argparse 生成の本文をそのまま
    # 載せていることを同時に固定する(用法の行が混じればここで落ちる)。
    assert capsys.readouterr().err == "error: unrecognized arguments: --bogus\n"
