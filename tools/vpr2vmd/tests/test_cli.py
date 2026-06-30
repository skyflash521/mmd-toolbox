"""vpr2vmd CLI 骨組みのテスト(vpr2vmd.md §4)。

範囲は CLI の起動・引数解析・検証と `--dry-run` の空実行(出力を書かない)に限る。
vpr 読み込み・口形イベント確定・VMD 生成の統合は test_convert.py が検証する。

終了コードは既存ツール(shakevmd 等)の規約に倣う:
0 正常 / 1 入力不正(欠落・非vpr 等) / 2 引数エラー(未知オプション・範囲不正・上書きガード) /
3 出力書き込み失敗。
"""

import pytest
from vpr import Note, Part, TempoEvent, Track, VprProject

from vpr2vmd import cli


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


def test_dry_run_writes_no_output(tmp_path):
    """--dry-run は出力 VMD を書かない。"""
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


def test_report_json_is_now_unknown_option(tmp_path):
    """--report-json は廃止。未知オプションとして引数エラー(コード2。vpr2vmd.md §4.2・§4.4)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--report-json", str(tmp_path / "rep.json"), "--dry-run"]) == 2


def test_track_accepts_non_integer_name(tmp_path):
    """--track は整数 INDEX だけでなく非整数の Track 名も受理する(vpr2vmd.md §4.2)。"""
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


def test_existing_output_requires_overwrite(tmp_path):
    """既存の出力 VMD は --overwrite 無しでは上書きしない(vpr2vmd.md §4.2)。"""
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")  # 既存出力(入力とは別パス)
    assert cli.main([src, "-o", out]) == 2


def test_overwrite_allows_existing_output(tmp_path):
    """--overwrite 指定時は既存出力があってもガードで弾かれない。"""
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


def test_overwrite_guard_blocks_input_overwrite(tmp_path):
    """出力先が入力と同一パスのとき、--overwrite 無しは引数エラー。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "-o", src, "--dry-run"]) == 2


def test_overwrite_guard_blocks_same_path_even_when_missing(tmp_path):
    """入力同一パス指定は、そのパスが未存在でも上書きガード(引数エラー)で弾く。

    存在確認(入力不正=1)より同一パス判定を先に行うため、未存在でも 1 でなく 2 になる。
    """
    missing = str(tmp_path / "missing.vpr")
    assert cli.main([missing, "-o", missing, "--dry-run"]) == 2


def test_model_name_over_20_bytes_is_arg_error(tmp_path):
    """--model-name が cp932 で 20 バイト超なら引数エラー(vpr2vmd.md §4.2)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "x" * 21, "--dry-run"]) == 2


def test_no_n_morph_accepted_in_dry_run(tmp_path):
    """--no-n-morph を受理し --dry-run で 0(vpr2vmd.md §4.2)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--no-n-morph", "--dry-run"]) == 0


def test_dry_run_plan_reflects_n_morph_on_by_default(tmp_path, capsys):
    """既定では「ん」モーフを使う。--dry-run の計画表示に n-morph on を出す。"""
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--dry-run"])
    assert "n-morph: on" in capsys.readouterr().out


def test_dry_run_plan_reflects_no_n_morph(tmp_path, capsys):
    """--no-n-morph 指定時は撥音を無音へ倒す旨(n-morph off)を計画表示に出す。"""
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--no-n-morph", "--dry-run"])
    assert "n-morph: off" in capsys.readouterr().out


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


def test_existing_default_output_requires_overwrite(tmp_path):
    """既定出力 <入力名>.vmd が既存なら、-o 無し・--overwrite 無しでも拒否する。"""
    src = _touch(tmp_path / "song.vpr")
    _touch(tmp_path / "song.vmd")  # 既定出力が既に存在
    assert cli.main([src]) == 2


def test_default_output_is_vmd_alongside_input(tmp_path):
    """既定出力は <入力名>.vmd。--dry-run では書かないがパス解決でエラーにならない。"""
    src = _touch(tmp_path / "song.vpr")
    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "song.vmd").exists()


# --- 調整パラメータの CLI オプション(vpr2vmd.md §4.2) ---


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
