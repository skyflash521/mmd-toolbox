"""vpr2vmd CLI 骨組みのテスト(vpr2vmd.md §4、実装計画 P-0)。

P-0 の範囲は CLI の起動・引数解析・検証と `--dry-run` の空実行(出力を書かない)に限る。
vpr 読み込み・口形イベント確定・VMD 生成は後続ステップ(P-1〜P-4)で検証する。

終了コードは既存ツール(shakevmd 等)の規約に倣う:
0 正常 / 1 入力不正(欠落・非vpr 等) / 2 引数エラー(未知オプション・範囲不正・上書きガード) /
3 出力書き込み失敗。
"""

import pytest

# P-0 実装前は vpr2vmd.cli が無く import で収集が失敗するため、実装が入るまでスキップする。
cli = pytest.importorskip("vpr2vmd.cli", reason="impl pending: P-0 CLI骨組み")


def _touch(path):
    path.write_bytes(b"")
    return str(path)


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
        "--report-json", str(tmp_path / "rep.json"),
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


def test_dry_run_writes_no_report_json(tmp_path):
    """--dry-run は report-json も書かない(出力せず、が原則。vpr2vmd.md §4.2)。"""
    src = _touch(tmp_path / "in.vpr")
    rep = tmp_path / "rep.json"
    rc = cli.main([src, "--report-json", str(rep), "--dry-run"])
    assert rc == 0
    assert not rep.exists()


def test_track_accepts_non_integer_name(tmp_path):
    """--track は整数 INDEX だけでなく非整数の Track 名も受理する(vpr2vmd.md §4.2)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--track", "Lead Vocal", "--dry-run"]) == 0


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


def test_model_name_over_20_bytes_is_arg_error(tmp_path):
    """--model-name が cp932 で 20 バイト超なら引数エラー(vpr2vmd.md §4.2)。"""
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "x" * 21, "--dry-run"]) == 2


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
