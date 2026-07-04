"""song2vmd CLI 骨組みのテスト(song2vmd.md §5・11章)。

範囲は CLI の起動・引数解析・検証・出力先解決・上書きガードと `--dry-run` の空実行(出力を書かない)に
限る。音声読み込み・口形イベント確定・lipsync 連携・VMD 生成は音声処理パイプラインの実装後に検証する。

終了コード(song2vmd.md 11章): 0 正常 / 1 入力不正 / 2 引数エラー(未知オプション・範囲不正・
上書きガード等) / 3 出力書き込み失敗 / 4 音声前段の外部依存の失敗 / 130 協調的な中断。
"""

import pytest

from song2vmd import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


# --- 主要オプションの受理と --dry-run の空実行 -------------------------------


def test_parses_full_option_set_in_dry_run(tmp_path):
    """主要オプションを一通り受理し、--dry-run が 0 で空実行する。"""
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    rc = cli.main([
        src,
        "-o", str(out),
        "--model-name", "Model",
        "--style", "ballad",
        "--separate-vocals", "always",
        "--separator", "audio-separator-htdemucs-ft",
        "--recognizer", "whisper-ctc-forcedalign",
        "--no-n-morph",
        "--vowel-gain", "1.0:0.9:0.8:0.7:0.6",
        "--open-max", "0.8",
        "--coarticulation", "2",
        "--anticipation", "1",
        "--min-hold", "3",
        "--intensity-curve", "0.5",
        "--silence-threshold", "0.05:0.12",
        "--max-duration", "120",
        "--keep-intermediate",
        "-v",
        "--dry-run",
    ])
    assert rc == 0


def test_dry_run_writes_no_output(tmp_path):
    """--dry-run は出力 VMD を書かない。"""
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


def test_default_output_is_vmd_alongside_input(tmp_path):
    """既定出力は <入力名>.vmd。--dry-run では書かないがパス解決でエラーにならない。"""
    src = _touch(tmp_path / "song.wav")
    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "song.vmd").exists()


def test_n_morph_defaults_to_on(tmp_path):
    """--n-morph/--no-n-morph 無指定の既定は on(撥音「ん」にんモーフを使う)。"""
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.n_morph is True


def test_no_n_morph_disables(tmp_path):
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--no-n-morph", "--dry-run"])
    assert args.n_morph is False


def test_vowel_gain_default(tmp_path):
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.vowel_gain == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_silence_threshold_default(tmp_path):
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.silence_threshold == (0.06, 0.10)


def test_style_default_is_pop(tmp_path):
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.style == "pop"


def test_open_max_and_preset_dependent_defaults_are_none(tmp_path):
    """プリセット依存の既定(--open-max・--coarticulation・--anticipation・--min-hold)は未指定時 None。"""
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.open_max is None
    assert args.coarticulation is None
    assert args.anticipation is None
    assert args.min_hold is None


# --- 引数エラー(argparse 検出。終了コード2)---------------------------------


def test_unknown_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--bogus"]) == 2


def test_missing_positional_is_arg_error():
    assert cli.main([]) == 2


def test_unknown_style_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--style", "nope", "--dry-run"]) == 2


def test_unknown_separate_vocals_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separate-vocals", "sometimes", "--dry-run"]) == 2


def test_unknown_separator_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separator", "nope", "--dry-run"]) == 2


def test_unknown_recognizer_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer", "nope", "--dry-run"]) == 2


@pytest.mark.parametrize("opt", ["--open-max", "--intensity-curve", "--max-duration"])
def test_float_option_non_numeric_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "abc", "--dry-run"]) == 2


@pytest.mark.parametrize("opt", ["--coarticulation", "--anticipation", "--min-hold"])
def test_int_option_non_numeric_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "abc", "--dry-run"]) == 2


@pytest.mark.parametrize("opt", ["--coarticulation", "--anticipation", "--min-hold"])
def test_int_option_negative_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "-1", "--dry-run"]) == 2


def test_open_max_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--open-max", "-0.1", "--dry-run"]) == 2


def test_open_max_over_one_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--open-max", "1.5", "--dry-run"]) == 2


def test_open_max_bounds_are_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--open-max", "0.0", "--dry-run"]) == 0
    assert cli.main([src, "--open-max", "1.0", "--dry-run"]) == 0


def test_intensity_curve_zero_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--intensity-curve", "0", "--dry-run"]) == 2


def test_intensity_curve_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--intensity-curve", "-0.5", "--dry-run"]) == 2


def test_max_duration_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--max-duration", "-1", "--dry-run"]) == 2


def test_max_duration_zero_is_accepted(tmp_path):
    """0 は長尺分割の無効化を意味する有効値(song2vmd.md 5.2)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--max-duration", "0", "--dry-run"]) == 0


@pytest.mark.parametrize("text", ["1:1:1:1", "1:1:1:1:1:1", "a:b:c:d:e", "1:1:1:1:-1"])
def test_vowel_gain_invalid_is_arg_error(tmp_path, text):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--vowel-gain", text, "--dry-run"]) == 2


def test_vowel_gain_valid_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--vowel-gain", "1.1:0.9:1.0:1.0:1.2", "--dry-run"]) == 0


@pytest.mark.parametrize("text", ["0.1", "0.1:0.2:0.3", "a:b", "-0.1:0.2", "1.1:0.2"])
def test_silence_threshold_malformed_is_arg_error(tmp_path, text):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--silence-threshold", text, "--dry-run"]) == 2


def test_silence_threshold_on_not_less_than_off_is_arg_error(tmp_path):
    """下降側(ON)は上昇側(OFF)未満でなければならない(無音ヒステリシスの意味上の制約)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--silence-threshold", "0.10:0.06", "--dry-run"]) == 2
    assert cli.main([src, "--silence-threshold", "0.10:0.10", "--dry-run"]) == 2


def test_model_name_over_20_bytes_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "x" * 21, "--dry-run"]) == 2


def test_model_name_multibyte_over_20_bytes_is_arg_error(tmp_path):
    """制約はバイト長。cp932 で2バイトの文字11個=22バイトは引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "あ" * 11, "--dry-run"]) == 2


def test_model_name_multibyte_at_20_byte_limit_is_accepted(tmp_path):
    """cp932 で2バイトの文字10個=20バイトちょうどは受理する。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "あ" * 10, "--dry-run"]) == 0


def test_model_name_non_cp932_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "\U0001F600", "--dry-run"]) == 2


def test_model_name_at_20_byte_limit_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "x" * 20, "--dry-run"]) == 0


# --- 上書きガード(song2vmd.md 5.3)-----------------------------------------


def test_overwrite_guard_blocks_input_overwrite(tmp_path):
    """出力先が入力と同一パスのとき、--overwrite 無しは引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", src, "--dry-run"]) == 2


def test_overwrite_guard_blocks_same_path_even_when_missing(tmp_path):
    """入力同一パス指定は、そのパスが未存在でも上書きガード(引数エラー)で弾く。"""
    missing = str(tmp_path / "missing.wav")
    assert cli.main([missing, "-o", missing, "--dry-run"]) == 2


def test_overwrite_flag_allows_input_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", src, "--overwrite", "--dry-run"]) == 0


def test_existing_separate_output_does_not_require_overwrite(tmp_path):
    """保護対象は入力ファイルに限る。別パスの既存出力ファイルはガード対象外(song2vmd.md 5.3)。"""
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")  # 既存だが入力とは別パス
    assert cli.main([src, "-o", out, "--dry-run"]) == 0


# --- メタ操作(--version・--help)--------------------------------------------


def test_version_prints_and_exits_zero(capsys):
    from song2vmd import __version__

    rc = cli.main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "song2vmd" in out and __version__ in out
    assert __version__ == "0.0.1"


def test_help_lists_key_flags(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    for flag in ("--machine", "--describe", "--quiet", "--version", "--style",
                 "--separate-vocals", "--separator", "--recognizer", "--vowel-gain",
                 "--silence-threshold", "--no-n-morph"):
        assert flag in text
