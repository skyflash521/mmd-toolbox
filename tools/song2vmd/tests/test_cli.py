"""song2vmd CLI 骨組みのテスト。

範囲は CLI の起動・引数解析・検証・出力先解決・上書きガードと `--dry-run` の(出力を書かない)経路に
限る。実際の音声処理パイプライン(`pipeline.run`)は決定論的なスタブに差し替え、引数の受理・検証・
出力先解決だけを対象にする(パイプラインの実データ配線は test_cli_run.py で検証する)。

終了コード: 0 正常 / 1 入力不正 / 2 引数エラー(未知オプション・範囲不正・
上書きガード等) / 3 出力書き込み失敗 / 4 音声前段の外部依存の失敗 / 130 協調的な中断。
"""

import re

import pytest

from song2vmd import cli
from song2vmd import events as _events
from song2vmd import pipeline as _pipeline
from song2vmd import report as _report
from vmd import VmdDocument


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _stub_pipeline_result():
    document = VmdDocument(model_name_raw=b"\x00" * 20, morph=[])
    diagnostics = _report.build_diagnostics(
        segments=[], mouth_events=[], mora_event_group_sizes=[],
        event_diagnostics=_events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0),
        backends={"separator": "audio-separator-htdemucs-ft", "recognizer": "openai/whisper-medium"},
        style="pop", separated=True, duration_sec=1.0, keys=0, forced_split=False,
    )
    return _pipeline.PipelineResult(document=document, diagnostics=diagnostics, sample_rate=44100, channels=2)


@pytest.fixture(autouse=True)
def _stub_pipeline_run(monkeypatch):
    """cli.py の引数解析・検証だけを対象にするため、実処理(pipeline.run)を決定論的スタブへ差し替える。"""
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: _stub_pipeline_result())


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
        "--recognizer-model-id", "openai/whisper-medium",
        "--recognizer-model-revision", "abc123",
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


def test_n_morph_defaults_to_off(tmp_path):
    """--n-morph/--no-n-morph 無指定の既定は off(撥音を無音に倒す)。"""
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.n_morph is False


def test_n_morph_flag_enables(tmp_path):
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--n-morph", "--dry-run"])
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


def test_recognizer_model_id_accepts_any_string(tmp_path):
    """--recognizer-model-id は安定idでなく自由な文字列。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-id", "anything/goes", "--dry-run"]) == 0


def test_recognizer_model_revision_without_model_id_is_arg_error(tmp_path):
    """--recognizer-model-revision だけの指定は対象が無く無意味なので引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-revision", "abc123", "--dry-run"]) == 2


def test_explicit_recognizer_retry_flag_is_accepted(tmp_path):
    """既定onの明示形 --recognizer-retry は単独で受理される。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-retry", "--dry-run"]) == 0


def test_forced_aligner_default_needs_no_sofa_args(tmp_path):
    """--forced-aligner既定(wav2vec2-ctc-forcedalign)は--sofa-*が一切無くても成功する。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--dry-run"]) == 0


def test_forced_aligner_sofa_without_sofa_python_is_arg_error(tmp_path):
    """--forced-aligner sofa-forcedalign選択時、--sofa-python欠落は引数エラー(最初の欠落を報告)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-root", "/sofa", "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 2


def test_forced_aligner_sofa_without_sofa_root_is_arg_error(tmp_path):
    """--sofa-pythonがあっても--sofa-root欠落は引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 2


def test_forced_aligner_sofa_without_checkpoint_is_arg_error(tmp_path):
    """--sofa-python・--sofa-rootがあっても--sofa-checkpoint欠落は引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa", "--dry-run",
    ]) == 2


def test_forced_aligner_sofa_with_all_sofa_args_is_accepted(tmp_path):
    """--forced-aligner sofa-forcedalign選択時、--sofa-*3つが揃えば成功する(--sofa-timeoutは既定値可)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 0


def test_forced_aligner_sofa_timeout_zero_is_arg_error(tmp_path):
    """--sofa-timeoutは正の数値のみ(0以下は無意味)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--sofa-timeout", "0", "--dry-run",
    ]) == 2


def test_forced_aligner_unknown_choice_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--forced-aligner", "unknown-aligner", "--dry-run"]) == 2


def test_describe_succeeds_without_forced_aligner_specified():
    """--describeは--forced-aligner未指定・--sofa-*無しでも成功する(音声を読まない自己記述)。"""
    assert cli.main(["--describe"]) == 0


@pytest.mark.parametrize("opt", ["--open-max", "--intensity-curve", "--max-duration"])
def test_float_option_non_numeric_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "abc", "--dry-run"]) == 2


@pytest.mark.parametrize("opt", ["--coarticulation", "--anticipation", "--min-hold"])
def test_int_option_non_numeric_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "abc", "--dry-run"]) == 2


@pytest.mark.parametrize("opt, value", [
    ("--open-max", "abc"), ("--open-max", "nan"), ("--open-max", "2"),
    ("--coarticulation", "abc"), ("--coarticulation", "1.5"), ("--coarticulation", "-1"),
    ("--vowel-gain", "a:1:1:1:1"), ("--vowel-gain", "1:1:1:1"), ("--vowel-gain", "-1:1:1:1:1"),
    ("--silence-threshold", "a:0.5"), ("--silence-threshold", "1.5:0.5"),
    ("--silence-threshold", "0.9:0.1"),
])
def test_option_value_error_states_the_reason_in_japanese(tmp_path, opt, value, capsys):
    # 引数エラーの理由は利用者に伝わる日本語で出す。検証子の実体がそのまま文字列化されると、
    # 実行ごとに変わるアドレスが理由の代わりに出て何も伝わらない。
    src = _touch(tmp_path / "in.wav")
    # 負数の書式に当てはまらない負の値(コロン区切りの複合値)は、単独で置くと argparse が
    # オプション名と解釈するので、等号で1トークンにして渡す。
    given = [f"{opt}={value}"] if value.startswith("-") else [opt, value]
    assert cli.main([src, *given, "--dry-run"]) == 2
    line = capsys.readouterr().err
    assert re.search(r"[぀-ヿ一-鿿]", line)
    assert "0x" not in line and " object at " not in line


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
    """0 は長尺分割の無効化を意味する有効値。"""
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


# --- 上書きガード -----------------------------------------------------------


def test_overwrite_guard_blocks_input_overwrite(tmp_path):
    """出力先が入力と同一パスのとき、--overwrite 無しは引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", src, "--dry-run"]) == 2


def test_overwrite_guard_allows_missing_same_path(tmp_path):
    """入力・出力が同一の未存在パスなら、出力先に既存ファイルが無いためガードは発火しない
    (後続の入力検証が別途コード1/4等で弾く)。"""
    missing = str(tmp_path / "missing.wav")
    assert cli.main([missing, "-o", missing, "--dry-run"]) != 2


def test_overwrite_flag_allows_input_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", src, "--overwrite", "--dry-run"]) == 0


def test_existing_separate_output_requires_overwrite(tmp_path):
    """保護対象は出力先の既存ファイル全般。入力と別パスの既存出力もガード対象。"""
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")  # 既存だが入力とは別パス
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_existing_separate_output_allowed_with_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")  # 既存だが入力とは別パス
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


# --- メタ操作(--version・--help)--------------------------------------------


def test_version_prints_and_exits_zero(capsys):
    from song2vmd import __version__

    rc = cli.main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "song2vmd" in out and __version__ in out


def test_help_lists_key_flags(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    for flag in ("--machine", "--describe", "--quiet", "--version", "--style",
                 "--separate-vocals", "--separator", "--recognizer-model-id", "--vowel-gain",
                 "--silence-threshold", "--no-n-morph"):
        assert flag in text
