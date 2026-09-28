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
    monkeypatch.setattr(cli._pipeline, "run", lambda *a, **k: _stub_pipeline_result())


def test_parses_full_option_set_in_dry_run(tmp_path):
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
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


def test_dry_run_without_output_option_writes_nothing_next_to_input(tmp_path):
    src = _touch(tmp_path / "song.wav")
    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "song.vmd").exists()


def test_default_output_replaces_input_extension_with_vmd(tmp_path):
    src = _touch(tmp_path / "song.mp3")
    rc = cli.main([src])
    assert rc == 0
    assert (tmp_path / "song.vmd").is_file()


def test_n_morph_defaults_to_off(tmp_path):
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


def test_preset_dependent_options_default_to_none(tmp_path):
    src = _touch(tmp_path / "in.wav")
    parser = cli._build_parser()
    args = parser.parse_args([src, "--dry-run"])
    assert args.open_max is None
    assert args.coarticulation is None
    assert args.anticipation is None
    assert args.min_hold is None


def test_unknown_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--bogus"]) == 2


def test_abbreviated_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--over", "--dry-run"]) == 2


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
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-id", "anything/goes", "--dry-run"]) == 0


def test_recognizer_model_revision_without_model_id_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-revision", "abc123", "--dry-run"]) == 2


def test_explicit_recognizer_retry_flag_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-retry", "--dry-run"]) == 0


def test_forced_aligner_default_needs_no_sofa_args(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--dry-run"]) == 0


def test_forced_aligner_sofa_without_sofa_python_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-root", "/sofa", "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 2


def test_forced_aligner_sofa_without_sofa_root_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 2


def test_forced_aligner_sofa_without_checkpoint_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa", "--dry-run",
    ]) == 2


def test_forced_aligner_sofa_with_three_sofa_paths_and_default_timeout_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 0


def test_forced_aligner_sofa_timeout_zero_is_arg_error(tmp_path):
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
def test_option_value_error_states_the_reason_in_japanese_without_object_repr(tmp_path, opt, value, capsys):
    src = _touch(tmp_path / "in.wav")
    # argparse は負数の書式に当てはまらない "-" 始まりの値をオプション名と解釈する。
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
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--silence-threshold", "0.10:0.06", "--dry-run"]) == 2
    assert cli.main([src, "--silence-threshold", "0.10:0.10", "--dry-run"]) == 2


def test_model_name_over_20_bytes_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "x" * 21, "--dry-run"]) == 2


def test_model_name_of_eleven_double_byte_characters_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "あ" * 11, "--dry-run"]) == 2


def test_model_name_of_ten_double_byte_characters_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "あ" * 10, "--dry-run"]) == 0


def test_model_name_non_cp932_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "\U0001F600", "--dry-run"]) == 2


def test_model_name_at_20_byte_limit_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--model-name", "x" * 20, "--dry-run"]) == 0


def test_overwrite_guard_blocks_output_equal_to_existing_input(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", src, "--dry-run"]) == 2


def test_overwrite_guard_allows_missing_same_path(tmp_path):
    missing = str(tmp_path / "missing.wav")
    assert cli.main([missing, "-o", missing, "--dry-run"]) != 2


def test_overwrite_flag_allows_input_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", src, "--overwrite", "--dry-run"]) == 0


def test_existing_separate_output_requires_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_existing_separate_output_allowed_with_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


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


def test_main_installs_the_sigbreak_handler(tmp_path, monkeypatch):
    installed = []
    monkeypatch.setattr(cli, "install_sigbreak_handler", lambda: installed.append(True))
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--dry-run"]) == 0
    assert installed == [True]
