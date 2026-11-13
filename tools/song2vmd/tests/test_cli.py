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


def test_default_output_replaces_input_extension_with_vmd(tmp_path):
    src = _touch(tmp_path / "song.mp3")
    rc = cli.main([src])
    assert rc == 0
    assert (tmp_path / "song.vmd").is_file()


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


def test_abbreviated_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--over", "--dry-run"]) == 2


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


def test_forced_aligner_unknown_choice_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--forced-aligner", "unknown-aligner", "--dry-run"]) == 2


def test_intensity_curve_non_numeric_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--intensity-curve", "abc", "--dry-run"]) == 2


@pytest.mark.parametrize("opt", ["--anticipation", "--min-hold"])
def test_int_option_non_numeric_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "abc", "--dry-run"]) == 2


def test_silence_threshold_on_not_below_off_states_the_reason_in_japanese_without_object_repr(tmp_path, capsys):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--silence-threshold", "0.9:0.1", "--dry-run"]) == 2
    line = capsys.readouterr().err
    assert re.search(r"[぀-ヿ一-鿿]", line)
    assert "0x" not in line and " object at " not in line


@pytest.mark.parametrize("opt", ["--anticipation", "--min-hold"])
def test_int_option_negative_is_arg_error(tmp_path, opt):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, "-1", "--dry-run"]) == 2


def test_open_max_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--open-max", "-0.1", "--dry-run"]) == 2


def test_open_max_bounds_are_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--open-max", "0.0", "--dry-run"]) == 0
    assert cli.main([src, "--open-max", "1.0", "--dry-run"]) == 0


def test_intensity_curve_zero_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--intensity-curve", "0", "--dry-run"]) == 2


def test_vowel_gain_with_six_elements_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--vowel-gain", "1:1:1:1:1:1", "--dry-run"]) == 2


def test_vowel_gain_valid_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--vowel-gain", "1.1:0.9:1.0:1.0:1.2", "--dry-run"]) == 0


@pytest.mark.parametrize("text", ["0.1", "0.1:0.2:0.3", "-0.1:0.2", "0.5:1.1"])
def test_silence_threshold_malformed_is_arg_error(tmp_path, text):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, f"--silence-threshold={text}", "--dry-run"]) == 2


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
