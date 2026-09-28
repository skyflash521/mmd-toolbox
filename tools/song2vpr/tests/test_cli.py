import re
import types

import pytest

from song2vpr import cli

from .support import front_stage_result


def _touch(path):
    path.write_bytes(b"")
    return str(path)


@pytest.fixture(autouse=True)
def _stub_front_stage_with_short_silence(monkeypatch):
    monkeypatch.setattr(cli, "_pipeline",
                        types.SimpleNamespace(run=lambda *a, **k: front_stage_result()))


def test_parses_full_option_set(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vpr"
    assert cli.main([
        src,
        "-o", str(out),
        "--separate-vocals", "always",
        "--separator", "audio-separator-htdemucs-ft",
        "--recognizer-model-id", "openai/whisper-medium",
        "--recognizer-model-revision", "abc123",
        "--recognizer-retry",
        "--forced-aligner", "wav2vec2-ctc-forcedalign",
        "--english-katakana-method", "arpakana",
        "--device", "cpu",
        "--max-duration", "120",
        "--keep-intermediate",
        "--quiet",
        "-v",
        "--dry-run",
    ]) == 0


def test_dry_run_writes_no_output(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vpr"
    assert cli.main([src, "-o", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_default_output_resolves_to_input_stem_vpr_and_trips_overwrite_guard(tmp_path):
    src = _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vpr")
    assert cli.main([src, "--dry-run"]) == 2


def test_default_output_does_not_collide_with_other_extension(tmp_path):
    src = _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vmd")
    assert cli.main([src, "--dry-run"]) == 0


def test_unknown_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--bogus"]) == 2


def test_missing_positional_is_arg_error():
    assert cli.main([]) == 2


def test_abbreviated_flag_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--over", "--dry-run"]) == 2


def test_unknown_separate_vocals_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separate-vocals", "sometimes", "--dry-run"]) == 2


def test_unknown_separator_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separator", "nope", "--dry-run"]) == 2


def test_unknown_forced_aligner_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--forced-aligner", "unknown-aligner", "--dry-run"]) == 2


def test_unknown_device_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--device", "gpu", "--dry-run"]) == 2


def test_recognizer_model_revision_without_model_id_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-revision", "abc123", "--dry-run"]) == 2


@pytest.mark.parametrize("value", ["always", "never"])
def test_separate_vocals_choices_are_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separate-vocals", value, "--dry-run"]) == 0


@pytest.mark.parametrize("value", ["auto", "cpu"])
def test_device_choices_are_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--device", value, "--dry-run"]) == 0


def test_unknown_english_katakana_method_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--english-katakana-method", "nope", "--dry-run"]) == 2


def test_no_recognizer_retry_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--no-recognizer-retry", "--dry-run"]) == 0


def test_sofa_timeout_positive_value_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--sofa-timeout", "60", "--dry-run",
    ]) == 0


def test_recognizer_model_id_accepts_any_string(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-id", "anything/goes", "--dry-run"]) == 0


@pytest.mark.parametrize("missing", ["--sofa-python", "--sofa-root", "--sofa-checkpoint"])
def test_forced_aligner_sofa_missing_required_arg_is_arg_error(tmp_path, missing):
    src = _touch(tmp_path / "in.wav")
    given = {"--sofa-python": "/venv/python", "--sofa-root": "/sofa", "--sofa-checkpoint": "/ckpt.ckpt"}
    del given[missing]
    argv = [src, "--forced-aligner", "sofa-forcedalign", "--dry-run"]
    for name, value in given.items():
        argv += [name, value]
    assert cli.main(argv) == 2


def test_forced_aligner_sofa_with_all_required_args_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 0


def test_max_duration_zero_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--max-duration", "0", "--dry-run"]) == 0


def test_max_duration_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--max-duration=-1", "--dry-run"]) == 2


@pytest.mark.parametrize("value", ["0.01", "300"])
def test_tempo_from_the_storable_minimum_upward_is_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--tempo", value, "--dry-run"]) == 0


@pytest.mark.parametrize("value", ["0", "0.009", "-120", "abc", "nan", "inf", "-inf"])
def test_tempo_below_the_storable_minimum_or_not_finite_is_arg_error(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, f"--tempo={value}", "--dry-run"]) == 2


@pytest.mark.parametrize("value", ["3/4", "6/8", "1/1", "4/128"])
def test_time_signature_with_a_power_of_two_denominator_is_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--time-signature", value, "--dry-run"]) == 0


@pytest.mark.parametrize("value", [
    pytest.param("4/3", id="denominator_not_power_of_two"),
    pytest.param("4/256", id="beat_not_whole_ticks"),
    pytest.param("0/4", id="numerator_not_positive"),
    pytest.param("4/0", id="denominator_not_positive"),
    pytest.param("4", id="not_numerator_slash_denominator"),
    pytest.param("a/4", id="numerator_not_integer"),
])
def test_time_signature_outside_the_accepted_set_is_arg_error(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--time-signature", value, "--dry-run"]) == 2


@pytest.mark.parametrize("opt, value", [
    ("--max-duration", "abc"), ("--max-duration", "nan"), ("--sofa-timeout", "abc"),
    ("--sofa-timeout", "0"),
])
def test_option_value_error_states_the_reason_in_japanese_without_object_repr(tmp_path, opt, value, capsys):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, value, "--dry-run"]) == 2
    line = capsys.readouterr().err
    assert re.search(r"[぀-ヿ一-鿿]", line)
    assert "0x" not in line and " object at " not in line


def test_existing_output_requires_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out]) == 2


def test_existing_output_allowed_with_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


def test_overwrite_guard_is_evaluated_in_dry_run(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_missing_output_path_is_not_blocked(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", str(tmp_path / "out.vpr"), "--dry-run"]) == 0


@pytest.mark.parametrize("extra", [[], ["--overwrite"]])
def test_output_directory_is_rejected_regardless_of_overwrite(tmp_path, extra):
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "dir"
    outdir.mkdir()
    assert cli.main([src, "-o", str(outdir), *extra, "--dry-run"]) == 2


def test_help_lists_key_flags(capsys):
    assert cli.main(["--help"]) == 0
    text = capsys.readouterr().out
    for flag in ("--output", "--overwrite", "--lyrics", "--tempo", "--time-signature",
                 "--dry-run", "--keep-intermediate", "--verbose",
                 "--quiet", "--machine", "--describe", "--version",
                 "--separate-vocals", "--separator", "--recognizer-model-id",
                 "--recognizer-model-revision", "--recognizer-retry", "--forced-aligner",
                 "--sofa-python", "--sofa-root", "--sofa-checkpoint", "--sofa-timeout",
                 "--english-katakana-method", "--device", "--max-duration"):
        assert flag in text
