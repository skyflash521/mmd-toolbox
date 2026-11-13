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


def test_default_output_resolves_to_input_stem_vpr_and_trips_overwrite_guard(tmp_path):
    src = _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vpr")
    assert cli.main([src, "--dry-run"]) == 2


def test_default_output_does_not_collide_with_other_extension(tmp_path):
    src = _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vmd")
    assert cli.main([src, "--dry-run"]) == 0


def test_abbreviated_flag_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--over", "--dry-run"]) == 2


@pytest.mark.parametrize("value", ["0.01"])
def test_tempo_from_the_storable_minimum_upward_is_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--tempo", value, "--dry-run"]) == 0


@pytest.mark.parametrize("value", ["0.009"])
def test_tempo_below_the_storable_minimum_or_not_finite_is_arg_error(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, f"--tempo={value}", "--dry-run"]) == 2


@pytest.mark.parametrize("value", ["1/1", "4/128"])
def test_time_signature_with_a_power_of_two_denominator_is_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--time-signature", value, "--dry-run"]) == 0


@pytest.mark.parametrize("value", [
    pytest.param("4/256", id="beat_not_whole_ticks"),
    pytest.param("0/4", id="numerator_not_positive"),
    pytest.param("4/0", id="denominator_not_positive"),
])
def test_time_signature_outside_the_accepted_set_is_arg_error(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--time-signature", value, "--dry-run"]) == 2
