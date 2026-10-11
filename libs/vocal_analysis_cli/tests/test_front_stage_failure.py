from pathlib import Path

import pytest

from vocal_analysis.front_stage import (
    IntermediateReadError,
    IntermediateWriteError,
    StageExecutionError,
)
from vocal_analysis.io import AudioLoadError
from vocal_analysis.phonemes import RecognitionError
from vocal_analysis.separator import SeparationError
from vocal_analysis_cli import FrontStageFailure, front_stage_failure


def _fields(failure):
    return (failure.code, failure.message, failure.exit_code,
            failure.field, failure.path, failure.stage)


def test_intermediate_read_error_is_not_audio_exit_1_with_path():
    exc = IntermediateReadError("読めない", path=Path("work") / "a.wav")

    failure = front_stage_failure(exc)

    assert isinstance(failure, FrontStageFailure)
    assert _fields(failure) == ("not_audio", "読めない", 1, None, str(Path("work") / "a.wav"), None)


def test_audio_load_error_not_audio_is_exit_1_on_input_field():
    exc = AudioLoadError("音声として読めない", reason="not_audio")

    assert _fields(front_stage_failure(exc)) == (
        "not_audio", "音声として読めない", 1, "input", None, None)


def test_audio_load_error_decoder_missing_is_exit_4_on_input_field():
    exc = AudioLoadError("ffmpeg が無い", reason="decoder_missing")

    assert _fields(front_stage_failure(exc)) == (
        "decoder_missing", "ffmpeg が無い", 4, "input", None, None)


def test_stage_execution_error_names_stage_by_shared_label():
    exc = StageExecutionError("メモリ不足", stage="separate")

    assert _fields(front_stage_failure(exc)) == (
        "stage_failed", "ボーカル分離に失敗しました: メモリ不足", 4, None, None, "separate")


def test_stage_execution_error_with_unknown_stage_uses_stage_id_as_label():
    exc = StageExecutionError("壊れた", stage="notes")

    failure = front_stage_failure(exc)

    assert failure.message == "notesに失敗しました: 壊れた"
    assert failure.stage == "notes"


def test_stage_execution_error_uses_extra_labels_for_tool_specific_stage():
    exc = StageExecutionError("壊れた", stage="events")

    failure = front_stage_failure(exc, extra_labels={"events": "口形イベント確定"})

    assert failure.message == "口形イベント確定に失敗しました: 壊れた"
    assert failure.stage == "events"


def test_separation_error_is_stage_failed_at_separate_with_message_as_is():
    assert _fields(front_stage_failure(SeparationError("分離できない"))) == (
        "stage_failed", "分離できない", 4, None, None, "separate")


def test_recognition_error_is_stage_failed_at_recognize_with_message_as_is():
    assert _fields(front_stage_failure(RecognitionError("認識できない"))) == (
        "stage_failed", "認識できない", 4, None, None, "recognize")


def test_intermediate_write_error_is_write_failed_exit_3_with_keep_intermediate_dir():
    failure = front_stage_failure(
        IntermediateWriteError("書けない"), keep_intermediate_dir="out.vmd.intermediate")

    assert _fields(failure) == (
        "write_failed", "書けない", 3, "--keep-intermediate", "out.vmd.intermediate", None)


def test_intermediate_write_error_without_dir_has_no_path():
    failure = front_stage_failure(IntermediateWriteError("書けない"))

    assert failure.field == "--keep-intermediate"
    assert failure.path is None


@pytest.mark.parametrize("exc", [
    ValueError("x"), OSError("x"), RuntimeError("x"), KeyboardInterrupt(), Exception("x"),
])
def test_unrelated_exception_maps_to_none(exc):
    assert front_stage_failure(exc) is None


def test_extra_labels_do_not_change_other_mappings():
    failure = front_stage_failure(
        SeparationError("分離できない"), extra_labels={"separate": "別名"})

    assert failure.message == "分離できない"
