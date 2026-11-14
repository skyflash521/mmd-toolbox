from dataclasses import dataclass

from .progress import stage_label


@dataclass(frozen=True)
class FrontStageFailure:
    code: str
    message: str
    exit_code: int
    field: str | None = None
    path: str | None = None
    stage: str | None = None


def front_stage_failure(exc, *, keep_intermediate_dir=None, extra_labels=None):
    from vocal_analysis.front_stage import (
        IntermediateReadError,
        IntermediateWriteError,
        StageExecutionError,
    )
    from vocal_analysis.io import AudioLoadError
    from vocal_analysis.phonemes import RecognitionError
    from vocal_analysis.separator import SeparationError

    if isinstance(exc, IntermediateReadError):
        return FrontStageFailure("not_audio", str(exc), 1, path=str(exc.path))
    if isinstance(exc, AudioLoadError):
        exit_code = 4 if exc.reason == "decoder_missing" else 1
        return FrontStageFailure(exc.reason, str(exc), exit_code, field="input")
    if isinstance(exc, StageExecutionError):
        label = stage_label(exc.stage, extra_labels)
        return FrontStageFailure(
            "stage_failed", f"{label}に失敗しました: {exc}", 4, stage=exc.stage)
    if isinstance(exc, SeparationError):
        return FrontStageFailure("stage_failed", str(exc), 4, stage="separate")
    if isinstance(exc, RecognitionError):
        return FrontStageFailure("stage_failed", str(exc), 4, stage="recognize")
    if isinstance(exc, IntermediateWriteError):
        return FrontStageFailure(
            "write_failed", str(exc), 3, field="--keep-intermediate", path=keep_intermediate_dir)
    return None
