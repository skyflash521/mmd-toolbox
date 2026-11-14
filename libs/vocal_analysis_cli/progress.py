from cli_progress_router import ProgressRouter
from vocal_analysis.quiet import STDERR_WRITE_LOCK

FRONT_STAGE_LABELS = {
    "load": "音声読み込み",
    "separate": "ボーカル分離",
    "recognize": "音素認識",
    "rms": "音量解析",
    "write": "書き出し",
}


def _labels(extra_labels):
    return {**FRONT_STAGE_LABELS, **(extra_labels or {})}


def stage_label(stage_id, extra_labels=None):
    return _labels(extra_labels).get(stage_id, stage_id)


def build_router(*, machine, quiet, emitter, stream, extra_labels=None):
    return ProgressRouter(
        machine=machine, quiet=quiet, emitter=emitter, stream=stream,
        labels=_labels(extra_labels),
        # Windows の実コンソールでは、fd 2 の差し替え中に別スレッドが書くと OSError(WinError 1)になる。
        write_lock=STDERR_WRITE_LOCK)
