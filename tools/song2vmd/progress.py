from cli_progress_router import ProgressRouter
from vocal_analysis.quiet import STDERR_WRITE_LOCK

STAGE_LABELS = {
    "load": "音声読み込み",
    "separate": "ボーカル分離",
    "recognize": "音素認識",
    "rms": "音量解析",
    "events": "口形イベント確定",
    "generate": "モーフ生成",
    "write": "書き出し",
}


def stage_label(stage_id):
    return STAGE_LABELS.get(stage_id, stage_id)


def build_router(*, machine, quiet, emitter, stream):
    return ProgressRouter(
        machine=machine, quiet=quiet, emitter=emitter, stream=stream,
        labels=STAGE_LABELS,
        # Windows の実コンソールでは、fd 2 の差し替え中に別スレッドが書くと OSError(WinError 1)になる。
        write_lock=STDERR_WRITE_LOCK)
