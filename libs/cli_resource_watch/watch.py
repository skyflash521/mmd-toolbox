from typing import NamedTuple

_MIB = 2**20

SWAP_INCREASE_FLOOR_MIB = 64
PROCESS_GROWTH_FLOOR_MIB = 64


def _default_gpu_probe():
    import torch

    if not torch.cuda.is_available():
        return None
    # mem_get_info は CUDA が未初期化なら初期化してから空きを返す。
    free, total = torch.cuda.mem_get_info()
    # max_memory_reserved はプロセス開始以降の最大値で、reset_peak_memory_stats を呼ぶまで下がらない。
    return free, total, torch.cuda.max_memory_reserved()


def _default_ram_probe():
    import psutil

    swap_bytes = psutil.swap_memory().used
    proc = psutil.Process()
    rss = 0
    for p in [proc] + proc.children(recursive=True):
        try:
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    return swap_bytes, rss, psutil.virtual_memory().total


class _GpuBaseline(NamedTuple):
    free_bytes: int
    total_bytes: int


class _RamBaseline(NamedTuple):
    swap_bytes: int
    rss_bytes: int


class _EmitFailed(Exception):
    def __init__(self, cause):
        super().__init__()
        self.cause = cause


class ResourceWatch:
    """emit_warning(code, fields) は警告が成立したときに呼ばれ、それが送出した例外は check の
    呼び出し元へそのまま届く。

    gpu_probe は (空き VRAM, VRAM 総量, GPU メモリ予約量の最大値) を、ram_probe は
    (スワップ使用量, 本プロセスと全子孫の常駐メモリの合計, 搭載 RAM) を、いずれもバイト単位で返す。
    gpu_probe は GPU を使えないとき None を返す。probe が None を返すか例外を送出すると、その側の
    判定は以後止まる。
    """

    def __init__(self, emit_warning, *, gpu_probe=None, ram_probe=None):
        self._emit = emit_warning
        self._gpu_probe = gpu_probe if gpu_probe is not None else _default_gpu_probe
        self._ram_probe = ram_probe if ram_probe is not None else _default_ram_probe
        self._fired = set()
        self._gpu_baseline = None
        self._gpu_disabled = False
        self._ram_baseline = None
        self._ram_disabled = False

    def check(self, stage_id):
        try:
            self._check_gpu(stage_id)
        except _EmitFailed as e:
            raise e.cause from None
        except Exception:
            self._gpu_disabled = True
        try:
            self._check_ram(stage_id)
        except _EmitFailed as e:
            raise e.cause from None
        except Exception:
            self._ram_disabled = True

    def _fire(self, code, fields):
        self._fired.add(code)
        try:
            self._emit(code, fields)
        except Exception as e:
            raise _EmitFailed(e) from None

    def _check_gpu(self, stage_id):
        if self._gpu_disabled or "gpu_memory_oversubscribed" in self._fired:
            return
        probed = self._gpu_probe()
        if probed is None:
            self._gpu_disabled = True
            return
        free_bytes, total_bytes, reserved_bytes = probed
        if self._gpu_baseline is None:
            self._gpu_baseline = _GpuBaseline(free_bytes, total_bytes)
            return
        baseline = self._gpu_baseline
        if reserved_bytes > baseline.free_bytes:
            self._fire(
                "gpu_memory_oversubscribed",
                {"stage": stage_id, "reserved_mib": reserved_bytes // _MIB,
                 "free_at_start_mib": baseline.free_bytes // _MIB,
                 "total_mib": baseline.total_bytes // _MIB},
            )

    def _check_ram(self, stage_id):
        if self._ram_disabled or "swap_detected" in self._fired:
            return
        swap_bytes, rss_bytes, total_bytes = self._ram_probe()
        if self._ram_baseline is None:
            self._ram_baseline = _RamBaseline(swap_bytes, rss_bytes)
            return
        swap_increase_bytes = swap_bytes - self._ram_baseline.swap_bytes
        if (swap_increase_bytes >= SWAP_INCREASE_FLOOR_MIB * _MIB
                and rss_bytes - self._ram_baseline.rss_bytes >= PROCESS_GROWTH_FLOOR_MIB * _MIB):
            self._fire(
                "swap_detected",
                {"stage": stage_id, "swap_increase_mib": swap_increase_bytes // _MIB,
                 "process_rss_mib": rss_bytes // _MIB, "ram_total_mib": total_bytes // _MIB},
            )


class ProgressWithResourceCheck:
    """reporter は stage(stage_id, **kwargs)・close()・summary(message) を持つこと。"""

    def __init__(self, reporter, watch):
        self._reporter = reporter
        self._watch = watch
        self._last_stage = None

    def stage(self, stage_id, **kwargs):
        if stage_id != self._last_stage:
            self._last_stage = stage_id
            self._watch.check(stage_id)
        self._reporter.stage(stage_id, **kwargs)

    def close(self):
        self._reporter.close()

    def summary(self, message):
        self._reporter.summary(message)
