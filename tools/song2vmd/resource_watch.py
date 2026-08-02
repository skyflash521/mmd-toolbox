"""song2vmd 資源逼迫の警告。

VRAM・RAMの逼迫が実際に起きた事実だけを観測して警告する(モデル・曲・機材ごとの需要見積りを
使わない)。判定は処理段の切り替わり時に行い、同一 code は1実行1回。観測手段が使えない環境では
該当側を黙って無効にする(警告が出ないだけで、処理・終了コード・出力へ影響しない副作用専用)。
"""

_MIB = 2**20

# スワップ増分のノイズ床と帰属条件(自プロセス増加)の下限。他アプリの背景動作によるゆらぎを
# 弾くためだけの定数で、判定結果を左右する需要見積りではない(初期値。実データで調整する)。
SWAP_INCREASE_FLOOR_MIB = 64
PROCESS_GROWTH_FLOOR_MIB = 64


def _default_gpu_probe():
    """(空きVRAM, VRAM総量, CUDAアロケータ要求量の最大値) をバイト値で返す。

    GPU(CUDA)が使えない環境では None(GPU側の観測を無効化する)。初回呼び出しは
    mem_get_info が CUDA を初期化するため、処理開始時点の空きの基準を取る役割を兼ねる。
    判定を切り捨て前のバイト値で行うため、MiB への丸めはしない(丸めは表示側)。
    """
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free, total, torch.cuda.max_memory_reserved()


def _default_ram_probe():
    """(スワップ使用量, 本プロセスツリーの使用量, 搭載RAM) をバイト値で返す。

    psutil が無い環境では ImportError がそのまま上がり、呼び出し側(ResourceWatch.check)が
    RAM側の観測を無効化する。ツリーの使用量は自プロセスと子孫(SOFAサブプロセス等)の合計。
    判定を切り捨て前のバイト値で行うため、MiB への丸めはしない(丸めは表示側)。
    """
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


class ResourceWatch:
    """段の切り替わりごとに逼迫の事実を判定し、成立時に emit_warning を code ごとに1回だけ呼ぶ。

    emit_warning(code, message, human_text, fields): message は機械モードの warning イベント本文、
    human_text は観測値併記の人間向け1行本文、fields はイベントへ追加する観測値。観測・判定中の
    例外は握りつぶして該当側を無効化する(副作用専用の保証)。
    """

    def __init__(self, emit_warning, *, gpu_probe=None, ram_probe=None):
        self._emit = emit_warning
        self._gpu_probe = gpu_probe if gpu_probe is not None else _default_gpu_probe
        self._ram_probe = ram_probe if ram_probe is not None else _default_ram_probe
        self._fired = set()
        self._gpu_start = None  # (free_at_start_bytes, total_bytes)。False は無効化
        self._ram_start = None  # (swap_start_bytes, rss_start_bytes)。False は無効化

    def check(self, stage_id):
        try:
            self._check_gpu(stage_id)
        except Exception:
            self._gpu_start = False
        try:
            self._check_ram(stage_id)
        except Exception:
            self._ram_start = False

    def _fire(self, code, message, human_text, fields):
        self._fired.add(code)
        self._emit(code, message, human_text, fields)

    def _check_gpu(self, stage_id):
        if self._gpu_start is False or "gpu_memory_oversubscribed" in self._fired:
            return
        probed = self._gpu_probe()
        if probed is None:
            self._gpu_start = False
            return
        free_bytes, total_bytes, reserved_bytes = probed
        if self._gpu_start is None:
            self._gpu_start = (free_bytes, total_bytes)
            return
        free_at_start_bytes, total = self._gpu_start
        # 判定は切り捨て前のバイト値で行う(MiB へ丸めてから比較すると1MiB未満の超過を
        # 取りこぼし、しきい値なしの算術判定に暗黙の許容を持ち込んでしまう)。
        if reserved_bytes > free_at_start_bytes:
            reserved_mib = reserved_bytes // _MIB
            free_at_start_mib = free_at_start_bytes // _MIB
            self._fire(
                "gpu_memory_oversubscribed",
                "GPUメモリの要求量が空き容量を超過しました",
                f"GPUメモリの要求量が空き容量を超過しました(要求 {reserved_mib}MiB / "
                f"開始時空き {free_at_start_mib}MiB)。超過分は主記憶へ退避され処理が大幅に低速化します。"
                f"--device cpu での実行を検討してください",
                {"stage": stage_id, "reserved_mib": reserved_mib,
                 "free_at_start_mib": free_at_start_mib, "total_mib": total // _MIB},
            )

    def _check_ram(self, stage_id):
        if self._ram_start is False or "swap_detected" in self._fired:
            return
        swap_bytes, rss_bytes, total_bytes = self._ram_probe()
        if self._ram_start is None:
            self._ram_start = (swap_bytes, rss_bytes)
            return
        swap_start_bytes, rss_start_bytes = self._ram_start
        swap_increase_bytes = swap_bytes - swap_start_bytes
        if (swap_increase_bytes >= SWAP_INCREASE_FLOOR_MIB * _MIB
                and rss_bytes - rss_start_bytes >= PROCESS_GROWTH_FLOOR_MIB * _MIB):
            swap_increase_mib = swap_increase_bytes // _MIB
            rss_mib = rss_bytes // _MIB
            self._fire(
                "swap_detected",
                "メモリ不足によるスワップを検出しました",
                f"メモリ不足によるスワップを検出しました(スワップ増加 {swap_increase_mib}MiB・"
                f"本プロセス使用 {rss_mib}MiB)。処理が大幅に低速化します。"
                f"他のアプリケーションを終了してメモリを確保してください",
                {"stage": stage_id, "swap_increase_mib": swap_increase_mib,
                 "process_rss_mib": rss_mib, "ram_total_mib": total_bytes // _MIB},
            )


class ProgressWithResourceCheck:
    """進捗の報告先への委譲ラッパー。段の切り替わり(stage id の変化)時に ResourceWatch.check を
    表示より先に走らせる(人間向け警告の発行側がライブ行を消してから書けるようにするための順序)。
    同一段内の進行更新(ダウンロード進捗等)では判定しない。
    """

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
