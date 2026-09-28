def _gpu_memory_oversubscribed(fields):
    message = "GPUメモリの要求量が空き容量を超過しました"
    return message, (
        f"{message}(要求 {fields['reserved_mib']}MiB / "
        f"開始時空き {fields['free_at_start_mib']}MiB)。超過分は主記憶へ退避され処理が大幅に低速化します。"
        f"--device cpu での実行を検討してください"
    )


def _swap_detected(fields):
    message = "メモリ不足によるスワップを検出しました"
    return message, (
        f"{message}(スワップ増加 {fields['swap_increase_mib']}MiB・"
        f"本プロセス使用 {fields['process_rss_mib']}MiB)。処理が大幅に低速化します。"
        f"他のアプリケーションを終了してメモリを確保してください"
    )


def _cpu_only_torch(fields):
    message = "導入されている torch では GPU を扱えません"
    return message, f"{message}(torch {fields['torch_version']} は CPU 専用版)。CPU で処理します"


def _cuda_unavailable(fields):
    message = "この GPU で使えない CUDA 版の torch が入っています"
    return message, (
        f"{message}(torch {fields['torch_version']})。別の CUDA のバージョンで入れ直してください"
    )


_BUILDERS = {
    "gpu_memory_oversubscribed": _gpu_memory_oversubscribed,
    "swap_detected": _swap_detected,
    "cpu_only_torch": _cpu_only_torch,
    "cuda_unavailable": _cuda_unavailable,
}


def warning_texts(code, fields):
    """戻り値は (機械モードの warning イベント本文, 人間向け1行)。"""
    return _BUILDERS[code](fields)
