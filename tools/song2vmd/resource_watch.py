"""song2vmd の資源逼迫・GPU 構成の警告の文言。

何を観測しどの事実で警告が成立するかは共有の cli_resource_watch が持つ。ここが持つのは、
成立した警告のコードと観測値から、機械モードの warning イベント本文と、観測値を併記した
人間向け1行を組み立てる部分だけ(勧める対処が song2vmd のオプション名に依存するため)。
"""


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
    """警告のコードと観測値から (機械モードの本文, 人間向け1行) を組み立てる。"""
    return _BUILDERS[code](fields)
